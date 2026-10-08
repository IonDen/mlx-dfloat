"""Group-size arithmetic and a phase-structured fit estimate (predictions, labelled as such)."""

from collections.abc import Container, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_dfloat.integrate.names import NameMap


def decoded_bytes(group: Any) -> int:
    """Bytes of BF16 a group decodes to: two per element, one element per sign-mantissa byte."""
    return 2 * int(group.tensors["sign_mantissa"].nbytes)


def largest_decoded_bytes(
    ckpt: Any, name_map: NameMap, *, skip: Container[str] = frozenset()
) -> dict[str, int]:
    """The largest decoded group per block kind, from the checkpoint headers (nothing is loaded).

    Groups named in ``skip`` (non-block groups, which have no kind) are left out.
    """
    largest: dict[str, int] = {}
    for name, group in ckpt.groups.items():
        if name in skip:
            continue
        kind = name_map.kind_of(name)
        largest[kind] = max(largest.get(kind, 0), decoded_bytes(group))
    return largest


FIT_RESERVE_BYTES = 2 * 1024**3
"""The fit rule's reserve for the OS below the recommended working set."""


def budget_for(recommended_bytes: int, *, reserve_bytes: int = FIT_RESERVE_BYTES) -> int:
    """The fit rule's budget on a Mac whose recommended working set is ``recommended_bytes``."""
    return recommended_bytes - reserve_bytes


def budget_bytes(*, reserve_bytes: int = FIT_RESERVE_BYTES) -> int:
    """The fit rule's budget: the device's recommended working set minus a reserve for the OS."""
    return budget_for(
        int(mx.device_info()["max_recommended_working_set_size"]), reserve_bytes=reserve_bytes
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class FitEstimate:
    """A predicted peak per phase against a budget; a prediction, not a measurement."""

    phases: dict[str, int]
    peak_phase: str
    peak_bytes: int
    budget_bytes: int

    @property
    def fits(self) -> bool:
        """Whether the peak phase stays within the budget."""
        return self.peak_bytes <= self.budget_bytes


def fit_estimate(phases: Mapping[str, Mapping[str, int]], *, budget_bytes: int) -> FitEstimate:
    """Sum each phase's terms; the peak is the largest phase."""
    totals = {phase: sum(terms.values()) for phase, terms in phases.items()}
    peak_phase = max(totals, key=totals.__getitem__)
    return FitEstimate(
        phases=totals,
        peak_phase=peak_phase,
        peak_bytes=totals[peak_phase],
        budget_bytes=budget_bytes,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CallPlan:
    """What one generate call will do about memory: the cache limit, the estimate, and whether the set is dropped before the VAE decode."""

    cache_limit: int
    estimate: FitEstimate
    drop_set_before_vae: bool


def safetensors_bytes(root: Path, *subdirs: str) -> int:
    """The size of every ``*.safetensors`` file directly under each ``root/subdir`` (missing subdirs count zero)."""
    return sum(
        p.stat().st_size
        for sub in subdirs
        for p in (root / sub).glob("*.safetensors")
        if p.is_file()
    )
