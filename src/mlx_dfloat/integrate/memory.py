"""Group-size arithmetic and a phase-structured fit estimate (predictions, labelled as such)."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from mlx_dfloat.integrate.names import NameMap


def decoded_bytes(group: Any) -> int:
    """Bytes of BF16 a group decodes to: two per element, one element per sign-mantissa byte."""
    return 2 * int(group.tensors["sign_mantissa"].nbytes)


def largest_decoded_bytes(ckpt: Any, name_map: NameMap) -> dict[str, int]:
    """The largest decoded group per block kind, from the checkpoint headers (nothing is loaded)."""
    largest: dict[str, int] = {}
    for name, group in ckpt.groups.items():
        kind = name_map.kind_of(name)
        largest[kind] = max(largest.get(kind, 0), decoded_bytes(group))
    return largest


def budget_bytes(*, reserve_bytes: int = 2 * 1024**3) -> int:
    """The fit rule's budget: the device's recommended working set minus a reserve for the OS."""
    return int(mx.device_info()["max_recommended_working_set_size"]) - reserve_bytes


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
