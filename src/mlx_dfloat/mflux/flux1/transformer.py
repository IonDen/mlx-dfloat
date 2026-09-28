"""mflux's FLUX.1 Transformer with the block seam composed in front of its two per-block hooks."""

from functools import cache
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError
from mlx_dfloat.format import DF11Checkpoint
from mlx_dfloat.integrate import seam
from mlx_dfloat.integrate.coverage import check_extras_cover, extras_plan, read_extra
from mlx_dfloat.integrate.names import NameMap, Shapes
from mlx_dfloat.integrate.placeholders import PLACEHOLDER, install_placeholders
from mlx_dfloat.integrate.providers import WeightProvider
from mlx_dfloat.mflux import require_mflux
from mlx_dfloat.mflux.flux1.names import (
    DOUBLE_PREFIX,
    DROPPED_EXTRAS,
    SINGLE_PREFIX,
    check_flux_groups,
    flux_name_map,
)

MAX_BUILD_ACTIVE_BYTES = 2 * 1024**3


class SeamMixin:
    """Overrides mflux's ``_apply_joint_transformer_block`` / ``_apply_single_transformer_block``.

    Compose it in front of mflux's ``Transformer`` (``seam_transformer_class``) or a fake with the same hooks.
    Call ``attach`` before the first step. A step is ``out = transformer(...)``; with ``verify_in_call`` the
    decode status words are checked before the call returns, otherwise call ``verify_step()`` after the
    step's final eval.
    """

    _seam: seam.SeamState

    def attach(
        self,
        provider: WeightProvider,
        shapes: Shapes,
        *,
        eval_policy: str = "per-block",
        tracer: seam.TraceHook | None = None,
        verify_in_call: bool = False,
    ) -> None:
        """Bind the provider, the block shapes (from ``install_placeholders``) and the eval policy."""
        self._seam = seam.attach_state(
            provider, shapes, eval_policy=eval_policy, tracer=tracer, verify_in_call=verify_in_call
        )

    def _state(self) -> seam.SeamState:
        try:
            return self._seam
        except AttributeError as exc:
            raise DFloatIntegrationError(
                "call attach(provider, shapes) before running a step"
            ) from exc

    def verify_step(self) -> None:
        """Run the provider's deferred checks; call it after the step's final ``mx.eval``."""
        seam.verify_step(self._state())

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run mflux's step; a failed step leaves no stale seam state."""
        state = self._state()
        seam.begin_step(state)
        try:
            out = super().__call__(*args, **kwargs)  # type: ignore[misc]
            return seam.end_step(state, out)
        except BaseException:
            seam.abort_step(state)
            raise

    def _apply_joint_transformer_block(self, idx: int, block: Any, **kwargs: Any) -> Any:
        return seam.run_block(
            self._state(),
            f"{DOUBLE_PREFIX}.{idx}",
            block,
            lambda: super(SeamMixin, self)._apply_joint_transformer_block(  # type: ignore[misc]
                idx=idx, block=block, **kwargs
            ),
        )

    def _apply_single_transformer_block(self, idx: int, block: Any, **kwargs: Any) -> Any:
        return seam.run_block(
            self._state(),
            f"{SINGLE_PREFIX}.{idx}",
            block,
            lambda: super(SeamMixin, self)._apply_single_transformer_block(  # type: ignore[misc]
                idx=idx, block=block, **kwargs
            ),
        )


@cache
def seam_transformer_class() -> type:
    """``SeamMixin`` composed in front of mflux's ``Transformer`` (imports mflux).

    Raises:
        DFloatDependencyError: The optional ``mflux`` extra is not installed.
    """
    require_mflux()
    from mflux.models.flux.model.flux_transformer.transformer import Transformer

    return type("SeamTransformer", (SeamMixin, Transformer), {})


def build_transformer(
    model_config: Any,
    ckpt: DF11Checkpoint,
    *,
    name_map: NameMap | None = None,
    n_double: int | None = None,
    n_single: int | None = None,
) -> tuple[Any, Shapes]:
    """Construct mflux's ``Transformer`` (seamed) with placeholders for the matrices and the extras loaded.

    Block counts come from the checkpoint's groups, or ``n_double``/``n_single`` for a reduced-depth build
    (each between zero and the checkpoint's own count). Every block matrix is a placeholder and every other
    parameter comes from the checkpoint's extras; the coverage of both is asserted, and so is the MLX
    active memory the build added (under ``MAX_BUILD_ACTIVE_BYTES``).

    Raises:
        DFloatFormatError: The groups are not FLUX.1's, an extra is not BF16 or has the wrong shape.
        DFloatDependencyError: The optional ``mflux`` extra is not installed.
        DFloatIntegrationError: A depth override is negative or exceeds the checkpoint's own block
            count, an uncovered parameter, an extra without a target, or too much active memory added
            by the build.
    """
    names = flux_name_map() if name_map is None else name_map
    ckpt_double, ckpt_single = check_flux_groups(ckpt)
    n_double = ckpt_double if n_double is None else n_double
    n_single = ckpt_single if n_single is None else n_single
    if n_double < 0 or n_single < 0:
        raise DFloatIntegrationError(
            f"asked for {n_double} double / {n_single} single blocks; a depth cannot be negative"
        )
    if n_double > ckpt_double or n_single > ckpt_single:
        raise DFloatIntegrationError(
            f"asked for {n_double} double / {n_single} single blocks; the checkpoint has "
            f"{ckpt_double} / {ckpt_single}"
        )
    before = int(mx.get_active_memory())
    transformer = seam_transformer_class()(
        model_config, num_transformer_blocks=n_double, num_single_transformer_blocks=n_single
    )
    shapes = install_placeholders(
        [
            (DOUBLE_PREFIX, transformer.transformer_blocks),
            (SINGLE_PREFIX, transformer.single_transformer_blocks),
        ],
        names,
    )
    matrix_paths = {f"{block}.{attr}.weight" for block, per in shapes.items() for attr in per}
    params = dict(tree_flatten(transformer.parameters()))
    plan = extras_plan(
        ckpt,
        names,
        counts={DOUBLE_PREFIX: n_double, SINGLE_PREFIX: n_single},
        dropped=DROPPED_EXTRAS,
    )
    check_extras_cover(params, (name for name, _path, _info in plan), matrix_paths)
    weights: list[tuple[str, mx.array]] = []
    for name, path, info in plan:
        array = read_extra(path, info)
        if tuple(array.shape) != tuple(params[name].shape):
            raise DFloatFormatError(
                f"{name}: extra has shape {tuple(array.shape)}, parameter {tuple(params[name].shape)}"
            )
        weights.append((name, array))
    transformer.load_weights(weights, strict=False)
    mx.eval([array for _name, array in weights], PLACEHOLDER)
    added = int(mx.get_active_memory()) - before
    if added >= MAX_BUILD_ACTIVE_BYTES:
        raise DFloatIntegrationError(
            f"extras added {added / 1024**3:.2f} GiB of active memory "
            f"(limit {MAX_BUILD_ACTIVE_BYTES / 1024**3:g} GiB)"
        )
    return transformer, shapes
