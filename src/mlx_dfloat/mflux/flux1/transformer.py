"""mflux's FLUX.1 Transformer with the block seam composed in front of its two per-block hooks."""

from functools import cache
from typing import Any

from mlx_dfloat.errors import DFloatIntegrationError
from mlx_dfloat.integrate import seam
from mlx_dfloat.integrate.names import Shapes
from mlx_dfloat.integrate.providers import WeightProvider

DOUBLE_PREFIX = "transformer_blocks"
SINGLE_PREFIX = "single_transformer_blocks"


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
    """``SeamMixin`` composed in front of mflux's ``Transformer`` (imports mflux)."""
    from mflux.models.flux.model.flux_transformer.transformer import Transformer

    return type("SeamTransformer", (SeamMixin, Transformer), {})
