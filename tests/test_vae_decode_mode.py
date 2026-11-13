"""The VAE decode mode: Z-Image, FLUX.2 Klein and ERNIE-Image stay on mflux's legacy decode.

mflux 0.24 gave those three VAEs a ``decode_mode``. The memory numbers and the image checks were measured on the
legacy path (one graph per image), so ``load_vae`` sets it explicitly instead of following the class default.
"""

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "mlx_dfloat"
VAE_CLASSES = {"VAE", "Flux2VAE", "Qwen21VAE", "QwenVAE"}


def _noop(*args, **kwargs):
    return None


def test_pin_decode_mode_sets_the_named_constant_of_the_decoder_class():
    # Bug caught: the helper hard-codes a string, or reads another constant of the decoder class.
    from mlx_dfloat.mflux._vae import pin_decode_mode

    class FakeDecoder:
        LEGACY = "x-legacy"
        LOW_MEMORY = "x-low"
        FULL_PRECISION = "x-full"

    vae = SimpleNamespace(decode_mode="x-low")
    pin_decode_mode(vae, decoder_cls=FakeDecoder)
    assert vae.decode_mode == "x-legacy"


# family -> (module, the VAE class's module and name, whether load_vae calls _require_replaced)
_PINNED = {
    "zimage": ("mlx_dfloat.mflux.zimage.init", "mflux.models.z_image.model.z_image_vae.vae", "VAE"),
    "flux2": ("mlx_dfloat.mflux.flux2.init", "mflux.models.flux2.model.flux2_vae.vae", "Flux2VAE"),
    "ernie": ("mlx_dfloat.mflux.ernie.init", "mflux.models.flux2.model.flux2_vae.vae", "Flux2VAE"),
}


def _no_weights(monkeypatch, module):
    """``load_vae`` without a base directory: the loaders and the replaced-weights check do nothing."""
    from mlx_dfloat.mflux import _hub

    monkeypatch.setattr(_hub, "load_into", _noop)
    for name in ("load_into", "_load_into", "_require_replaced"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, _noop)


@pytest.mark.mflux
@pytest.mark.parametrize("family", sorted(_PINNED))
def test_each_familys_load_vae_ends_on_the_legacy_decode(family, monkeypatch, tmp_path):
    # Bug caught: the explicit pin dropped, so the product follows whatever default upstream ships (simulated here by
    # flipping the class default to the strip decode the stock initializers choose).
    from mflux.models.common.vae.block_decoder import VAEBlockDecoder

    module_name, vae_module, vae_class = _PINNED[family]
    module = importlib.import_module(module_name)
    _no_weights(monkeypatch, module)
    monkeypatch.setattr(
        getattr(importlib.import_module(vae_module), vae_class), "decode_mode", "low_memory"
    )
    vae = module.load_vae(tmp_path)
    assert vae.decode_mode == "legacy"
    assert VAEBlockDecoder.LEGACY == "legacy"


@pytest.mark.mflux
@pytest.mark.parametrize("family", ["flux1", "qwen21", "krea2"])
def test_the_other_three_vaes_have_no_decode_mode(family, monkeypatch, tmp_path):
    # Bug caught: upstream gives one of these VAEs a decode mode too, and the product never pinned it.
    module = importlib.import_module(f"mlx_dfloat.mflux.{family}.init")
    _no_weights(monkeypatch, module)
    vae = module.load_vae(tmp_path)
    assert not hasattr(vae, "decode_mode")


def _raise_block_decoder(*args, **kwargs):
    raise AssertionError("the legacy decode must not run VAEBlockDecoder")


def _bits_equal(a, b):
    return a.shape == b.shape and bool(mx.array_equal(a.view(mx.uint32), b.view(mx.uint32)))


@pytest.mark.mflux
def test_upstream_legacy_branch_is_the_023_computation_zimage(monkeypatch):
    # Bug caught: upstream removes or rewires the legacy branch of the Z-Image VAE (or changes its scale and shift),
    # so "legacy" no longer decodes what the 0.23 memory and image numbers were measured on.
    from mflux.models.common.vae.block_decoder import VAEBlockDecoder
    from mflux.models.z_image.model.z_image_vae.vae import VAE
    from tests._qwen21_tiny import mflux_uncompiled

    monkeypatch.setattr(VAEBlockDecoder, "decode", staticmethod(_raise_block_decoder))
    with mflux_uncompiled():
        mx.random.seed(0)
        vae = VAE()
        vae.decode_mode = VAEBlockDecoder.LEGACY
        latents = mx.random.normal((1, 16, 1, 8, 8))
        got = vae.decode(latents)
        # mflux 0.23.0 z_image_vae/vae.py:28-33, scaling 0.3611 and shift 0.1159 from the same file.
        want = vae.decoder((latents[:, :, 0, :, :] / 0.3611) + 0.1159)[:, :, None, :, :]
        mx.eval(got, want)
    assert got.shape == (1, 3, 1, 64, 64)
    assert _bits_equal(got, want)


@pytest.mark.mflux
def test_upstream_legacy_branch_is_the_023_computation_flux2(monkeypatch):
    # Bug caught: upstream removes or rewires the legacy branch of the FLUX.2 VAE (ERNIE-Image's VAE too), so
    # "legacy" no longer decodes what the 0.23 memory and image numbers were measured on.
    from mflux.models.common.vae.block_decoder import VAEBlockDecoder
    from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
    from tests._qwen21_tiny import mflux_uncompiled

    monkeypatch.setattr(VAEBlockDecoder, "decode", staticmethod(_raise_block_decoder))
    with mflux_uncompiled():
        mx.random.seed(0)
        vae = Flux2VAE()
        vae.decode_mode = VAEBlockDecoder.LEGACY
        latents = mx.random.normal((1, 32, 8, 8))
        got = vae.decode(latents)
        # mflux 0.23.0 flux2_vae/vae.py:39-47, scaling 1.0 and shift 0.0 from the same file.
        x = mx.transpose((latents / 1.0) + 0.0, (0, 2, 3, 1))
        x = mx.transpose(vae.post_quant_conv(x), (0, 3, 1, 2))
        want = vae.decoder(x)
        mx.eval(got, want)
    assert got.shape == (1, 3, 64, 64)
    assert _bits_equal(got, want)


def _constructions_outside_load_vae(root: Path) -> list[str]:
    """``file:line`` of every call to a VAE class name that is not inside a function named ``load_vae``."""
    found = []

    def visit(node: ast.AST, inside: bool, path: Path) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inside = inside or node.name == "load_vae"
        if isinstance(node, ast.Call) and not inside:
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if name in VAE_CLASSES:
                found.append(f"{path.relative_to(root)}:{node.lineno}")
        for child in ast.iter_child_nodes(node):
            visit(child, inside, path)

    for path in sorted(root.rglob("*.py")):
        visit(ast.parse(path.read_text(encoding="utf-8")), False, path)
    return found


def test_load_vae_is_the_only_vae_constructor_in_src():
    # Bug caught: a second construction path (a reload helper, a script entry) builds a VAE without the pin and
    # silently takes the upstream class default.
    assert _constructions_outside_load_vae(SRC) == []


def test_the_constructor_scan_finds_a_vae_built_outside_load_vae(tmp_path):
    # Bug caught: the scan above passing because it never sees a call (wrong names, wrong node type, or every
    # function counted as load_vae).
    (tmp_path / "a.py").write_text(
        "def load_vae():\n    return VAE()\n\ndef reload():\n    return m.Flux2VAE()\n\nx = QwenVAE()\n",
        encoding="utf-8",
    )
    assert _constructions_outside_load_vae(tmp_path) == ["a.py:5", "a.py:7"]


def _product_vae(family, monkeypatch, tmp_path):
    """The VAE our ``load_vae`` builds, with the class default flipped to the strip decode the stock initializers set."""
    module_name, vae_module, vae_class = _PINNED[family]
    module = importlib.import_module(module_name)
    _no_weights(monkeypatch, module)
    monkeypatch.setattr(
        getattr(importlib.import_module(vae_module), vae_class), "decode_mode", "low_memory"
    )
    return module.load_vae(tmp_path)


@pytest.mark.mflux
def test_flux2_packed_decode_stays_legacy(monkeypatch, tmp_path):
    # Bug caught: the pin set on the VAE but the product path (Flux2Klein -> vae.decode_packed_latents ->
    # VAEUtil.decode -> vae.decode) still reaching VAEBlockDecoder, e.g. a decode mode read from somewhere else.
    from mflux.models.common.vae.block_decoder import VAEBlockDecoder
    from tests._qwen21_tiny import mflux_uncompiled

    monkeypatch.setattr(VAEBlockDecoder, "decode", staticmethod(_raise_block_decoder))
    with mflux_uncompiled():
        vae = _product_vae("flux2", monkeypatch, tmp_path)
        mx.random.seed(0)
        # flux2_vae/vae.py:60-87 (0.24.0): packed [B, 128, h, w] unpatchifies to [B, 32, 2h, 2w], decoded 8x.
        got = vae.decode_packed_latents(mx.random.normal((1, 128, 4, 4)))
        mx.eval(got)
    assert got.shape == (1, 3, 64, 64)


@pytest.mark.mflux
def test_zimage_vae_util_decode_stays_legacy(monkeypatch, tmp_path):
    # Bug caught: as above for Z-Image, whose pipeline decodes with VAEUtil.decode(vae=self.vae, ...)
    # (z_image.py:214 in 0.24.0).
    from mflux.models.common.vae.block_decoder import VAEBlockDecoder
    from mflux.models.common.vae.vae_util import VAEUtil
    from tests._qwen21_tiny import mflux_uncompiled

    monkeypatch.setattr(VAEBlockDecoder, "decode", staticmethod(_raise_block_decoder))
    with mflux_uncompiled():
        vae = _product_vae("zimage", monkeypatch, tmp_path)
        mx.random.seed(0)
        got = VAEUtil.decode(vae=vae, latent=mx.random.normal((1, 16, 1, 8, 8)), tiling_config=None)
        mx.eval(got)
    assert got.shape == (1, 3, 64, 64)
