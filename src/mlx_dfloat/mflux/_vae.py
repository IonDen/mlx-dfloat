"""The VAE decode mode the adapters run: mflux's legacy decode, set explicitly.

mflux 0.24 gave the Z-Image and FLUX.2 VAEs (ERNIE-Image uses the FLUX.2 one) a ``decode_mode``: ``legacy`` decodes the
whole image in one graph, the other modes decode block by block. The memory numbers and image checks of these families
were measured on the legacy decode, so ``load_vae`` pins it instead of following the class default.
"""

from typing import Any

DECODE_MODE_NAME: str = "LEGACY"


def pin_decode_mode(vae: Any, *, decoder_cls: Any = None) -> None:
    """Set ``vae.decode_mode`` to the decoder class's ``LEGACY`` constant.

    ``decoder_cls`` defaults to mflux's ``VAEBlockDecoder``, imported here so this module loads without mflux.
    """
    if decoder_cls is None:
        from mflux.models.common.vae.block_decoder import VAEBlockDecoder

        decoder_cls = VAEBlockDecoder
    vae.decode_mode = getattr(decoder_cls, DECODE_MODE_NAME)
