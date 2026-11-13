"""A recorder of the BF16 bits each seamed block holds while mflux's own block code runs (the mflux lane only)."""

import mlx.core as mx
import numpy as np


def _module_at(block, sub):
    module = block
    for part in sub.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def record_block_bits(monkeypatch, lists, subs):
    """Patch each block class under the seam to record, as a seamed block runs, the bits of its matrices.

    ``lists`` is the transformer's ``block_lists`` (``<kind>`` and its blocks, run order); ``subs`` maps each block
    name ``<kind>.<idx>`` to its matrix sub-paths, which are mflux's own module paths inside the block (read with
    ``getattr``, never through our name map). Returns the list the calls append to: ``(block name, {sub: uint16})``.
    Only the seamed blocks of ``lists`` record, so a stock transformer of the same classes run in the same test
    adds nothing.
    """
    names = {
        id(block): f"{kind}.{idx}" for kind, blocks in lists for idx, block in enumerate(blocks)
    }
    seen = []
    # The seamed class is (BlockSeam, <mflux block class>): its call reaches the mflux class's __call__ with the
    # block's weights assigned, so the record is what mflux's block code computes with.
    for cls in {type(block).__bases__[-1] for _kind, blocks in lists for block in blocks}:
        real = cls.__call__

        def recording(self, *args, _real=real, **kwargs):
            name = names.get(id(self))
            if name is not None:
                seen.append(
                    (
                        name,
                        {
                            sub: np.array(_module_at(self, sub).weight.view(mx.uint16))
                            for sub in subs[name]
                        },
                    )
                )
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(cls, "__call__", recording)
    return seen
