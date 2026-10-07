This directory holds the two groups the GPU decoder is checked against before its first real decode. The first
(`qwen3_4b_layer0_4blocks.npz` and its `.json` provenance record) is a slice of four compressed blocks of one matrix
from the DFloat11/Qwen3-4B-DF11 checkpoint, with the matching BF16 values read from Qwen/Qwen3-4B. Qwen3-4B is licensed
under the Apache License, Version 2.0, and the DFloat11 checkpoint is derived from it. The second (`long_codes.npz`) was
built with the DFloat11 encoder (Apache License, Version 2.0) from values chosen so the codes reach 32 bits and the
lookup tables chain four levels deep, which the Qwen3-4B slice does not reach. It spans more than seven blocks and
includes one block too dense for the staged path's threadgroup buffer.

Each file stores the six group arrays and `expected_bf16`, the BF16 bit patterns the group must decode to: the original
Qwen3-4B weights for the slice, and the encoder's input for the long-code group. Neither was produced by the decoder
under test. The files are loaded with `numpy.load(..., allow_pickle=False)`.
