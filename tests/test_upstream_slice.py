import hashlib
import json
from pathlib import Path

import numpy as np

from mlx_dfloat.format import GroupArrays
from mlx_dfloat.reference import decode_group, max_code_length

FIXTURE = Path(__file__).parent / "fixtures" / "upstream" / "qwen3_4b_layer0_4blocks"
PROVENANCE = json.loads(FIXTURE.with_suffix(".json").read_text())


def test_fixture_integrity_matches_provenance():
    # Integrity only: guards against a corrupted file, not a wrong one.
    assert (
        hashlib.sha256(FIXTURE.with_suffix(".npz").read_bytes()).hexdigest()
        == PROVENANCE["npz_sha256"]
    )


def test_reference_decodes_upstream_bytes_to_the_original_bf16():
    """Upstream's encoder produced these bytes; the expected values are Qwen3-4B's own q_proj weights.

    The stream keeps 8 bytes of block 4 because block 3's last codes read up to 31 bits past the
    block boundary; without them the slice would decode wrongly with no error.
    """
    data = np.load(FIXTURE.with_suffix(".npz"))
    arrays = GroupArrays(
        encoded_exponent=data["encoded_exponent"],
        sign_mantissa=data["sign_mantissa"],
        luts=data["luts"],
        gaps=data["gaps"],
        output_positions=data["output_positions"].astype(np.uint32),
        split_positions=data["split_positions"],
    )
    assert arrays.n_bytes % 4096 == 8
    assert max_code_length(arrays.luts) == PROVENANCE["max_code_length"]
    # The slice is truncated mid-stream (no real EOF), so only the stream-end byte count is skipped;
    # gap continuity, block starts and the tail-entry checks all still run.
    out = decode_group(arrays, name="qwen3-4b layer 0 slice", check_stream_end=False)
    np.testing.assert_array_equal(out, data["expected_bf16"])
