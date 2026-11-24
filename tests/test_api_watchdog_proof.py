"""The pure parts of the API watchdog's hardware proof: the per-phase ceilings, the unit verdicts, resume.

The synthetic peaks are in MiB so every expected stop line and ceiling can be worked by hand in the comments,
never taken from the code. The offline test feeds every committed FLUX.2 Klein 4B record through the derivation.
No test here imports mflux, builds a model or runs a unit.
"""

import json
from pathlib import Path
from typing import Any

import pytest

GIB = 1_073_741_824
MIB = 1_048_576
REPO = Path(__file__).resolve().parent.parent

# The headroom rule from the proof's spec: at least 0.4 GiB (the records differ by about 0.3 GiB between runs).
MIN_HEADROOM = 0.4 * GIB

PEAKS: dict[str, Any] = {
    "label": "sampled at phase boundaries",
    "encode": {"footprint_start": 400 * MIB, "mlx_peak": 6000 * MIB, "footprint_end": 6400 * MIB},
    # Set load opens on the encoder's residue (6300 MiB), above its own top (5600 MiB).
    "set_load": {
        "footprint_start": 6300 * MIB,
        "mlx_peak": 5000 * MIB,
        "footprint_end": 5600 * MIB,
    },
    "denoise": {
        "footprint_start": 5600 * MIB,
        "mlx_peak": 6600 * MIB,
        "footprint_end": 7600 * MIB,
        "watched_peak": 7800 * MIB,
    },
    "vae": {
        "footprint_start": 7600 * MIB,
        "mlx_peak": 11000 * MIB,
        "footprint_end": 6000 * MIB,
        "watched_peak": 11500 * MIB,
    },
}


def _peaks(**changes: dict[str, int]) -> dict[str, Any]:
    peaks = {k: dict(v) if isinstance(v, dict) else v for k, v in PEAKS.items()}
    for phase, fields in changes.items():
        peaks[phase].update(fields)
    return peaks


def test_raise_units_stop_half_a_margin_under_the_phase_peak_and_set_load_mid_climb():
    # Bug: the stop line placed at the phase midpoint for encode, denoise or the VAE (ceiling under the VAE's sampled
    # peak), or the set load's high taken from its start footprint (the encoder's residue), or a sampled peak
    # ignored; or set load placed half a margin under its top, where the line is crossed only in the last group or
    # the non-block decode after it, with no set-load checkpoint left.
    from scripts._api_watchdog_proof import ceilings_for

    ceilings = ceilings_for(PEAKS, stop_margin=1024 * MIB)

    assert ceilings == {
        # hi = end 6400; stop 6400 - 512 = 5888; ceiling 5888 + 1024 = 6912
        "encode": (6912 * MIB, 1024 * MIB),
        # lo = encode start 400, hi = end 5600 (the start 6300 is the encoder's residue);
        # stop mid-climb (400 + 5600) / 2 = 3000; ceiling 3000 + 1024 = 4024
        "set_load": (4024 * MIB, 1024 * MIB),
        # hi = sampled 7800; stop 7288; ceiling 8312
        "denoise": (8312 * MIB, 1024 * MIB),
        # hi = sampled 11500; stop 10988; ceiling 12012
        "vae": (12012 * MIB, 1024 * MIB),
    }


def test_a_margin_of_twice_the_headroom_or_less_places_the_line_for_an_exit():
    # Bug: U5's 64 MiB margin placed like a raise (32 MiB under the peak: never isolable, so every run stops at U0),
    # or the exit window not centred between the earlier peak and the VAE's peak.
    from scripts._api_watchdog_proof import ceilings_for

    # VAE: lo = max(set load 6300, denoise 7800) = 7800, hi = 11500; stop (7800 + 11500 - 64) // 2 = 9618
    assert ceilings_for(PEAKS, stop_margin=64 * MIB)["vae"] == (9682 * MIB, 64 * MIB)


def test_the_mode_switch_sits_at_twice_the_headroom():
    # Bug: `>` turned into `>=` in the mode choice, so a margin of exactly twice the headroom places a raise whose
    # peak sits only one headroom under the ceiling.
    from scripts._api_watchdog_proof import ceilings_for

    headroom = 100 * MIB
    # raise: VAE stop 11500 - 100 = 11400 (margin 200 + 2 bytes: half is 100 MiB + 1)
    raise_line = ceilings_for(PEAKS, stop_margin=200 * MIB + 2, headroom=headroom)["vae"]
    assert raise_line == (11500 * MIB - (100 * MIB + 1) + 200 * MIB + 2, 200 * MIB + 2)
    # exit: stop (7800 + 11500 - 200) // 2 = 9550, ceiling 9750
    assert ceilings_for(PEAKS, stop_margin=200 * MIB, headroom=headroom)["vae"] == (
        9750 * MIB,
        200 * MIB,
    )


@pytest.mark.parametrize("field", ["footprint_start", "watched_after"])
@pytest.mark.parametrize(
    ("residue", "derived"),
    [
        # denoise stop 7288; the earlier peak is set load's; 7288 - 410 = 6878 is the boundary
        (6878 * MIB - 1, True),
        (6878 * MIB, False),
    ],
)
def test_a_stop_line_within_headroom_of_an_earlier_peak_is_not_isolable(field, residue, derived):
    # Bug: `>` turned into `>=`, the headroom dropped, or set load's start footprint (or the stretch sampled after
    # it) left out of the earlier peak: a unit armed through set load would trip there, not in denoise.
    from scripts._api_watchdog_proof import ceilings_for

    ceilings = ceilings_for(_peaks(set_load={field: residue}), stop_margin=1024 * MIB)

    assert (ceilings["denoise"] is not None) is derived


def test_the_headroom_default_is_at_least_0_4_gib():
    # Bug: the headroom constant shrunk under the run-to-run spread of the records.
    from scripts._api_watchdog_proof import HEADROOM

    assert HEADROOM >= MIN_HEADROOM


@pytest.mark.parametrize(
    ("settled", "derived"),
    [
        # stop = (settled + 5600) // 2 must pass settled + 410, so 5600 - settled must pass 820 after the floor:
        # settled 4780 - 2 bytes gives stop 5190 MiB - 1, 410 MiB + 1 over it; 4780 - 1 byte gives exactly 410 MiB
        (4780 * MIB - 2, True),
        (4780 * MIB - 1, False),
    ],
)
def test_set_load_takes_its_low_bound_from_the_settled_reading(settled, derived):
    # Bug: the settled reading ignored (the encode start, 400 MiB, used instead), or compared with `>=`.
    from scripts._api_watchdog_proof import ceilings_for

    ceiling = ceilings_for(PEAKS, stop_margin=1024 * MIB, settled=settled)["set_load"]

    assert (ceiling is not None) is derived
    if derived:
        # stop 5190 MiB - 1, ceiling 5190 + 1024 = 6214 MiB - 1
        assert ceiling == (6214 * MIB - 1, 1024 * MIB)


@pytest.mark.parametrize(
    ("peaks", "expected_stop"),
    [
        # the overall peak is the VAE's sampled 11500; stop 11500 + 1024 = 12524
        (PEAKS, 12524 * MIB),
        # the overall peak sits in the stretch after the denoise (12000), not in the VAE
        (_peaks(denoise={"watched_after": 12000 * MIB}), 13024 * MIB),
        # the overall peak is the encode's start footprint (13000)
        (_peaks(encode={"footprint_start": 13000 * MIB}), 14024 * MIB),
    ],
)
def test_the_retry_line_sits_1_gib_over_the_baselines_overall_peak(peaks, expected_stop):
    # Bug: the retry line taken from one phase only (the VAE), a sampled field left out, the headroom dropped, or the
    # ceiling not one margin over the stop: then a retry whose drain freed nothing still completes, unseen.
    from scripts._api_watchdog_proof import retry_line

    assert retry_line(peaks, stop_margin=1024 * MIB) == (expected_stop + 1024 * MIB, 1024 * MIB)


def test_no_retry_line_without_a_phase_record():
    # Bug: a KeyError or a line at 0 + headroom (every retry trips) when the baseline kept no peaks.
    from scripts._api_watchdog_proof import retry_line

    assert retry_line({"label": "x"}, stop_margin=1024 * MIB) is None


@pytest.mark.parametrize(
    ("hi", "derived"),
    [
        # lo 7800, margin 64: centred, so stop - lo = hi - ceiling = (hi - 7800 - 64) / 2 must pass 410: hi > 8684
        (8684 * MIB, False),
        (8684 * MIB + 2, True),
    ],
)
def test_an_exit_line_needs_headroom_on_both_sides(hi, derived):
    # Bug: the exit placement accepted with the peak too close to the ceiling (U5 might end by a raise instead).
    from scripts._api_watchdog_proof import ceilings_for

    peaks = _peaks(
        vae={"watched_peak": hi, "mlx_peak": 0, "footprint_start": 0, "footprint_end": 0}
    )

    assert (ceilings_for(peaks, stop_margin=64 * MIB)["vae"] is not None) is derived


@pytest.mark.parametrize(
    ("dropped", "expected"),
    [
        # Set load's planning low bound is the encode's start footprint.
        ("encode", {"encode": None, "set_load": None, "denoise": "derived", "vae": "derived"}),
        ("set_load", {"encode": "derived", "set_load": None, "denoise": None, "vae": None}),
        ("denoise", {"encode": "derived", "set_load": "derived", "denoise": None, "vae": None}),
    ],
)
def test_a_missing_earlier_phase_record_gives_no_ceiling(dropped, expected):
    # Bug: a KeyError on a missing record, or a low bound taken from only the phases that are there.
    from scripts._api_watchdog_proof import ceilings_for

    peaks = {name: record for name, record in PEAKS.items() if name != dropped}

    ceilings = ceilings_for(peaks, stop_margin=GIB)

    assert {p: None if c is None else "derived" for p, c in ceilings.items()} == expected


def test_phase_peaks_keep_each_phase_and_the_stretch_after_it():
    # Bug: a sample taken between phases counted in the next phase, or dropped (the earlier peak a later unit must
    # clear would miss the encoder's residue), or a lower sample overwriting a higher one.
    from scripts._api_watchdog_proof import PhasePeaks

    sampled = PhasePeaks()
    for phase, watched in [
        (None, 50),  # before the first phase: no phase to charge
        ("encode", 10),
        ("encode", 30),
        ("encode", 20),
        (None, 25),
        (None, 40),
        ("set_load", 5),
    ]:
        sampled.add(phase, watched)

    merged = sampled.into({"label": "x", "encode": {"mlx_peak": 1}, "set_load": {"mlx_peak": 2}})

    assert merged == {
        "label": "x",
        "encode": {"mlx_peak": 1, "watched_peak": 30, "watched_after": 40},
        "set_load": {"mlx_peak": 2, "watched_peak": 5},
    }


def test_the_sampler_samples_on_its_thread_and_once_more_at_stop():
    # Bug: the thread never samples (the U0 peaks are boundary-only again), or stop() skips its last sample, so
    # the stretch after the decode is lost.
    import threading

    from scripts._api_watchdog_proof import PhaseSampler

    reads: list[int] = []
    two = threading.Event()
    phase = ["vae"]

    def read() -> int:
        reads.append(len(reads) + 1)
        if len(reads) >= 2:
            two.set()
        return reads[-1]

    sampler = PhaseSampler(read, lambda: phase[0], interval=0.001).start()
    assert two.wait(timeout=5)
    phase[0] = None  # the call has closed its last phase
    sampler.stop()

    assert sampler.sampled.peaks["vae"]["watched_peak"] >= 2
    assert sampler.sampled.peaks["vae"]["watched_after"] == reads[-1]


# ---- unit verdicts ----


def _exited(code, *, phase, escalation, reason="memory"):
    artifact = {"reason": reason, "escalation": escalation, "context": {"phase": phase}}
    return {"exit_code": code, "result": None, "artifact": artifact}


def _baseline(code, *, completed, image_saved):
    return {
        "exit_code": code,
        "result": {"completed": completed, "image_saved": image_saved},
        "artifact": None,
    }


def _raised(*, phase, site, reason="memory", retry=None, after_abort=None):
    raised = {"type": "DFloatWatchdogError", "reason": reason, "phase": phase, "site": site}
    # The armed line: ceiling 4096 MiB, margin 1024 MiB, so the stop line is 3072 MiB.
    result: dict[str, Any] = {"ceiling": [4096 * MIB, 1024 * MIB], "raised": raised}
    if after_abort is not None:
        result["watched_after_abort"] = after_abort
    if retry is not None:
        result["retry"] = retry
    return {"exit_code": 0, "result": result, "artifact": None}


DONE = {"completed": True, "raised": None}
# The watched memory after the stopped call's cleanup, 1024 MiB under the stop line.
RELEASED = 2048 * MIB
RETRY_RAISED = {
    "type": "DFloatWatchdogError",
    "reason": "memory",
    "phase": "vae",
    "site": "call_end",
}


@pytest.mark.parametrize(
    ("unit", "outcome", "expected"),
    [
        ("U1", _raised(phase="encode", site="prompt_encoded"), "pass"),
        ("U1", _raised(phase="encode", site="encoder_loaded"), "pass"),
        ("U1", _raised(phase="encode", site="call_start"), "fail"),  # stopped before any work
        ("U1", _raised(phase="set_load", site="set_group"), "fail"),
        ("U1", _exited(70, phase="encode", escalation="ceiling"), "fail"),
        ("U2", _raised(phase="set_load", site="set_load_start"), "pass"),
        ("U2", _raised(phase="set_load", site="set_group"), "pass"),
        ("U2", _raised(phase="set_load", site="call_start"), "fail"),
        ("U2", _raised(phase="encode", site="encoder_loaded"), "fail"),
        ("U2", _raised(phase="set_load", site="block_start"), "fail"),
        ("U2", _raised(phase="set_load", site="set_group", reason="wall"), "fail"),
        ("U3", _raised(phase="denoise", site="block_start"), "pass"),
        ("U3", _raised(phase="denoise", site="block_end"), "pass"),
        ("U3", _raised(phase="denoise", site="call_start"), "fail"),
        ("U3", _raised(phase="vae", site="call_end"), "fail"),
        ("U3", _raised(phase="set_load", site="block_start"), "fail"),
        ("U4", _raised(phase="vae", site="call_end"), "pass"),
        # The finished image must be discarded by a raise: an exit proves the same thing U5 does.
        ("U4", _exited(70, phase="vae", escalation="ceiling"), "fail"),
        ("U4", _raised(phase="vae", site="call_start"), "fail"),
        ("U4", _raised(phase="vae", site="block_end"), "fail"),
        ("U4", _raised(phase="denoise", site="call_end"), "fail"),
        ("U5", _exited(70, phase="vae", escalation="ceiling"), "pass"),
        # A memory trip ends by grace only when the sampler died, so a grace exit is not the watchdog working.
        ("U5", _exited(70, phase="vae", escalation="grace"), "fail"),
        ("U5", _exited(70, phase="vae", escalation=None), "fail"),
        ("U5", _exited(70, phase="denoise", escalation="ceiling"), "fail"),
        ("U5", _exited(71, phase="vae", escalation="grace", reason="wall"), "fail"),
        ("U5", _raised(phase="vae", site="call_end"), "fail"),
        (
            "U6",
            _raised(phase="set_load", site="set_group", retry=DONE, after_abort=RELEASED),
            "pass",
        ),
        # the stopped call's memory came back to 1 byte under the 3072 MiB stop line: released
        (
            "U6",
            _raised(phase="set_load", site="set_group", retry=DONE, after_abort=3072 * MIB - 1),
            "pass",
        ),
        # on or over the stop line after the cleanup: the set's pages were kept, even if the retry completes
        (
            "U6",
            _raised(phase="set_load", site="set_group", retry=DONE, after_abort=3072 * MIB),
            "fail",
        ),
        (
            "U6",
            _raised(phase="set_load", site="set_group", retry=DONE, after_abort=5000 * MIB),
            "fail",
        ),
        (
            "U6",
            _raised(phase="set_load", site="set_group", retry=DONE),
            "fail",
        ),  # no reading after the abort
        (
            "U6",
            _raised(
                phase="set_load",
                site="set_group",
                retry={"completed": False, "raised": RETRY_RAISED},
                after_abort=RELEASED,
            ),
            "fail",
        ),
        (
            "U6",
            _raised(phase="set_load", site="set_load_start", retry=DONE, after_abort=RELEASED),
            "pass",
        ),
        (
            "U6",
            _raised(phase="set_load", site="set_group", after_abort=RELEASED),
            "fail",
        ),  # no retry ran
        (
            "U6",
            _raised(
                phase="set_load", site="set_group", retry={"completed": False}, after_abort=RELEASED
            ),
            "fail",
        ),
        (
            "U6",
            _raised(phase="denoise", site="block_start", retry=DONE, after_abort=RELEASED),
            "fail",
        ),
        (
            "U6",
            {"exit_code": 0, "result": {"ceiling": [1, 1], "raised": None, "retry": DONE}},
            "fail",
        ),
        ("U6", _exited(70, phase="set_load", escalation="ceiling"), "fail"),
        ("U3", {"exit_code": 0, "result": {"ceiling": None}, "artifact": None}, "not isolable"),
        ("U0", _baseline(0, completed=True, image_saved=True), "pass"),
        ("U0", _baseline(70, completed=True, image_saved=True), "fail"),
        ("U0", _baseline(0, completed=False, image_saved=True), "fail"),
        ("U0", _baseline(0, completed=True, image_saved=False), "fail"),
    ],
)
def test_unit_verdicts(unit, outcome, expected):
    # Bug: the verdict ignores the phase, the site or the trip's reason; U4 passes on an exit; U6 passes without its
    # completed retry, without the abort before it, or with the watched memory after the abort at or over its stop line
    # (the stopped call kept its pages), or without that reading; a raise at call_start (before any work) passes; "not
    # isolable" reads as a pass; or the baseline passes on a non-zero exit, an unfinished run or an unsaved image.
    from scripts._api_watchdog_proof import unit_verdict

    assert unit_verdict(unit, outcome) == expected


@pytest.mark.parametrize("unit", ["U7", "u1", ""])
def test_an_unknown_unit_is_refused(unit):
    # Bug: the unit check missing, so an unknown unit falls through to PHASE_OF and raises KeyError (or a lowercase
    # name is accepted), not the documented ValueError.
    from scripts._api_watchdog_proof import unit_verdict

    with pytest.raises(ValueError, match="unknown unit"):
        unit_verdict(unit, _baseline(0, completed=True, image_saved=True))


def test_a_resumed_run_skips_finished_units(tmp_path: Path):
    # Bug: a finished unit runs again (its JSON ignored), or a half-written one counts as finished.
    from scripts._api_watchdog_proof import pending_units

    (tmp_path / "U1.json").write_text(json.dumps({"unit": "U1", "verdict": "pass"}))
    (tmp_path / "U2.json.tmp").write_text("{")
    (tmp_path / "U3.outcome.json").write_text("{}")

    assert pending_units(tmp_path) == ["U0", "U2", "U3", "U4", "U5", "U6"]


# ---- the committed FLUX.2 Klein 4B records ----

RECORDS = [
    "bench/results/tiers/flux2-klein-4b-1024.json",
    "bench/results/tiers/flux2-klein-base-4b-1024.json",
    "bench/results/tiers/flux2-klein-base-4b-1024-tier16.json",
    "bench/results/tiers/flux2-klein-base-4b-1024-tier24.json",
    "bench/results/identity/flux2-klein-4b-1024/df11-result.json",
    "bench/results/identity/flux2-klein-base-4b-1024/df11-result.json",
]
# The 16 GB tier frees the transformer set before the decode: its VAE peak (6.93 GiB) is under the denoise
# peak (7.61 GiB), so no line can stop the decode without stopping denoise first. The proof runs the default tier.
VAE_BELOW_DENOISE = {"bench/results/tiers/flux2-klein-base-4b-1024-tier16.json"}


def _find_peaks(data: Any) -> dict[str, Any] | None:
    if isinstance(data, dict):
        if isinstance(data.get("peaks"), dict) and "encode" in data["peaks"]:
            return data["peaks"]
        for value in data.values():
            if (found := _find_peaks(value)) is not None:
                return found
    return None


def _load(rel: str) -> tuple[dict[str, Any], int]:
    """A committed record's boundary peaks, with its call-wide sampled peak charged to its highest phase.

    These records predate the per-phase sampled peak. The call-wide ``footprint_peak_bytes`` (the watchdog's
    0.05 s samples) is above every boundary reading, so it is charged to the phase whose boundary readings are
    highest; without it, the VAE's high would be 0.9-1.0 GiB under what the watchdog sees.
    """
    data = json.loads((REPO / rel).read_text())
    peaks = {k: dict(v) if isinstance(v, dict) else v for k, v in _find_peaks(data).items()}  # type: ignore[union-attr]
    call_peak = int(data["footprint_peak_bytes"])
    phases = ("encode", "set_load", "denoise", "vae")
    top = max(
        phases,
        key=lambda p: max(peaks[p][k] for k in ("footprint_start", "mlx_peak", "footprint_end")),
    )
    peaks[top]["watched_peak"] = call_peak
    return peaks, call_peak


def _all(record: dict[str, int]) -> int:
    return max(v for k, v in record.items() if k != "active_at_start")


def _own(phase: str, record: dict[str, int]) -> int:
    if phase == "set_load":  # its start footprint is the encoder's residue in a baseline run
        return max(record["mlx_peak"], record["footprint_end"])
    return _all(record)


def _earlier(phase: str, peaks: dict[str, Any]) -> int:
    if phase in ("encode", "set_load"):
        return peaks["encode"]["footprint_start"]
    if phase == "denoise":
        return _all(peaks["set_load"])
    return max(_all(peaks["set_load"]), _all(peaks["denoise"]))


@pytest.mark.parametrize("rel", RECORDS)
def test_every_committed_klein_4b_record_places_a_line_for_each_unit(rel):
    # Bug: the derivation cannot stop a unit on the records this model really produces (the old midpoint rule left
    # set load not isolable six times out of six and put the VAE ceiling under the sampled decode peak).
    import scripts.prove_api_watchdog as prove
    from scripts._api_watchdog_proof import PHASE_OF, ceilings_for

    peaks, _ = _load(rel)
    for unit, phase in PHASE_OF.items():
        margin = prove.ESCALATION_MARGIN if unit == "U5" else GIB
        line = ceilings_for(peaks, stop_margin=margin)[phase]
        if phase == "vae" and rel in VAE_BELOW_DENOISE:
            assert line is None, unit
            continue
        assert line is not None, unit
        ceiling, stop_margin = line
        stop = ceiling - stop_margin
        lo, hi = _earlier(phase, peaks), _own(phase, peaks[phase])
        # the earlier phases and the arm-time reading stay under the line
        assert stop > lo + MIN_HEADROOM, unit
        if unit == "U5":
            assert hi > ceiling + MIN_HEADROOM, unit  # the decode passes the hard line: exit 70
        elif phase == "set_load":
            # crossed before the tail with no set-load checkpoint: the last group plus the non-block decode
            assert hi - stop > _set_load_tail(rel), unit
        else:
            # crosses the stop, not the ceiling
            assert stop + MIN_HEADROOM < hi < ceiling - MIN_HEADROOM, unit
    expected_blocked = ["U4", "U5"] if rel in VAE_BELOW_DENOISE else []
    assert prove._not_isolable(peaks) == expected_blocked


# Compressed group sizes in bytes, in the order the set loads them (sorted names), read from the safetensors
# headers of mingyi456/FLUX.2-klein-4B-DF11@d29a2c2 (distilled) and mingyi456/FLUX.2-klein-base-4B-DF11@b887c73.
# Five non-block groups, single blocks 0, 1, 10-19, 2-9, then double blocks 0-4.
GROUPS_DISTILLED = (
    *(31102869, 77274555, 76422697, 27252494, 40906061),
    *(166173729, 166061116, 166101474, 166241621, 166283109, 166291721, 166407701, 166590094),
    *(166618477, 166820360, 166856951, 167054860, 166293159, 166684440, 167202190, 167388461),
    *(166153645, 166069826, 165987779, 166100410),
    *(331860073, 331018225, 331084614, 331171115, 331565605),
)
GROUPS_BASE = (
    *(31097515, 77270676, 76426672, 27248241, 40908632),
    *(166172845, 166060147, 166101998, 166244554, 166284512, 166293371, 166411738, 166595908),
    *(166623546, 166828945, 166861230, 167052344, 166292112, 166685588, 167201234, 167389902),
    *(166149637, 166069235, 165988347, 166100828),
    *(331857016, 331013479, 331079435, 331164200, 331559145),
)


def _groups(rel: str) -> tuple[int, ...]:
    return GROUPS_BASE if "base" in rel else GROUPS_DISTILLED


def _sizes(rel: str) -> dict[str, int]:
    data = json.loads((REPO / rel).read_text())

    def find(node: Any) -> dict[str, int] | None:
        if isinstance(node, dict):
            if isinstance(node.get("sizes"), dict) and "nonblock" in node["sizes"]:
                return node["sizes"]
            for value in node.values():
                if (found := find(value)) is not None:
                    return found
        return None

    sizes = find(data)
    assert sizes is not None, rel
    return sizes


def _set_load_tail(rel: str) -> int:
    return _sizes(rel)["nonblock"] + _groups(rel)[-1]


@pytest.mark.parametrize("rel", RECORDS)
@pytest.mark.parametrize("planned", [True, False], ids=["planned", "settled"])
def test_the_set_load_line_is_crossed_with_set_load_checkpoints_left(rel, planned):
    # Bug: the set-load line placed half a margin under the phase top: the climb passes it only while the last group
    # is read or the non-block groups decode, the next checkpoint is block_start, and the unit fails; or so late in
    # the climb that the raise one group after the crossing lands within the headroom of the ceiling.
    from scripts._api_watchdog_proof import ceilings_for

    peaks, _ = _load(rel)
    groups = _groups(rel)
    sizes = _sizes(rel)
    # Each read is one copy into MLX and the freed compressed non-block groups stay counted, so the watched memory
    # at the checkpoint before group i is the start plus the groups read so far; the start is the phase end less
    # every group and the decoded non-block weights (the unit's settled reading after its pre-encode).
    start = peaks["set_load"]["footprint_end"] - sizes["compressed"] - sizes["nonblock"]
    line = ceilings_for(peaks, stop_margin=GIB, settled=None if planned else start)["set_load"]
    assert line is not None
    ceiling, margin = line
    stop = ceiling - margin
    readings = [start + sum(groups[:i]) for i in range(len(groups))]
    over = [i for i, reading in enumerate(readings) if reading >= stop]
    # A second set-group checkpoint over the line: the sampler may latch only after the first one has run.
    assert len(over) >= 2, (stop / GIB, readings[-1] / GIB)
    late = readings[over[1]]
    assert ceiling - late >= MIN_HEADROOM, (stop / GIB, late / GIB, ceiling / GIB)
