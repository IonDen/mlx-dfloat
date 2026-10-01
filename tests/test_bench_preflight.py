import pytest

from mlx_dfloat.bench.capped import GIB
from mlx_dfloat.bench.preflight import (
    HEAVY_PATTERNS,
    Preflight,
    check,
    parse_ac,
    parse_batt,
    parse_clamshell,
    parse_memory_pressure,
    parse_ps,
    parse_therm,
    sample,
)

# REAL captures (this Mac, 2026-09-29, on battery, lid open, pasted verbatim):
#   BATT_DISCHARGING_REAL, AC_NONE_REAL, THERM_CLEAN, CLAMSHELL_OPEN, PRESSURE (head line + tail),
#   PS_REAL_HEAD (the first lines of `ps -Ao pid=,rss=,command=`).
# SYNTHETIC (states this Mac was not in; same line format as the real battery line, built by hand):
#   BATT_AC, BATT_NOT_CHARGING, BATT_CHARGING, THERM_LIMITED, CLAMSHELL_CLOSED, AC_WATTS_SYNTHETIC, PS.
# `pmset -g ac` printed no Wattage line on this Mac (no adapter attached), so parse_ac of the real
# output is None; the Wattage format below is synthetic and only pins the regex.
BATT_DISCHARGING_REAL = (
    "Now drawing from 'Battery Power'\n"
    " -InternalBattery-0 (id=24576099)\t100%; discharging; 6:34 remaining present: true\n"
)
BATT_AC = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=24576099)\t100%; charged; 0:00 remaining present: true\n"
BATT_DISCHARGING = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=24576099)\t63%; discharging; 3:10 remaining present: true\n"
BATT_NOT_CHARGING = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=24576099)\t38%; AC attached; not charging present: true\n"
BATT_CHARGING = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=24576099)\t80%; charging; 0:45 remaining present: true\n"
AC_NONE_REAL = "No adapter attached.\n"
AC_WATTS_SYNTHETIC = "Wattage = 96W\nCurrent = 4800mA\nVoltage = 20000mV\n"
THERM_CLEAN = (
    "Note: No thermal warning level has been recorded\n"
    "Note: No performance warning level has been recorded\n"
    "Note: No CPU power status has been recorded\n"
)
THERM_LIMITED = "CPU_Scheduler_Limit \t= 100\nCPU_Available_CPUs \t= 10\nCPU_Speed_Limit \t= 65\n"
CLAMSHELL_OPEN = '  |   "AppleClamshellCausesSleep" = Yes\n  |   "AppleClamshellState" = No\n'
CLAMSHELL_CLOSED = '  |   "AppleClamshellCausesSleep" = Yes\n  |   "AppleClamshellState" = Yes\n'
PRESSURE = (
    "The system has 34359738368 (2097152 pages with a page size of 16384).\n"
    "\n"
    "Stats: \n"
    "Pages speculative: 31727 \n"
    "Pages throttled: 0 \n"
    "Pages wired down: 208850 \n"
    "\n"
    "Compressor Stats:\n"
    "Pages used by compressor: 152998 \n"
    "Pages decompressed: 784609531 \n"
    "Pages compressed: 823091156 \n"
    "\n"
    "File I/O:\n"
    "Pageins: 370890042 \n"
    "Pageouts: 421404 \n"
    "\n"
    "System-wide memory free percentage: 82%\n"
)
PS_REAL_HEAD = (
    "    1  21184 /sbin/launchd\n"
    "  359  39024 /usr/libexec/logd\n"
    "  362   8864 /usr/libexec/UserEventAgent (System)\n"
)
PS = (
    "  123  2048 /usr/bin/python3 -m scripts.bench_flux_step --mode df11\n"
    "  124 4194304 /Users/x/.venv/bin/python -m scripts.bench_flux_step --mode control\n"
    "  125 4194304 /Applications/ClaudeCode.app/Contents/MacOS/claude\n"
    "  126 4194304 /usr/bin/ssh host\n"
)


def test_parse_batt_reads_source_percentage_and_charging_state():
    # Red if: `ac` is true whenever "Power" appears; the id digits are read as the percentage;
    # "not charging" or "discharging" is reported as charging because it contains "charging".
    assert parse_batt(BATT_DISCHARGING_REAL) == (False, 100, "no")
    assert parse_batt(BATT_AC) == (True, 100, "full")
    assert parse_batt(BATT_DISCHARGING) == (False, 63, "no")
    assert parse_batt(BATT_NOT_CHARGING) == (True, 38, "no")
    assert parse_batt(BATT_CHARGING) == (True, 80, "yes")


def test_parse_batt_without_a_source_line_is_all_none():
    # Red if: an unreadable probe is parsed as "on battery" (False) instead of None.
    assert parse_batt("") == (None, None, None)
    assert parse_batt("garbage\n") == (None, None, None)


def test_parse_ac_reads_the_wattage_or_none():
    # Red if: the regex drops the digits, or "No adapter attached." parses to a number.
    assert parse_ac(AC_WATTS_SYNTHETIC) == 96
    assert parse_ac(AC_NONE_REAL) is None


def test_parse_therm_reads_the_speed_limit_only_when_recorded():
    # Red if: the clean "No CPU power status" notes parse to a limit, or the limit is missed.
    assert parse_therm(THERM_CLEAN) is None
    assert parse_therm(THERM_LIMITED) == 65
    assert parse_therm("") is None


def test_parse_clamshell_reads_the_state_not_the_causes_sleep_key():
    # Red if: a first-match regex on "AppleClamshell" reads CausesSleep = Yes as "closed".
    assert parse_clamshell(CLAMSHELL_OPEN) is True
    assert parse_clamshell(CLAMSHELL_CLOSED) is False
    assert parse_clamshell("") is None


def test_parse_memory_pressure_reads_the_free_percentage_not_a_page_count():
    # Red if: the regex grabs the first number in the page-size line or a "Pages ..." count.
    assert parse_memory_pressure(PRESSURE) == 82
    assert parse_memory_pressure("garbage") is None


def test_parse_ps_lists_heavy_processes_above_the_threshold_and_excludes_the_editor():
    # Red if: RSS is read as bytes instead of KiB (nothing crosses 1 GiB), the exclude patterns
    # are ignored (this session's own process reads as busy), or a non-matching command is listed.
    busy = parse_ps(PS, patterns=HEAVY_PATTERNS, min_rss_bytes=GIB)
    assert busy == ("/Users/x/.venv/bin/python -m scripts.bench_flux_step --mode control",)


def test_parse_ps_threshold_is_inclusive_and_real_output_lists_nothing():
    # Red if: the comparison is `>` instead of `>=` (1 GiB == 1048576 KiB), or system daemons match.
    at_line = "  7 1048576 /usr/bin/python -m scripts.bench_flux_step\n"
    assert parse_ps(at_line, patterns=HEAVY_PATTERNS) == (
        "/usr/bin/python -m scripts.bench_flux_step",
    )
    below = "  7 1048575 /usr/bin/python -m scripts.bench_flux_step\n"
    assert parse_ps(below, patterns=HEAVY_PATTERNS) == ()
    assert parse_ps(PS_REAL_HEAD, patterns=HEAVY_PATTERNS, min_rss_bytes=0) == ()


def _p(**over: object) -> Preflight:
    base: dict[str, object] = {
        "ac_power": True,
        "battery_percent": 100,
        "charging": "full",
        "charger_watts": 96,
        "cpu_speed_limit": None,
        "lid_open": True,
        "free_disk_bytes": 124 * GIB,
        "memory_free_percent": 88,
        "busy_processes": (),
    }
    base.update(over)
    return Preflight(**base)  # type: ignore[arg-type]


def test_a_healthy_laptop_sample_passes():
    # Red if: any gate fires on a healthy sample.
    assert check(_p()) == []


@pytest.mark.parametrize(
    ("over", "gate"),
    [
        ({"ac_power": False}, "ac_power"),
        ({"battery_percent": 39}, "battery"),
        ({"battery_percent": 49, "charging": "no"}, "not_charging"),
        ({"cpu_speed_limit": 99}, "cpu_speed_limit"),
        ({"lid_open": False}, "lid"),
        ({"free_disk_bytes": 20 * GIB - 1}, "free_disk"),
        ({"memory_free_percent": 19}, "memory_free"),
        ({"busy_processes": ("python -m scripts.bench_flux_step",)}, "busy"),
        ({"ac_power": None}, "unreadable:ac_power"),
        ({"free_disk_bytes": None}, "unreadable:free_disk_bytes"),
        ({"memory_free_percent": None}, "unreadable:memory_free_percent"),
        ({"busy_processes": None}, "unreadable:busy_processes"),
        ({"lid_open": None}, "unreadable:lid_open"),
    ],
)
def test_each_gate_fires_by_name(over, gate):
    # Red if: the named branch in `check` is removed or renamed.
    assert gate in check(_p(**over))


@pytest.mark.parametrize(
    "over",
    [
        {"battery_percent": 40},
        {"memory_free_percent": 20},
        {"free_disk_bytes": 20 * GIB},
        {"cpu_speed_limit": 100},
        {"battery_percent": 100, "charging": "full"},
        {"battery_percent": 80, "charging": "yes"},
    ],
)
def test_the_boundaries_pass(over):
    # Red if: a floor becomes exclusive (`<=`), or "full" at 100 % trips not_charging.
    assert check(_p(**over)) == []


def test_a_desktop_sample_has_no_battery_or_lid_gates():
    # Red if: `lid` or `unreadable:lid_open` applies without a battery (a desktop has no lid sensor).
    desktop = _p(
        battery_percent=None,
        charging=None,
        lid_open=None,
        charger_watts=None,
        busy_processes=(),
    )
    assert check(desktop) == []


def test_not_charging_needs_ac_power():
    # Red if: not_charging fires on battery power (only `ac_power` should) or at 100 % on AC.
    assert check(_p(ac_power=False, battery_percent=80, charging="no")) == ["ac_power"]
    assert check(_p(battery_percent=100, charging="no")) == []


@pytest.mark.parametrize(("percent", "fires"), [(49, True), (50, False), (80, False)])
def test_not_charging_fires_only_below_half_charge(percent, fires):
    # macOS Optimized Battery Charging holds a MacBook on AC at 80 % "not charging"; the operating
    # rule is that 50 % or more may run. Red if: the floor is dropped (80 % refused, the typical
    # laptop on AC) or made inclusive (50 % refused).
    failed = check(_p(battery_percent=percent, charging="no"))
    assert ("not_charging" in failed) is fires


def test_the_not_charging_floor_is_a_parameter():
    # Red if: the 50 % floor is hardcoded instead of read from the parameter.
    assert "not_charging" in check(_p(battery_percent=80, charging="no"), min_not_charging=81)


def _runner(outputs):
    def run(*argv: str) -> str:
        return outputs.get(argv[0], "")

    return run


def test_sample_assembles_the_parsed_probes():
    # Red if: a probe is wired to the wrong parser or argv, so a field lands in another's slot.
    ps_text = "  9 4194304 /usr/bin/python -m scripts.bench_flux_step\n" + PS_REAL_HEAD
    seen: list[tuple[str, ...]] = []

    def run(*argv: str) -> str:
        seen.append(argv)
        if argv[:3] == ("pmset", "-g", "batt"):
            return BATT_CHARGING
        if argv[:3] == ("pmset", "-g", "ac"):
            return AC_WATTS_SYNTHETIC
        if argv[:3] == ("pmset", "-g", "therm"):
            return THERM_LIMITED
        if argv[0] == "ioreg":
            return CLAMSHELL_OPEN
        if argv[0] == "memory_pressure":
            return PRESSURE
        if argv[0] == "ps":
            return ps_text
        return ""

    got = sample(run=run)
    assert got.ac_power is True
    assert got.battery_percent == 80
    assert got.charging == "yes"
    assert got.charger_watts == 96
    assert got.cpu_speed_limit == 65
    assert got.lid_open is True
    assert got.memory_free_percent == 82
    assert got.busy_processes == ("/usr/bin/python -m scripts.bench_flux_step",)
    assert isinstance(got.free_disk_bytes, int)
    assert got.free_disk_bytes > 0
    assert len(seen) == 6


def test_sample_with_every_probe_empty_is_unreadable_not_healthy():
    # Red if: an empty probe parses to a healthy value (e.g. busy_processes == () instead of None).
    got = sample(run=_runner({}))
    assert got.ac_power is None
    assert got.busy_processes is None
    assert got.memory_free_percent is None
    assert "unreadable:ac_power" in check(got)
    assert "unreadable:busy_processes" in check(got)


def test_as_dict_carries_every_field():
    # Red if: a field is dropped from the run record.
    d = _p().as_dict()
    assert d["battery_percent"] == 100
    assert set(d) == {
        "ac_power",
        "battery_percent",
        "charging",
        "charger_watts",
        "cpu_speed_limit",
        "lid_open",
        "free_disk_bytes",
        "memory_free_percent",
        "busy_processes",
    }
