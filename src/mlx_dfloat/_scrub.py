"""Home-directory scrubbing for JSON the tools write (result files are committed to a public repo)."""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def scrub_home(value: Any, home: str = str(Path.home())) -> Any:
    """``value`` with every occurrence of ``home`` as a whole path component written as ``~``.

    Walks dicts, lists and tuples (tuples come back as lists, as JSON writes them); a sibling
    directory that merely shares the prefix (``/home/ab2`` for ``/home/ab``) is left alone.
    """
    home = home.rstrip("/")
    if not home:
        return value
    pattern = re.compile(re.escape(home) + r"(?=/|\s|$)")
    return _scrub(value, pattern)


def _scrub(value: Any, pattern: re.Pattern[str]) -> Any:
    if isinstance(value, str):
        return pattern.sub("~", value)
    if isinstance(value, Mapping):
        return {k: _scrub(v, pattern) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_scrub(v, pattern) for v in value]
    return value
