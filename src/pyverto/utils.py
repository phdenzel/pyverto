# SPDX-FileCopyrightText: 2025-present phdenzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of pyverto
# SPDX-License-Identifier: MIT
"""Utility functions for locating, parsing, and writing __version__."""

from pathlib import Path
import re
from pyverto.regexp import VERSION_RE


def load_tomllib():
    """Return a tomllib-compatible module (fallback for Python 3.10 is tomli)."""
    try:
        import tomllib

        return tomllib
    except ImportError:
        try:
            import tomli as tomllib

            return tomllib
        except ImportError:
            raise SystemExit(
                "Error: This program requires either tomllib or tomli but neither is available"
            )


def find_version_file() -> Path | None:
    """Locate the file containing the __version__ variable.

    Checks common locations:
      - **/__about__.py
      - **/__init__.py
    """
    candidates = []

    # Search for common candidates
    for pattern in [
        "src/**/__about__.py",
        "src/**/__init__.py",
        "**/__about__.py",
        "**/__init__.py",
    ]:
        candidates.extend(Path().glob(pattern))

    # Filter only those actually containing __version__
    candidates = [p for p in candidates if p.is_file() and "__version__" in p.read_text()]

    # Prefer __about__.py over __init__.py
    for p in candidates:
        if p.name == "__about__.py":
            return p
    return candidates[0] if candidates else None


def get_current_version(version_file: Path) -> str:
    """Fetch version string from version file."""
    content = version_file.read_text()
    _match = re.search(VERSION_RE, content)
    if not _match:
        raise ValueError(f"Could not find __version__ in {version_file}")
    return _match.group(1)


def write_version(version_file: Path, new_version: str):
    """Write new version string to version file."""
    content = version_file.read_text()
    new_content = re.sub(VERSION_RE, f'__version__ = "{new_version}"', content)
    version_file.write_text(new_content)


def parse_version(v: str) -> tuple[int, int, int, str | None, int | None, int | None, str, str]:
    """Parse into a version tuple (major, minor, micro, suffix, number, post).

    Example: 1.2.3-beta4+post2 -> (1, 2, 3, 'beta', 4, 2)
    """
    re_full = re.compile(
        r"^(?P<main>\d+\.\d+\.\d+)"
        r"(?P<suffix_sep>[-\.]?)"
        r"(?P<suffix>(alpha|beta|rc|dev))?"
        r"(?P<suffix_num>\d*)"
        r"(?P<post_sep>(\+|[-\.]))?"
        r"(?P<post>post\d+)?$"
    )
    m = re_full.match(v)
    if not m:
        raise ValueError(f"Invalid version string: {v}")

    major, minor, micro = map(int, m.group("main").split("."))
    label = m.group("suffix")
    num = int(m.group("suffix_num")) if m.group("suffix_num") else (1 if label else None)
    post = m.group("post")
    post_n = int(post[4:]) if post else None
    pre_sep = m.group("suffix_sep") if m.group("suffix_sep") else ""
    post_sep = m.group("post_sep") if m.group("post_sep") else ""
    return major, minor, micro, label, num, post_n, pre_sep, post_sep


def format_version(
    major: int,
    minor: int,
    micro: int,
    label: str | None = None,
    num: int | None = None,
    post: int | None = None,
    pre_separator: str = "-",
    post_separator: str = "+",
):
    """Format a version tuple string corresponding to the version tuple."""
    v = f"{major}.{minor}.{micro}"
    if label:
        v += f"{pre_separator}{label}{num if num is not None else 0}"
    if post:
        v += f"{post_separator}post{post}"
    return v
