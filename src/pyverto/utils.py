# SPDX-FileCopyrightText: 2025-present phdenzel <phdenzel@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Utility functions for locating, parsing, and writing __version__."""

from pathlib import Path
import re
from pyverto.regexp import VERSION_RE


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
    candidates = [
        p for p in candidates if p.is_file() and "__version__" in p.read_text()
    ]

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


def parse_version(v: str) -> tuple[int, int, int, str | None, int | None, int | None]:
    """Parse into a version tuple (major, minor, micro, suffix, number, post).

    Example: 1.2.3-beta4+post2 -> (1, 2, 3, 'beta', 4, 2)
    """
    base, post = (v.split("+", 1) + [None])[:2]
    post_n = int(post[4:]) if post and post.startswith("post") else None
    main, *suffix = re.split(r"-(alpha|beta|rc|dev)(\d*)", base)
    major, minor, micro = map(int, main.split("."))
    if suffix:
        label, num = suffix[0], int(suffix[1] or 1)
    else:
        label, num = None, None
    return major, minor, micro, label, num, post_n


def format_version(
    major: int,
    minor: int,
    micro: int,
    label: str | None = None,
    num: int | None = None,
    post: int | None = None,
):
    """Format a version tuple string corresponding to the version tuple."""
    v = f"{major}.{minor}.{micro}"
    if label:
        v += f"-{label}{num if num is not None else 0}"
    if post:
        v += f"+post{post}"
    return v
