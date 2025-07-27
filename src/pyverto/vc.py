# SPDX-FileCopyrightText: 2025-present phdenzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of pyverto. Distributed as-is with no warranty.
# SPDX-License-Identifier: MIT
"""Functions for handling version control features."""

from pathlib import Path
from git import Repo, InvalidGitRepositoryError


def git_commit_and_tag(
        version_file: Path, version: str, old_version: str | None = None, tag: bool = True,
):
    """Commit the version file and tag with new version."""
    try:
        repo = Repo(Path().resolve(), search_parent_directories=True)
    except InvalidGitRepositoryError:
        raise SystemExit("Error: Not a Git repository, cannot commit/tag.")

    repo.index.add([str(version_file)])
    if old_version is None:
        old_version = ""
    repo.index.commit(f"Bump version: {old_version} → {version}")
    if tag:
        repo.create_tag(f"v{version}")
    print(f"Committed{' and tagged ' if tag else ' '} v{version}")
