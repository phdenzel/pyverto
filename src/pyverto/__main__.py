#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025-present phdenzel <phdenzel@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Version management for any Python project.

Usage:
  pyverto [command] [--commit]

Commands:
  version    Print out current version
  release    Remove any pre-release/dev/post suffix (finalize version)
  major      Increment the major version
  minor      Increment the minor version
  micro      Increment the micro (patch) version
  alpha      Convert to or increment alpha pre-release
  beta       Convert to or increment beta pre-release
  pre        Convert to or increment rc (release candidate)
  rev        Increment post-release (+postN)
  dev        Convert to or increment dev release (-devN)

Examples:
  pyverto dev
  pyverto pre --commit
"""

import argparse
from pyverto.utils import (
    find_version_file,
    get_current_version,
    write_version,
    parse_version,
    format_version,
)
from pyverto.vc import git_commit_and_tag


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Version incrementor.")
    parser.add_argument(
        "command",
        choices=[
            "version",
            "release",
            "major",
            "minor",
            "micro",
            "alpha",
            "beta",
            "pre",
            "rev",
            "dev",
        ],
        help="Version bump type",
    )
    parser.add_argument("--commit", action="store_true", help="Commit & tag in git")
    parser.add_argument("--no-tag", action="store_true", help="Do not tag when committing")
    args = parser.parse_args()
    return args


def bump(command: str, current_version: str):
    """Bump version in various ways.

    Args:
        command: The manner how the version is incremented.
        current_version: Version string to be incremented.
    """
    major, minor, micro, label, num, post = parse_version(current_version)
    if command == "version":
        return format_version(major, minor, micro, label, num, post)
    if command == "release":
        return format_version(major, minor, micro)
    if command == "major":
        return format_version(major + 1, 0, 0)
    if command == "minor":
        return format_version(major, minor + 1, 0)
    if command in ("micro", "patch"):
        return format_version(major, minor, micro + 1)
    if command in ("alpha", "beta", "pre"):
        stage = {"pre": "rc"}.get(command, command)
        if label == stage:
            num = (num or 0) + 1
        else:
            num = 0
        return format_version(major, minor, micro, stage, num)
    if command == "rev":
        return format_version(major, minor, micro, label, num, (post or 0) + 1)
    if command == "dev":
        if label == "dev":
            num = (num or 0) + 1
        else:
            num = 0
        return format_version(major, minor, micro, "dev", num)
    raise ValueError(f"Unknown command: {command}")


def main():
    """Main entry point."""
    args = parse_args()
    version_file = find_version_file()
    if not version_file:
        raise SystemExit("Error: Could not locate a file with __version__.")
    current_version = get_current_version(version_file)
    if args.command == "version":
        print(current_version)
    else:
        new_version = bump(args.command, current_version)
        write_version(version_file, new_version)
        print(f"Bumped version in {version_file}: {current_version} → {new_version}")
        if args.commit:
            git_commit_and_tag(version_file, new_version, current_version, tag=(not args.no_tag))


if __name__ == "__main__":
    main()
