# SPDX-FileCopyrightText: 2025-present phdenzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of pyverto
# SPDX-License-Identifier: MIT
"""Manage headers in project files."""

import datetime
from pathlib import Path
from git import Repo, InvalidGitRepositoryError
from pyverto.utils import load_tomllib
tomllib = load_tomllib()


IGNORED_DIRS = {"tests", "test", "docs", "examples", "scripts", "dist", ".github", ".venv", "venv"}


def get_project_name(base_path: Path | str = ".") -> str | list[str]:
    """Extract the project name from pyproject.toml or other sources."""
    base_path = Path(base_path) if base_path is not None else Path(".")
    pyproject = base_path / "pyproject.toml"
    setup_cfg = base_path / "setup.cfg"
    src_path = base_path / "src"
    # search in pyproject.toml
    if pyproject.exists():
        data = tomllib.loads(pyproject.read_text())
        if "project" in data and "name" in data["project"]:
            return data["project"]["name"]
    # search in setup.cfg
    if setup_cfg.exists():
        for line in setup_cfg.read_text().splitlines():
            if line.strip().startswith("name ="):
                return line.split("=", 1)[1].strip()
    # search in src/** or any folder containing __init__.py
    for pkg_dir in [src_path, base_path]:
        if pkg_dir.exists():
            for path in pkg_dir.iterdir():
                if (
                    path.is_dir()
                    and path.name not in IGNORED_DIRS
                    and (path / "__init__.py").exists()
                ):
                    return path.name
    # search git repository name
    # Repo(base_path, search_parent_directories=True)
    try:
        repo = Repo(base_path, search_parent_directories=True)
        return Path(repo.working_tree_dir).name
    except InvalidGitRepositoryError:
        pass

    # Last resort
    return base_path.resolve().name


def generate_default_header(pyproject_path: Path, defaults: dict = {}) -> str:
    """Build default SPDX header from pyproject.toml."""
    data = tomllib.loads(pyproject_path.read_text())
    project = data.get("project", {})

    year = datetime.datetime.now().year
    # Extract project name
    name = project.get("name", "this project")

    # Extract author info
    authors = project.get("authors", None)
    if authors and isinstance(authors, list):
        author_name = authors[0].get("name", "author")
        author_email = authors[0].get("email", "author@example.com")
    else:
        author_name = "author"
        author_email = "author@example.com"

    # Extract license info
    license_text = project.get("license", None) or defaults.get("license", "MIT")

    return (
        f"# SPDX-FileCopyrightText: {year}-present {author_name} <{author_email}>\n"
        f"# SPDX-FileNotice: Part of {name}\n"
        f"# SPDX-License-Identifier: {license_text}"
    )


def insert_header(file_path: Path, header_text: str):
    """Insert or replace SPDX header in a project python file."""
    lines = file_path.read_text().splitlines()
    new_lines = []
    shebang = None

    # Detect shebang
    if lines and lines[0].startswith("#!"):
        shebang = lines.pop(0)

    # Strip any existing header block
    while lines and lines[0].startswith("#"):
        lines.pop(0)

    # Place header before module docstring
    if lines and (lines[0].startswith('"""') or lines[0].startswith("'''")):
        # Insert before docstring
        new_lines = ([shebang] if shebang else []) + [header_text] + lines
    else:
        # Insert at top (after shebang, if present)
        new_lines = ([shebang] if shebang else []) + [header_text] + lines

    file_path.write_text("\n".join(new_lines) + "\n")
