# SPDX-FileCopyrightText: 2025-present phdenzel <phdenzel@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Unit tests for pyverto."""

import sys
import re
import types
import textwrap
from pathlib import Path
import importlib
import builtins
from unittest.mock import MagicMock, patch
import pytest
import pyverto.utils as utils
import pyverto.vc as vc
import pyverto.header as header
import pyverto.__main__ as main_mod


def test_version_string_is_defined():
    """Ensure __version__ string is defined."""
    from pyverto import __about__

    # Ensure __version__ exists and is a non-empty string
    assert isinstance(__about__.__version__, str)
    assert __about__.__version__.strip() != ""


@pytest.fixture
def temp_version_file(tmp_path):
    """Set up temporary __about__.py with a __version__ variable."""
    f = tmp_path / "__about__.py"
    f.write_text('__version__ = "1.2.3"')
    return f


@pytest.fixture
def temp_init_file(tmp_path):
    """Set up temporary mypkg/__init__.py with a __version__ variable."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    f = pkg / "__init__.py"
    f.write_text('__version__ = "0.1.0"')
    return f


def test_find_version_file_prefers_about(tmp_path):
    """Test `find_version_file` case: Priority __about__.py > __init__.py."""
    about = tmp_path / "__about__.py"
    initf = tmp_path / "__init__.py"
    about.write_text('__version__ = "1.0.0"')
    initf.write_text('__version__ = "0.1.0"')

    # Patch Path.glob to search tmp_path
    with patch("pyverto.utils.Path.glob", side_effect=lambda p: [about, initf]):
        found = utils.find_version_file()
        assert found == about


def test_find_version_file_fallback_init(temp_init_file):
    """Test `find_version_file` case: __init__.py if __about__.py not found."""
    # Only __init__.py present
    with patch("pyverto.utils.Path.glob", side_effect=lambda p: [temp_init_file]):
        found = utils.find_version_file()
        assert found == temp_init_file


def test_get_current_version_reads_version(temp_version_file):
    """Test `get_current_version` from __about__.py."""
    assert utils.get_current_version(temp_version_file) == "1.2.3"


def test_get_current_version_raises_if_missing(tmp_path):
    """Test `get_current_version` case: failure if none found."""
    f = tmp_path / "__about__.py"
    f.write_text("# no version here")
    with pytest.raises(ValueError):
        utils.get_current_version(f)


def test_write_version_updates_file(temp_version_file):
    """Test `write_version` in __about__.py."""
    utils.write_version(temp_version_file, "2.0.0")
    content = temp_version_file.read_text()
    assert re.search(r'__version__ = "2\.0\.0"', content)


@pytest.mark.parametrize(
    "v,expected",
    [
        ("1.2.3", (1, 2, 3, None, None, None, "", "")),
        ("1.2.3-alpha1", (1, 2, 3, "alpha", 1, None, "-", "")),
        ("1.2.3-beta4+post2", (1, 2, 3, "beta", 4, 2, "-", "+")),
        ("1.0.0-rc", (1, 0, 0, "rc", 1, None, "-", "")),
        ("1.0.0-dev5", (1, 0, 0, "dev", 5, None, "-", "")),
        ("1.0.3.dev2", (1, 0, 3, "dev", 2, None, ".", "")),
        ("1.0.3dev2.post1", (1, 0, 3, "dev", 2, 1, "", ".")),
    ],
)
def test_parse_version(v, expected):
    """Test `parse_version` for various inputs."""
    out = utils.parse_version(v)
    assert out == expected


def test_parse_invalid_version():
    """Test `parse_version` case: ValueError."""
    with pytest.raises(ValueError):
        utils.parse_version("Hello World!")


def test_format_version_all_fields():
    """Test `format_version` case: full input."""
    v = utils.format_version(1, 2, 3, "beta", 4, 2)
    assert v == "1.2.3-beta4+post2"


def test_format_version_no_suffix():
    """Test `format_version` case: minimal input."""
    v = utils.format_version(1, 0, 0)
    assert v == "1.0.0"


def test_parse_args_with_command(monkeypatch):
    """Test CLI commands: case minor + --commit flag."""
    test_args = ["pyverto", "minor", "--commit"]
    monkeypatch.setattr(sys, "argv", test_args)

    args = main_mod.parse_args()
    assert args.command == "minor"
    assert args.commit is True


def test_parse_args_without_commit(monkeypatch):
    """Test CLI commands: case dev."""
    test_args = ["pyverto", "dev"]
    monkeypatch.setattr(sys, "argv", test_args)

    args = main_mod.parse_args()
    assert args.command == "dev"
    assert args.commit is False


@pytest.mark.parametrize(
    "cmd,current,expected",
    [
        ("version", "1.2.3-dev2", "1.2.3-dev2"),
        ("release", "1.2.3-dev2", "1.2.3"),
        ("major", "1.2.3", "2.0.0"),
        ("minor", "1.2.3", "1.3.0"),
        ("micro", "1.2.3", "1.2.4"),
        ("alpha", "1.2.3", "1.2.3.alpha0"),
        ("alpha", "1.2.3-alpha0", "1.2.3-alpha1"),
        ("alpha", "1.2.3.alpha1", "1.2.3.alpha2"),
        ("beta", "1.2.3-beta0", "1.2.3-beta1"),
        ("pre", "1.2.3", "1.2.3.rc0"),
        ("pre", "1.2.3-rc0", "1.2.3-rc1"),
        ("rev", "1.2.3", "1.2.3-post1"),
        ("rev", "1.2.3+post2", "1.2.3+post3"),
        ("dev", "1.2.3", "1.2.3.dev0"),
        ("dev", "1.2.3-dev0", "1.2.3-dev1"),
        ("dev", "1.2.3.dev1", "1.2.3.dev2"),
        ("dev", "1.2.3dev0", "1.2.3dev1"),
        ("rev", "1.2.3.dev0", "1.2.3.dev0-post1"),
        ("rev", "1.2.3.dev0-post1", "1.2.3.dev0-post2"),
    ],
)
def test_bump_variants(cmd, current, expected):
    """Test `bump` for various inputs."""
    assert main_mod.bump(cmd, current) == expected


def test_bump_invalid_command():
    """Test `bump` case: failure if command not known."""
    with pytest.raises(ValueError):
        main_mod.bump("foobar", "1.2.3")


def test_git_commit_and_tag_success(monkeypatch, temp_version_file, capfd):
    """Test `git_commit_and_tag` case: valid input."""
    repo_mock = MagicMock()
    monkeypatch.setattr(vc, "Repo", lambda *a, **kw: repo_mock)

    vc.git_commit_and_tag(temp_version_file, "2.0.0", "1.0.0")
    out, err = capfd.readouterr()
    assert "2.0.0" in out

    repo_mock.index.add.assert_called_with([str(temp_version_file)])
    repo_mock.index.commit.assert_called_once()
    repo_mock.create_tag.assert_called_with("v2.0.0")


def test_git_commit_and_tag_success_no_tag(monkeypatch, temp_version_file, capfd):
    """Test `git_commit_and_tag` case: valid input."""
    repo_mock = MagicMock()
    monkeypatch.setattr(vc, "Repo", lambda *a, **kw: repo_mock)

    vc.git_commit_and_tag(temp_version_file, "2.0.0", "1.0.0", tag=False)
    out, err = capfd.readouterr()
    assert "2.0.0" in out

    repo_mock.index.add.assert_called_with([str(temp_version_file)])
    repo_mock.index.commit.assert_called_once()
    repo_mock.create_tag.assert_not_called()


def test_git_commit_and_tag_missing_old_version(monkeypatch, temp_version_file, capfd):
    """Test `git_commit_and_tag` case: valid input."""
    repo_mock = MagicMock()
    monkeypatch.setattr(vc, "Repo", lambda *a, **kw: repo_mock)

    vc.git_commit_and_tag(temp_version_file, "2.0.0")
    out, err = capfd.readouterr()
    assert "2.0.0" in out

    repo_mock.index.add.assert_called_with([str(temp_version_file)])
    repo_mock.index.commit.assert_called_once()
    repo_mock.create_tag.assert_called_with("v2.0.0")


def test_git_commit_and_tag_not_a_repo(monkeypatch, temp_version_file):
    """Test `git_commit_and_tag` case: failure if not a git repo."""
    monkeypatch.setattr(vc, "Repo", MagicMock(side_effect=vc.InvalidGitRepositoryError))
    with pytest.raises(SystemExit):
        vc.git_commit_and_tag(temp_version_file, "2.0.0")


def _patch_import(monkeypatch, missing=(), tomllib_fake=None, tomli_fake=None):
    """Helper to patch import behavior for tomllib/tomli."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in missing:
            raise ImportError(f"Simulated missing {name}")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import

    # Inject fake modules if provided
    if tomllib_fake:
        monkeypatch.setitem(sys.modules, "tomllib", tomllib_fake)
    else:
        sys.modules.pop("tomllib", None)

    if tomli_fake:
        monkeypatch.setitem(sys.modules, "tomli", tomli_fake)
    else:
        sys.modules.pop("tomli", None)

    # Reload utils so it picks up this environment
    importlib.reload(utils)
    return real_import


def test_load_tomllib_prefers_tomllib(monkeypatch):
    """Test `load_tomllib` case: tomllib available."""
    fake_tomllib = types.SimpleNamespace(loads=lambda _: {"ok": True})
    real_import = _patch_import(monkeypatch, missing=("tomli",), tomllib_fake=fake_tomllib)

    try:
        mod = utils.load_tomllib()
        assert mod is fake_tomllib
        assert mod.loads("dummy") == {"ok": True}
    finally:
        builtins.__import__ = real_import


def test_load_tomllib_falls_back_to_tomli(monkeypatch):
    """Test `load_tomllib` case: tomli available."""
    fake_tomli = types.SimpleNamespace(loads=lambda _: {"fallback": True})
    real_import = _patch_import(monkeypatch, missing=("tomllib",), tomli_fake=fake_tomli)

    try:
        mod = utils.load_tomllib()
        assert mod is fake_tomli
        assert mod.loads("dummy") == {"fallback": True}
    finally:
        builtins.__import__ = real_import


def test_load_tomllib_raises_if_both_missing(monkeypatch):
    """Test `load_tomllib` case: none availble."""
    real_import = _patch_import(monkeypatch, missing=("tomllib", "tomli"))

    try:
        with pytest.raises(SystemExit) as excinfo:
            utils.load_tomllib()
        assert "tomllib or tomli" in str(excinfo.value)
    finally:
        builtins.__import__ = real_import


def test_project_name_from_pyproject(tmp_path):
    """Test `get_project_name` case: pyproject.toml."""
    pyproj = tmp_path / "pyproject.toml"
    pyproj.write_text(
        textwrap.dedent("""
    [project]
    name = "my-project"
    """)
    )
    assert header.get_project_name(tmp_path) == "my-project"


def test_project_name_from_setup_cfg_instead_pyproject(tmp_path):
    """Test `get_project_name` case: setup.cfg."""
    pyproj = tmp_path / "pyproject.toml"
    pyproj.write_text(
        textwrap.dedent("""
    [project]
    notname = "my-project"
    """)
    )
    setup_cfg = tmp_path / "setup.cfg"
    setup_cfg.write_text(
        textwrap.dedent("""
    [metadata]
    name = legacy-project
    """)
    )
    assert header.get_project_name(tmp_path) == "legacy-project"


def test_project_name_from_setup_cfg(tmp_path):
    """Test `get_project_name` case: setup.cfg."""
    setup_cfg = tmp_path / "setup.cfg"
    setup_cfg.write_text(
        textwrap.dedent("""
    [metadata]
    name = legacy-project
    """)
    )
    assert header.get_project_name(tmp_path) == "legacy-project"


def test_project_name_from_src_package_instead_setup_cfg(tmp_path):
    """Test `get_project_name` case: src/**."""
    setup_cfg = tmp_path / "setup.cfg"
    setup_cfg.write_text(
        textwrap.dedent("""
    [metadata]
    noname = legacy-project
    """)
    )
    src_dir = tmp_path / "src"
    pkg_dir = src_dir / "realpkg"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("# package")
    assert header.get_project_name(tmp_path) == "realpkg"


def test_project_name_from_flat_package(tmp_path):
    """Test `get_project_name` case: src/flatpkg."""
    pkg_dir = tmp_path / "flatpkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("# flat package")
    assert header.get_project_name(tmp_path) == "flatpkg"


def test_project_name_skips_tests_docs_examples(tmp_path):
    """Test `get_project_name` case: actualpkg/."""
    # These should be ignored as possible names
    for name in ["tests", "docs", "examples"]:
        d = tmp_path / name
        d.mkdir()
        (d / "__init__.py").write_text("# dummy")

    real_pkg = tmp_path / "actualpkg"
    real_pkg.mkdir()
    (real_pkg / "__init__.py").write_text("# real package")

    assert header.get_project_name(tmp_path) == "actualpkg"


def test_project_name_from_git(monkeypatch, tmp_path):
    """Test `get_project_name` case: git repo."""
    repo_mock = MagicMock()
    repo_dir = tmp_path / "reponame"
    repo_dir.mkdir()
    repo_mock.working_tree_dir = str(repo_dir)

    # Patch Repo to return our fake repo
    monkeypatch.setattr("pyverto.header.Repo", lambda *a, **kw: repo_mock)
    monkeypatch.setattr("pyverto.header.InvalidGitRepositoryError", Exception)

    # Remove pyproject/setup.cfg and no package dirs
    print(Path(repo_mock.working_tree_dir).name)
    assert header.get_project_name(tmp_path) == "reponame"


def test_project_name_defaults_to_dir_name(monkeypatch, tmp_path):
    """Test `get_project_name` case: base_path (last resort)."""
    # Simulate no metadata and no git repo
    monkeypatch.setattr("git.Repo", MagicMock(side_effect=Exception))
    assert header.get_project_name(tmp_path) == tmp_path.name


def test_project_name_prioritizes_pyproject_over_others(tmp_path):
    """Test `get_project_name` case: pyproject.toml and others."""
    # Set up multiple sources, should prefer pyproject.toml
    pyproj = tmp_path / "pyproject.toml"
    pyproj.write_text(
        textwrap.dedent("""
    [project]
    name = "preferred-project"
    """)
    )
    setup_cfg = tmp_path / "setup.cfg"
    setup_cfg.write_text("[metadata]\nname = other-project")
    pkg_dir = tmp_path / "mypkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("# pkg")
    assert header.get_project_name(tmp_path) == "preferred-project"


def test_generate_default_header_reads_pyproject(tmp_path):
    """Test `generate_default_header` case: defaults from pyproject.toml."""
    pyproj = tmp_path / "pyproject.toml"
    pyproj.write_text(
        textwrap.dedent("""\
    [project]
    name = "mytool"
    authors = [{name = "Jane Doe", email = "jane@example.com"}]
    license = { text = "MIT" }
    """)
    )
    hdr = header.generate_default_header(pyproj)
    assert "Jane Doe" in hdr
    assert "jane@example.com" in hdr
    assert "MIT" in hdr


def test_generate_default_header_reads_pyproject_no_authors(tmp_path):
    """Test `generate_default_header` case: defaults from pyproject.toml."""
    pyproj = tmp_path / "pyproject.toml"
    pyproj.write_text(
        textwrap.dedent("""\
    [project]
    name = "mytool"
    license = { text = "MIT" }
    """)
    )
    hdr = header.generate_default_header(pyproj)
    print(hdr)
    assert "author" in hdr
    assert "author@example.com" in hdr
    assert "MIT" in hdr


def test_generate_default_header_reads_pyproject_alt_license(tmp_path):
    """Test `generate_default_header` case: defaults from pyproject.toml."""
    pyproj = tmp_path / "pyproject.toml"
    pyproj.write_text(
        textwrap.dedent("""\
    [project]
    name = "mytool"
    authors = [{name = "Jane Doe", email = "jane@example.com"}]
    license = "MIT"
    """)
    )
    hdr = header.generate_default_header(pyproj)
    assert "Jane Doe" in hdr
    assert "MIT" in hdr


def test_insert_header_with_shebang_and_docstring(tmp_path):
    """Test `insert_header` case: shebang + docstring."""
    src = tmp_path / "test.py"
    original_text = textwrap.dedent("""\
    #!/usr/bin/env python3
    \"\"\"Module docstring\"\"\"

    print("hi")
    """)
    src.write_text(original_text)
    header_txt = "# SPDX-License-Identifier: MIT"
    header.insert_header(src, header_txt)
    text = src.read_text()
    # Header must be after shebang but before docstring
    lines = text.splitlines()
    print()
    print("-" * 5)
    print(original_text)
    print("-" * 5)
    print(text)
    assert lines[0].startswith("#!")
    assert lines[1] == header_txt


def test_insert_header_replaces_existing_spdx(tmp_path):
    """Test `insert_header` case: replace existing spdx header."""
    src = tmp_path / "test.py"
    original_text = textwrap.dedent("""\
    # SPDX-License-Identifier: Apache-2.0
    print("hi")
    """)
    src.write_text(original_text)
    new_header = "# SPDX-License-Identifier: MIT"
    header.insert_header(src, new_header)
    text = src.read_text()
    print()
    print("-" * 5)
    print(original_text)
    print("-" * 5)
    print(text)
    assert text.startswith(new_header)
    assert "Apache-2.0" not in text


def test_insert_header_no_shebang_or_docstring(tmp_path):
    """Test `insert_header` case: no shebang, no docstring."""
    src = tmp_path / "test.py"
    original_text = "print('hello')\n"
    src.write_text(original_text)
    header.insert_header(src, "# HEADER")
    text = src.read_text()
    print()
    print("-" * 5)
    print(original_text)
    print("-" * 5)
    print(text)
    assert text.startswith("# HEADER\nprint(")


def test_edit_header_with_header_file(tmp_path):
    """Test `edit_header` case: header_file."""
    # Prepare a header file
    header_file = tmp_path / "header.txt"
    header_file.write_text("# Custom Header")

    # Create a dummy Python file
    pyfile = tmp_path / "src" / "pkg"
    pyfile.mkdir(parents=True)
    file_path = pyfile / "file.py"
    file_path.write_text("print('hi')\n")

    # Patch insert_header to verify it is called
    with (
        patch.object(main_mod, "insert_header") as mock_insert,
    ):
        main_mod.edit_header(header_file=header_file, dry_run=False)

        mock_insert.assert_called_once()
        called_file, called_header = mock_insert.call_args[0]
        assert str(called_file).endswith("file.py")
        assert called_header == "# Custom Header"


def test_edit_header_with_direct_header_text(tmp_path):
    """Test `edit_header` case: header_text."""
    pyfile = tmp_path / "src" / "pkg"
    pyfile.mkdir(parents=True)
    file_path = pyfile / "file.py"
    file_path.write_text("print('hi')\n")

    # Patch insert_header to verify it is called
    with (
        patch.object(main_mod, "insert_header") as mock_insert,
    ):
        main_mod.edit_header(header_text="# Direct Header", dry_run=False)

        mock_insert.assert_called_once()
        called_file, called_header = mock_insert.call_args[0]
        assert called_header == "# Direct Header"


def test_edit_header_dry_run_prints_files_and_header(tmp_path, capfd):
    """Test `edit_header` case: dry_run."""
    pyfile = tmp_path / "src" / "pkg"
    pyfile.mkdir(parents=True)
    file_path = pyfile / "file.py"
    file_path.write_text("print('hi')\n")

    # Simulate default header by patching generate_default_header/get_project_name
    with (
        patch.object(main_mod, "generate_default_header", return_value="# Default Header"),
        patch.object(main_mod, "insert_header") as mock_insert,
    ):
        main_mod.edit_header(dry_run=True)

        mock_insert.assert_not_called()
        out, _ = capfd.readouterr()
        assert "file.py" in out
        assert "# Default Header" in out


def test_edit_header_skips_hidden_files(tmp_path, capfd):
    """Test `edit_header` case: hidden files."""
    pyfile = tmp_path / "src" / "pkg"
    pyfile.mkdir(parents=True)
    file_path = pyfile / "file.py"
    file_path.write_text("print('hi')\n")
    file_path2 = pyfile / ".hidden_file.py"
    file_path2.write_text("print('easter_egg')\n")

    # Simulate default header by patching generate_default_header/get_project_name
    with (
        patch.object(main_mod, "generate_default_header", return_value="# Default Header"),
        patch.object(main_mod, "insert_header") as mock_insert,
    ):
        main_mod.edit_header(dry_run=True)

        mock_insert.assert_not_called()
        out, _ = capfd.readouterr()
        assert "file.py" in out
        assert ".hidden_file.py" not in out
        assert "# Default Header" in out


def test_edit_header_raises_without_pyproject(tmp_path):
    """Test `edit_header` case: no pyproject failure."""
    pkgpath = tmp_path / "src" / "pkg"
    pkgpath.mkdir(parents=True)

    def fake_exists(path_self):
        return False

    # No header_file, no header_text, and pyproject.toml missing
    with (
        patch("pyverto.header.get_project_name", return_value="pkg"),
        patch("pathlib.Path.exists", new=fake_exists),
        pytest.raises(SystemExit),
    ):
        main_mod.edit_header(None, None, dry_run=True)


def test_edit_header_uses_generated_header(tmp_path):
    """Test `edit_header` case: fake pyproject."""
    # Create dummy pyproject.toml and Python file
    fake_pyproj = tmp_path / "pyproject.toml"
    original_text = textwrap.dedent("""
    [project]
    name = "mypkg"
    authors = [{name = "John", email = "john@example.com"}]
    license = {text = "MIT"}
    """)
    fake_pyproj.write_text(original_text)
    pkg_dir = tmp_path / "mypkg"
    pkg_dir.mkdir()
    file_path = pkg_dir / "file.py"
    file_path.write_text("print('hi')\n")

    def fake_path(arg: str = None):
        # If asking for "pyproject.toml", return fake one
        if arg and "pyproject.toml" in arg:
            return fake_pyproj
        return Path(arg) if arg is not None else Path()

    # Patch insert_header to capture calls
    with (
        patch.object(main_mod, "insert_header") as mock_insert,
        patch.object(main_mod, "Path", side_effect=fake_path),
    ):
        main_mod.edit_header(dry_run=False)

    mock_insert.assert_called_once()
    called_file, called_header = mock_insert.call_args[0]
    assert str(called_file).endswith("file.py")
    assert "SPDX-License-Identifier" in called_header
    assert "John" in called_header





def test_main_function_current_version(capfd):
    """Test `main` case: valid version."""
    fake_version_file = Path("fake_pkg/__about__.py")

    # Patch dependencies used inside main()
    with (
        patch.object(main_mod, "find_version_file", return_value=fake_version_file),
        patch.object(main_mod, "get_current_version", return_value="1.0.0"),
        patch.object(main_mod, "parse_args") as mock_args,
    ):
        # Simulate CLI arguments: "version"
        mock_args.return_value.command = "version"

        # Run main()
        main_mod.main()

        # Capture printed output
        out, err = capfd.readouterr()
        assert "1.0.0" in out


def test_main_function_dry_run_header(capfd):
    """Test `main` case: valid version."""
    # fake_version_file = Path("fake_pkg/__about__.py")

    # Patch dependencies used inside main()
    with (
        # patch.object(main_mod, "edit_header") as mock_edit,
        patch.object(main_mod, "insert_header") as mock_insert,
        patch.object(main_mod, "parse_args") as mock_args,
    ):
        # Simulate CLI arguments: "header"
        mock_args.return_value.command = "header"
        mock_args.return_value.dry_run = True

        # Run main()
        main_mod.main()

        mock_insert.assert_not_called()


def test_main_function_bumps_and_commits(capfd):
    """Test `main` case: valid bump + commit + tag."""
    fake_version_file = Path("fake_pkg/__about__.py")

    # Patch dependencies used inside main()
    with (
        patch.object(main_mod, "find_version_file", return_value=fake_version_file),
        patch.object(main_mod, "get_current_version", return_value="1.0.0"),
        patch.object(main_mod, "write_version") as mock_write,
        patch.object(main_mod, "git_commit_and_tag") as mock_git,
        patch.object(main_mod, "parse_args") as mock_args,
    ):
        # Simulate CLI arguments: bump "minor" and commit
        mock_args.return_value.command = "minor"
        mock_args.return_value.commit = True
        mock_args.return_value.no_tag = False
        mock_args.return_value.dry_run = False

        # Run main()
        main_mod.main()

        # Assertions
        mock_write.assert_called_once_with(fake_version_file, "1.1.0")
        mock_git.assert_called_once_with(fake_version_file, "1.1.0", "1.0.0", tag=True)

        # Capture printed output
        out, err = capfd.readouterr()
        assert "Bumped version in" in out
        assert "1.0.0 → 1.1.0" in out


def test_main_function_bumps_and_commits_not_tag(capfd):
    """Test `main` case: valid bump + commit (not tag)."""
    fake_version_file = Path("fake_pkg/__about__.py")

    # Patch dependencies used inside main()
    with (
        patch.object(main_mod, "find_version_file", return_value=fake_version_file),
        patch.object(main_mod, "get_current_version", return_value="1.0.0"),
        patch.object(main_mod, "write_version") as mock_write,
        patch.object(main_mod, "git_commit_and_tag") as mock_git,
        patch.object(main_mod, "parse_args") as mock_args,
    ):
        # Simulate CLI arguments: bump "minor" and commit
        mock_args.return_value.command = "minor"
        mock_args.return_value.commit = True
        mock_args.return_value.no_tag = True
        mock_args.return_value.dry_run = False

        # Run main()
        main_mod.main()

        # Assertions
        mock_write.assert_called_once_with(fake_version_file, "1.1.0")
        mock_git.assert_called_once_with(fake_version_file, "1.1.0", "1.0.0", tag=False)

        # Capture printed output
        out, err = capfd.readouterr()
        assert "Bumped version in" in out
        assert "1.0.0 → 1.1.0" in out


def test_main_function_bumps_no_commit(capfd):
    """Test `main` case: valid bump (no commit)."""
    fake_version_file = Path("fake_pkg/__about__.py")

    # Patch dependencies used inside main()
    with (
        patch.object(main_mod, "find_version_file", return_value=fake_version_file),
        patch.object(main_mod, "get_current_version", return_value="1.0.0"),
        patch.object(main_mod, "write_version") as mock_write,
        patch.object(main_mod, "parse_args") as mock_args,
    ):
        # Simulate CLI arguments: bump "minor" and commit
        mock_args.return_value.command = "minor"
        mock_args.return_value.commit = False
        mock_args.return_value.dry_run = False

        # Run main()
        main_mod.main()

        # Assertions
        mock_write.assert_called_once_with(fake_version_file, "1.1.0")

        # Capture printed output
        out, err = capfd.readouterr()
        assert "Bumped version in" in out
        assert "1.0.0 → 1.1.0" in out


def test_main_function_bumps_dry_run(capfd):
    """Test `main` case: dry-run bump."""
    fake_version_file = Path("fake_pkg/__about__.py")

    # Patch dependencies used inside main()
    with (
        patch.object(main_mod, "find_version_file", return_value=fake_version_file),
        patch.object(main_mod, "get_current_version", return_value="1.0.0"),
        patch.object(main_mod, "write_version") as mock_write,
        patch.object(main_mod, "parse_args") as mock_args,
    ):
        # Simulate CLI arguments: bump "minor" and commit
        mock_args.return_value.command = "minor"
        mock_args.return_value.commit = False
        mock_args.return_value.dry_run = True

        # Run main()
        main_mod.main()

        # Assertions
        mock_write.assert_not_called()

        # Capture printed output
        out, err = capfd.readouterr()
        assert "Bumped version in" in out
        assert "1.0.0 → 1.1.0" in out


def test_main_function_exit_syserr(capfd):
    """Test `main` case: valid run."""
    # Patch dependencies used inside main()
    with (
        patch.object(main_mod, "find_version_file", return_value=None),
        patch.object(main_mod, "parse_args") as mock_args,
    ):
        # Simulate CLI arbitrary arguments
        mock_args.return_value.command = "minor"
        mock_args.return_value.commit = False

        # Expect SystemExit
        with pytest.raises(SystemExit) as excinfo:
            main_mod.main()

        # Assertions
        out, err = capfd.readouterr()
        assert "Could not locate a file with __version__" in str(excinfo.value) or out
