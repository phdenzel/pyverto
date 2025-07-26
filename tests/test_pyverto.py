# SPDX-FileCopyrightText: 2025-present phdenzel <phdenzel@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Unit tests for pyverto."""

import sys
import re
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from pyverto import utils
from pyverto import vc
import pyverto.__main__ as main_mod


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
        ("1.2.3", (1, 2, 3, None, None, None)),
        ("1.2.3-alpha1", (1, 2, 3, "alpha", 1, None)),
        ("1.2.3-beta4+post2", (1, 2, 3, "beta", 4, 2)),
        ("1.0.0-rc", (1, 0, 0, "rc", 1, None)),
        ("1.0.0-dev5", (1, 0, 0, "dev", 5, None)),
    ],
)
def test_parse_version(v, expected):
    """Test `parse_version` for various inputs."""
    assert utils.parse_version(v) == expected


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
        ("alpha", "1.2.3", "1.2.3-alpha0"),
        ("alpha", "1.2.3-alpha0", "1.2.3-alpha1"),
        ("beta", "1.2.3-beta0", "1.2.3-beta1"),
        ("pre", "1.2.3", "1.2.3-rc0"),
        ("pre", "1.2.3-rc0", "1.2.3-rc1"),
        ("rev", "1.2.3", "1.2.3+post1"),
        ("rev", "1.2.3+post2", "1.2.3+post3"),
        ("dev", "1.2.3", "1.2.3-dev0"),
        ("dev", "1.2.3-dev0", "1.2.3-dev1"),
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

def test_main_function_current_version(capfd):
    """Test `main` case: valid version."""
    fake_version_file = Path("fake_pkg/__about__.py")

    # Patch dependencies used inside main()
    with patch.object(main_mod, "find_version_file", return_value=fake_version_file), \
         patch.object(main_mod, "get_current_version", return_value="1.0.0"), \
         patch.object(main_mod, "parse_args") as mock_args:
        
        # Simulate CLI arguments: "version"
        mock_args.return_value.command = "version"

        # Run main()
        main_mod.main()

        # Capture printed output
        out, err = capfd.readouterr()
        assert "1.0.0" in out
        

def test_main_function_bumps_and_commits(capfd):
    """Test `main` case: valid bump + commit."""
    fake_version_file = Path("fake_pkg/__about__.py")

    # Patch dependencies used inside main()
    with patch.object(main_mod, "find_version_file", return_value=fake_version_file), \
         patch.object(main_mod, "get_current_version", return_value="1.0.0"), \
         patch.object(main_mod, "write_version") as mock_write, \
         patch.object(main_mod, "git_commit_and_tag") as mock_git, \
         patch.object(main_mod, "parse_args") as mock_args:
        
        # Simulate CLI arguments: bump "minor" and commit
        mock_args.return_value.command = "minor"
        mock_args.return_value.commit = True

        # Run main()
        main_mod.main()

        # Assertions
        mock_write.assert_called_once_with(fake_version_file, "1.1.0")
        mock_git.assert_called_once_with(fake_version_file, "1.1.0", "1.0.0")

        # Capture printed output
        out, err = capfd.readouterr()
        assert "Bumped version in" in out
        assert "1.0.0 → 1.1.0" in out


def test_main_function_bumps_no_commit(capfd):
    """Test `main` case: valid bump (no commit)."""
    fake_version_file = Path("fake_pkg/__about__.py")

    # Patch dependencies used inside main()
    with patch.object(main_mod, "find_version_file", return_value=fake_version_file), \
         patch.object(main_mod, "get_current_version", return_value="1.0.0"), \
         patch.object(main_mod, "write_version") as mock_write, \
         patch.object(main_mod, "parse_args") as mock_args:
        
        # Simulate CLI arguments: bump "minor" and commit
        mock_args.return_value.command = "minor"
        mock_args.return_value.commit = False

        # Run main()
        main_mod.main()

        # Assertions
        mock_write.assert_called_once_with(fake_version_file, "1.1.0")

        # Capture printed output
        out, err = capfd.readouterr()
        assert "Bumped version in" in out
        assert "1.0.0 → 1.1.0" in out


def test_main_function_exit_syserr(capfd):
    """Test `main` case: valid run."""
    # Patch dependencies used inside main()
    with patch.object(main_mod, "find_version_file", return_value=None), \
         patch.object(main_mod, "parse_args") as mock_args:
        
        # Simulate CLI arbitrary arguments
        mock_args.return_value.command = "minor"
        mock_args.return_value.commit = False

        # Expect SystemExit
        with pytest.raises(SystemExit) as excinfo:
            main_mod.main()

        # Assertions
        out, err = capfd.readouterr()
        assert "Could not locate a file with __version__" in str(excinfo.value) or out
