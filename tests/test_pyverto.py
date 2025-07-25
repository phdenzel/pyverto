import re
from unittest.mock import MagicMock, patch
import pytest
from pyverto import utils
from pyverto import vc
import pyverto.__main__ as main_mod


@pytest.fixture
def temp_version_file(tmp_path):
    f = tmp_path / "__about__.py"
    f.write_text('__version__ = "1.2.3"')
    return f


@pytest.fixture
def temp_init_file(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    f = pkg / "__init__.py"
    f.write_text('__version__ = "0.1.0"')
    return f


def test_find_version_file_prefers_about(tmp_path):
    about = tmp_path / "__about__.py"
    initf = tmp_path / "__init__.py"
    about.write_text('__version__ = "1.0.0"')
    initf.write_text('__version__ = "0.1.0"')

    # Patch Path.glob to search tmp_path
    with patch("pyverto.utils.Path.glob", side_effect=lambda p: [about, initf]):
        found = utils.find_version_file()
        assert found == about


def test_find_version_file_fallback_init(temp_init_file):
    # Only __init__.py present
    with patch("pyverto.utils.Path.glob", side_effect=lambda p: [temp_init_file]):
        found = utils.find_version_file()
        assert found == temp_init_file


def test_get_current_version_reads_version(temp_version_file):
    assert utils.get_current_version(temp_version_file) == "1.2.3"


def test_get_current_version_raises_if_missing(tmp_path):
    f = tmp_path / "__about__.py"
    f.write_text("# no version here")
    with pytest.raises(ValueError):
        utils.get_current_version(f)


def test_write_version_updates_file(temp_version_file):
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
    assert utils.parse_version(v) == expected


def test_format_version_all_fields():
    v = utils.format_version(1, 2, 3, "beta", 4, 2)
    assert v == "1.2.3-beta4+post2"


def test_format_version_no_suffix():
    v = utils.format_version(1, 0, 0)
    assert v == "1.0.0"


# ------------------------
# __main__.py bump tests
# ------------------------

@pytest.mark.parametrize(
    "cmd,current,expected",
    [
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
    assert main_mod.bump(cmd, current) == expected


def test_bump_invalid_command():
    with pytest.raises(ValueError):
        main_mod.bump("foobar", "1.2.3")


# ------------------------
# vc.py git_commit_and_tag tests
# ------------------------

def test_git_commit_and_tag_success(monkeypatch, temp_version_file):
    repo_mock = MagicMock()
    monkeypatch.setattr(vc, "Repo", lambda *a, **kw: repo_mock)

    vc.git_commit_and_tag(temp_version_file, "2.0.0", "1.0.0")

    repo_mock.index.add.assert_called_with([str(temp_version_file)])
    repo_mock.index.commit.assert_called_once()
    repo_mock.create_tag.assert_called_with("v2.0.0")


def test_git_commit_and_tag_not_a_repo(monkeypatch, temp_version_file):
    monkeypatch.setattr(vc, "Repo", MagicMock(side_effect=vc.InvalidGitRepositoryError))
    with pytest.raises(SystemExit):
        vc.git_commit_and_tag(temp_version_file, "2.0.0")
