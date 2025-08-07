"""Setup configuration for pytests."""

import pytest
from pathlib import Path
from unittest.mock import patch

@pytest.fixture(autouse=True)
def patch_rglob_to_tmp(tmp_path):
    """Auto-patch Path.rglob to only search within tmp_path (not the real repo)."""
    original_rglob = Path.rglob

    def fake_rglob(self, pattern):
        # Force all rglob() calls to search inside tmp_path
        return original_rglob(tmp_path, pattern)

    with patch("pathlib.Path.rglob", new=fake_rglob):
        yield
