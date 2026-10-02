"""
The runnable examples in the Markdown documentation must still produce
exactly the output recorded next to them (see scripts/check_doc_examples.py).
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKER = REPO_ROOT / "scripts" / "check_doc_examples.py"

pytestmark = pytest.mark.skipif(
    not CHECKER.exists() or not (REPO_ROOT / "docs" / "fixtures").is_dir(),
    reason="documentation sources not available (installed package)",
)


def test_documentation_examples_match_recorded_output():
    completed = subprocess.run(
        [sys.executable, str(CHECKER)], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
