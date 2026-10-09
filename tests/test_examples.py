"""
Keeps examples/ executable: every example script must run cleanly against
the committed sample data, and that data must match its generator.
"""

import filecmp
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_SCRIPTS = sorted(EXAMPLES_DIR.glob("0*.py"))

pytestmark = pytest.mark.skipif(not EXAMPLES_DIR.is_dir(), reason="examples/ not available (installed package)")


def _run(script: Path, cwd: Path) -> str:
    completed = subprocess.run(
        [sys.executable, str(script)], cwd=cwd, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS, ids=lambda p: p.name)
def test_example_runs(script, tmp_path):
    copy = tmp_path / "examples"
    shutil.copytree(EXAMPLES_DIR, copy, ignore=shutil.ignore_patterns("output", "__pycache__"))
    assert _run(copy / script.name, tmp_path)


def test_boundary_example_shows_label_change(tmp_path):
    output = _run(EXAMPLES_DIR / "02_window_boundaries.py", tmp_path)
    assert "[t, t+24h)          label=0" in output
    assert "(t, t+24h]         label=1" in output
    assert "(t, t+23:59:59]    label=0" in output


def test_multiclass_example_shows_class_labels(tmp_path):
    output = _run(EXAMPLES_DIR / "09_multiclass_labels.py", tmp_path)
    assert "label counts: NF=16, A=0, B=24, C=2, M=54, X=22" in output
    assert "binary M label == (multiclass in {M, X}) for every image: True" in output
    assert "2024-03-01 12:00:00         13600     M               X" in output


def test_sample_data_matches_generator(tmp_path):
    copy = tmp_path / "data"
    shutil.copytree(EXAMPLES_DIR / "data", copy)
    _run(copy / "make_sample_data.py", tmp_path)
    for committed in (EXAMPLES_DIR / "data").glob("*.*"):
        if committed.suffix in (".csv", ".txt"):
            assert filecmp.cmp(committed, copy / committed.name, shallow=False), committed.name
