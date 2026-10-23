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
# Examples 07-08 (PyTorch) and 10-11 (TensorFlow) need downloaded real
# images and a framework; they are checked in test_pytorch_examples.py and
# test_tensorflow_examples.py instead.
REAL_DATA_EXAMPLES = {
    "07_pytorch_full_disk_images.py", "08_pytorch_full_disk_sequences.py",
    "10_tensorflow_full_disk_images.py", "11_tensorflow_full_disk_sequences.py",
}
EXAMPLE_SCRIPTS = sorted(p for p in EXAMPLES_DIR.glob("[0-9][0-9]_*.py") if p.name not in REAL_DATA_EXAMPLES)

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
    assert "label=0  starts exactly at t+24h: excluded" in output
    assert "label=1  window 10 min longer: included" in output
    assert "label=0  peak exactly at t+w: excluded" in output
    assert "label=0  starts exactly at t: excluded" in output


def test_multiclass_example_shows_class_labels(tmp_path):
    output = _run(EXAMPLES_DIR / "09_multiclass_labels.py", tmp_path)
    assert "label counts: NF=18, A=0, B=23, C=2, M=54, X=21" in output
    assert "binary M label == (multiclass in {M, X}) for every image: True" in output
    assert "2024-03-01 12:00:00         13600     M               X" in output


def test_cumulative_index_example_shows_paper_value_and_labels(tmp_path):
    output = _run(EXAMPLES_DIR / "13_cumulative_index.py", tmp_path)
    assert "cumulative index: 48.5" in output
    assert "label (index >= 10): 1" in output
    assert "label (index >= 20): 1" in output
    assert "label (index >= 50): 0" in output
    assert "empty window: 0.0" in output
    assert "A/B-only window: 0.0" in output


def test_sample_data_matches_generator(tmp_path):
    copy = tmp_path / "data"
    shutil.copytree(EXAMPLES_DIR / "data", copy)
    _run(copy / "make_sample_data.py", tmp_path)
    for committed in (EXAMPLES_DIR / "data").glob("*.*"):
        if committed.suffix in (".csv", ".txt"):
            assert filecmp.cmp(committed, copy / committed.name, shallow=False), committed.name
