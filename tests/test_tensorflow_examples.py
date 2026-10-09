"""
Checks for the real-image TensorFlow examples (examples 10 and 11).

The layout conversion and the sequence checks need neither TensorFlow nor
images. Running the examples needs TensorFlow, Pillow with JPEG 2000
support and the images from examples/real_data/fetch_real_data.py; the
comparison with the PyTorch examples (07, 08) also needs PyTorch. Tests
whose requirements are missing are skipped, so the base suite runs without
any framework installed.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
REAL_DATA = EXAMPLES / "real_data"

pytestmark = pytest.mark.skipif(not REAL_DATA.is_dir(), reason="examples/ not available (installed package)")


def _load(name):
    spec = importlib.util.spec_from_file_location(f"example_{name[:2]}", EXAMPLES / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _has(module):
    return importlib.util.find_spec(module) is not None


def _images_available():
    if not REAL_DATA.is_dir():
        return False
    manifest = pd.read_csv(REAL_DATA / "image_manifest.csv")
    return all((REAL_DATA / p).exists() for p in manifest.loc[manifest["status"] == "ok", "image_path"])


needs_tf = pytest.mark.skipif(not _has("tensorflow") or not _images_available(),
                              reason="needs tensorflow and the downloaded real images")
needs_tf_and_torch = pytest.mark.skipif(not _has("tensorflow") or not _has("torch") or not _images_available(),
                                        reason="needs tensorflow, torch and the downloaded real images")


# ── Without TensorFlow ───────────────────────────────────────────────────

def test_channels_first_is_converted_to_channels_last():
    images = _load("10_tensorflow_full_disk_images.py")
    chw = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)  # C=2, H=3, W=4
    hwc = images.to_channels_last(chw)
    assert hwc.shape == (3, 4, 2) and hwc.flags["C_CONTIGUOUS"]
    for c in range(2):
        np.testing.assert_array_equal(hwc[:, :, c], chw[c])
    with pytest.raises(ValueError):
        images.to_channels_last(chw[0])


def test_sequence_checks_reject_bad_rows_before_decoding():
    sequences = _load("11_tensorflow_full_disk_sequences.py")
    t = pd.to_datetime(["2017-09-03 00:59:49.9", "2017-09-03 01:59:49.9", "2017-09-03 02:59:49.9"])
    good = dict(timestamps=list(t), image_path=["a", "b", "c"], n_images=3, sequence_end=t[-1])
    cadence = pd.Timedelta("1h")

    def row(**changes):
        return pd.Series({**good, **changes})

    off_grid = list(t[:2]) + [t[2] + pd.Timedelta("100ms")]
    with pytest.raises(ValueError, match="not 0 days 01:00:00 apart"):
        sequences.load_sequence(row(timestamps=off_grid, sequence_end=off_grid[-1]), REAL_DATA, cadence)
    with pytest.raises(ValueError, match="not 0 days 01:00:00 apart"):
        sequences.load_sequence(row(timestamps=list(reversed(t)), sequence_end=t[0]), REAL_DATA, cadence)
    with pytest.raises(ValueError, match="last image"):
        sequences.load_sequence(row(sequence_end=t[1]), REAL_DATA, cadence)
    with pytest.raises(ValueError, match="paths"):
        sequences.load_sequence(row(image_path=["a", "b"]), REAL_DATA, cadence)


def test_tensorflow_shapes_follow_the_shared_decoder():
    images = _load("10_tensorflow_full_disk_images.py")
    sequences = _load("11_tensorflow_full_disk_sequences.py")
    assert images.IMAGE_SHAPE == (256, 256, 1)
    assert sequences.SEQUENCE_SHAPE == (3, 256, 256, 1)


# ── With TensorFlow ──────────────────────────────────────────────────────

def _run(script, output):
    completed = subprocess.run([sys.executable, str(EXAMPLES / script), "--output", str(output)],
                               capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


@needs_tf
def test_tensorflow_image_example_runs(tmp_path):
    output = _run("10_tensorflow_full_disk_images.py", tmp_path)
    assert "batch 0: images [4, 256, 256, 1] float32" in output
    assert "batch 1: images [4, 256, 256, 1] float32" in output
    assert "batch 2: images [1, 256, 256, 1] float32" in output
    assert "All 9 labels match an independent recomputation from the catalog." in output
    assert "9 images: 4 positive, 5 negative; all checks passed" in output


@needs_tf
def test_tensorflow_sequence_example_runs(tmp_path):
    output = _run("11_tensorflow_full_disk_sequences.py", tmp_path)
    assert "candidates 8, emitted 2, skipped 6" in output
    assert "batch 0: images [2, 3, 256, 256, 1] float32  labels [0, 0]" in output
    assert "2 sequences: 0 positive, 2 negative; all checks passed" in output


@needs_tf_and_torch
def test_tensorflow_and_pytorch_examples_produce_the_same_label_tables(tmp_path):
    for torch_script, tf_script, torch_table, tf_table in [
        ("07_pytorch_full_disk_images.py", "10_tensorflow_full_disk_images.py",
         "07_image_labels.csv", "10_image_labels.csv"),
        ("08_pytorch_full_disk_sequences.py", "11_tensorflow_full_disk_sequences.py",
         "08_sequence_labels.csv", "11_sequence_labels.csv"),
    ]:
        _run(torch_script, tmp_path / "torch")
        _run(tf_script, tmp_path / "tf")
        assert (tmp_path / "torch" / torch_table).read_text() == (tmp_path / "tf" / tf_table).read_text()


@needs_tf_and_torch
def test_tensorflow_and_pytorch_loaders_yield_the_same_samples(tmp_path):
    torch_images = _load("07_pytorch_full_disk_images.py")
    tf_images = _load("10_tensorflow_full_disk_images.py")
    torch_sequences = _load("08_pytorch_full_disk_sequences.py")
    tf_sequences = _load("11_tensorflow_full_disk_sequences.py")
    real = tf_images.real
    import flare_indexer as fidx

    index_path = tmp_path / "index.csv"
    real.build_image_index(index_path)
    options = dict(prediction_window="24h", strategy=fidx.BinaryThresholdStrategy("M"), target="full_disk",
                   event_time="start", interval_mode="right_closed")
    labels = fidx.DatasetBuilder(include_metadata=True, **options).build(index_path, real.CATALOG)
    sequences = fidx.DatasetBuilder(cadence="1h", observation_window="3h", sliding_window="1h", **options) \
        .build(index_path, real.CATALOG)

    torch_dataset = torch_images.make_dataset_class()(labels, real.REAL_DATA)
    tf_samples = list(tf_images.make_dataset(labels, real.REAL_DATA).as_numpy_iterator())
    assert len(torch_dataset) == len(tf_samples) == 9
    for i, (tf_image, tf_label, tf_time, tf_path) in enumerate(tf_samples):
        image, label, timestamp, path = torch_dataset[i]
        np.testing.assert_array_equal(image.permute(1, 2, 0).numpy(), tf_image)       # [C,H,W] -> [H,W,C]
        assert (int(label), timestamp, path) == (int(tf_label), tf_time.decode(), tf_path.decode())

    torch_dataset = torch_sequences.make_dataset_class()(sequences, real.REAL_DATA, "1h")
    tf_samples = list(tf_sequences.make_dataset(sequences, real.REAL_DATA, "1h").as_numpy_iterator())
    assert len(torch_dataset) == len(tf_samples) == 2
    for i, (tf_images_t, tf_label, tf_end, tf_paths) in enumerate(tf_samples):
        images, label, end, paths = torch_dataset[i]
        np.testing.assert_array_equal(images.permute(0, 2, 3, 1).numpy(), tf_images_t)  # [T,C,H,W] -> [T,H,W,C]
        assert (int(label), end, paths) == (int(tf_label), tf_end.decode(), [p.decode() for p in tf_paths])
        assert end == real.format_time(sequences["sequence_end"].iloc[i])
