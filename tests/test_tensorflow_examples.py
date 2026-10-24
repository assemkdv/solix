"""
Checks for the real-image TensorFlow examples (examples 08, 09 and 11).

The layout conversion and the sequence checks (shared helpers in
examples/real_data) need neither TensorFlow nor images. Running 08 and 09
needs TensorFlow, Pillow with JPEG 2000 support and the images from
examples/real_data/fetch_real_data.py; 11 needs TensorFlow, astropy and the
SHARP patches. The comparisons with the PyTorch examples (06, 07, 10) also
need PyTorch. Tests
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
from flare_indexer.builder import DatasetBuilder

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
REAL_DATA = EXAMPLES / "real_data"

pytestmark = pytest.mark.skipif(not REAL_DATA.is_dir(), reason="examples/ not available (installed package)")

if REAL_DATA.is_dir():
    sys.path.insert(0, str(REAL_DATA))
    import hmi_jp2
    import real_examples as real


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


needs_tf_only = pytest.mark.skipif(not _has("tensorflow"), reason="needs tensorflow")
needs_tf = pytest.mark.skipif(not _has("tensorflow") or not _images_available(),
                              reason="needs tensorflow and the downloaded real images")
needs_tf_and_torch = pytest.mark.skipif(not _has("tensorflow") or not _has("torch") or not _images_available(),
                                        reason="needs tensorflow, torch and the downloaded real images")


# ── Without TensorFlow ───────────────────────────────────────────────────

def test_channels_first_is_converted_to_channels_last():
    chw = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)  # C=2, H=3, W=4
    hwc = hmi_jp2.to_channels_last(chw)
    assert hwc.shape == (3, 4, 2) and hwc.flags["C_CONTIGUOUS"]
    for c in range(2):
        np.testing.assert_array_equal(hwc[:, :, c], chw[c])
    with pytest.raises(ValueError):
        hmi_jp2.to_channels_last(chw[0])


def test_sequence_checks_reject_bad_rows_before_decoding():
    t = pd.to_datetime(["2017-09-03 00:59:49.9", "2017-09-03 01:59:49.9", "2017-09-03 02:59:49.9"])
    good = dict(timestamps=list(t), image_path=["a", "b", "c"], n_images=3, sequence_end=t[-1])
    cadence = pd.Timedelta("1h")

    def row(**changes):
        return pd.Series({**good, **changes})

    real.check_sequence(row(), cadence)
    off_grid = list(t[:2]) + [t[2] + pd.Timedelta("100ms")]
    with pytest.raises(ValueError, match="not 0 days 01:00:00 apart"):
        real.check_sequence(row(timestamps=off_grid, sequence_end=off_grid[-1]), cadence)
    with pytest.raises(ValueError, match="not 0 days 01:00:00 apart"):
        real.check_sequence(row(timestamps=list(reversed(t)), sequence_end=t[0]), cadence)
    with pytest.raises(ValueError, match="last image"):
        real.check_sequence(row(sequence_end=t[1]), cadence)
    with pytest.raises(ValueError, match="paths"):
        real.check_sequence(row(image_path=["a", "b"]), cadence)


# ── With TensorFlow ──────────────────────────────────────────────────────

@needs_tf_only
def test_tensorflow_shapes_follow_the_shared_decoder():
    images = _load("08_tensorflow_images.py")
    sequences = _load("09_tensorflow_sequences.py")
    assert images.IMAGE_SHAPE == (256, 256, 1)
    assert sequences.SEQUENCE_SHAPE == (3, 256, 256, 1)


def _run(script, output):
    completed = subprocess.run([sys.executable, str(EXAMPLES / script), "--output", str(output)],
                               capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


@needs_tf
def test_tensorflow_image_example_runs(tmp_path):
    output = _run("08_tensorflow_images.py", tmp_path)
    assert "batch 0: images [4, 256, 256, 1] float32" in output
    assert "batch 1: images [4, 256, 256, 1] float32" in output
    assert "batch 2: images [1, 256, 256, 1] float32" in output
    assert "All 9 labels match an independent recomputation from the catalog." in output
    assert "9 images: 4 positive, 5 negative; all checks passed" in output


@needs_tf
def test_tensorflow_sequence_example_runs(tmp_path):
    output = _run("09_tensorflow_sequences.py", tmp_path)
    assert "candidates 8, emitted 2, skipped 6" in output
    assert "batch 0: images [2, 3, 256, 256, 1] float32  labels [0, 0]" in output
    assert "2 sequences: 0 positive, 2 negative; all checks passed" in output


@needs_tf_and_torch
def test_tensorflow_and_pytorch_examples_produce_the_same_label_tables(tmp_path):
    for torch_script, tf_script, torch_table, tf_table in [
        ("06_pytorch_images.py", "08_tensorflow_images.py",
         "06_image_labels.csv", "08_image_labels.csv"),
        ("07_pytorch_sequences.py", "09_tensorflow_sequences.py",
         "07_sequence_labels.csv", "09_sequence_labels.csv"),
    ]:
        _run(torch_script, tmp_path / "torch")
        _run(tf_script, tmp_path / "tf")
        assert (tmp_path / "torch" / torch_table).read_text() == (tmp_path / "tf" / tf_table).read_text()


@needs_tf_and_torch
def test_tensorflow_and_pytorch_loaders_yield_the_same_samples(tmp_path):
    torch_images = _load("06_pytorch_images.py")
    tf_images = _load("08_tensorflow_images.py")
    torch_sequences = _load("07_pytorch_sequences.py")
    tf_sequences = _load("09_tensorflow_sequences.py")
    real = tf_images.real
    import flare_indexer as fidx

    index_path = tmp_path / "index.csv"
    real.build_image_index(index_path)
    options = dict(prediction_window="24h", strategy=fidx.BinaryThresholdStrategy("M"), target="full_disk",
                   event_time="start")
    labels = DatasetBuilder(include_metadata=True, **options).build(index_path, real.CATALOG)
    sequences = DatasetBuilder(cadence="1h", observation_window="3h", sliding_window="1h", **options) \
        .build(index_path, real.CATALOG)

    torch_dataset = torch_images.FlareDataset(labels, real.REAL_DATA)
    tf_dataset = tf_images.FlareDataset(labels, real.REAL_DATA, batch_size=4)
    tf_samples = [sample for i in range(len(tf_dataset)) for sample in zip(*tf_dataset[i])]
    assert len(torch_dataset) == len(tf_samples) == 9
    for i, (tf_image, tf_label) in enumerate(tf_samples):
        image, label = torch_dataset[i]
        np.testing.assert_array_equal(image.permute(1, 2, 0).numpy(), tf_image)       # [C,H,W] -> [H,W,C]
        assert int(label) == int(tf_label) == labels["label"].iloc[i]

    torch_dataset = torch_sequences.FlareSequenceDataset(sequences, real.REAL_DATA, "1h")
    tf_dataset = tf_sequences.FlareSequenceDataset(sequences, real.REAL_DATA, "1h", batch_size=2)
    tf_samples = [sample for i in range(len(tf_dataset)) for sample in zip(*tf_dataset[i])]
    assert len(torch_dataset) == len(tf_samples) == 2
    for i, (tf_images_t, tf_label) in enumerate(tf_samples):
        images, label = torch_dataset[i]
        np.testing.assert_array_equal(images.permute(0, 2, 3, 1).numpy(), tf_images_t)  # [T,C,H,W] -> [T,H,W,C]
        assert int(label) == int(tf_label) == sequences["label"].iloc[i]


@needs_tf
def test_tensorflow_datasets_train_with_model_fit(tmp_path):
    tf_images = _load("08_tensorflow_images.py")
    tf_sequences = _load("09_tensorflow_sequences.py")
    real = tf_images.real
    import flare_indexer as fidx
    from tensorflow import keras

    index_path = tmp_path / "index.csv"
    real.build_image_index(index_path)
    options = dict(prediction_window="24h", strategy=fidx.BinaryThresholdStrategy("M"), target="full_disk",
                   event_time="start")
    labels = DatasetBuilder(include_metadata=True, **options).build(index_path, real.CATALOG)
    sequences = DatasetBuilder(cadence="1h", observation_window="3h", sliding_window="1h", **options) \
        .build(index_path, real.CATALOG)

    for shape, dataset in [
        (tf_images.IMAGE_SHAPE, tf_images.FlareDataset(labels, real.REAL_DATA, batch_size=4)),
        (tf_sequences.SEQUENCE_SHAPE, tf_sequences.FlareSequenceDataset(sequences, real.REAL_DATA, "1h", batch_size=2)),
    ]:
        model = keras.Sequential([keras.Input(shape), keras.layers.Flatten(), keras.layers.Dense(1, activation="sigmoid")])
        model.compile(optimizer="adam", loss="binary_crossentropy")
        history = model.fit(dataset, epochs=1, verbose=0)
        assert np.isfinite(history.history["loss"][0])


# ── Active-region patches (example 11) ───────────────────────────────────

def _patches_available():
    manifest = pd.read_csv(REAL_DATA / "sharp_manifest.csv")
    return all((REAL_DATA / p).exists() for p in manifest["image_path"])


needs_tf_and_patches = pytest.mark.skipif(
    not _has("tensorflow") or not _has("astropy") or not _patches_available(),
    reason="needs tensorflow, astropy and the SHARP patches")


@needs_tf_and_patches
def test_tensorflow_active_region_example_runs(tmp_path):
    output = _run("11_tensorflow_active_region.py", tmp_path)
    assert "batch 3: patches [4, 128, 128, 1] float32  regions ['12673', '12675', '12673', '12675']  " \
           "labels [1, 0, 1, 0]" in output
    assert "16 patches: 2 positive, 14 negative; all checks passed" in output


@needs_tf_and_patches
@pytest.mark.skipif(not _has("torch"), reason="needs torch")
def test_tensorflow_and_pytorch_active_region_loaders_yield_the_same_patches(tmp_path):
    torch_example = _load("10_pytorch_active_region.py")
    tf_example = _load("11_tensorflow_active_region.py")
    real = tf_example.real
    import flare_indexer as fidx

    index_path = tmp_path / "patch_index.csv"
    real.build_patch_index(index_path)
    labels = fidx.ActiveRegionSingleImage("24h", fidx.BinaryThresholdStrategy("M"), include_metadata=True) \
        .build(index_path, real.CATALOG)
    torch_dataset = torch_example.FlareDataset(labels, real.REAL_DATA)
    tf_dataset = tf_example.FlareDataset(labels, real.REAL_DATA, batch_size=4)
    tf_samples = [sample for i in range(len(tf_dataset)) for sample in zip(*tf_dataset[i])]
    assert len(torch_dataset) == len(tf_samples) == 16
    for i, (tf_patch, tf_label) in enumerate(tf_samples):
        patch, label = torch_dataset[i]
        np.testing.assert_array_equal(patch.permute(1, 2, 0).numpy(), tf_patch)   # [C,H,W] -> [H,W,C]
        assert int(label) == int(tf_label) == labels["label"].iloc[i]
