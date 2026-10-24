"""
Example 9 -- real full-disk image sequences on a time grid, loaded with TensorFlow.

The TensorFlow counterpart of example 8: same images, same catalog, same
labeling and grid settings, same checks. Only the loader differs.

Each sample is T = 3 images one hour apart (observation_window 3h,
cadence 1h); a new sequence starts every hour (sliding_window 1h). A
sequence is labeled from its FINAL observation time t_end: 1 if an M-class
or stronger flare starts in (t_end, t_end + 24h], else 0.

The grid uses full-precision DATE_OBS times and has no tolerance. With
this data only 2 of 8 candidate sequences are complete (see example 8 for
why: images 0.1 s off the grid, a missing hour, and an image at
09:00:34.9); both are labeled 0. Missing grid images are never replaced.

Steps
1. Image index from the manifest (same as examples 6, 7 and 8).
2. Sequences and labels: flare_indexer.FullDiskSequence on a time grid.
3. CSV round trip of the list-valued columns (timestamps, image_path).
4. Loading: a tf.data.Dataset that decodes the T images of each sequence,
   oldest first, converts each from [C, H, W] to [H, W, C], stacks them
   to [T, H, W, C], and yields batches [B, T, H, W, C] = [B, 3, 256, 256, 1].
5. Checks: order and spacing of every sequence, header time of every
   image, batch shapes, order of paths and labels against the sequence
   table, and every label recomputed from the catalog without flare_indexer.

Setup:  pip install tensorflow pillow        # TensorFlow needs Python <= 3.13
        python examples/real_data/fetch_real_data.py
Run:    python examples/09_tensorflow_sequences.py
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import flare_indexer as fidx

EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES / "real_data"))

import hmi_jp2  # noqa: E402
import real_examples as real  # noqa: E402

PREDICTION_WINDOW = "24h"
CADENCE = "1h"
OBSERVATION_WINDOW = "3h"
SLIDING_WINDOW = "1h"
BATCH_SIZE = 2
N_IMAGES = round(pd.Timedelta(OBSERVATION_WINDOW) / pd.Timedelta(CADENCE))
SEQUENCE_SHAPE = (N_IMAGES, hmi_jp2.IMAGE_SIZE, hmi_jp2.IMAGE_SIZE, 1)  # [T, H, W, C]


def _image_example():
    """Reuse example 8's channel conversion and per-image loading (header check + decode)."""
    spec = importlib.util.spec_from_file_location("example_10", EXAMPLES / "08_tensorflow_images.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


images_example = _image_example()


def load_sequence(row, image_root: Path, cadence: pd.Timedelta) -> np.ndarray:
    """Check one sequence row and return float32 [T, H, W, C], oldest image first."""
    timestamps, paths = row.timestamps, row.image_path
    if len(timestamps) != len(paths) or len(paths) != row.n_images:
        raise ValueError(f"sequence ending {row.sequence_end}: {len(timestamps)} timestamps, {len(paths)} paths")
    if any(later - earlier != cadence for earlier, later in zip(timestamps, timestamps[1:])):
        raise ValueError(f"sequence ending {row.sequence_end}: images are not {cadence} apart: {timestamps}")
    if timestamps[-1] != row.sequence_end:
        raise ValueError(f"sequence ending {row.sequence_end}: last image is at {timestamps[-1]}")
    frames = [images_example.load_image(image_root / p, t) for t, p in zip(timestamps, paths)]
    return np.stack(frames)


def make_dataset(sequences: pd.DataFrame, image_root: Path, cadence):
    """tf.data.Dataset of (images [T, H, W, C], label, sequence_end, image_paths), in table order."""
    import tensorflow as tf

    cadence = pd.Timedelta(cadence)

    def samples():
        for row in sequences.itertuples(index=False):
            yield (load_sequence(row, image_root, cadence), np.int32(row.label),
                   real.format_time(row.sequence_end), np.array(row.image_path))

    return tf.data.Dataset.from_generator(samples, output_signature=(
        tf.TensorSpec(shape=SEQUENCE_SHAPE, dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(N_IMAGES,), dtype=tf.string),
    ))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXAMPLES / "output")
    args = parser.parse_args(argv)
    real.require_optional_dependencies("tensorflow")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "09_image_index.csv"
    sequences_path = args.output / "09_sequence_labels.csv"

    # 1. Image index from actual, full-precision observation times.
    index = real.build_image_index(index_path)
    provenance = real.check_catalog_coverage(index["timestamp"], PREDICTION_WINDOW)
    print(f"TensorFlow {tf.__version__}")
    print(f"Image index: {len(index)} images -> {index_path}")
    print(f"Catalog: {provenance['catalog_rows']} flares, queried {provenance['query_start']} .. "
          f"{provenance['query_end']}; covers every [t, t+{PREDICTION_WINDOW}) window")

    # 2. Sequences and labels.
    builder = fidx.FullDiskSequence(
        prediction_window=PREDICTION_WINDOW,
        strategy=fidx.BinaryThresholdStrategy("M"),  # 1 if any flare >= M1.0 in the window
        event_time="start",
        cadence=CADENCE,
        observation_window=OBSERVATION_WINDOW,
        sliding_window=SLIDING_WINDOW,
    )
    sequences, report = builder.build(index_path, real.CATALOG, return_report=True)
    print(f"\nTime grid: inferred source cadence {report.inferred_source_cadence}, start {report.effective_start}")
    print(f"candidates {report.total_candidate_sequences}, emitted {report.emitted_sequences}, "
          f"skipped {report.skipped_sequences}")
    for start, missing in zip(report.skipped_sequence_starts, report.missing_timestamps):
        print(f"  skipped sequence starting {start}: no image at {missing}")
    assert report.emitted_sequences == len(sequences) > 0

    # 3. CSV round trip of the list-valued columns.
    real.save_sequences(sequences, sequences_path)
    reloaded = real.load_sequences(sequences_path)
    pd.testing.assert_frame_equal(reloaded, sequences, check_dtype=False)
    print(f"\nSequences -> {sequences_path} (reloaded and compared: identical)")
    print(sequences[["sequence_start", "sequence_end", "n_images", "label"]].to_string(index=False))

    # 4. tf.data pipeline.
    dataset = make_dataset(reloaded, real.REAL_DATA, CADENCE).batch(BATCH_SIZE)
    print(f"\ntf.data.Dataset.from_generator(...).batch({BATCH_SIZE})  element_spec images: "
          f"{dataset.element_spec[0]}")
    seen = []
    for number, (images, labels, ends, paths) in enumerate(dataset):
        assert images.shape.rank == 5 and tuple(images.shape[1:]) == SEQUENCE_SHAPE
        assert images.dtype == tf.float32
        assert 0.0 <= float(tf.reduce_min(images)) and float(tf.reduce_max(images)) <= 1.0
        ends = [e.decode() for e in ends.numpy()]
        paths = [tuple(p.decode() for p in row) for row in paths.numpy()]
        print(f"batch {number}: images {images.shape.as_list()} {images.dtype.name}  labels {labels.numpy().tolist()}")
        for end, label, sequence_paths in zip(ends, labels.numpy().tolist(), paths):
            print(f"    sequence_end {end}  label {label}  {[Path(p).name for p in sequence_paths]}")
        seen += list(zip(ends, labels.numpy().tolist(), paths))
    expected = list(zip(sequences["sequence_end"].map(real.format_time), sequences["label"],
                        map(tuple, sequences["image_path"])))
    assert seen == expected, "tf.data order or labels differ from the sequence table"

    # 5. Independent label check from the final observation time of each
    #    sequence: the full-precision DATE_OBS of its last image.
    catalog = real.load_catalog()
    date_obs = real.date_obs_times()
    for _, row in sequences.iterrows():
        assert [date_obs[p] for p in row["image_path"]] == row["timestamps"], row
        t = date_obs[row["image_path"][-1]]
        assert t == row["sequence_end"], (t, row)
        assert real.independent_m_label(catalog, t, PREDICTION_WINDOW)[0] == row["label"], (t, row)
    print(f"\nAll {len(sequences)} labels match an independent recomputation from the catalog.")

    print(f"\n{len(sequences)} sequences: {int(sequences['label'].sum())} positive, "
          f"{int((sequences['label'] == 0).sum())} negative; all checks passed")


if __name__ == "__main__":
    main()
