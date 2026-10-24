"""
Example 9 -- real full-disk image sequences on a time grid, loaded with TensorFlow.

The TensorFlow counterpart of example 7: same images, same catalog, same
labeling and grid settings, same label checks. Only the loader differs.

Each sample is T = 3 images one hour apart (observation_window 3h,
cadence 1h); a new sequence starts every hour (sliding_window 1h). A
sequence is labeled from its FINAL observation time t_end: 1 if an M-class
or stronger flare starts in [t_end, t_end + 24h), else 0.

The grid uses full-precision DATE_OBS times and has no tolerance. With
this data only 2 of 8 candidate sequences are complete (see example 7 for
why: images 0.1 s off the grid, a missing hour, and an image at
09:00:34.9); both are labeled 0. Missing grid images are never replaced.

Steps
1. Image index from the manifest (same as examples 6, 7 and 8).
2. Sequences and labels: flare_indexer.FullDiskSequence on a time grid.
3. CSV round trip of the list-valued columns (timestamps, image_path).
4. Loading: FlareSequenceDataset, a keras.utils.PyDataset that gives one
   batch of (images, labels) per item, oldest image first, as
   [B, T, H, W, C] = [B, 3, 256, 256, 1]. It can be passed straight to
   model.fit.
5. Checks: order and spacing of every sequence, header time of every
   image, batch shapes, order of paths and labels against the sequence
   table, and every label recomputed from the catalog without flare_indexer.

Setup:  pip install ".[tensorflow-examples]"        # TensorFlow needs Python <= 3.13
        python examples/real_data/fetch_real_data.py     # downloads about 10 MB once
Run:    python examples/09_tensorflow_sequences.py
"""

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

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


class FlareSequenceDataset(keras.utils.PyDataset):
    """
    One batch of labeled sequences per item: (images [B, T, H, W, C],
    labels [B]), oldest image first.

    sequences is the table built by flare_indexer, one row per sequence
    with lists of timestamps and image paths and one label; image_root is
    the folder the image paths are relative to.
    """

    def __init__(self, sequences: pd.DataFrame, image_root: Path, cadence, batch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.sequences = sequences.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.cadence = pd.Timedelta(cadence)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.sequences) / self.batch_size)

    def __getitem__(self, i):
        rows = self.sequences.iloc[i * self.batch_size:(i + 1) * self.batch_size]
        batch = []
        for _, row in rows.iterrows():
            real.check_sequence(row, self.cadence)
            batch.append(np.stack([hmi_jp2.to_channels_last(hmi_jp2.load_image(self.image_root / path, t))
                                   for t, path in zip(row["timestamps"], row["image_path"])]))
        return np.stack(batch), rows["label"].to_numpy(dtype=np.int32)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXAMPLES / "output")
    args = parser.parse_args(argv)
    real.require_optional_dependencies()

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

    # 4. Dataset.
    dataset = FlareSequenceDataset(reloaded, real.REAL_DATA, CADENCE, batch_size=BATCH_SIZE)
    print(f"\nFlareSequenceDataset(batch_size={BATCH_SIZE}): {len(dataset)} batches")
    seen = []
    for number in range(len(dataset)):
        images, labels = dataset[number]
        assert images.ndim == 5 and images.shape[1:] == SEQUENCE_SHAPE
        assert images.dtype == np.float32 and 0.0 <= images.min() and images.max() <= 1.0
        print(f"batch {number}: images {list(images.shape)} {images.dtype}  labels {labels.tolist()}")
        rows = sequences.iloc[len(seen):len(seen) + len(labels)]
        for end, label, sequence_paths in zip(rows["sequence_end"], labels.tolist(), rows["image_path"]):
            print(f"    sequence_end {real.format_time(end)}  label {label}  {[Path(p).name for p in sequence_paths]}")
        seen += labels.tolist()
    assert seen == sequences["label"].tolist(), "dataset order or labels differ from the sequence table"

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
