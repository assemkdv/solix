"""
Example 10 -- real full-disk images, one label per image, loaded with TensorFlow.

The TensorFlow counterpart of example 7: same images, same catalog, same
labeling settings, same checks. Only the loader differs.

Question answered for every image: "Does an M-class or stronger flare start
in (t, t + 24h), where t is the image's full-precision observation time
(DATE_OBS)?"

Steps
1. Image index from examples/real_data/image_manifest.csv (timestamps are
   the full-precision DATE_OBS read from each file's header).
2. Labels: flare_indexer.DatasetBuilder with every labeling choice explicit.
3. Loading: a tf.data.Dataset that decodes each JPEG 2000 file with the
   shared decoder (channel-first [C, H, W]), converts it explicitly to
   TensorFlow's channel-last layout [H, W, C], and yields batches
   [B, H, W, C] = [B, 256, 256, 1].
4. Checks: header time vs. index time for every loaded image, batch shapes,
   order of image paths, timestamps and labels against the label table, and
   every label recomputed from the catalog without flare_indexer.

Setup:  pip install tensorflow pillow        # TensorFlow needs Python <= 3.13
        python examples/real_data/fetch_real_data.py
Run:    python examples/10_tensorflow_full_disk_images.py
"""

import argparse
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
BATCH_SIZE = 4
IMAGE_SHAPE = (hmi_jp2.IMAGE_SIZE, hmi_jp2.IMAGE_SIZE, 1)  # [H, W, C]


def to_channels_last(array: np.ndarray) -> np.ndarray:
    """[C, H, W] (the shared decoder's layout) -> [H, W, C] (TensorFlow's layout)."""
    if array.ndim != 3:
        raise ValueError(f"expected [C, H, W], got shape {array.shape}")
    return np.ascontiguousarray(np.transpose(array, (1, 2, 0)))


def load_image(path: Path, expected_time: pd.Timestamp) -> np.ndarray:
    """Check the header time, decode, and return float32 [H, W, C]."""
    observed = hmi_jp2.observation_time(hmi_jp2.read_jp2_header(path))
    if observed != expected_time:
        raise ValueError(f"{path}: header time {observed} != index time {expected_time}")
    return to_channels_last(hmi_jp2.decode_magnetogram(path))


def make_dataset(labels: pd.DataFrame, image_root: Path):
    """tf.data.Dataset of (image [H, W, C], label, timestamp, image_path), in label-table order."""
    import tensorflow as tf

    def samples():
        for row in labels.itertuples(index=False):
            image = load_image(image_root / row.image_path, row.timestamp)
            yield image, np.int32(row.label), real.format_time(row.timestamp), row.image_path

    return tf.data.Dataset.from_generator(samples, output_signature=(
        tf.TensorSpec(shape=IMAGE_SHAPE, dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
    ))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXAMPLES / "output")
    args = parser.parse_args(argv)
    real.require_optional_dependencies("tensorflow")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "10_image_index.csv"
    labels_path = args.output / "10_image_labels.csv"

    # 1. Image index from actual, full-precision observation times.
    index = real.build_image_index(index_path)
    provenance = real.check_catalog_coverage(index["timestamp"], PREDICTION_WINDOW)
    print(f"TensorFlow {tf.__version__}")
    print(f"Image index: {len(index)} images -> {index_path}")
    print(f"Catalog: {provenance['catalog_rows']} flares, queried {provenance['query_start']} .. "
          f"{provenance['query_end']}; covers every (t, t+{PREDICTION_WINDOW}) window")

    # 2. Labels.
    builder = fidx.DatasetBuilder(
        prediction_window=PREDICTION_WINDOW,
        strategy=fidx.BinaryThresholdStrategy("M"),  # 1 if any flare >= M1.0 in the window
        target="full_disk",
        event_time="start",
        include_metadata=True,                       # keep image_path
    )
    labels = builder.build(index_path, real.CATALOG)
    labels.to_csv(labels_path, index=False)
    assert labels["image_path"].tolist() == index["image_path"].tolist()
    print(f"\nLabels -> {labels_path}")
    print(labels.to_string(index=False))

    # 3. tf.data pipeline.
    dataset = make_dataset(labels, real.REAL_DATA).batch(BATCH_SIZE)
    print(f"\ntf.data.Dataset.from_generator(...).batch({BATCH_SIZE})  element_spec images: "
          f"{dataset.element_spec[0]}")
    seen = []
    for number, (images, batch_labels, timestamps, paths) in enumerate(dataset):
        assert images.shape.rank == 4 and tuple(images.shape[1:]) == IMAGE_SHAPE
        assert images.dtype == tf.float32
        assert 0.0 <= float(tf.reduce_min(images)) and float(tf.reduce_max(images)) <= 1.0
        timestamps = [t.decode() for t in timestamps.numpy()]
        paths = [p.decode() for p in paths.numpy()]
        print(f"batch {number}: images {images.shape.as_list()} {images.dtype.name} "
              f"range [{float(tf.reduce_min(images)):.3f}, {float(tf.reduce_max(images)):.3f}]  "
              f"labels {batch_labels.numpy().tolist()}")
        for t, path in zip(timestamps, paths):
            print(f"    {t}  {path}")
        seen += list(zip(timestamps, paths, batch_labels.numpy().tolist()))
    expected = list(zip(labels["timestamp"].map(real.format_time), labels["image_path"], labels["label"]))
    assert seen == expected, "tf.data order or labels differ from the label table"

    # 4. Independent label check: no flare_indexer, just the catalog rows.
    catalog = real.load_catalog()
    date_obs = real.date_obs_times()
    for _, row in labels.iterrows():
        t = date_obs[row["image_path"]]
        assert t == row["timestamp"], (t, row)
        assert real.independent_m_label(catalog, t, PREDICTION_WINDOW)[0] == row["label"], (t, row)
    print(f"\nAll {len(labels)} labels match an independent recomputation from the catalog.")

    print(f"\n{len(labels)} images: {int(labels['label'].sum())} positive, "
          f"{int((labels['label'] == 0).sum())} negative; all checks passed")


if __name__ == "__main__":
    main()
