"""
Example 8 -- real full-disk images, one label per image, loaded with TensorFlow.

The TensorFlow counterpart of example 6: same images, same catalog, same
labeling settings, same checks. Only the loader differs.

Question answered for every image: "Does an M-class or stronger flare start
in [t, t + 24h), where t is the image's full-precision observation time
(DATE_OBS)?"

Steps
1. Image index from examples/real_data/image_manifest.csv (timestamps are
   the full-precision DATE_OBS read from each file's header).
2. Labels: flare_indexer.FullDiskSingleImage with every labeling choice explicit.
3. Loading: FlareDataset, a keras.utils.PyDataset (TensorFlow's
   counterpart of a torch Dataset). Keras asks it for whole batches, so
   each item is (images, labels) for one batch, with images in
   TensorFlow's channel-last layout [B, H, W, C] = [B, 256, 256, 1]. It can
   be passed straight to model.fit.
4. Checks: header time vs. index time for every loaded image, batch shapes,
   order of image paths, timestamps and labels against the label table, and
   every label recomputed from the catalog without flare_indexer.

Setup:  pip install tensorflow pillow        # TensorFlow needs Python <= 3.13
        python examples/real_data/fetch_real_data.py
Run:    python examples/08_tensorflow_images.py
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
BATCH_SIZE = 4
IMAGE_SHAPE = (hmi_jp2.IMAGE_SIZE, hmi_jp2.IMAGE_SIZE, 1)  # [H, W, C]


class FlareDataset(keras.utils.PyDataset):
    """
    One batch of labeled images per item: (images [B, H, W, C], labels [B]).

    labels is the table built by flare_indexer, one row per image with
    timestamp, image_path and label; image_root is the folder the image
    paths are relative to.
    """

    def __init__(self, labels: pd.DataFrame, image_root: Path, batch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.labels = labels.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.labels) / self.batch_size)

    def __getitem__(self, i):
        rows = self.labels.iloc[i * self.batch_size:(i + 1) * self.batch_size]
        images = np.stack([hmi_jp2.to_channels_last(hmi_jp2.load_image(self.image_root / path, t))
                           for t, path in zip(rows["timestamp"], rows["image_path"])])
        return images, rows["label"].to_numpy(dtype=np.int32)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXAMPLES / "output")
    args = parser.parse_args(argv)
    real.require_optional_dependencies("tensorflow")

    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "08_image_index.csv"
    labels_path = args.output / "08_image_labels.csv"

    # 1. Image index from actual, full-precision observation times.
    index = real.build_image_index(index_path)
    provenance = real.check_catalog_coverage(index["timestamp"], PREDICTION_WINDOW)
    print(f"TensorFlow {tf.__version__}")
    print(f"Image index: {len(index)} images -> {index_path}")
    print(f"Catalog: {provenance['catalog_rows']} flares, queried {provenance['query_start']} .. "
          f"{provenance['query_end']}; covers every [t, t+{PREDICTION_WINDOW}) window")

    # 2. Labels.
    builder = fidx.FullDiskSingleImage(
        prediction_window=PREDICTION_WINDOW,
        strategy=fidx.BinaryThresholdStrategy("M"),  # 1 if any flare >= M1.0 in the window
        event_time="start",
        include_metadata=True,                       # keep image_path
    )
    labels = builder.build(index_path, real.CATALOG)
    labels.to_csv(labels_path, index=False)
    assert labels["image_path"].tolist() == index["image_path"].tolist()
    print(f"\nLabels -> {labels_path}")
    print(labels.to_string(index=False))

    # 3. Dataset.
    dataset = FlareDataset(labels, real.REAL_DATA, batch_size=BATCH_SIZE)
    print(f"\nFlareDataset(batch_size={BATCH_SIZE}): {len(dataset)} batches")
    seen = []
    for number in range(len(dataset)):
        images, batch_labels = dataset[number]
        assert images.ndim == 4 and images.shape[1:] == IMAGE_SHAPE
        assert images.dtype == np.float32 and 0.0 <= images.min() and images.max() <= 1.0
        print(f"batch {number}: images {list(images.shape)} {images.dtype} "
              f"range [{images.min():.3f}, {images.max():.3f}]  labels {batch_labels.tolist()}")
        rows = labels.iloc[len(seen):len(seen) + len(batch_labels)]
        for t, path in zip(rows["timestamp"], rows["image_path"]):
            print(f"    {real.format_time(t)}  {path}")
        seen += batch_labels.tolist()
    assert seen == labels["label"].tolist(), "dataset order or labels differ from the label table"

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
