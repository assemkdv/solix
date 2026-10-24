"""
Example 11 -- real active-region patches, one label per patch, loaded with TensorFlow.

The TensorFlow counterpart of example 10: same patches, same catalog, same
labeling settings, same label checks. Only the loader differs.

Steps
1. Image index from examples/real_data/sharp_manifest.csv (timestamps are
   DATE-OBS and regions NOAA_AR, both read from each patch's header).
2. Labels: flare_indexer.ActiveRegionSingleImage.
3. Loading: FlareDataset, a keras.utils.PyDataset. Keras asks it for whole
   batches, so each item is (patches, labels) for one batch, with patches
   in TensorFlow's channel-last layout [B, H, W, C] = [B, 128, 128, 1]. It
   can be passed straight to model.fit.
4. Checks: header time vs. index time for every patch, batch shapes, and
   every label recomputed from the catalog, region by region, without
   flare_indexer.

Setup:  pip install ".[tensorflow-examples]"        # TensorFlow needs Python <= 3.13
Run:    python examples/11_tensorflow_active_region.py
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
import sharp_fits  # noqa: E402

PREDICTION_WINDOW = "24h"
BATCH_SIZE = 4
PATCH_SHAPE = (sharp_fits.PATCH_SIZE, sharp_fits.PATCH_SIZE, 1)  # [H, W, C]


class FlareDataset(keras.utils.PyDataset):
    """
    One batch of labeled active-region patches per item:
    (patches [B, H, W, C], labels [B]).

    labels is the table built by flare_indexer, one row per patch with
    timestamp, active_region, image_path and label; image_root is the
    folder the image paths are relative to.
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
        patches = np.stack([hmi_jp2.to_channels_last(sharp_fits.load_patch(self.image_root / path, t))
                            for t, path in zip(rows["timestamp"], rows["image_path"])])
        return patches, rows["label"].to_numpy(dtype=np.int32)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXAMPLES / "output")
    args = parser.parse_args(argv)
    real.require_optional_dependencies(images="fits")

    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "11_patch_index.csv"
    labels_path = args.output / "11_patch_labels.csv"

    # 1. Image index from the patches' own headers.
    index = real.build_patch_index(index_path)
    provenance = real.check_catalog_coverage(index["timestamp"], PREDICTION_WINDOW)
    regions = sorted(int(r) for r in index["active_region"].unique())
    print(f"TensorFlow {tf.__version__}")
    print(f"Image index: {len(index)} patches of regions {regions} -> {index_path}")
    print(f"Catalog: {provenance['catalog_rows']} flares, queried {provenance['query_start']} .. "
          f"{provenance['query_end']}; covers every [t, t+{PREDICTION_WINDOW}) window")

    # 2. Labels: only each patch's own region's flares count.
    builder = fidx.ActiveRegionSingleImage(
        prediction_window=PREDICTION_WINDOW,
        strategy=fidx.BinaryThresholdStrategy("M"),  # 1 if a flare >= M1.0 from this region starts in the window
        include_metadata=True,                       # keep image_path
        include_flares=True,                         # show which flares made each label
    )
    labels = builder.build(index_path, real.CATALOG)
    labels.to_csv(labels_path, index=False)
    print(f"\nLabels -> {labels_path}")
    print(labels.drop(columns="image_path").to_string(index=False))

    # 3. Dataset.
    dataset = FlareDataset(labels, real.REAL_DATA, batch_size=BATCH_SIZE)
    print(f"\nFlareDataset(batch_size={BATCH_SIZE}): {len(dataset)} batches")
    seen = []
    for number in range(len(dataset)):
        patches, batch_labels = dataset[number]
        assert patches.ndim == 4 and patches.shape[1:] == PATCH_SHAPE
        assert patches.dtype == np.float32 and 0.0 <= patches.min() and patches.max() <= 1.0
        rows = labels.iloc[len(seen):len(seen) + len(batch_labels)]
        print(f"batch {number}: patches {list(patches.shape)} {patches.dtype}  "
              f"regions {rows['active_region'].tolist()}  labels {batch_labels.tolist()}")
        seen += batch_labels.tolist()
    assert seen == labels["label"].tolist(), "dataset order or labels differ from the label table"

    # 4. Independent label check, region by region: no flare_indexer, just the catalog rows.
    catalog = real.load_catalog()
    for _, row in labels.iterrows():
        expected, _ = real.independent_m_label(catalog, row["timestamp"], PREDICTION_WINDOW, row["active_region"])
        assert expected == row["label"], row
    print(f"\nAll {len(labels)} labels match an independent recomputation from the catalog.")

    print(f"\n{len(labels)} patches: {int(labels['label'].sum())} positive, "
          f"{int((labels['label'] == 0).sum())} negative; all checks passed")


if __name__ == "__main__":
    main()
