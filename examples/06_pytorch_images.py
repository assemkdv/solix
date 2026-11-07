"""
Example 6 -- real full-disk images, one label per image, loaded with PyTorch.

Question answered for every image: "Does an M-class or stronger flare start
in [t, t + 24h), where t is the image's own observation time?"

Steps
1. Image index: built from examples/real_data/image_manifest.csv, which
   fetch_real_data.py wrote from each downloaded file's header (DATE_OBS).
2. Labels: flare_indexer.FullDiskSingleImage with every labeling choice explicit.
3. Loading: FlareDataset, a torch Dataset that gives one (image, label)
   per row of the label table, and a DataLoader that yields image batches
   shaped [B, C, H, W] = [B, 1, 256, 256].
4. Checks: header time vs. index time for every loaded image, and every
   label recomputed from the catalog without flare_indexer (one positive
   and one negative are printed with their flares).

Data: 10 SDO/HMI magnetograms requested hourly on 2017-09-03 (9 available)
and the HEK/SWPC GOES flare list for 2017-09-01 .. 2017-09-06. See
examples/real_data/README.md.

Setup:  pip install ".[pytorch-examples]"           # torch, pillow, astropy
        python examples/real_data/fetch_real_data.py     # downloads about 10 MB once
Run:    python examples/06_pytorch_images.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import flare_indexer as fidx

EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES / "real_data"))

import hmi_jp2  # noqa: E402
import real_examples as real  # noqa: E402

PREDICTION_WINDOW = "24h"
BATCH_SIZE = 4


class FlareDataset(Dataset):
    """
    One labeled image per item: (image [C, H, W], label).

    labels is the table built by flare_indexer, one row per image with
    timestamp, image_path and label; image_root is the folder the image
    paths are relative to.
    """

    def __init__(self, labels: pd.DataFrame, image_root: Path):
        self.labels = labels.reset_index(drop=True)
        self.image_root = Path(image_root)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        row = self.labels.iloc[i]
        image = hmi_jp2.load_image(self.image_root / row["image_path"], row["timestamp"])
        return torch.from_numpy(image), torch.tensor(int(row["label"]))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXAMPLES / "output")
    args = parser.parse_args(argv)
    real.require_optional_dependencies()

    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "06_image_index.csv"
    labels_path = args.output / "06_image_labels.csv"

    # 1. Image index from actual observation times.
    index = real.build_image_index(index_path)
    provenance = real.check_catalog_coverage(index["timestamp"], PREDICTION_WINDOW)
    print(f"Image index: {len(index)} images -> {index_path}")
    print(f"Catalog: {provenance['catalog_rows']} flares, queried {provenance['query_start']} .. "
          f"{provenance['query_end']}; covers every [t, t+{PREDICTION_WINDOW}) window")

    # 2. Labels.
    builder = fidx.FullDiskSingleImage(
        prediction_window=PREDICTION_WINDOW,
        reducer=fidx.MaxFluxReducer(),               # the strongest flare in the window
        labeler=fidx.BinaryThresholdLabeler("M"),    # 1 if any flare >= M1.0 in the window
        include_metadata=True,                       # keep image_path
    )
    labels = builder.build(index_path, real.CATALOG)
    labels.to_csv(labels_path, index=False)
    assert labels["image_path"].tolist() == index["image_path"].tolist()
    print(f"\nLabels -> {labels_path}")
    print(labels.to_string(index=False))

    # 3. Dataset and DataLoader.
    dataset = FlareDataset(labels, real.REAL_DATA)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"\nDataLoader(FlareDataset, batch_size={BATCH_SIZE}, shuffle=False)")
    seen = []
    for number, (images, batch_labels) in enumerate(loader):
        assert images.dim() == 4 and images.shape[1:] == (1, hmi_jp2.IMAGE_SIZE, hmi_jp2.IMAGE_SIZE)
        assert images.dtype == torch.float32 and 0.0 <= images.min() and images.max() <= 1.0
        print(f"batch {number}: images {list(images.shape)} {images.dtype} "
              f"range [{images.min():.3f}, {images.max():.3f}]  labels {batch_labels.tolist()}")
        rows = labels.iloc[len(seen):len(seen) + len(batch_labels)]
        for t, path in zip(rows["timestamp"], rows["image_path"]):
            print(f"    {real.format_time(t)}  {path}")
        seen += batch_labels.tolist()
    assert seen == labels["label"].tolist(), "DataLoader order or labels differ from the label table"

    # 4. Independent label check: no flare_indexer, just the catalog rows.
    #    The reference time is the full-precision DATE_OBS parsed straight
    #    from the manifest, which must equal the label table's timestamp.
    catalog = real.load_catalog()
    date_obs = real.date_obs_times()
    for _, row in labels.iterrows():
        t = date_obs[row["image_path"]]
        assert t == row["timestamp"], (t, row)
        assert real.independent_m_label(catalog, t, PREDICTION_WINDOW)[0] == row["label"], (t, row)
    print(f"\nAll {len(labels)} labels match an independent recomputation from the catalog.")
    print("Independent check against catalog rows (start_time in [t, t+24h), class M or X):")
    for _, row in pd.concat([labels[labels["label"] == 1].head(1), labels[labels["label"] == 0].tail(1)]).iterrows():
        expected_label, strong = real.independent_m_label(catalog, row["timestamp"], PREDICTION_WINDOW)
        assert expected_label == row["label"], row
        events = ", ".join(f"{c} at {s}" for c, s in zip(strong["class"], strong["start_time"])) or "none"
        print(f"  t={row['timestamp']}  package label={row['label']}  catalog M/X flares: {events}")

    print(f"\n{len(labels)} images: {int(labels['label'].sum())} positive, "
          f"{int((labels['label'] == 0).sum())} negative; all checks passed")


if __name__ == "__main__":
    main()
