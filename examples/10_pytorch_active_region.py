"""
Example 10 -- real active-region patches, one label per patch, loaded with PyTorch.

Question answered for every patch: "Does an M-class or stronger flare from
THIS patch's own region start in [t, t + 24h), where t is the patch's
observation time?" Two patches observed at the same time can get
different labels, because only each region's own flares count.

Steps
1. Image index: built from examples/real_data/sharp_manifest.csv, which
   fetch_sharp_patches.py wrote from each patch's own header (DATE-OBS and
   NOAA_AR).
2. Labels: flare_indexer.ActiveRegionSingleImage.
3. Loading: FlareDataset, a torch Dataset that gives one (patch, label) per
   row of the label table, and a DataLoader that yields batches shaped
   [B, C, H, W] = [B, 1, 128, 128].
4. Checks: header time vs. index time for every patch, and every label
   recomputed from the catalog, region by region, without flare_indexer.

Data: SDO/HMI SHARP magnetograms (hmi.sharp_cea_720s) of HARP 7115 =
NOAA 12673 and HARP 7120 = NOAA 12675, recorded hourly from 00:00 to 09:00
TAI on 2017-09-03 (no records at 07:00 and 08:00 TAI; the timestamps used
are each file's DATE-OBS in UTC, about 78 s earlier), and the HEK/SWPC GOES
flare list for 2017-09-01 .. 2017-09-06. The patches are in examples/real_data/sharp_patches/.

Setup:  pip install ".[pytorch-examples]"           # torch, pillow, astropy
Run:    python examples/10_pytorch_active_region.py
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

import real_examples as real  # noqa: E402
import sharp_fits  # noqa: E402

PREDICTION_WINDOW = "24h"
BATCH_SIZE = 4


class FlareDataset(Dataset):
    """
    One labeled active-region patch per item: (patch [C, H, W], label).

    labels is the table built by flare_indexer, one row per patch with
    timestamp, active_region, image_path and label; image_root is the
    folder the image paths are relative to.
    """

    def __init__(self, labels: pd.DataFrame, image_root: Path):
        self.labels = labels.reset_index(drop=True)
        self.image_root = Path(image_root)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        row = self.labels.iloc[i]
        patch = sharp_fits.load_patch(self.image_root / row["image_path"], row["timestamp"])
        return torch.from_numpy(patch), torch.tensor(int(row["label"]))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXAMPLES / "output")
    args = parser.parse_args(argv)
    real.require_optional_dependencies(images="fits")

    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "10_patch_index.csv"
    labels_path = args.output / "10_patch_labels.csv"

    # 1. Image index from the patches' own headers.
    index = real.build_patch_index(index_path)
    provenance = real.check_catalog_coverage(index["timestamp"], PREDICTION_WINDOW)
    regions = sorted(int(r) for r in index["active_region"].unique())
    print(f"Image index: {len(index)} patches of regions {regions} -> {index_path}")
    print(f"Catalog: {provenance['catalog_rows']} flares, queried {provenance['query_start']} .. "
          f"{provenance['query_end']}; covers every [t, t+{PREDICTION_WINDOW}) window")

    # 2. Labels: only each patch's own region's flares count.
    builder = fidx.ActiveRegionSingleImage(
        prediction_window=PREDICTION_WINDOW,
        reducer=fidx.MaxFluxReducer(),               # the strongest flare in the window
        labeler=fidx.BinaryThresholdLabeler("M"),    # 1 if a flare >= M1.0 from this region starts in the window
        include_metadata=True,                       # keep image_path
        include_flares=True,                         # show which flares made each label
    )
    labels = builder.build(index_path, real.CATALOG)
    labels.to_csv(labels_path, index=False)
    print(f"\nLabels -> {labels_path}")
    print(labels.drop(columns="image_path").to_string(index=False))

    # 3. Dataset and DataLoader.
    dataset = FlareDataset(labels, real.REAL_DATA)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    size = sharp_fits.PATCH_SIZE
    print(f"\nDataLoader(FlareDataset, batch_size={BATCH_SIZE}, shuffle=False)")
    seen = []
    for number, (patches, batch_labels) in enumerate(loader):
        assert patches.dim() == 4 and patches.shape[1:] == (1, size, size)
        assert patches.dtype == torch.float32 and 0.0 <= patches.min() and patches.max() <= 1.0
        rows = labels.iloc[len(seen):len(seen) + len(batch_labels)]
        regions = rows["active_region"].tolist()
        print(f"batch {number}: patches {list(patches.shape)} {patches.dtype}  "
              f"regions {regions}  labels {batch_labels.tolist()}")
        seen += batch_labels.tolist()
    assert seen == labels["label"].tolist(), "DataLoader order or labels differ from the label table"

    # 4. Independent label check, region by region: no flare_indexer, just the catalog rows.
    catalog = real.load_catalog()
    for _, row in labels.iterrows():
        expected, _ = real.independent_m_label(catalog, row["timestamp"], PREDICTION_WINDOW, row["active_region"])
        assert expected == row["label"], row
    print(f"\nAll {len(labels)} labels match an independent recomputation from the catalog.")
    for region, group in labels.groupby("active_region"):
        print(f"  region {region}: {int(group['label'].sum())} positive, {int((group['label'] == 0).sum())} negative")

    print(f"\n{len(labels)} patches: {int(labels['label'].sum())} positive, "
          f"{int((labels['label'] == 0).sum())} negative; all checks passed")


if __name__ == "__main__":
    main()
