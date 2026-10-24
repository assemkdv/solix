"""
Example 7 -- real full-disk image sequences on a time grid, loaded with PyTorch.

Each sample is T = 3 images one hour apart (observation_window 3h,
cadence 1h); a new sequence starts every hour (sliding_window 1h). A
sequence is labeled from its FINAL observation time t_end: 1 if an M-class
or stronger flare starts in [t_end, t_end + 24h), else 0.

The grid uses the images' full-precision observation times (DATE_OBS,
e.g. 00:59:49.9), starting at the first image. Three grid points in this
data have no image, and flare_indexer skips every sequence that needs one
of them; no nearby image is used in its place and no time is rounded:
- 04:59:49.9: the image for that hour was observed at 04:59:50.0, 0.1 s
  later. It stays in the index and is labeled on its own (example 6), but
  it is not on this grid;
- 06:59:49.9: the request for 07:00 had no image within 60 s (the closest
  was 07:36:35), so fetch_real_data.py recorded it as missing;
- 08:59:49.9: the image near 09:00 was observed at 09:00:34.9.
With this data only 2 of 8 candidate sequences are complete, and both are
labeled 0.

Steps
1. Image index from the manifest (same as example 6).
2. Sequences and labels: flare_indexer.FullDiskSequence on a time grid.
3. CSV round trip: the list-valued columns (timestamps, image_path) are
   written as JSON lists and parsed back, and the reloaded table must equal
   the original.
4. Loading: FlareSequenceDataset, a torch Dataset that gives one
   (images, label) per sequence, oldest image first, and a DataLoader
   yielding [B, T, C, H, W] = [B, 3, 1, 256, 256].
5. Checks: order and spacing of every sequence, header time of every
   image, and the label of every sequence recomputed from the catalog
   without flare_indexer.

Setup:  pip install torch pillow
        python examples/real_data/fetch_real_data.py     # downloads ~10 MB once
Run:    python examples/07_pytorch_sequences.py
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
CADENCE = "1h"
OBSERVATION_WINDOW = "3h"
SLIDING_WINDOW = "1h"
BATCH_SIZE = 2


class FlareSequenceDataset(Dataset):
    """
    One labeled sequence per item: (images [T, C, H, W], label), oldest
    image first.

    sequences is the table built by flare_indexer, one row per sequence
    with lists of timestamps and image paths and one label; image_root is
    the folder the image paths are relative to.
    """

    def __init__(self, sequences: pd.DataFrame, image_root: Path, cadence):
        self.sequences = sequences.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.cadence = pd.Timedelta(cadence)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        row = self.sequences.iloc[i]
        real.check_sequence(row, self.cadence)
        frames = [hmi_jp2.load_image(self.image_root / path, t)
                  for t, path in zip(row["timestamps"], row["image_path"])]
        images = torch.stack([torch.from_numpy(frame) for frame in frames])
        return images, torch.tensor(int(row["label"]))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXAMPLES / "output")
    args = parser.parse_args(argv)
    real.require_optional_dependencies()

    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "07_image_index.csv"
    sequences_path = args.output / "07_sequence_labels.csv"

    # 1. Image index from actual observation times.
    index = real.build_image_index(index_path)
    provenance = real.check_catalog_coverage(index["timestamp"], PREDICTION_WINDOW)
    print(f"Image index: {len(index)} images -> {index_path}")
    print(index["timestamp"].astype(str).to_string(index=False, header=False))
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
    with pd.option_context("display.width", 200, "display.max_colwidth", 200):
        print(sequences[["sequence_start", "sequence_end", "n_images", "label"]].to_string(index=False))

    # 4. Dataset and DataLoader.
    dataset = FlareSequenceDataset(reloaded, real.REAL_DATA, CADENCE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    n_images = round(pd.Timedelta(OBSERVATION_WINDOW) / pd.Timedelta(CADENCE))
    print(f"\nDataLoader(FlareSequenceDataset, batch_size={BATCH_SIZE}, shuffle=False)")
    seen = []
    for number, (images, labels) in enumerate(loader):
        assert images.dim() == 5 and images.shape[1:] == (n_images, 1, hmi_jp2.IMAGE_SIZE, hmi_jp2.IMAGE_SIZE)
        assert images.dtype == torch.float32 and 0.0 <= images.min() and images.max() <= 1.0
        print(f"batch {number}: images {list(images.shape)} {images.dtype}  labels {labels.tolist()}")
        rows = sequences.iloc[len(seen):len(seen) + len(labels)]
        for end, label, sequence_paths in zip(rows["sequence_end"], labels.tolist(), rows["image_path"]):
            print(f"    sequence_end {real.format_time(end)}  label {label}  {[Path(p).name for p in sequence_paths]}")
        seen += labels.tolist()
    assert seen == sequences["label"].tolist(), "DataLoader order or labels differ from the sequence table"

    # 5. Independent label check from the final observation time of each
    #    sequence: the full-precision DATE_OBS of its last image.
    catalog = real.load_catalog()
    date_obs = real.date_obs_times()
    for _, row in sequences.iterrows():
        t = date_obs[row["image_path"][-1]]
        assert t == row["sequence_end"], (t, row)
        assert real.independent_m_label(catalog, t, PREDICTION_WINDOW)[0] == row["label"], (t, row)
    print(f"\nAll {len(sequences)} labels match an independent recomputation from the catalog.")
    print("Independent check against catalog rows (start_time in [t_end, t_end+24h), class M or X):")
    for _, row in pd.concat([sequences[sequences["label"] == 1].head(1),
                             sequences[sequences["label"] == 0].tail(1)]).iterrows():
        _, strong = real.independent_m_label(catalog, row["sequence_end"], PREDICTION_WINDOW)
        events = ", ".join(f"{c} at {s}" for c, s in zip(strong["class"], strong["start_time"])) or "none"
        print(f"  t_end={row['sequence_end']}  package label={row['label']}  catalog M/X flares: {events}")

    print(f"\n{len(sequences)} sequences: {int(sequences['label'].sum())} positive, "
          f"{int((sequences['label'] == 0).sum())} negative; all checks passed")


if __name__ == "__main__":
    main()