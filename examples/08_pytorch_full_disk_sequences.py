"""
Example 8 -- real full-disk image sequences on a time grid, loaded with PyTorch.

Each sample is T = 3 images one hour apart (observation_window 3h,
cadence 1h); a new sequence starts every hour (sliding_window 1h). A
sequence is labeled from its FINAL observation time t_end: 1 if an M-class
or stronger flare starts in (t_end, t_end + 24h], else 0.

The grid uses the images' full-precision observation times (DATE_OBS,
e.g. 00:59:49.9), starting at the first image. Three grid points in this
data have no image, and flare_indexer skips every sequence that needs one
of them; no nearby image is used in its place and no time is rounded:
- 04:59:49.9: the image for that hour was observed at 04:59:50.0, 0.1 s
  later. It stays in the index and is labeled on its own (example 7), but
  it is not on this grid;
- 06:59:49.9: the request for 07:00 had no image within 60 s (the closest
  was 07:36:35), so fetch_real_data.py recorded it as missing;
- 08:59:49.9: the image near 09:00 was observed at 09:00:34.9.
With this data only 2 of 8 candidate sequences are complete, and both are
labeled 0. (An earlier version of this example rounded DATE_OBS to whole
seconds, which put 00:59:49.9 ... 03:59:49.9 and 04:59:50.0 on one
whole-second grid and produced 4 sequences, including a positive one;
that result depended on the rounding.)

Steps
1. Image index from the manifest (same as example 7).
2. Sequences and labels: DatasetBuilder in time-based mode.
3. CSV round trip: the list-valued columns (timestamps, image_path) are
   written as JSON lists and parsed back, and the reloaded table must equal
   the original.
4. Loading: a torch Dataset that decodes the T images of each sequence in
   order and a DataLoader yielding [B, T, C, H, W] = [B, 3, 1, 256, 256].
5. Checks: order and spacing of every sequence, header time of every
   image, and the label of every sequence recomputed from the catalog
   without flare_indexer.

Setup:  pip install torch pillow
        python examples/real_data/fetch_real_data.py     # downloads ~10 MB once
Run:    python examples/08_pytorch_full_disk_sequences.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

import flare_indexer as fidx

EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES / "real_data"))

import hmi_jp2  # noqa: E402
import real_examples as real  # noqa: E402
from real_examples import load_sequences, save_sequences  # noqa: E402,F401

PREDICTION_WINDOW = "24h"
CADENCE = "1h"
OBSERVATION_WINDOW = "3h"
SLIDING_WINDOW = "1h"
BATCH_SIZE = 2


def make_dataset_class():
    import torch
    from torch.utils.data import Dataset

    class FullDiskSequenceDataset(Dataset):
        """One labeled sequence per item: (images [T, C, H, W], label, sequence_end, image_paths)."""

        def __init__(self, sequences: pd.DataFrame, image_root: Path, cadence):
            self.sequences = sequences.reset_index(drop=True)
            self.image_root = image_root
            self.cadence = pd.Timedelta(cadence)

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, i):
            row = self.sequences.iloc[i]
            timestamps, paths = row["timestamps"], row["image_path"]
            if len(timestamps) != len(paths) or len(paths) != row["n_images"]:
                raise ValueError(f"sequence {i}: {len(timestamps)} timestamps, {len(paths)} paths")
            if any(later - earlier != self.cadence for earlier, later in zip(timestamps, timestamps[1:])):
                raise ValueError(f"sequence {i}: images are not {self.cadence} apart: {timestamps}")
            if timestamps[-1] != row["sequence_end"]:
                raise ValueError(f"sequence {i}: last image {timestamps[-1]} != sequence_end {row['sequence_end']}")

            frames = []
            for t, relative_path in zip(timestamps, paths):
                path = self.image_root / relative_path
                observed = hmi_jp2.observation_time(hmi_jp2.read_jp2_header(path))
                if observed != t:
                    raise ValueError(f"{path}: header time {observed} != sequence time {t}")
                frames.append(torch.from_numpy(hmi_jp2.decode_magnetogram(path)))
            images = torch.stack(frames)  # [T, C, H, W], oldest first
            label = torch.tensor(int(row["label"]), dtype=torch.long)
            return images, label, real.format_time(row["sequence_end"]), list(paths)

    return FullDiskSequenceDataset


def collate(batch):
    import torch

    images, labels, ends, paths = zip(*batch)
    return torch.stack(images), torch.stack(labels), list(ends), list(paths)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXAMPLES / "output")
    args = parser.parse_args(argv)
    real.require_optional_dependencies()
    import torch
    from torch.utils.data import DataLoader

    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "08_image_index.csv"
    sequences_path = args.output / "08_sequence_labels.csv"

    # 1. Image index from actual observation times.
    index = real.build_image_index(index_path)
    provenance = real.check_catalog_coverage(index["timestamp"], PREDICTION_WINDOW)
    print(f"Image index: {len(index)} images -> {index_path}")
    print(index["timestamp"].astype(str).to_string(index=False, header=False))
    print(f"Catalog: {provenance['catalog_rows']} flares, queried {provenance['query_start']} .. "
          f"{provenance['query_end']}; covers every (t, t+{PREDICTION_WINDOW}) window")

    # 2. Sequences and labels.
    builder = fidx.DatasetBuilder(
        prediction_window=PREDICTION_WINDOW,
        strategy=fidx.BinaryThresholdStrategy("M"),  # 1 if any flare >= M1.0 in the window
        target="full_disk",
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
    save_sequences(sequences, sequences_path)
    reloaded = load_sequences(sequences_path)
    pd.testing.assert_frame_equal(reloaded, sequences, check_dtype=False)
    print(f"\nSequences -> {sequences_path} (reloaded and compared: identical)")
    with pd.option_context("display.width", 200, "display.max_colwidth", 200):
        print(sequences[["sequence_start", "sequence_end", "n_images", "label"]].to_string(index=False))

    # 4. Dataset and DataLoader.
    dataset = make_dataset_class()(reloaded, real.REAL_DATA, CADENCE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate)
    n_images = round(pd.Timedelta(OBSERVATION_WINDOW) / pd.Timedelta(CADENCE))
    print(f"\nDataLoader(batch_size={BATCH_SIZE}, shuffle=False)")
    seen = []
    for number, (images, labels, ends, paths) in enumerate(loader):
        assert images.dim() == 5 and images.shape[1:] == (n_images, 1, hmi_jp2.IMAGE_SIZE, hmi_jp2.IMAGE_SIZE)
        assert images.dtype == torch.float32 and 0.0 <= images.min() and images.max() <= 1.0
        print(f"batch {number}: images {list(images.shape)} {images.dtype}  labels {labels.tolist()}")
        for end, label, sequence_paths in zip(ends, labels.tolist(), paths):
            print(f"    sequence_end {end}  label {label}  {[Path(p).name for p in sequence_paths]}")
        seen += list(zip(ends, labels.tolist(), map(tuple, paths)))
    expected = list(zip(sequences["sequence_end"].map(real.format_time), sequences["label"], map(tuple, sequences["image_path"])))
    assert seen == expected, "DataLoader order or labels differ from the sequence table"

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