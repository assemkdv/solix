"""
Example 12 -- any file type: .npy files, labeled and loaded with PyTorch.

SOLIX never opens your files. It only needs their times, so the
same steps work for images, .npy arrays or anything else. Here each
instance is a .npy file that holds one random number.

Steps
1. Files: 100 .npy files, one per hour from 2024-03-01 00:00, each named
   by its time (20240301_000000.npy) and holding one random number.
2. Index: one row per file, with the time read from the file name and the
   file's path.
3. Labels: FullDiskSingleInstance (one label per file) and FullDiskSequence
   (one label per sequence of 6 files 1 hour apart, a new sequence every
   6 hours), both from the flares that start in the 24 hours after t (for
   a sequence, t is the time of its last file). The flare catalog is examples/data/flare_catalog.csv.
4. Loading: a torch Dataset that opens the .npy file(s) of each row, and a
   DataLoader that yields batches of values and labels.
5. Checks: every loaded value is the value that was written, and the
   batches give the labels in the order of the label table.

Setup:  pip install torch
Run:    python examples/12_pytorch_npy.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    sys.exit("This example needs PyTorch: pip install torch")

import solix

EXAMPLES = Path(__file__).resolve().parent
CATALOG = EXAMPLES / "data" / "flare_catalog.csv"

N_FILES = 100
FIRST_TIME = pd.Timestamp("2024-03-01 00:00:00")
FILE_CADENCE = pd.Timedelta("1h")
NAME_FORMAT = "%Y%m%d_%H%M%S"         # 2024-03-01 00:00 -> 20240301_000000.npy
SEED = 0


def write_npy_files(folder: Path) -> dict:
    """Write one .npy file per hour, each holding one random number; returns {file name: value}."""
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.npy"):     # files from an earlier run would join the index
        old.unlink()
    rng = np.random.default_rng(SEED)
    written = {}
    for i in range(N_FILES):
        name = f"{FIRST_TIME + i * FILE_CADENCE:{NAME_FORMAT}}.npy"
        value = rng.random(1, dtype=np.float32)          # shape (1,)
        np.save(folder / name, value)
        written[name] = value
    return written


def build_index(folder: Path, root: Path, index_path: Path) -> pd.DataFrame:
    """The index SOLIX reads: the time of each file, from its name, and its path."""
    files = sorted(folder.glob("*.npy"))
    index = pd.DataFrame({
        "timestamp": pd.to_datetime([f.stem for f in files], format=NAME_FORMAT),
        "file_path": [f.relative_to(root).as_posix() for f in files],
    })
    index.to_csv(index_path, index=False)
    return index


class NpyDataset(Dataset):
    """One item per row of the label table: (value [1], label)."""

    def __init__(self, labels: pd.DataFrame, root: Path):
        self.labels = labels.reset_index(drop=True)
        self.root = Path(root)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        row = self.labels.iloc[i]
        value = np.load(self.root / row["file_path"])
        return torch.from_numpy(value), torch.tensor(int(row["label"]))


class NpySequenceDataset(NpyDataset):
    """One item per sequence: (values [T, 1], label), oldest file first."""

    def __getitem__(self, i):
        row = self.labels.iloc[i]
        values = np.stack([np.load(self.root / path) for path in row["file_path"]])
        return torch.from_numpy(values), torch.tensor(int(row["label"]))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXAMPLES / "output")
    args = parser.parse_args(argv)
    root = args.output
    folder = root / "12_npy"
    index_path = root / "12_index.csv"

    # 1. Files.
    written = write_npy_files(folder)
    names = sorted(written)
    print(f"Files: {len(written)} .npy files -> {folder}")
    print(f"  {names[0]} .. {names[-1]}, one number each, e.g. {names[0]} holds {written[names[0]][0]:.4f}")

    # 2. Index.
    index = build_index(folder, root, index_path)
    print(f"\nIndex: {len(index)} instances -> {index_path}")
    print(index.head(3).to_string(index=False))

    # 3. Labels, the same way for one file or a sequence of files.
    labels = solix.FullDiskSingleInstance(
        prediction_window="24h",                     # flares that start in [t, t + 24h)
        reducer=solix.MaxFluxReducer(),              # the strongest of them
        labeler=solix.BinaryThresholdLabeler("M"),   # 1 if it is M1.0 or stronger, else 0
        include_metadata=True,                       # keep file_path
    ).build(index_path, CATALOG)
    print(f"\nSingle instances: {len(labels)} labels, {int(labels['label'].sum())} positive")
    print(labels.head(3).to_string(index=False))

    sequences = solix.FullDiskSequence(
        prediction_window="24h",
        reducer=solix.MaxFluxReducer(),
        labeler=solix.BinaryThresholdLabeler("M"),
        cadence="1h",                 # time between the files of a sequence
        observation_window="6h",      # one sequence covers 6 h: 6 files
        sliding_window="6h",          # a new sequence every 6 h
    ).build(index_path, CATALOG)
    print(f"\nSequences: {len(sequences)} sequences of 6 files, {int(sequences['label'].sum())} positive")
    with pd.option_context("display.width", 200):
        print(sequences[["sequence_start", "sequence_end", "n_instances", "label"]].head(3).to_string(index=False))

    # 4. Dataset and DataLoader.
    print("\nDataLoader(NpyDataset, batch_size=16, shuffle=False)")
    seen = []
    for number, (values, batch_labels) in enumerate(DataLoader(NpyDataset(labels, root), batch_size=16)):
        assert values.shape[1:] == (1,) and values.dtype == torch.float32
        print(f"batch {number}: values {list(values.shape)}  labels {batch_labels.tolist()}")
        seen.append(values)
        if number == 0:
            print(f"    first values {[round(v, 4) for v in values[:4, 0].tolist()]}")
    loaded = torch.cat(seen)

    print("\nDataLoader(NpySequenceDataset, batch_size=4, shuffle=False)")
    seen_labels = []
    for number, (values, batch_labels) in enumerate(DataLoader(NpySequenceDataset(sequences, root), batch_size=4)):
        assert values.shape[1:] == (6, 1) and values.dtype == torch.float32
        print(f"batch {number}: values {list(values.shape)}  labels {batch_labels.tolist()}")
        seen_labels += batch_labels.tolist()

    # 5. Checks.
    expected = np.concatenate([written[Path(p).name] for p in labels["file_path"]])
    assert np.array_equal(loaded[:, 0].numpy(), expected), "loaded values differ from the written ones"
    assert seen_labels == sequences["label"].tolist(), "sequence batches differ from the label table"
    for _, row in sequences.iterrows():
        assert [Path(p).name for p in row["file_path"]] == [f"{t:{NAME_FORMAT}}.npy" for t in row["timestamps"]]
    print(f"\n{len(labels)} files and {len(sequences)} sequences loaded; all checks passed")


if __name__ == "__main__":
    main()
