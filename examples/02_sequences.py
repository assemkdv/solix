"""
Example 2 -- labeling sequences of full-disk images.

A sequence is a set of images on an exact time grid: `cadence` is the time
between images, `observation_window` the time one sequence covers, and
`sliding_window` the step to the next sequence. A sequence is labeled
from its LAST image, and a sequence with a missing image is skipped,
never filled in.

The sample image index has a gap (2024-03-02 05:00 and 06:00 are missing).

Run:  python examples/02_sequences.py
"""

from pathlib import Path

import pandas as pd

import flare_indexer as fidx

DATA = Path(__file__).parent / "data"
INDEX = DATA / "full_disk_image_index.csv"
CATALOG = DATA / "flare_catalog.csv"
COLUMNS = ["sequence_start", "sequence_end", "n_images", "label"]


def main():
    print("6 images every 2h (12h observation window), new sequence every 6h")
    sequences, report = fidx.FullDiskSequence(
        "24h", reducer=fidx.MaxFluxReducer(), labeler=fidx.BinaryThresholdLabeler("M"),
        cadence="2h", observation_window="12h", sliding_window="6h",
    ).build(INDEX, CATALOG, return_report=True)
    with pd.option_context("display.width", 120):
        print(sequences[COLUMNS].to_string(index=False))
    print(f"\ninferred source cadence: {report.inferred_source_cadence}")
    print(f"candidates: {report.total_candidate_sequences}, emitted: {report.emitted_sequences}, "
          f"skipped: {report.skipped_sequences}")
    print(f"skipped starts: {[str(t) for t in report.skipped_sequence_starts]}")
    print(f"first missing image per skip: {[str(t) for t in report.missing_timestamps]}")

    print("\nEach sequence also keeps the per-image paths, e.g. the first one:")
    print(sequences["image_path"].iloc[0])


if __name__ == "__main__":
    main()
