"""
Example 2 -- labeling sequences of images (full-disk).

A sequence is labeled from its LAST image: the prediction window starts at
`sequence_end`. Two ways to build sequences:

- row-based:  every `sequence_length` consecutive rows (optionally checking
              that images are exactly `cadence_minutes` apart);
- time-based: an exact timestamp grid start + i * cadence; a sequence
              with any missing image is skipped, never interpolated.

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
    print("Row-based: 3 consecutive hourly images, stride 3, must be 60 min apart")
    row_based = fidx.DatasetBuilder(
        prediction_window=24, strategy=fidx.BinaryThresholdStrategy("M"),
        sequence_length=3, stride=3, cadence_minutes=60,
    ).build(INDEX, CATALOG)
    print(row_based[COLUMNS].head(6).to_string(index=False))
    print(f"... {len(row_based)} sequences\n")

    print("Time-based: 6 images every 2h (12h observation window), new sequence every 6h")
    time_based, report = fidx.DatasetBuilder(
        prediction_window=24, strategy=fidx.BinaryThresholdStrategy("M"),
        cadence="2h", observation_window="12h", sliding_window="6h",
    ).build(INDEX, CATALOG, return_report=True)
    with pd.option_context("display.width", 120):
        print(time_based[COLUMNS].to_string(index=False))
    print(f"\ninferred source cadence: {report.inferred_source_cadence}")
    print(f"candidates: {report.total_candidate_sequences}, emitted: {report.emitted_sequences}, "
          f"skipped: {report.skipped_sequences}")
    print(f"skipped starts: {[str(t) for t in report.skipped_sequence_starts]}")
    print(f"first missing image per skip: {[str(t) for t in report.missing_timestamps]}")

    print("\nEach sequence also keeps the per-image paths, e.g. the first one:")
    print(time_based["image_path"].iloc[0])


if __name__ == "__main__":
    main()
