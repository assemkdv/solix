"""
Example 2 -- labeling sequences of full-disk instances.

A sequence is a set of instances on an exact time grid: `cadence` is the
time between instances, `observation_window` the time one sequence covers,
and `sliding_window` the step to the next sequence. A sequence is labeled
from its LAST instance, and a sequence with a missing instance is skipped,
never filled in.

The sample index has a gap (2024-03-02 05:00 and 06:00 are missing).

Run:  python examples/02_sequences.py
"""

from pathlib import Path

import pandas as pd

import solix

DATA = Path(__file__).parent / "data"
INDEX = DATA / "full_disk_index.csv"
CATALOG = DATA / "flare_catalog.csv"
COLUMNS = ["sequence_start", "sequence_end", "n_instances", "label"]


def main():
    print("6 instances every 2h (12h observation window), new sequence every 6h")
    sequences, report = solix.FullDiskSequence(
        "24h", reducer=solix.MaxFluxReducer(), labeler=solix.BinaryThresholdLabeler("M"),
        cadence="2h", observation_window="12h", sliding_window="6h",
    ).build(INDEX, CATALOG, return_report=True)
    with pd.option_context("display.width", 120):
        print(sequences[COLUMNS].to_string(index=False))
    print(f"\ninferred source cadence: {report.inferred_source_cadence}")
    print(f"candidates: {report.total_candidate_sequences}, emitted: {report.emitted_sequences}, "
          f"skipped: {report.skipped_sequences}")
    print(f"skipped starts: {[str(t) for t in report.skipped_sequence_starts]}")
    print(f"first missing instance per skip: {[str(t) for t in report.missing_timestamps]}")

    print("\nEach sequence also keeps the per-instance paths, e.g. the first one:")
    print(sequences["image_path"].iloc[0])


if __name__ == "__main__":
    main()
