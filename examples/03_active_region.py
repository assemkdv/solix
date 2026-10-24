"""
Example 3 -- active-region (AR) labels.

With target="active_region", an image patch only gets flares from its own
NOAA active region. This example shows:

1. two regions observed at the same time get different labels;
2. what happens to a patch with an unknown region (missing_active_region);
3. catalog flares with no region can never be matched (a warning says so);
4. sequences are built separately for each region.

Run:  python examples/03_active_region.py
"""

import warnings
from pathlib import Path

import flare_indexer as fidx

DATA = Path(__file__).parent / "data"
INDEX = DATA / "active_region_image_index.csv"
CATALOG = DATA / "flare_catalog.csv"
OUTPUT = Path(__file__).parent / "output"


def build(**kwargs):
    return fidx.ActiveRegionSingleImage(
        prediction_window=24, strategy=fidx.BinaryThresholdStrategy("M"),
        event_time="start", include_metadata=True, **kwargs,
    ).build(INDEX, CATALOG)


def main():
    OUTPUT.mkdir(exist_ok=True)

    print("1) Default policy (missing_active_region='negative'):")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        labels = build()
    print(labels[["timestamp", "active_region", "label", "image_path"]].head(8).to_string(index=False))
    print("\nWarnings raised:")
    for w in caught:
        print(f"  - {w.message}")

    print("\n2) missing_active_region='drop' leaves the unknown patch out:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dropped = build(missing_active_region="drop")
    print(f"  rows: {len(labels)} -> {len(dropped)}")
    dropped.to_csv(OUTPUT / "03_active_region_labels.csv", index=False)

    print("\n   missing_active_region='error' stops instead:")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            build(missing_active_region="error")
    except ValueError as exc:
        print(f"  ValueError: {str(exc)[:110]}...")

    print("\n3) Sequences are built per region: two regions share timestamps, and the")
    print("   unknown patch never joins a region's sequence (it is dropped here).")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sequences, report = fidx.ActiveRegionSequence(
            prediction_window=24, strategy=fidx.BinaryThresholdStrategy("M"),
            event_time="start", missing_active_region="drop",
            cadence="6h", observation_window="12h", sliding_window="6h",
        ).build(INDEX, CATALOG, return_report=True)
    print(sequences[["sequence_start", "sequence_end", "active_region", "label"]].head(8).to_string(index=False))
    print(f"   ... {len(sequences)} sequences from {report.region_count} regions, "
          f"{report.skipped_sequences} skipped for missing images")

if __name__ == "__main__":
    main()
