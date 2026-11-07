"""
Example 3 -- active-region (AR) labels.

With ActiveRegionSingleInstance and ActiveRegionSequence, an instance only
gets flares from its own NOAA active region. This example prints three parts:

1) two regions observed at the same time get different labels, and two
   warnings: an instance with an unknown region is labeled 0, and catalog
   flares with no region can never be matched;
2) missing_active_region="drop" leaves the unknown instance out, and "error"
   stops instead;
3) sequences are built separately for each region.

Run:  python examples/03_active_region.py
"""

import warnings
from pathlib import Path

import solix

DATA = Path(__file__).parent / "data"
INDEX = DATA / "active_region_index.csv"
CATALOG = DATA / "flare_catalog.csv"
OUTPUT = Path(__file__).parent / "output"


def build(**kwargs):
    return solix.ActiveRegionSingleInstance(
        prediction_window=24, reducer=solix.MaxFluxReducer(), labeler=solix.BinaryThresholdLabeler("M"),
        include_metadata=True, **kwargs,
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

    print("\n2) missing_active_region='drop' leaves the unknown instance out:")
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
    print("   unknown instance never joins a region's sequence (it is dropped here).")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sequences, report = solix.ActiveRegionSequence(
            prediction_window=24, reducer=solix.MaxFluxReducer(), labeler=solix.BinaryThresholdLabeler("M"),
            missing_active_region="drop",
            cadence="6h", observation_window="12h", sliding_window="6h",
        ).build(INDEX, CATALOG, return_report=True)
    print(sequences[["sequence_start", "sequence_end", "active_region", "label"]].head(8).to_string(index=False))
    print(f"   ... {len(sequences)} sequences from {report.region_count} regions, "
          f"{report.skipped_sequences} skipped for missing instances")


if __name__ == "__main__":
    main()
