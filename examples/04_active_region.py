"""
Example 4 -- active-region (AR) labels.

With target="active_region", an image patch only gets flares from its own
NOAA active region. This example shows:

1. two regions observed at the same time get different labels;
2. what happens to a patch with an unknown region (missing_active_region);
3. catalog flares with no region can never be matched (a warning says so);
4. how to build AR sequences today: one region at a time.

Run:  python examples/04_active_region.py
"""

import warnings
from pathlib import Path

import pandas as pd

import flare_indexer as fidx

DATA = Path(__file__).parent / "data"
INDEX = DATA / "active_region_image_index.csv"
CATALOG = DATA / "flare_catalog.csv"
OUTPUT = Path(__file__).parent / "output"


def build(**kwargs):
    return fidx.DatasetBuilder(
        prediction_window=24, strategy=fidx.BinaryThresholdStrategy("M"),
        target="active_region", event_time="start", interval_mode="right_closed",
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

    print("\n2) missing_active_region='drop' leaves the unknown patch out:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dropped = build(missing_active_region="drop")
    print(f"  rows: {len(labels)} -> {len(dropped)}")
    dropped.to_csv(OUTPUT / "04_active_region_labels.csv", index=False)

    print("\n   missing_active_region='error' stops instead:")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            build(missing_active_region="error")
    except ValueError as exc:
        print(f"  ValueError: {str(exc)[:110]}...")

    print("\n3) Sequences: the builder needs unique timestamps, so split by region first.")
    index = pd.read_csv(INDEX, dtype={"active_region": "string"})
    for region, rows in index.dropna(subset=["active_region"]).groupby("active_region"):
        region_index = OUTPUT / f"04_index_region_{region}.csv"
        rows.to_csv(region_index, index=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sequences = fidx.DatasetBuilder(
                prediction_window=24, strategy=fidx.BinaryThresholdStrategy("M"),
                target="active_region", event_time="start", interval_mode="right_closed",
                cadence="6h", observation_window="12h", sliding_window="6h",
            ).build(region_index, CATALOG)
        print(f"\n  region {region}:")
        print(sequences[["sequence_start", "sequence_end", "active_region", "label"]].to_string(index=False))


if __name__ == "__main__":
    main()
