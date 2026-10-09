"""
Example 9 -- multiclass labels: NF, A, B, C, M or X.

Question answered for every image: "What is the GOES class of the
strongest flare that starts in (t, t + 24h]?" ("NF" if there is none.)

The label is the class of the single flux value MaxFluxReducer selects,
i.e. the strongest flare in the window. It does not record which other
classes occurred or how many flares there were.

The active-region part warns that two sample flares have no region: they
cannot match any patch (see docs/decisions.md, D6).

Run:  python examples/09_multiclass_labels.py
"""

from pathlib import Path

import pandas as pd

import flare_indexer as fidx

DATA = Path(__file__).parent / "data"
OUTPUT = Path(__file__).parent / "output"
CATALOG = DATA / "flare_catalog.csv"
OPTIONS = dict(prediction_window="24h", event_time="start", interval_mode="right_closed")


def main():
    OUTPUT.mkdir(exist_ok=True)
    multiclass = fidx.DatasetBuildingPipeline(reducer=fidx.MaxFluxReducer(), labeler=fidx.MulticlassLabeler())

    # Full-disk, one label per image; the binary and flux labels for comparison.
    labels = fidx.DatasetBuilder(strategy=multiclass, include_metadata=True, **OPTIONS) \
        .build(DATA / "full_disk_image_index.csv", CATALOG)
    labels["max_flux"] = fidx.DatasetBuilder(strategy=fidx.MaxFlareStrategy(), **OPTIONS) \
        .build(DATA / "full_disk_image_index.csv", CATALOG)["label"]
    labels["binary_M"] = fidx.DatasetBuilder(strategy=fidx.BinaryThresholdStrategy("M"), **OPTIONS) \
        .build(DATA / "full_disk_image_index.csv", CATALOG)["label"]
    labels.to_csv(OUTPUT / "09_multiclass_labels.csv", index=False)

    print("Full-disk images (every 12th shown)")
    with pd.option_context("display.width", 120):
        print(labels[["timestamp", "label", "max_flux", "binary_M"]].iloc[::12].to_string(index=False))
    counts = labels["label"].value_counts().reindex(fidx.MulticlassLabeler.CLASSES, fill_value=0)
    print("\nlabel counts:", ", ".join(f"{c}={n}" for c, n in counts.items()))
    print("binary M label == (multiclass in {M, X}) for every image:",
          (labels["binary_M"] == labels["label"].isin(["M", "X"]).astype(int)).all())

    # Time-based sequences: 6 images every 2h, labeled from the last image.
    sequences, report = fidx.DatasetBuilder(
        strategy=multiclass, cadence="2h", observation_window="12h", sliding_window="12h", **OPTIONS,
    ).build(DATA / "full_disk_image_index.csv", CATALOG, return_report=True)
    print(f"\nTime-based sequences ({report.emitted_sequences} emitted, {report.skipped_sequences} skipped)")
    print(sequences[["sequence_start", "sequence_end", "n_images", "label"]].to_string(index=False))

    # Active regions: each patch only sees its own region's flares.
    ar = fidx.DatasetBuilder(
        strategy=multiclass, target="active_region", missing_active_region="drop", include_metadata=True, **OPTIONS,
    ).build(DATA / "active_region_image_index.csv", CATALOG)
    ar["full_disk_label"] = ar["timestamp"].map(labels.set_index("timestamp")["label"])
    comparable = ar[ar["full_disk_label"].notna()]  # the full-disk index has a gap at 03-02 05:00-06:00
    differ = comparable[comparable["label"] != comparable["full_disk_label"]]
    print(f"\nActive-region patches: {len(differ)} of {len(comparable)} with a full-disk image at the same "
          "time get a different label, e.g.")
    print(differ[["timestamp", "active_region", "label", "full_disk_label"]].head(6).to_string(index=False))


if __name__ == "__main__":
    main()
