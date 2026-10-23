"""
Example 1 -- full-disk, single-image labels.

Question answered for every image: "Did an M-class or stronger flare
start in the 24 hours after this image?"

Run:  python examples/01_full_disk.py
"""

from pathlib import Path

import pandas as pd

import flare_indexer as fidx

DATA = Path(__file__).parent / "data"
OUTPUT = Path(__file__).parent / "output"


def main():
    OUTPUT.mkdir(exist_ok=True)

    builder = fidx.DatasetBuilder(
        prediction_window=24,                       # hours (or a duration such as "24h")
        strategy=fidx.BinaryThresholdStrategy("M"),  # 1 if any flare >= M1.0, else 0
        target="full_disk",                          # any flare on the Sun counts
        event_time="start",                          # compare the flare's start_time ...
        include_metadata=True,                       # keep image_path in the output
    )
    labels = builder.build(DATA / "full_disk_image_index.csv", DATA / "flare_catalog.csv")

    # A second builder gives the strongest flare's flux instead of 0/1.
    flux = fidx.DatasetBuilder(
        prediction_window=24, strategy=fidx.MaxFlareStrategy(),
        event_time="start",
    ).build(DATA / "full_disk_image_index.csv", DATA / "flare_catalog.csv")
    labels["max_flux"] = flux["label"]

    labels.to_csv(OUTPUT / "01_full_disk_labels.csv", index=False)

    with pd.option_context("display.width", 120):
        print(labels.iloc[::8].to_string(index=False))
    print(f"\n{len(labels)} images: {int(labels['label'].sum())} positive, "
          f"{int((labels['label'] == 0).sum())} negative")
    print(f"Written to {OUTPUT / '01_full_disk_labels.csv'}")


if __name__ == "__main__":
    main()
