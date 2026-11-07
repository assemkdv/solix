"""
Example 1 -- full-disk, single-instance labels.

Question answered for every instance (here, one HMI image each): "Did an
M-class or stronger flare start in the 24 hours after this instance?"

Run:  python examples/01_full_disk.py
"""

from pathlib import Path

import pandas as pd

import solix

DATA = Path(__file__).parent / "data"
OUTPUT = Path(__file__).parent / "output"


def main():
    OUTPUT.mkdir(exist_ok=True)

    builder = solix.FullDiskSingleInstance(
        prediction_window="24h",                     # a number means hours: 24 is the same
        reducer=solix.MaxFluxReducer(),              # the strongest flare in the window
        labeler=solix.BinaryThresholdLabeler("M"),   # 1 if a flare >= M1.0 starts in the window, else 0
        include_metadata=True,                       # keep image_path in the output
    )
    labels = builder.build(DATA / "full_disk_index.csv", DATA / "flare_catalog.csv")

    # A second builder gives the strongest flare's flux instead of 0/1.
    flux = solix.FullDiskSingleInstance(
        prediction_window="24h", reducer=solix.MaxFluxReducer(), labeler=solix.RegressionLabeler(),
    ).build(DATA / "full_disk_index.csv", DATA / "flare_catalog.csv")
    labels["max_flux"] = flux["label"]

    labels.to_csv(OUTPUT / "01_full_disk_labels.csv", index=False)

    with pd.option_context("display.width", 120):
        print(labels.iloc[::8].to_string(index=False))
    print(f"\n{len(labels)} instances: {int(labels['label'].sum())} positive, "
          f"{int((labels['label'] == 0).sum())} negative")
    print(f"Written to {OUTPUT / '01_full_disk_labels.csv'}")


if __name__ == "__main__":
    main()
