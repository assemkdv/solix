"""
Regenerates the small, fully synthetic sample datasets used by examples/.

None of these rows are real observations. The flares are invented so that
each example shows one behavior clearly (a flare that crosses midnight, a
flare exactly 24 hours after an image, a strong flare with no active
region, a gap in the image series, ...).

Usage:
    python examples/data/make_sample_data.py
"""

from pathlib import Path

import pandas as pd

from flare_indexer import adapt_goes_catalog

DATA_DIR = Path(__file__).parent

# GOES-catalog-shaped flares (same column names as an integrated
# GOES catalog, goes_flares_integrated.csv; `ssw_flare_id` is an extra column that
# adapt_goes_catalog ignores).
FLARES = [
    # start,              peak,                 end,                  class, NOAA AR, lon,  lat
    ("2024-03-01 04:00:00", "2024-03-01 04:12:00", "2024-03-01 04:30:00", "C3.2", 13600, -20.0, 12.0),
    ("2024-03-01 13:20:00", "2024-03-01 13:35:00", "2024-03-01 13:50:00", "M1.4", 13600, -18.5, 12.0),
    # Crosses midnight: start and peak fall on different calendar dates.
    ("2024-03-01 23:52:00", "2024-03-02 00:07:00", "2024-03-02 00:20:00", "M2.7", 13601, 35.0, -8.0),
    ("2024-03-02 10:00:00", "2024-03-02 10:15:00", "2024-03-02 10:40:00", "X1.1", 13601, 36.0, -8.0),
    # No NOAA active region recorded (common for real catalogs).
    ("2024-03-02 12:00:00", "2024-03-02 12:09:00", "2024-03-02 12:25:00", "C8.0", None, None, None),
    ("2024-03-03 18:00:00", "2024-03-03 18:05:00", "2024-03-03 18:15:00", "B5.0", 13600, 10.0, 12.5),
    # Starts exactly 24h after the 2024-03-04 00:00 image: outside that
    # image's (t, t+24h) window, inside the window of the 2024-03-05 00:00 image.
    ("2024-03-05 00:00:00", "2024-03-05 00:10:00", "2024-03-05 00:25:00", "M1.0", 13600, 25.0, 12.5),
    # A strong flare with no active region.
    ("2024-03-05 20:00:00", "2024-03-05 20:20:00", "2024-03-05 20:45:00", "M5.0", None, None, None),
]


def _hmi_path(ts: pd.Timestamp) -> str:
    return f"hmi/{ts:%Y/%m/%d}/HMI.m{ts:%Y.%m.%d_%H.%M.%S}.jpg"


def main():
    goes = pd.DataFrame(
        FLARES,
        columns=["start_time", "peak_time", "end_time", "goes_class", "noaa_active_region", "fl_lon", "fl_lat"],
    )
    goes["ssw_flare_id"] = [f"synthetic_{i:02d}" for i in range(len(goes))]
    goes.to_csv(DATA_DIR / "goes_flares_sample.csv", index=False)

    adapt_goes_catalog(DATA_DIR / "goes_flares_sample.csv", DATA_DIR / "flare_catalog.csv")

    # Hourly full-disk images for five days, with 05:00 and 06:00 on
    # 2024-03-02 missing (a data gap).
    hours = pd.date_range("2024-03-01 00:00", "2024-03-05 23:00", freq="1h")
    hours = hours[~hours.isin(pd.to_datetime(["2024-03-02 05:00", "2024-03-02 06:00"]))]
    pd.DataFrame({"timestamp": hours, "image_path": [_hmi_path(t) for t in hours]}).to_csv(
        DATA_DIR / "full_disk_image_index.csv", index=False,
    )

    # Headerless HMI path list, like totalfiles_jpg_512.csv: mostly images,
    # plus one non-image path that a directory walk would also pick up.
    listing = [f"/data/hmi_jpgs_512/{_hmi_path(t)[4:]}" for t in hours[:6]]
    listing.insert(3, "/data/hmi_jpgs_512/labels_store/Fold1_train.csv")
    (DATA_DIR / "hmi_image_list.txt").write_text("\n".join(listing) + "\n")

    # Active-region patches every 6 hours for two regions observed at the
    # same timestamps, plus one patch whose region is unknown.
    patches = []
    for t in pd.date_range("2024-03-01 00:00", "2024-03-05 18:00", freq="6h"):
        for region in (13600, 13601):
            patches.append({
                "timestamp": t,
                "active_region": region,
                "image_path": f"patches/{region}/patch_{region}_{t:%Y%m%d_%H%M}.jpg",
            })
    patches.append({
        "timestamp": pd.Timestamp("2024-03-02 00:00"),
        "active_region": None,
        "image_path": "patches/unknown/patch_unknown_20240302_0000.jpg",
    })
    ar_index = pd.DataFrame(patches).sort_values(["timestamp", "image_path"], kind="stable")
    ar_index["active_region"] = ar_index["active_region"].astype("Int64")
    ar_index.to_csv(DATA_DIR / "active_region_image_index.csv", index=False)

    print(f"Sample data written to {DATA_DIR}")


if __name__ == "__main__":
    main()
