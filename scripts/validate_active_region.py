"""
Reproduce the published arpatch-sfp active-region (AR) labels.

This checks the label files published in chetrajpandey/arpatch-sfp. It is
NOT a validation of the complete original SHARP patch dataset: the
published files contain only patches that passed the legacy filters (see
docs/validation/active_region.md), and the unfiltered patch table is not
public.

What it does:
1. Reads the published label files. Default: the four
   data_labels/-90_to_90/Fold{1..4}_test.csv files, which partition the
   published -90..90 degree dataset by the month of harp_start
   (step5_CV_lon_filter.py). Pass --labels to use other files, e.g.
   train/val/test.csv (train.csv is an undersampled subset of Fold1+Fold2;
   val.csv and test.csv equal Fold3 and Fold4).
2. Builds a HARP-keyed flare catalog exactly as step1_harp_to_noaa.py
   does: GOES flares joined to HARP numbers through allharp2noaa.csv.
3. Re-implements the legacy labeling rule of step3_labeling.py -- flares
   of the same HARP with start_time in [t - 1 day, t], goes_class = the
   alphabetically largest class string, label = goes_class >= "M1.0" --
   and compares both the `label` and `goes_class` columns, file by file
   (the published "FQ" marker counts as "no flare").
4. Labels every patch with flare_indexer (target="active_region",
   start_time, (t, t+window], M threshold) and compares the binary label.

All inputs are pinned to fixed commits, so the comparison is repeatable.

Usage:
    python scripts/validate_active_region.py --summary docs/validation/active_region_summary.json
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from flare_indexer import BinaryThresholdStrategy, DatasetBuilder, FluxConverter

from validate_full_disk import CACHE_DIR, _resolve_source, _sha256

ARPATCH_COMMIT = "ee20210dd6c6aa0396f1f4b8575ad05e9d0142b6"
_BASE = f"https://raw.githubusercontent.com/chetrajpandey/arpatch-sfp/{ARPATCH_COMMIT}/data_process/data_labeling/"
DEFAULT_LABEL_URLS = [_BASE + f"data_labels/-90_to_90/Fold{i}_test.csv" for i in range(1, 5)]
DEFAULT_HARP_MAP_URL = _BASE + "allharp2noaa.csv"
DEFAULT_EVENT_CATALOG_URL = _BASE + "goes_flares_integrated.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABEL_URLS,
                        help="Published AR label CSVs (local paths or URLs).")
    parser.add_argument("--harp-map", default=DEFAULT_HARP_MAP_URL, help="allharp2noaa.csv (path or URL).")
    parser.add_argument("--event-catalog", default=DEFAULT_EVENT_CATALOG_URL,
                        help="goes_flares_integrated.csv (path or URL).")
    parser.add_argument("--prediction-window", default="24h", help="Forward window for flare_indexer.")
    parser.add_argument("--summary", default=None, help="Optional JSON summary output path.")
    return parser.parse_args()


def _cache_name(source: str) -> str:
    parts = source.rstrip("/").split("/")
    return "ar_" + "_".join(parts[-2:]).replace("-", "m") if "data_labels" in source else parts[-1]


def harp_keyed_catalog(goes_path: Path, harp_map_path: Path) -> tuple[pd.DataFrame, dict]:
    goes = pd.read_csv(goes_path)
    harp = pd.read_csv(harp_map_path)

    harp["NOAA_ARS"] = harp["NOAA_ARS"].astype(str).str.strip('"').str.split(",")
    harp = harp.explode("NOAA_ARS")
    harp["NOAA_ARS"] = pd.to_numeric(harp["NOAA_ARS"], errors="coerce")
    harp = harp.dropna(subset=["NOAA_ARS", "DEF_HARPNUM"])
    harp = harp.astype({"NOAA_ARS": int, "DEF_HARPNUM": int})[["DEF_HARPNUM", "NOAA_ARS"]].drop_duplicates()

    goes = goes.reset_index().rename(columns={"index": "flare_row"})
    with_noaa = goes[goes["noaa_active_region"].notna()].copy()
    with_noaa["noaa"] = with_noaa["noaa_active_region"].astype(int)
    joined = with_noaa.merge(harp, left_on="noaa", right_on="NOAA_ARS", how="inner")

    catalog = pd.DataFrame({
        "start_time": pd.to_datetime(joined["start_time"]),
        "peak_time": pd.to_datetime(joined["peak_time"]),
        "end_time": pd.to_datetime(joined["end_time"]),
        "class": joined["goes_class"],
        "active_region": joined["DEF_HARPNUM"].astype(str),
    }).sort_values("start_time", kind="stable").reset_index(drop=True)

    mx = goes["goes_class"].str[0].isin(["M", "X"])
    stats = {
        "goes_flares": len(goes),
        "goes_m_or_x_flares": int(mx.sum()),
        "flares_without_noaa_region": int(goes["noaa_active_region"].isna().sum()),
        "m_or_x_flares_without_noaa_region": int((mx & goes["noaa_active_region"].isna()).sum()),
        "flares_with_noaa_but_no_harp": int(with_noaa.loc[~with_noaa["noaa"].isin(harp["NOAA_ARS"]), "flare_row"].nunique()),
        "flare_harp_pairs": len(joined),
        "distinct_flares_mapped": int(joined["flare_row"].nunique()),
        "max_noaa_regions_per_harp": int(harp.groupby("DEF_HARPNUM")["NOAA_ARS"].nunique().max()),
        "max_harps_per_noaa_region": int(harp.groupby("NOAA_ARS")["DEF_HARPNUM"].nunique().max()),
        "first_flare_start": str(catalog["start_time"].min()),
        "last_flare_start": str(catalog["start_time"].max()),
    }
    return catalog, stats


def legacy_lookback(index: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """
    step3_labeling.py + step4_with_lon_filtering.py, per patch:
    goes_class = max of the class strings (alphabetical, as pandas .max()
    on strings) of same-HARP flares with start_time in [t - 1 day, t];
    label = goes_class >= "M1.0" (alphabetical) and not empty.
    Also returns the numeric-maximum class, to show where the alphabetical
    choice differs.
    """
    converter = FluxConverter()
    lexical = np.full(len(index), "", dtype=object)
    numeric = np.full(len(index), "", dtype=object)
    for harp, rows in index.groupby("active_region"):
        flares = catalog[catalog["active_region"] == harp]
        if flares.empty:
            continue
        starts = flares["start_time"].to_numpy()
        classes = flares["class"].to_numpy()
        times = rows["timestamp"].to_numpy()
        lo = np.searchsorted(starts, times - np.timedelta64(1, "D"), side="left")
        hi = np.searchsorted(starts, times, side="right")
        for position, a, b in zip(rows.index, lo, hi):
            if b > a:
                window = classes[a:b]
                lexical[position] = max(window)
                numeric[position] = max(window, key=converter.to_flux)
    labels = np.array([int(c != "" and c >= "M1.0") for c in lexical])
    return pd.DataFrame({"goes_class": lexical, "numeric_max_class": numeric, "label": labels}, index=index.index)


def _describe_file(source: str, path: Path, frame: pd.DataFrame) -> dict:
    return {
        "source": source,
        "sha256": _sha256(path),
        "rows": len(frame),
        "positives": int(frame["label"].sum()),
        "harps": int(frame["harpnumber"].nunique()),
        "first_timestamp": str(frame["timestamp"].min()),
        "last_timestamp": str(frame["timestamp"].max()),
        "harp_start_months": sorted(int(m) for m in pd.to_datetime(frame["harp_start"]).dt.month.unique()),
        "lon_fwt_range": [round(float(frame["LON_FWT"].min()), 3), round(float(frame["LON_FWT"].max()), 3)]
        if "LON_FWT" in frame else None,
        "duplicate_harp_timestamp_rows": int(frame.duplicated(["harpnumber", "timestamp"]).sum()),
    }


def _agreement(name: str, ours: np.ndarray, theirs: np.ndarray) -> dict:
    agree = int((ours == theirs).sum())
    result = {
        "agree": agree,
        "total": len(ours),
        "disagree": len(ours) - agree,
    }
    print(f"  {name}: {agree}/{len(ours)} agree ({100 * agree / len(ours):.3f}%)")
    return result


def main():
    args = parse_args()
    t0 = time.monotonic()

    label_paths = [_resolve_source(src, _cache_name(src)) for src in args.labels]
    harp_map_path = _resolve_source(args.harp_map, "allharp2noaa.csv")
    goes_path = _resolve_source(args.event_catalog, "goes_flares_integrated.csv")

    frames = []
    files = []
    for source, path in zip(args.labels, label_paths):
        frame = pd.read_csv(path, low_memory=False)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        files.append(_describe_file(source, path, frame))
        frame["source_file"] = len(files) - 1
        frames.append(frame)
    published = pd.concat(frames, ignore_index=True)
    duplicates_across_files = int(published.duplicated(["harpnumber", "timestamp"]).sum())

    index = pd.DataFrame({
        "timestamp": published["timestamp"],
        "active_region": published["harpnumber"].astype(int).astype(str),
        "image_path": published["SHARP_FILE"],
    })
    print(f"Published AR patches: {len(index)} rows from {len(files)} file(s), "
          f"{index['active_region'].nunique()} HARPs, {index['timestamp'].min()} -> {index['timestamp'].max()}")
    print(f"  duplicate (harpnumber, timestamp) rows across files: {duplicates_across_files}")
    print(f"  max patches at one timestamp: {int(index.groupby('timestamp').size().max())}")

    catalog, catalog_stats = harp_keyed_catalog(goes_path, harp_map_path)
    print("HARP-keyed catalog:", catalog_stats)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    index_path = CACHE_DIR / "ar_image_index.csv"
    catalog_path = CACHE_DIR / "ar_flare_catalog_harp.csv"
    index.to_csv(index_path, index=False)
    catalog.to_csv(catalog_path, index=False)

    published_label = published["label"].astype(int).to_numpy()
    # The published files write "FQ" (flare-quiet) where step3 wrote an
    # empty class (step6_undersample.py uses the same marker). Treat it, and
    # a blank cell, as "no flare in the window".
    raw_class = published["goes_class"].fillna("").astype(str)
    no_flare_markers = raw_class.isin(["", "FQ"])
    published_class = raw_class.where(~no_flare_markers, "").to_numpy()

    print("Legacy look-back rule [t - 1 day, t] (step3_labeling.py):")
    lookback = legacy_lookback(index, catalog)
    lookback_label = _agreement("label", lookback["label"].to_numpy(), published_label)
    lookback_class = _agreement("goes_class string", lookback["goes_class"].to_numpy(), published_class)
    lexical_vs_numeric = int((lookback["goes_class"] != lookback["numeric_max_class"]).sum())
    print(f"  rows where the alphabetical max class differs from the numeric max: {lexical_vs_numeric}")

    print(f"flare_indexer forward (t, t+{args.prediction_window}]:")
    builder = DatasetBuilder(
        prediction_window=args.prediction_window, strategy=BinaryThresholdStrategy("M"),
        target="active_region", event_time="start", interval_mode="right_closed",
        missing_active_region="error", include_metadata=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the HARP-keyed catalog has a region for every row
        generated = builder.build(index_path, catalog_path)
    forward = generated["label"].astype(int).to_numpy()
    forward_label = _agreement("label", forward, published_label)

    per_file = []
    for i, description in enumerate(files):
        mask = published["source_file"].to_numpy() == i
        per_file.append({
            "source": description["source"],
            "rows": int(mask.sum()),
            "lookback_label_agree": int((lookback["label"].to_numpy()[mask] == published_label[mask]).sum()),
            "lookback_goes_class_agree": int((lookback["goes_class"].to_numpy()[mask] == published_class[mask]).sum()),
            "forward_label_agree": int((forward[mask] == published_label[mask]).sum()),
            "forward_positives": int(forward[mask].sum()),
        })

    summary = {
        "inputs": {
            "arpatch_sfp_commit": ARPATCH_COMMIT,
            "label_files": files,
            "harp_map": {"source": args.harp_map, "sha256": _sha256(harp_map_path)},
            "event_catalog": {"source": args.event_catalog, "sha256": _sha256(goes_path)},
        },
        "patches": len(index),
        "harps": int(index["active_region"].nunique()),
        "duplicate_rows_across_files": duplicates_across_files,
        "published_positives": int(published_label.sum()),
        "published_no_flare_rows": {"FQ": int((raw_class == "FQ").sum()), "blank": int((raw_class == "").sum())},
        "catalog": catalog_stats,
        "legacy_lookback": {
            "rule": "same HARP, start_time in [t - 1 day, t]; goes_class = alphabetical max; label = goes_class >= 'M1.0'",
            "label": lookback_label,
            "goes_class": lookback_class,
            "positives": int(lookback["label"].sum()),
            "alphabetical_vs_numeric_max_differences": lexical_vs_numeric,
        },
        "flare_indexer_forward": {
            "settings": f"target=active_region, event_time=start, (t, t+{args.prediction_window}], M threshold",
            "label": forward_label,
            "positives": int(forward.sum()),
            "generated_1_published_0": int(((forward == 1) & (published_label == 0)).sum()),
            "generated_0_published_1": int(((forward == 0) & (published_label == 1)).sum()),
        },
        "per_file": per_file,
    }
    print(f"Runtime: {time.monotonic() - t0:.1f}s")

    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Summary written to: {args.summary}")


if __name__ == "__main__":
    main()
