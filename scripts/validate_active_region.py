"""
Active-region (AR) real-data check against the published arpatch-sfp labels.

What it does:
1. Builds an image index from the published AR label files
   (train/val/test.csv: harpnumber, timestamp, SHARP_FILE, goes_class,
   label, ...). These are the patches that survived the legacy width and
   longitude filters, not every SHARP patch.
2. Builds a HARP-keyed flare catalog: GOES flares are joined to HARP
   numbers through allharp2noaa.csv (one HARP can cover several NOAA
   regions), and adapt_goes_catalog()-style columns are written with
   active_region = HARP number.
3. Labels every patch with flare_indexer (target="active_region",
   start_time, (t, t+window], M threshold) and compares with the
   published binary labels.
4. Re-implements the legacy rule from step3_labeling.py -- flares of the
   same HARP with start_time in [t - 24h, t] -- and reports how many
   published labels it reproduces, to show which window the published
   labels actually use.

flare_indexer only supports windows after the image time, so step 3 is
not expected to reproduce the published labels if they use step 4's
look-back window; that difference is a scientific decision, not a bug to
patch here. See docs/active_region_workflow.md.

Usage:
    python scripts/validate_active_region.py
    python scripts/validate_active_region.py --summary docs/validation/active_region_summary.json
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import pandas as pd

from flare_indexer import BinaryThresholdStrategy, DatasetBuilder, FluxConverter

from validate_full_disk import CACHE_DIR, _resolve_source, _sha256

_BASE = "https://raw.githubusercontent.com/chetrajpandey/arpatch-sfp/main/data_process/data_labeling/"
DEFAULT_LABEL_URLS = [_BASE + f"data_labels/{split}.csv" for split in ("train", "val", "test")]
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
    }).sort_values("start_time", kind="stable")

    stats = {
        "goes_flares": len(goes),
        "flares_without_noaa_region": int(goes["noaa_active_region"].isna().sum()),
        "flares_with_noaa_but_no_harp": int(with_noaa.loc[~with_noaa["noaa"].isin(harp["NOAA_ARS"]), "flare_row"].nunique()),
        "flare_harp_pairs": len(joined),
        "distinct_flares_mapped": int(joined["flare_row"].nunique()),
        "max_noaa_regions_per_harp": int(harp.groupby("DEF_HARPNUM")["NOAA_ARS"].nunique().max()),
        "max_harps_per_noaa_region": int(harp.groupby("NOAA_ARS")["DEF_HARPNUM"].nunique().max()),
    }
    return catalog, stats


def legacy_lookback_labels(index: pd.DataFrame, catalog: pd.DataFrame) -> pd.Series:
    """step3_labeling.py rule: same HARP, start_time in [t - 24h, t], then >= M1.0."""
    converter = FluxConverter()
    by_harp = {harp: rows for harp, rows in catalog.groupby("active_region")}
    labels = []
    for harp, t in zip(index["active_region"], index["timestamp"]):
        rows = by_harp.get(harp)
        if rows is None:
            labels.append(0)
            continue
        in_window = rows[(rows["start_time"] >= t - pd.Timedelta(hours=24)) & (rows["start_time"] <= t)]
        labels.append(int(any(converter.to_flux(c) >= 1e-5 for c in in_window["class"])))
    return pd.Series(labels, index=index.index)


def main():
    args = parse_args()
    t0 = time.monotonic()

    label_paths = [_resolve_source(src, f"ar_{Path(src).stem}.csv") for src in args.labels]
    harp_map_path = _resolve_source(args.harp_map, "allharp2noaa.csv")
    goes_path = _resolve_source(args.event_catalog, "goes_flares_integrated.csv")

    published = pd.concat([pd.read_csv(p) for p in label_paths], ignore_index=True)
    published["timestamp"] = pd.to_datetime(published["timestamp"])
    index = pd.DataFrame({
        "timestamp": published["timestamp"],
        "active_region": published["harpnumber"].astype(int).astype(str),
        "image_path": published["SHARP_FILE"],
    })
    print(f"Published AR patches: {len(index)} from {index['active_region'].nunique()} HARPs, "
          f"{index['timestamp'].min()} -> {index['timestamp'].max()}")
    print(f"  max patches at one timestamp: {int(index.groupby('timestamp').size().max())}")

    catalog, catalog_stats = harp_keyed_catalog(goes_path, harp_map_path)
    print("HARP-keyed catalog:", catalog_stats)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    index_path = CACHE_DIR / "ar_image_index.csv"
    catalog_path = CACHE_DIR / "ar_flare_catalog_harp.csv"
    index.to_csv(index_path, index=False)
    catalog.to_csv(catalog_path, index=False)

    builder = DatasetBuilder(
        prediction_window=args.prediction_window, strategy=BinaryThresholdStrategy("M"),
        target="active_region", event_time="start", interval_mode="right_closed",
        missing_active_region="error", include_metadata=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # catalog rows all have a HARP here
        generated = builder.build(index_path, catalog_path)

    published_label = published["label"].astype(int).to_numpy()
    forward = generated["label"].astype(int).to_numpy()
    lookback = legacy_lookback_labels(index, catalog).to_numpy()

    def compare(name, labels):
        agree = int((labels == published_label).sum())
        result = {
            "agreement": agree,
            "total": len(labels),
            "positives": int(labels.sum()),
            "generated_1_published_0": int(((labels == 1) & (published_label == 0)).sum()),
            "generated_0_published_1": int(((labels == 0) & (published_label == 1)).sum()),
        }
        print(f"{name}: {agree}/{len(labels)} agree ({100 * agree / len(labels):.3f}%), "
              f"positives {result['positives']} vs published {int(published_label.sum())}")
        return result

    summary = {
        "inputs": {
            "labels": [{"source": s, "sha256": _sha256(p)} for s, p in zip(args.labels, label_paths)],
            "harp_map": {"source": args.harp_map, "sha256": _sha256(harp_map_path)},
            "event_catalog": {"source": args.event_catalog, "sha256": _sha256(goes_path)},
        },
        "patches": len(index),
        "harps": int(index["active_region"].nunique()),
        "published_positives": int(published_label.sum()),
        "catalog": catalog_stats,
        "flare_indexer_forward": {
            "settings": f"target=active_region, event_time=start, (t, t+{args.prediction_window}], M threshold",
            **compare(f"flare_indexer forward (t, t+{args.prediction_window}]", forward),
        },
        "legacy_rule_lookback": {
            "settings": "same HARP, start_time in [t-24h, t], M threshold (step3_labeling.py)",
            **compare("legacy look-back rule [t-24h, t]", lookback),
        },
    }
    print(f"Runtime: {time.monotonic() - t0:.1f}s")

    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Summary written to: {args.summary}")


if __name__ == "__main__":
    main()
