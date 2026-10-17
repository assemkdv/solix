"""
Validation of the cumulative class index against the reference calculation
and the published SuryaBench labels.

What is checked
---------------
1. Reference calculation. For an hourly grid of timestamps, the package
   (DatasetBuilder + CumulativeClassIndexReducer) is compared with a literal
   re-implementation of NASA-IMPACT/SuryaBench flare/flare_index_creator.py:
   sub-class values C=n, M=10n, X=100n (A/B -> None, skipped by the sum),
   flares with  t < event_starttime <= t + window, window 23:59:59, stride
   1 h, binary label index >= 10. Raw-index and binary-label differences are
   reported separately.
2. Published labels. The hourly rows of the public SuryaBench flare dataset
   (huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-flare-forecasting,
   CC BY 4.0; columns timestamp, max_goes_class, cumulative_index, label_max,
   label_cum) are joined on timestamp and compared with the package output,
   once with the lab's goes_flares_integrated.csv as input (2010-2018) and
   once with SuryaBench's own flare_catalog_2010-2024.csv (2010-2024).
   Unmatched timestamps and every disagreement are counted; the first
   disagreements are printed with the flares the package matched, so each
   can be investigated. The rule is never adjusted to force agreement.

Window convention: the reference code's (t, t+23:59:59] on start time is
used because it is what produced the published numbers. It is not the only
defensible choice; the paper writes [t, t+24h] (closed at both ends), which
is not expressible with the package's two interval modes. Use
--prediction-window / --interval-mode to test other conventions.

Catalog handling: the lab catalog gets the documented peak_time corrections
of scripts/catalog_corrections.py (they change peaks only; start-time
matching is unaffected). The SuryaBench catalog is renamed to the lab column
names and adapted with adapt_goes_catalog; rows whose peak time is outside
[start, end] or whose end precedes the start (the same upstream HEK error)
get that time blanked so the catalog loads; start times are never changed.
One row with class "A0.0" (zero multiplier, rejected by the package; an A
value adds nothing in SuryaBench either) is dropped. Counts are reported.

Usage:
    python scripts/validate_cumulative_index.py
    python scripts/validate_cumulative_index.py --summary docs/validation/cumulative_index_summary.json
    python scripts/validate_cumulative_index.py --catalog lab --no-published
    python scripts/validate_cumulative_index.py --prediction-window 24h --threshold 10 --threshold 20

Downloads are cached in scripts/.cache/ (gitignored).
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from flare_indexer import (
    CumulativeClassIndexReducer, DatasetBuilder, DatasetBuildingPipeline, EventMatcher,
    InvalidGOESClassError, RegressionLabeler,
)
from flare_indexer.classifier import parse_goes_class
from flare_indexer.events import _parse_prediction_window
from flare_indexer.loaders import adapt_goes_catalog

from catalog_corrections import apply_catalog_corrections, print_corrections
from validate_full_disk import CACHE_DIR, DEFAULT_EVENT_CATALOG_URL, _resolve_source, _sha256

SURYABENCH_REPO = "NASA-IMPACT/SuryaBench"
SURYABENCH_CATALOG_PATH = "flare/flare_catalog_2010-2024.csv"
SURYABENCH_SCRIPT_PATH = "flare/flare_index_creator.py"
HF_DATASET = "nasa-ibm-ai4science/surya-bench-flare-forecasting"
DEFAULT_LABELS_URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/data.csv"

# Column names of SuryaBench's catalog -> the lab catalog names adapt_goes_catalog expects.
SURYABENCH_RENAME = {
    "event_starttime": "start_time", "event_peaktime": "peak_time", "event_endtime": "end_time",
    "fl_goescls": "goes_class", "ar_noaanum": "noaa_active_region",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", choices=["lab", "suryabench", "both"], default="both",
                        help="Which flare catalog(s) to label (default both).")
    parser.add_argument("--event-catalog", default=DEFAULT_EVENT_CATALOG_URL,
                        help="Lab goes_flares_integrated.csv (path or URL).")
    parser.add_argument("--suryabench-catalog", default=None,
                        help="SuryaBench flare_catalog_2010-2024.csv (path or URL). Default: the file at the "
                             "latest commit that touched it, resolved through the GitHub API and recorded.")
    parser.add_argument("--suryabench-labels", default=DEFAULT_LABELS_URL,
                        help="Published SuryaBench data.csv (path or URL).")
    parser.add_argument("--no-published", action="store_true", help="Skip the comparison with published labels.")
    parser.add_argument("--prediction-window", default="23h59min59s",
                        help="Window length (default 23h59min59s, the reference convention).")
    parser.add_argument("--interval-mode", choices=["left_closed", "right_closed"], default="right_closed")
    parser.add_argument("--cadence", default="1h", help="Grid step (default 1h).")
    parser.add_argument("--threshold", type=float, action="append", default=None,
                        help="Binary threshold(s) on the index (repeatable; default 10 and 20).")
    parser.add_argument("--atol", type=float, default=1e-9, help="Absolute tolerance for raw-index agreement.")
    parser.add_argument("--max-examples", type=int, default=10, help="Disagreements to print per comparison.")
    parser.add_argument("--skip-catalog-corrections", action="store_true",
                        help="Do not apply scripts/catalog_corrections.py to the lab catalog.")
    parser.add_argument("--summary", default=None, help="Optional JSON summary output path.")
    return parser.parse_args()


# --- sources -----------------------------------------------------------------

def _github_json(url: str):
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "flare-indexer-validation"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def resolve_suryabench_catalog(explicit: str | None) -> tuple[Path, dict]:
    """Return (local path, provenance). Pins the raw URL to a commit SHA when the API answers."""
    if explicit is not None:
        path = _resolve_source(explicit, "suryabench_flare_catalog_2010-2024.csv")
        return path, {"source": explicit, "revision": "as given"}
    provenance = {"repository": SURYABENCH_REPO, "path": SURYABENCH_CATALOG_PATH}
    try:
        commits = _github_json(f"https://api.github.com/repos/{SURYABENCH_REPO}/commits"
                               f"?path={SURYABENCH_CATALOG_PATH}&per_page=1")
        sha = commits[0]["sha"]
        provenance.update(revision=sha, revision_date=commits[0]["commit"]["committer"]["date"])
        script = _github_json(f"https://api.github.com/repos/{SURYABENCH_REPO}/commits"
                              f"?path={SURYABENCH_SCRIPT_PATH}&per_page=1")
        provenance["flare_index_creator_revision"] = script[0]["sha"]
    except Exception as exc:  # rate limit or offline: fall back to the branch head, say so
        sha = "main"
        provenance.update(revision="main (unpinned: GitHub API unavailable)", api_error=str(exc))
    url = f"https://raw.githubusercontent.com/{SURYABENCH_REPO}/{sha}/{SURYABENCH_CATALOG_PATH}"
    provenance["source"] = url
    path = _resolve_source(url, f"suryabench_flare_catalog_2010-2024_{sha[:12]}.csv")
    return path, provenance


def resolve_published_labels(source: str) -> tuple[Path, dict]:
    provenance = {"source": source, "license": "CC BY 4.0", "dataset": HF_DATASET}
    if source == DEFAULT_LABELS_URL:
        try:
            info = _github_json(f"https://huggingface.co/api/datasets/{HF_DATASET}")
            provenance.update(revision=info.get("sha"), last_modified=info.get("lastModified"))
        except Exception as exc:
            provenance.update(revision="main (unpinned: Hugging Face API unavailable)", api_error=str(exc))
    path = _resolve_source(source, "suryabench_data.csv")
    return path, provenance


# --- catalogs ------------------------------------------------------------------

def prepare_lab_catalog(source: str, skip_corrections: bool) -> tuple[Path, dict]:
    raw_path = _resolve_source(source, "goes_flares_integrated.csv")
    report = {"source": source, "sha256": _sha256(raw_path)}
    if skip_corrections:
        corrected = raw_path
        report["catalog_corrections"] = "skipped"
        report["modifications"] = "none"
    else:
        corrected, corrections = apply_catalog_corrections(raw_path, CACHE_DIR / "goes_flares_integrated_corrected.csv")
        print_corrections(corrections)
        report["catalog_corrections"] = corrections
        report["modifications"] = ("7 documented peak_time corrections (scripts/catalog_corrections.py), applied to a "
                                   "copy; start times unchanged, so start-time matching is unaffected")
    adapted_path = CACHE_DIR / "cumulative_lab_catalog_adapted.csv"
    adapted = adapt_goes_catalog(corrected, adapted_path)
    report.update(_catalog_stats(adapted))
    return adapted_path, report


def prepare_suryabench_catalog(path: Path) -> tuple[Path, dict]:
    df = pd.read_csv(path)
    report = {"sha256": _sha256(path), "rows": len(df)}
    missing = set(SURYABENCH_RENAME) - set(df.columns)
    if missing:
        raise SystemExit(f"SuryaBench catalog lacks expected columns: {sorted(missing)}")
    df = df.rename(columns=SURYABENCH_RENAME)
    for column in ("start_time", "peak_time", "end_time"):
        df[column] = pd.to_datetime(df[column], errors="raise")
    # Classes the package rejects (e.g. "A0.0": zero multiplier). SuryaBench's own
    # regex accepts them; an A/B value contributes None (nothing) to its sum, so
    # dropping such a row cannot change the index. Every dropped row is reported.
    invalid = []
    for position, value in df["goes_class"].items():
        try:
            parse_goes_class(value)
        except InvalidGOESClassError as exc:
            invalid.append({"csv_line": int(position) + 2, "start_time": str(df.at[position, "start_time"]),
                            "goes_class": str(value), "error": str(exc)})
    report["invalid_class_rows_dropped"] = invalid
    if invalid:
        df = df.drop(index=[row["csv_line"] - 2 for row in invalid])
    # Same upstream HEK error as the lab file: peaks/ends before the start. Blank
    # the offending time (never the start) so the matcher accepts the file.
    bad_end = df["end_time"] < df["start_time"]
    peak_before_start = df["peak_time"] < df["start_time"]
    peak_after_end = df["peak_time"] > df["end_time"]
    bad_peak = peak_before_start | peak_after_end
    report["peak_out_of_order_blanked"] = int(bad_peak.sum())
    report["peak_before_start_blanked"] = int(peak_before_start.sum())
    report["peak_after_end_blanked"] = int(peak_after_end.sum())
    report["end_before_start_blanked"] = int(bad_end.sum())
    blanked = df.loc[bad_peak | bad_end, ["start_time", "peak_time", "end_time", "goes_class"]].astype(str)
    blanked["reason"] = np.where(peak_before_start[bad_peak | bad_end], "peak before start",
                                 np.where(peak_after_end[bad_peak | bad_end], "peak after end", "end before start"))
    report["blanked_rows"] = blanked.to_dict("records")
    df.loc[bad_peak, "peak_time"] = pd.NaT
    df.loc[bad_end, "end_time"] = pd.NaT
    report["modifications"] = (
        "benchmark-only: columns renamed to the lab names; peak times outside [start, end] blanked "
        f"({int(bad_peak.sum())} rows: {int(peak_before_start.sum())} before start, {int(peak_after_end.sum())} "
        f"after end); end times before start blanked ({int(bad_end.sum())} rows); rows with a class the package "
        f"rejects dropped ({len(invalid)}); ar_noaanum passed through adapt_goes_catalog as text; start times and "
        "classes of kept rows unchanged")
    renamed_path = CACHE_DIR / "suryabench_catalog_lab_columns.csv"
    df.to_csv(renamed_path, index=False)
    adapted_path = CACHE_DIR / "cumulative_suryabench_catalog_adapted.csv"
    adapted = adapt_goes_catalog(renamed_path, adapted_path)
    report.update(_catalog_stats(adapted))
    return adapted_path, report


def _catalog_stats(adapted: pd.DataFrame) -> dict:
    letters = adapted["class"].str[0].str.upper()
    return {
        "flares": len(adapted),
        "first_start": str(adapted["start_time"].min()),
        "last_start": str(adapted["start_time"].max()),
        "classes": {k: int(v) for k, v in letters.value_counts().sort_index().items()},
    }


# --- grids and computations ----------------------------------------------------------

def hourly_grid(first: pd.Timestamp, last: pd.Timestamp, cadence: str) -> pd.DatetimeIndex:
    start = first.floor("D")
    return pd.date_range(start, last.ceil("D"), freq=cadence, inclusive="left")


def package_index(index_path: Path, catalog_path: Path, window: str, interval_mode: str) -> pd.Series:
    pipeline = DatasetBuildingPipeline(CumulativeClassIndexReducer(), RegressionLabeler())
    builder = DatasetBuilder(prediction_window=window, strategy=pipeline,
                             event_time="start", interval_mode=interval_mode)
    result = builder.build(index_path, catalog_path)
    return pd.Series(result["label"].to_numpy(dtype=float), index=pd.DatetimeIndex(result["timestamp"]))


def reference_index(catalog_path: Path, grid: pd.DatetimeIndex, window: pd.Timedelta, interval_mode: str) -> pd.Series:
    """Literal SuryaBench rolling_window(): C=n, M=10n, X=100n, A/B -> None, window (t, t+w] by start time."""
    df = pd.read_csv(catalog_path, usecols=["start_time", "class"])
    df["start_time"] = pd.to_datetime(df["start_time"])
    df = df.sort_values("start_time", kind="stable").reset_index(drop=True)
    classes = df["class"].str.strip().str.upper()
    if not classes.str.match(r"^[ABCMX]\d+(\.\d+)?$").all():
        bad = classes[~classes.str.match(r"^[ABCMX]\d+(\.\d+)?$")]
        raise SystemExit(f"reference: classes without a number cannot be parsed like SuryaBench: {bad.unique()[:5]}")
    numeric = classes.str[1:].astype(float)
    sub_cls = np.select(
        [classes.str.startswith("C"), classes.str.startswith("M"), classes.str.startswith("X")],
        [numeric, 10 * numeric, 100 * numeric], default=None,
    )
    sub_series = pd.Series(sub_cls, dtype=object)
    starts = df["start_time"].to_numpy(dtype="datetime64[ns]")
    grid_np = grid.to_numpy(dtype="datetime64[ns]")
    window_np = np.timedelta64(window.to_timedelta64())
    if interval_mode == "right_closed":      # t < start <= t + w   (SuryaBench)
        lo = np.searchsorted(starts, grid_np, side="right")
        hi = np.searchsorted(starts, grid_np + window_np, side="right")
    else:                                    # t <= start < t + w
        lo = np.searchsorted(starts, grid_np, side="left")
        hi = np.searchsorted(starts, grid_np + window_np, side="left")
    values = np.empty(len(grid), dtype=float)
    for i, (a, b) in enumerate(zip(lo, hi)):
        values[i] = sub_series.iloc[a:b].sum() if b > a else 0.0   # same call as the reference script
    return pd.Series(values, index=grid)


def _aligned_finite(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series, dict]:
    """Align two series on identical, unique indexes and drop non-finite pairs explicitly.

    Comparing NaN silently yields False (counted as "different" or as label 0), so
    non-finite values are excluded and counted instead of being folded into totals.
    """
    if not a.index.equals(b.index):
        raise SystemExit("internal error: series to compare are not aligned on the same timestamps")
    if not a.index.is_unique:
        raise SystemExit("internal error: duplicate timestamps in a comparison series")
    finite = np.isfinite(a.to_numpy(dtype=float)) & np.isfinite(b.to_numpy(dtype=float))
    excluded = {
        "nonfinite_excluded": int((~finite).sum()),
        "nonfinite_a": int((~np.isfinite(a.to_numpy(dtype=float))).sum()),
        "nonfinite_b": int((~np.isfinite(b.to_numpy(dtype=float))).sum()),
    }
    return a[finite], b[finite], excluded


def compare_raw(a: pd.Series, b: pd.Series, atol: float) -> dict:
    total = int(len(a))
    a, b, excluded = _aligned_finite(a, b)
    diff = (a - b).abs()
    return {
        "rows": total,
        "rows_compared": int(len(a)),
        **excluded,
        "exactly_equal": int((a.to_numpy() == b.to_numpy()).sum()),
        "within_atol": int((diff <= atol).sum()),
        "differing_beyond_atol": int((diff > atol).sum()),
        "max_abs_difference": float(diff.max()) if len(diff) else 0.0,
        "atol": atol,
    }


def compare_labels(a: pd.Series, b: pd.Series, threshold: float) -> dict:
    total = int(len(a))
    a, b, excluded = _aligned_finite(a, b)
    la = (a >= threshold).astype(int)
    lb = (b >= threshold).astype(int)
    return {
        "threshold": threshold,
        "rows": total,
        "rows_compared": int(len(la)),
        **excluded,
        "agree": int((la == lb).sum()),
        "disagree": int((la != lb).sum()),
        "a_positive": int(la.sum()),
        "b_positive": int(lb.sum()),
        "a1_b0": int(((la == 1) & (lb == 0)).sum()),
        "a0_b1": int(((la == 0) & (lb == 1)).sum()),
    }


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.4f} %" if whole else "n/a"


def describe_disagreements(package: pd.Series, other: pd.Series, matcher: EventMatcher, window: str,
                           interval_mode: str, atol: float, limit: int, other_name: str) -> list[dict]:
    diff = (package - other).abs()
    bad = diff[diff > atol].sort_index()
    examples = []
    for ts in bad.index[:limit]:
        events = matcher.query(ts, window, event_time="start", interval_mode=interval_mode)
        examples.append({
            "timestamp": str(ts),
            "package_index": float(package[ts]),
            f"{other_name}_index": float(other[ts]),
            "package_flares": [f"{e.start_time} {e.goes_class}" for e in events],
        })
    return examples


def threshold_flip_examples(package: pd.Series, reference: pd.Series, threshold: float, limit: int) -> list[dict]:
    """Rows where the decimal package sum and the float reference sum fall on different sides of a threshold."""
    mask = (package >= threshold).astype(int) != (reference >= threshold).astype(int)
    rows = []
    for ts in package.index[mask.to_numpy()][:limit]:
        rows.append({"timestamp": str(ts), "package_index": repr(float(package[ts])),
                     "reference_index": repr(float(reference[ts]))})
    return rows


def by_year(package: pd.Series, other: pd.Series, atol: float, threshold: float) -> list[dict]:
    rows = []
    frame = pd.DataFrame({"p": package, "o": other})
    for year, part in frame.groupby(frame.index.year):
        rows.append({
            "year": int(year), "rows": int(len(part)),
            "raw_beyond_atol": int(((part["p"] - part["o"]).abs() > atol).sum()),
            "label_disagree": int(((part["p"] >= threshold).astype(int) != (part["o"] >= threshold).astype(int)).sum()),
        })
    return rows


# --- main ---------------------------------------------------------------------------

def main():
    args = parse_args()
    t0 = time.monotonic()
    thresholds = args.threshold or [10.0, 20.0]
    window_td = _parse_prediction_window(args.prediction_window)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Window: event_time=start, interval_mode={args.interval_mode}, prediction_window={args.prediction_window} "
          f"({window_td}), cadence={args.cadence}, thresholds={thresholds}, atol={args.atol}")
    print("Reference convention of the lab notebooks and SuryaBench: t < start <= t + 23:59:59. "
          "The paper's closed [t, t+24h] is not expressible with the two interval modes.\n")

    summary = {
        "settings": {
            "event_time": "start", "interval_mode": args.interval_mode,
            "prediction_window": args.prediction_window, "prediction_window_seconds": window_td.total_seconds(),
            "cadence": args.cadence, "thresholds": thresholds, "atol": args.atol,
            "paper_interval_note": "[t, t+24h] (closed) is not expressible; right_closed gives (t, t+w].",
        },
        "catalogs": {}, "reference_comparison": {}, "published_comparison": {},
    }

    published = None
    if not args.no_published:
        labels_path, labels_prov = resolve_published_labels(args.suryabench_labels)
        published_df = pd.read_csv(labels_path)
        published_df["timestamp"] = pd.to_datetime(published_df["timestamp"], errors="raise")
        published_df = published_df.sort_values("timestamp").reset_index(drop=True)
        dup = int(published_df["timestamp"].duplicated().sum())
        if dup:
            # A duplicated timestamp would be expanded by .loc and inflate every count.
            raise SystemExit(f"published labels have {dup} duplicated timestamps; decide how to treat them "
                             f"before comparing (first: {published_df.loc[published_df['timestamp'].duplicated(), 'timestamp'].head(3).tolist()})")
        raw_index = pd.to_numeric(published_df["cumulative_index"], errors="coerce").to_numpy(dtype=float)
        nonfinite_published = int((~np.isfinite(raw_index)).sum())
        label_values = published_df["label_cum"]
        bad_labels = int((~label_values.isin([0, 1])).sum())
        if bad_labels:
            raise SystemExit(f"published label_cum has {bad_labels} values outside {{0, 1}}")
        published = pd.Series(raw_index, index=pd.DatetimeIndex(published_df["timestamp"]))
        published_label = pd.Series(label_values.to_numpy(dtype=int), index=published.index)
        summary["published_labels"] = {
            **labels_prov, "sha256": _sha256(labels_path), "rows": len(published_df),
            "duplicate_timestamps": dup, "nonfinite_cumulative_index": nonfinite_published,
            "first_timestamp": str(published.index.min()), "last_timestamp": str(published.index.max()),
            "label_cum_positive": int(published_label.sum()),
            "label_cum_consistent_with_index_ge_10": int(((published >= 10).astype(int) == published_label).sum()),
            "modifications": "none; file used as downloaded (max_goes_class and label_max are not compared)",
        }
        print(f"Published SuryaBench labels: {len(published_df)} rows, {published.index.min()} -> {published.index.max()}, "
              f"label_cum=1: {int(published_label.sum())}, duplicates: {dup}, non-finite indices: {nonfinite_published}, "
              f"label_cum == (index >= 10) on {summary['published_labels']['label_cum_consistent_with_index_ge_10']} rows\n")

    catalogs = []
    if args.catalog in ("lab", "both"):
        catalogs.append(("lab", lambda: prepare_lab_catalog(args.event_catalog, args.skip_catalog_corrections)))
    if args.catalog in ("suryabench", "both"):
        def _sb():
            path, prov = resolve_suryabench_catalog(args.suryabench_catalog)
            adapted, report = prepare_suryabench_catalog(path)
            return adapted, {**prov, **report}
        catalogs.append(("suryabench", _sb))

    for name, prepare in catalogs:
        print(f"=== Catalog: {name} ===")
        catalog_path, report = prepare()
        summary["catalogs"][name] = report
        print(f"  {report['flares']} flares, {report['first_start']} -> {report['last_start']}, classes {report['classes']}")
        if name == "suryabench":
            print(f"  benchmark-only modifications: columns renamed; peak times blanked {report['peak_out_of_order_blanked']} "
                  f"({report['peak_before_start_blanked']} before start, {report['peak_after_end_blanked']} after end); "
                  f"end before start blanked {report['end_before_start_blanked']}; start times untouched; "
                  f"invalid-class rows dropped: {len(report['invalid_class_rows_dropped'])} "
                  f"{[r['goes_class'] + ' at ' + r['start_time'] for r in report['invalid_class_rows_dropped']]}")
        report["coverage"] = ("unknown: first_start/last_start are the bounds of the recorded events, "
                              "not evidence of observational coverage")

        first = pd.Timestamp(report["first_start"])
        last = pd.Timestamp(report["last_start"])
        if published is not None:
            first, last = min(first, published.index.min()), max(last, published.index.max())
        grid = hourly_grid(first, last, args.cadence)
        if not grid.is_unique:
            raise SystemExit("internal error: grid has duplicate timestamps")
        index_path = CACHE_DIR / f"cumulative_grid_{name}.csv"
        pd.DataFrame({"timestamp": grid}).to_csv(index_path, index=False)
        print(f"  grid: {len(grid)} timestamps, {grid[0]} -> {grid[-1]}")

        t1 = time.monotonic()
        pkg = package_index(index_path, catalog_path, args.prediction_window, args.interval_mode)
        if not pkg.index.equals(grid):
            raise SystemExit("internal error: package output is not aligned with the grid")
        if not np.isfinite(pkg.to_numpy()).all():
            raise SystemExit("internal error: package produced a non-finite index")
        print(f"  package: {len(pkg)} rows in {time.monotonic() - t1:.1f}s, index>0 on {int((pkg > 0).sum())}, "
              f"max {pkg.max():g}")
        t1 = time.monotonic()
        ref = reference_index(catalog_path, grid, window_td, args.interval_mode)
        print(f"  reference (SuryaBench re-implementation): {len(ref)} rows in {time.monotonic() - t1:.1f}s")

        raw = compare_raw(pkg, ref, args.atol)
        labels = [compare_labels(pkg, ref, t) for t in thresholds]
        # Threshold flips between the decimal package sum and the float reference sum.
        flips = {str(t): threshold_flip_examples(pkg, ref, t, args.max_examples) for t in thresholds}
        summary["reference_comparison"][name] = {
            "raw_index": raw, "labels": labels, "threshold_flip_examples_decimal_vs_float": flips,
            "grid_first": str(grid[0]), "grid_last": str(grid[-1]), "grid_rows": int(len(grid)),
        }
        print(f"  raw index vs reference: exactly equal {raw['exactly_equal']}/{raw['rows']}, "
              f"within atol {raw['within_atol']}/{raw['rows']}, max |diff| {raw['max_abs_difference']:.3g}")
        for lab in labels:
            print(f"  label (>= {lab['threshold']:g}) vs reference: agree {lab['agree']}/{lab['rows']} "
                  f"({_pct(lab['agree'], lab['rows'])}), package=1/ref=0 {lab['a1_b0']}, package=0/ref=1 {lab['a0_b1']}")
            for flip in flips[str(lab["threshold"])]:
                print(f"    flip at {flip['timestamp']}: package {flip['package_index']}, "
                      f"float reference {flip['reference_index']}")

        if published is not None:
            common = pkg.index.intersection(published.index)
            only_pkg = pkg.index.difference(published.index)
            only_pub = published.index.difference(pkg.index)
            # Comparison bounds: the first and last recorded event start times. They
            # bound the events the package can see; they say nothing about whether the
            # instrument was observing (coverage is unknown). A window that extends past
            # the last recorded event may still contain earlier events, so its value is
            # not necessarily 0; those windows are reported separately, with their
            # non-zero count, rather than treated as "no data".
            cat_first, cat_last = pd.Timestamp(report["first_start"]), pd.Timestamp(report["last_start"])
            within_mask = (common >= cat_first) & (common + window_td <= cat_last)
            within = common[within_mask]
            past = common[~within_mask]
            matcher = EventMatcher(catalog_path)
            comparison = {
                "matched_timestamps": int(len(common)),
                "matched_range": [str(common.min()), str(common.max())] if len(common) else None,
                "package_only_timestamps": int(len(only_pkg)),
                "package_only_range": [str(only_pkg.min()), str(only_pkg.max())] if len(only_pkg) else None,
                "published_only_timestamps": int(len(only_pub)),
                "published_only_range": [str(only_pub.min()), str(only_pub.max())] if len(only_pub) else None,
                "published_only_sample": [str(t) for t in only_pub[:10]],
                "catalog_event_bounds": [str(cat_first), str(cat_last)],
                "coverage": "unknown (event bounds are not observational coverage)",
                "within_bounds_definition": "first recorded start <= t and t + window <= last recorded start",
                "within_bounds_timestamps": int(len(within)),
                "past_bounds_timestamps": int(len(past)),
                "past_bounds_range": [str(past.min()), str(past.max())] if len(past) else None,
                "past_bounds_package_nonzero": int((pkg.loc[past] > 0).sum()) if len(past) else 0,
                "past_bounds_published_nonzero": int((published.loc[past] > 0).sum()) if len(past) else 0,
            }
            for scope, idx in (("within_bounds", within), ("all_matched", common)):
                p, q = pkg.loc[idx], published.loc[idx]
                pkg_label10 = (p >= 10).astype(int)
                pub_label = published_label.loc[idx]
                comparison[scope] = {
                    "rows": int(len(idx)),
                    "raw_index": compare_raw(p, q, args.atol),
                    "labels_recomputed_from_index": [compare_labels(p, q, t) for t in thresholds],
                    "label_cum_published_vs_package_ge_10": {
                        "rows": int(len(idx)), "agree": int((pkg_label10 == pub_label).sum()),
                        "disagree": int((pkg_label10 != pub_label).sum()),
                        "package1_published0": int(((pkg_label10 == 1) & (pub_label == 0)).sum()),
                        "package0_published1": int(((pkg_label10 == 0) & (pub_label == 1)).sum()),
                    },
                    "by_year": by_year(p, q, args.atol, 10.0),
                }
            comparison["disagreement_examples_within_bounds"] = describe_disagreements(
                pkg.loc[within], published.loc[within], matcher, args.prediction_window, args.interval_mode,
                args.atol, args.max_examples, "published")
            summary["published_comparison"][name] = comparison

            print(f"  --- published SuryaBench labels ({name} catalog as input) ---")
            print(f"  matched timestamps {len(common)} ({common.min()} -> {common.max()}); "
                  f"package-only {len(only_pkg)}, published-only {len(only_pub)}")
            print(f"  recorded-event bounds {cat_first} -> {cat_last} (coverage unknown): windows within bounds "
                  f"{len(within)}, windows extending past the last recorded event {len(past)}"
                  + (f" ({past.min()} -> {past.max()}; package non-zero on {comparison['past_bounds_package_nonzero']}, "
                     f"published non-zero on {comparison['past_bounds_published_nonzero']})" if len(past) else ""))
            for scope in ("within_bounds", "all_matched"):
                c = comparison[scope]
                r = c["raw_index"]
                print(f"  [{scope}: {c['rows']} rows, {r['rows_compared']} compared, {r['nonfinite_excluded']} non-finite excluded]")
                print(f"    raw cumulative_index: exactly equal {r['exactly_equal']}/{r['rows_compared']} "
                      f"({_pct(r['exactly_equal'], r['rows_compared'])}), within atol {r['within_atol']}, "
                      f"beyond atol {r['differing_beyond_atol']}, max |diff| {r['max_abs_difference']:.4g}")
                for lab in c["labels_recomputed_from_index"]:
                    print(f"    label (>= {lab['threshold']:g}) from both indices: agree {lab['agree']}/{lab['rows']} "
                          f"({_pct(lab['agree'], lab['rows'])}), package=1/pub=0 {lab['a1_b0']}, "
                          f"package=0/pub=1 {lab['a0_b1']}")
                lc = c["label_cum_published_vs_package_ge_10"]
                print(f"    published label_cum vs package (>= 10): agree {lc['agree']}/{lc['rows']} "
                      f"({_pct(lc['agree'], lc['rows'])}), package=1/pub=0 {lc['package1_published0']}, "
                      f"package=0/pub=1 {lc['package0_published1']}")
                years = [y for y in c["by_year"] if y["raw_beyond_atol"]]
                if years:
                    print("    raw differences by year: " + ", ".join(f"{y['year']}: {y['raw_beyond_atol']}" for y in years))
            for ex in comparison["disagreement_examples_within_bounds"]:
                print(f"    {ex['timestamp']}: package {ex['package_index']:g}, published {ex['published_index']:g}, "
                      f"flares {ex['package_flares']}")
        print()

    summary["runtime_seconds"] = round(time.monotonic() - t0, 1)
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Summary written to {args.summary}")
    print(f"Done in {summary['runtime_seconds']}s")


if __name__ == "__main__":
    main()
