import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Union

import warnings

import numpy as np
import pandas as pd

from .goes_class import FluxConverter, validate_goes_class_column
from .events import (
    _none_for_missing, _normalize_active_region, _parse_full_timestamp_column, _rename_columns, _validate_time_order,
    CATALOG_COLUMNS, XRAY_FLUX_COLUMNS,
)

# File names like "HMI.m2010.12.21_21.00.00.jpg" (the basename only); the
# date itself is checked when the timestamp is built.
_FILENAME_PATTERN = re.compile(
    r"^HMI\.m(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<day>\d{2})_"
    r"(?P<hour>\d{2})\.(?P<minute>\d{2})\.(?P<second>\d{2})\.jpg$"
)

GOES_CATALOG_REQUIRED_COLUMNS = {
    "start_time",
    "peak_time",
    "end_time",
    "goes_class",
    "noaa_active_region",
}
GOES_CATALOG_OPTIONAL_COLUMNS = ("fl_lon", "fl_lat")
# Source column name -> catalog column name for the relative X-ray flux increase.
GOES_CATALOG_RXFI_COLUMN = ("relative_X-ray_flux_increase", "rxfi")

VALID_DUPLICATE_POLICIES = {"error", "first", "last"}

# NOAA region numbers passed 9999 on this date (region 10000). Many flare
# lists kept four digits, so a later region 13324 is written 3324.
NOAA_FIVE_DIGIT_START = pd.Timestamp("2002-06-14")

# How many excluded / duplicate rows a report lists (its counts are always exact).
_MAX_REPORT_SAMPLES = 200


@dataclass
class ImageIndexReport:
    """
    Validation report for build_image_index_from_filenames().

    total_input_rows : how many non-blank lines/paths were given.
    valid_image_rows_before_dedup : how many parsed as valid HMI image
        filenames, before duplicate-timestamp handling.
    excluded_row_count : total rows that didn't parse as a valid HMI image
        filename (malformed shape, or shape-valid but an invalid calendar
        date/time). Always accurate, even if `excluded_rows` is truncated.
    excluded_rows : up to the first `_MAX_REPORT_SAMPLES` excluded raw path
        strings, for inspection.
    duplicate_timestamp_count : number of distinct timestamps that appeared
        more than once among the valid rows.
    duplicate_rows : up to the first `_MAX_REPORT_SAMPLES` image_path values
        involved in a duplicate timestamp, for inspection.
    final_row_count : row count of the returned DataFrame, after applying
        duplicate_policy.
    duplicate_policy : the duplicate_policy of the call. With "error" a
        report is only returned when there were no duplicates.
    """

    total_input_rows: int
    valid_image_rows_before_dedup: int
    excluded_row_count: int
    duplicate_policy: str
    excluded_rows: list = field(default_factory=list)
    duplicate_timestamp_count: int = 0
    duplicate_rows: list = field(default_factory=list)
    final_row_count: int = 0


def _read_paths_file(path: Path) -> list[str]:
    # utf-8-sig transparently strips a leading BOM if present (some sources
    # export these path lists with one) and behaves like plain utf-8 otherwise.
    with open(path, encoding="utf-8-sig") as f:
        return [line.strip() for line in f if line.strip()]


def build_image_index_from_filenames(
    paths: Union[str, Path, Iterable[str]],
    output_path: Union[str, Path, None] = None,
    strict: bool = True,
    duplicate_policy: str = "error",
    return_report: bool = False,
):
    """
    Build an image index (the timestamp CSV the dataset classes read) from HMI image file paths.

    `paths` is either an iterable of file paths, or a path to a headerless
    single-column text/CSV file listing one path per line (a UTF-8 BOM at
    the start of such a file, if present, is stripped automatically). Each
    filename is expected to look like
    "HMI.m{YYYY}.{MM}.{DD}_{HH}.{MM}.{SS}.jpg"; directory depth doesn't
    matter, only the basename is parsed.

    strict controls rows that are not valid HMI image file names:
    - True (default): raise ValueError naming the first one.
    - False: leave them out; return_report=True lists them.

    duplicate_policy controls what happens when multiple rows parse to the
    exact same timestamp:
    - "error" (default): raise ValueError naming the duplicate timestamps.
    - "first": keep the first-encountered row for each duplicated timestamp.
    - "last": keep the last-encountered row for each duplicated timestamp.

    Returns a DataFrame with a `timestamp` column (pandas datetime, sorted
    ascending) and an
    `image_path` column preserving the original path. If return_report is
    True, instead returns (DataFrame, ImageIndexReport).
    """
    for name, value in (("strict", strict), ("return_report", return_report)):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be True or False, got {value!r}")
    if duplicate_policy not in VALID_DUPLICATE_POLICIES:
        raise ValueError(
            f"duplicate_policy must be one of {sorted(VALID_DUPLICATE_POLICIES)}, "
            f"got {duplicate_policy!r}"
        )

    if isinstance(paths, (str, Path)):
        paths = _read_paths_file(Path(paths))
    else:
        # Same rules as for a file: surrounding spaces removed, blank items skipped.
        paths = [str(path).strip() for path in paths if str(path).strip()]

    total_input_rows = len(paths)
    records = []
    excluded_rows: list[str] = []

    for path in paths:
        filename = Path(path).name
        match = _FILENAME_PATTERN.match(filename)
        timestamp = None
        if match:
            try:
                timestamp = pd.Timestamp(
                    year=int(match["year"]),
                    month=int(match["month"]),
                    day=int(match["day"]),
                    hour=int(match["hour"]),
                    minute=int(match["minute"]),
                    second=int(match["second"]),
                )
            except ValueError as exc:
                if strict:
                    raise ValueError(
                        f"Filename {filename!r} (from path {str(path)!r}) has the right shape "
                        f"but an invalid date/time: {exc}"
                    ) from None
        if timestamp is None:
            if match is None and strict:
                raise ValueError(
                    f"Filename {filename!r} (from path {str(path)!r}) does not match "
                    "the expected pattern 'HMI.m{YYYY}.{MM}.{DD}_{HH}.{MM}.{SS}.jpg'"
                )
            excluded_rows.append(str(path))
            continue
        records.append({"timestamp": timestamp, "image_path": str(path)})

    valid_image_rows_before_dedup = len(records)

    df = pd.DataFrame(records, columns=["timestamp", "image_path"])
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    dup_mask = df["timestamp"].duplicated(keep=False)
    duplicate_timestamp_count = int(df.loc[dup_mask, "timestamp"].nunique())
    duplicate_rows_sample = df.loc[dup_mask, "image_path"].tolist()[:_MAX_REPORT_SAMPLES]

    if dup_mask.any():
        if duplicate_policy == "error":
            dupe_timestamps = sorted(df.loc[dup_mask, "timestamp"].unique())
            raise ValueError(
                f"Image paths contain {len(dupe_timestamps)} duplicate timestamp(s): "
                f"{', '.join(str(t) for t in dupe_timestamps[:10])}"
                + (", ..." if len(dupe_timestamps) > 10 else "")
            )
        keep = "first" if duplicate_policy == "first" else "last"
        df = df.drop_duplicates(subset="timestamp", keep=keep).reset_index(drop=True)

    if output_path is not None:
        df.to_csv(output_path, index=False)

    if not return_report:
        return df

    report = ImageIndexReport(
        total_input_rows=total_input_rows,
        valid_image_rows_before_dedup=valid_image_rows_before_dedup,
        excluded_row_count=len(excluded_rows),
        excluded_rows=excluded_rows[:_MAX_REPORT_SAMPLES],
        duplicate_timestamp_count=duplicate_timestamp_count,
        duplicate_rows=duplicate_rows_sample,
        final_row_count=len(df),
        duplicate_policy=duplicate_policy,
    )
    return df, report


def adapt_goes_catalog(
    input_path: Union[str, Path],
    output_path: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """
    Adapt an integrated GOES flare catalog into
    the full-timestamp `start_time, peak_time, end_time, class,
    active_region[, fl_lon, fl_lat, rxfi]` schema EventMatcher accepts directly.

    Source columns used: `start_time`, `peak_time`, `end_time` (full
    datetime strings), `goes_class`, `noaa_active_region` (nullable float),
    and, if present, `fl_lon`/`fl_lat` and `relative_X-ray_flux_increase`
    (written as `rxfi`). All other source columns are
    ignored. `noaa_active_region` is normalized to the same string-or-None
    convention as `events._normalize_active_region`.

    Each timestamp keeps its own complete date, so a flare that crosses
    midnight (start_time on one date, peak_time/end_time on the next) is
    written as it happened.

    Four-digit region numbers of flares that start on or after 2002-06-14
    get the leading 1 back (3324 -> 13324): NOAA's numbers passed 9999
    that day, and many flare lists kept only four digits. If a row then
    equals another row in start_time, peak_time, end_time, class and
    active_region, it is the same flare listed twice and is dropped. A
    warning gives both counts.
    """
    input_path = Path(input_path)

    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError:
        raise ValueError(
            f"GOES catalog at {input_path} is empty: no header row / columns found."
        ) from None

    missing = GOES_CATALOG_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"GOES catalog at {input_path} is missing required column(s): "
            f"{', '.join(sorted(missing))}"
        )

    validate_goes_class_column(df["goes_class"], input_path, column="goes_class")

    # Same parsing rules as EventMatcher: numeric columns refused, UTC
    # offsets converted to UTC and dropped, errors name file and column.
    adapted = pd.DataFrame({
        "start_time": _parse_full_timestamp_column(df, "start_time", input_path, what="GOES catalog"),
        "peak_time": _parse_full_timestamp_column(df, "peak_time", input_path, what="GOES catalog"),
        "end_time": _parse_full_timestamp_column(df, "end_time", input_path, what="GOES catalog"),
        "class": df["goes_class"],
        "active_region": _none_for_missing(df["noaa_active_region"].map(_normalize_active_region)),
    })

    for optional_column in GOES_CATALOG_OPTIONAL_COLUMNS:
        if optional_column in df.columns:
            adapted[optional_column] = df[optional_column]
    source_rxfi, rxfi = GOES_CATALOG_RXFI_COLUMN
    if source_rxfi in df.columns:
        adapted[rxfi] = df[source_rxfi]

    adapted = _restore_five_digit_regions(adapted, input_path)

    if output_path is not None:
        adapted.to_csv(output_path, index=False)

    return adapted


def _restore_five_digit_regions(catalog: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Add the leading 1 to four-digit regions after 2002-06-14 and drop fixed rows that then equal another row."""
    def is_four_digit(region):
        return isinstance(region, str) and region.isdigit() and 1 <= int(region) <= 9999

    fixed = catalog["active_region"].map(is_four_digit) & (catalog["start_time"] >= NOAA_FIVE_DIGIT_START)
    if not fixed.any():
        return catalog
    catalog = catalog.copy()
    catalog.loc[fixed, "active_region"] = catalog.loc[fixed, "active_region"].map(lambda r: str(int(r) + 10000))

    # A fixed row that now equals an unfixed row (or an earlier fixed row)
    # is the same flare listed twice; the other copy is kept.
    keys = ["start_time", "peak_time", "end_time", "class", "active_region"]
    order = fixed.astype(int).sort_values(kind="stable").index
    copies = catalog.loc[order].duplicated(subset=keys, keep="first").reindex(catalog.index)
    dropped = copies & fixed
    warnings.warn(
        f"{int(fixed.sum())} flare(s) in {path} starting on or after 2002-06-14 had a four-digit "
        f"active_region and got the leading 1 back (e.g. 3324 -> 13324); {int(dropped.sum())} of them "
        "then equaled another row (same start, peak, end, class and region) and were dropped as copies.",
        UserWarning,
        stacklevel=3,
    )
    catalog = catalog[~dropped].reset_index(drop=True)
    catalog["active_region"] = _none_for_missing(catalog["active_region"])
    return catalog


# Flux values at or below this are treated as bad readings.
XRAY_FLUX_FLOOR = 1e-9
BACKGROUND_LOOKBACK = pd.Timedelta(hours=24)


def add_relative_flux(
    catalog_path: Union[str, Path],
    xray_flux_path: Union[str, Path],
    output_path: Union[str, Path, None] = None,
    verbose: bool = False,
    *,
    catalog_columns=None,
    xray_columns=None,
) -> pd.DataFrame:
    """
    Add each flare's background X-ray flux and relative X-ray flux
    increase (rxfi) to a flare catalog.

    catalog_path is a catalog in the package schema (start_time,
    peak_time, end_time, class, active_region, ...). xray_flux_path is a
    CSV of GOES 1-8 Angstrom X-ray flux with the columns time and flux
    (W/m^2, one value per minute).

    The background of a flare:

    1. flux values at or below 1e-9 W/m^2 are dropped;
    2. every minute from start_time to end_time of any catalog flare is
       dropped (a flare with a blank start_time or end_time is skipped here);
    3. of the minutes from start_time - 24 h to start_time (both
       included), only values below the flux at start_time are kept;
    4. the background is the mean of the kept values.

    rxfi = peak flux of the flare's GOES class / background. A flare with
    no flux value at its start_time, or no kept value, gets a blank
    background and rxfi, and a warning counts such flares.

    Returns the catalog with two added columns, background_flux and rxfi,
    in its original row order; writes it to output_path if given.
    verbose=True prints a short summary (flares, flux minutes, how many
    flares got an rxfi); the result is the same either way.

    catalog_columns and xray_columns let the files use their own column
    names: {your column name: package column name}, e.g.
    xray_columns={"time_tag": "time", "xrsb_flux": "flux"}. The returned
    catalog uses the package's names, so it can go straight to build().
    """
    if not isinstance(verbose, bool):
        raise ValueError(f"verbose must be True or False, got {verbose!r}")
    catalog_path = Path(catalog_path)
    xray_flux_path = Path(xray_flux_path)

    catalog = _read_csv(catalog_path, "Flare catalog")
    catalog = _rename_columns(catalog, catalog_columns, CATALOG_COLUMNS, catalog_path, "Flare catalog", "catalog_columns")
    missing = {"start_time", "end_time", "class"} - set(catalog.columns)
    if missing:
        raise ValueError(
            f"Flare catalog at {catalog_path} is missing required column(s): {', '.join(sorted(missing))}"
        )
    existing = [c for c in ("background_flux", "rxfi") if c in catalog.columns]
    if existing:
        raise ValueError(
            f"Flare catalog at {catalog_path} already has {', '.join(existing)}; "
            "remove the column(s) to compute them again."
        )
    validate_goes_class_column(catalog["class"], catalog_path, column="class")
    starts = _parse_full_timestamp_column(catalog, "start_time", catalog_path)
    ends = _parse_full_timestamp_column(catalog, "end_time", catalog_path)
    peaks = (_parse_full_timestamp_column(catalog, "peak_time", catalog_path) if "peak_time" in catalog.columns
             else pd.Series(pd.NaT, index=catalog.index))
    # The same start <= peak <= end check as build(), so a catalog it accepts loads there too.
    _validate_time_order(pd.DataFrame({"start_time": starts, "peak_time": peaks, "end_time": ends}), catalog_path)

    times, raw = _read_xray_flux(xray_flux_path, xray_columns)
    if verbose:
        print(f"Flare catalog: {len(catalog):,} flares ({catalog_path})")
        span = f", {pd.Timestamp(times[0]):%Y-%m-%d} to {pd.Timestamp(times[-1]):%Y-%m-%d}" if len(times) else ""
        print(f"X-ray flux: {len(times):,} minutes{span} ({xray_flux_path})")

    # Steps 1 and 2: the flux that may count as background.
    usable = np.where(raw > XRAY_FLUX_FLOOR, raw, np.nan)
    flaring = np.zeros(len(times) + 1, dtype=np.int64)
    both = starts.notna() & ends.notna()
    lo = np.searchsorted(times, starts[both].to_numpy(), side="left")
    hi = np.searchsorted(times, ends[both].to_numpy(), side="right")
    np.add.at(flaring, lo, 1)
    np.add.at(flaring, hi, -1)
    usable[np.cumsum(flaring)[:-1] > 0] = np.nan

    # Steps 3 and 4, one flare at a time.
    background = np.full(len(catalog), np.nan)
    for row, start in enumerate(starts):
        if pd.isna(start):
            continue
        at_start = np.searchsorted(times, np.datetime64(start), side="left")
        if at_start == len(times) or times[at_start] != np.datetime64(start) or np.isnan(raw[at_start]):
            continue
        first = np.searchsorted(times, np.datetime64(start - BACKGROUND_LOOKBACK), side="left")
        window = usable[first:at_start + 1]
        kept = window[window < raw[at_start]]
        if len(kept):
            background[row] = kept.mean()

    converter = FluxConverter()
    peak_flux = catalog["class"].map(converter.to_flux).to_numpy(dtype=float)
    result = catalog.copy()
    result["background_flux"] = background
    result["rxfi"] = peak_flux / background

    unknown = int(np.isnan(background).sum())
    if verbose:
        print(f"Result: rxfi computed for {len(catalog) - unknown:,} flares; {unknown:,} without "
              "(no flux at start or no usable minutes before it)")
    if unknown:
        warnings.warn(
            f"{unknown} of {len(catalog)} flare(s) in {catalog_path} have no rxfi: no X-ray flux value "
            "at their start_time, or no usable value in the 24 hours before it.",
            UserWarning,
            stacklevel=2,
        )

    if output_path is not None:
        result.to_csv(output_path, index=False)
    return result


def _read_csv(path: Path, what: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"{what} at {path} is empty: no header row / columns found.") from None


def _read_xray_flux(path: Path, columns=None) -> tuple[np.ndarray, np.ndarray]:
    """Times (sorted) and flux values of an X-ray flux CSV with columns time, flux."""
    df = _read_csv(path, "X-ray flux file")
    df = _rename_columns(df, columns, XRAY_FLUX_COLUMNS, path, "X-ray flux file", "xray_columns")
    missing = {"time", "flux"} - set(df.columns)
    if missing:
        raise ValueError(f"X-ray flux file at {path} is missing required column(s): {', '.join(sorted(missing))}")

    times = _parse_full_timestamp_column(df, "time", path, what="X-ray flux file")
    if times.isna().any():
        position = int(np.flatnonzero(times.isna().to_numpy())[0])
        raise ValueError(f"X-ray flux file at {path}, column 'time', data row {position + 1}: the time is blank.")
    if times.duplicated().any():
        position = int(np.flatnonzero(times.duplicated().to_numpy())[0])
        raise ValueError(
            f"X-ray flux file at {path}, column 'time', data row {position + 1}: "
            f"{times.iloc[position]} appears more than once."
        )

    flux = pd.to_numeric(df["flux"], errors="coerce")
    bad = df["flux"].notna() & flux.isna()
    if bad.any():
        position = int(np.flatnonzero(bad.to_numpy())[0])
        raise ValueError(
            f"X-ray flux file at {path}, column 'flux', data row {position + 1}: "
            f"expected a number or a blank cell, got {df['flux'].iloc[position]!r}."
        )

    order = np.argsort(times.to_numpy(), kind="stable")
    return times.to_numpy()[order], flux.to_numpy(dtype=float)[order]
