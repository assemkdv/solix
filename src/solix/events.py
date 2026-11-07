import datetime
import numbers
import re
from collections.abc import Mapping

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from .goes_class import validate_goes_class_column

# A bare number written as text ("24", "1.5", "2.4e1"): pandas would read
# it as nanoseconds, so such a duration string is rejected and a unit asked for.
_UNITLESS_NUMBER_TEXT = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")

# Region text that is an integer, possibly with a ".0" tail as pandas writes
# a float column ("13700", "13700.0", "013700", "+13700").
_INTEGER_REGION_TEXT = re.compile(r"^[+-]?\d+(?:\.0*)?$")

# A date-time text that ends in an explicit UTC offset: "Z", "+00:00",
# "+0500", "-07:00", or hours only after a time ("10:00:00+02"). Used to
# tell offset-aware cells from naive ones.
_OFFSET_SUFFIX = re.compile(
    r"(?:Z|[+-]\d{2}:?\d{2}|\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s?[+-]\d{2})$", re.IGNORECASE
)


@dataclass
class FlareEvent:
    """One flare from the catalog."""

    peak_time: pd.Timestamp
    goes_class: str                  # e.g. "M2.3"
    start_time: pd.Timestamp
    active_region: str | None        # NOAA region, or None if unknown
    end_time: pd.Timestamp | None = None
    fl_lon: float | None = None      # heliographic longitude, if the catalog has it
    fl_lat: float | None = None      # heliographic latitude, if the catalog has it
    rxfi: float | None = None        # relative X-ray flux increase, if the catalog has it


def _parse_prediction_window(value) -> pd.Timedelta:
    """
    Normalize a prediction window into a strictly positive pd.Timedelta.

    A plain number (int or float) is hours. Anything else is parsed as a
    pandas duration, e.g. "24h" or a pd.Timedelta.
    """
    if isinstance(value, bool):
        raise ValueError(f"prediction_window must be a number of hours or a duration, got {value!r}")
    _reject_unitless_duration_text(value, "prediction_window")
    try:
        if isinstance(value, numbers.Real) and not isinstance(value, np.timedelta64):
            window = pd.Timedelta(hours=float(value))
        else:
            window = pd.Timedelta(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(
            f"prediction_window must be a number of hours or a duration (e.g. 24, '24h', "
            f"'23h59min59s'), got {value!r}: {exc}"
        ) from None
    if pd.isna(window) or window <= pd.Timedelta(0):
        raise ValueError(f"prediction_window must be > 0, got {value!r}")
    return window


def _reject_unitless_duration_text(value, name: str) -> None:
    """
    Raise ValueError for a duration given as bare numeric text ("24",
    "1.5", "2.4e1"). pandas reads such text as nanoseconds, which would
    silently give an (almost) empty window; a unit must be written
    ("24h"). Numbers that are not text are left to the caller, which
    gives them their documented meaning (hours for prediction_window).
    """
    if isinstance(value, str) and _UNITLESS_NUMBER_TEXT.match(value.strip()):
        raise ValueError(
            f"{name} {value!r} has no unit, so pandas would read it as nanoseconds. "
            f"Write a duration with a unit, e.g. '{value.strip()}h', or pass a number of hours."
        )


def _normalize_active_region(value) -> str | None:
    """
    Normalize a region identifier to the text form used for matching.

    The same value can arrive as 13700 (int column), 13700.0 (float
    column, because pandas uses float64 for a column with blanks),
    "13700" or "13700.0" (object column, because some cell in the column
    is text), or " 13700 ". All of these become "13700"; the integer is
    taken from the text itself, never through a float, so no precision is
    lost. Non-numeric text ("AR13700", "13700.5") is kept as written,
    with surrounding whitespace removed. Missing values (NaN, None, blank
    text) become None.

    Text with commas lists several regions, e.g. a HARP patch that covers
    NOAA regions 11160, 11161 and 11162: "11160, 11161.0,11162" becomes
    "11160,11161,11162". Each part is normalized as above; blank parts
    and repeats are dropped, and the parts are sorted (numbers by value),
    so the same regions written in another order give the same text.
    """
    if value is None or isinstance(value, bool):
        return None if value is None else str(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if "," in text:
            parts = []
            for part in text.split(","):
                normalized = _normalize_active_region(part)
                if normalized is not None and normalized not in parts:
                    parts.append(normalized)
            parts.sort(key=lambda part: (0, int(part), "") if part.lstrip("-").isdigit() else (1, 0, part))
            return ",".join(parts) if parts else None
        if _INTEGER_REGION_TEXT.match(text):
            return str(int(text.split(".")[0]))
        return text
    if pd.isna(value):
        return None
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        as_float = float(value)
        if as_float.is_integer():
            return str(int(as_float))
        return str(value)
    return _normalize_active_region(str(value))


def _shares_region(normalized: str | None, wanted: set) -> bool:
    """True if a normalized region (one region or a comma-separated list) has a region in wanted."""
    return normalized is not None and not wanted.isdisjoint(normalized.split(","))


def _none_for_missing(values: pd.Series) -> pd.Series:
    """Missing values as None (pandas 3 would otherwise store NaN in a text column)."""
    return values.astype(object).where(values.notna(), None)


def _to_naive_utc(timestamp: pd.Timestamp) -> pd.Timestamp:
    """
    Apply the package's time-zone policy to one timestamp: a value with an
    explicit UTC offset is converted to UTC and the offset dropped; a
    naive value is returned unchanged and is assumed to be on the
    catalog's time scale (UTC for GOES). This is a calendar conversion
    only; it never changes the time scale (no TAI-to-UTC correction).
    """
    if timestamp.tzinfo is not None:
        return timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _to_naive_utc_series(parsed: pd.Series) -> pd.Series:
    """Column version of _to_naive_utc (see there for the policy)."""
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        raise ValueError(
            "time column did not parse to a single datetime dtype "
            f"(got {parsed.dtype}); every cell must use the same convention."
        )
    if parsed.dt.tz is not None:
        return parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    return parsed


def _parse_time_text_column(series: pd.Series, column: str, what: str, path) -> pd.Series:
    """
    Parse one column of date-time text into naive timestamps under the
    package's time-zone policy (see _to_naive_utc):

    - every non-blank cell naive  -> parsed as written;
    - every non-blank cell with an explicit UTC offset ("Z", "+05:00";
      different offsets may be mixed, e.g. across a daylight-saving
      change) -> each instant converted to UTC, offset dropped;
    - naive and offset cells mixed in one column -> ValueError naming the
      first row of each kind, because the naive cells' zone is unknown;
    - a numeric column -> ValueError (see _reject_numeric_time_column);
    - blank cells -> NaT in every case.

    `what` and `path` name the file in messages ("Flare catalog at ...").
    """
    _reject_numeric_time_column(series, column, what, path)
    present = series.notna().to_numpy()
    positions = np.flatnonzero(present)
    text = series.iloc[positions].astype(str).str.strip()
    has_offset = text.str.contains(_OFFSET_SUFFIX).to_numpy() if len(text) else np.zeros(0, dtype=bool)
    if has_offset.any() and not has_offset.all():
        first_naive = int(positions[~has_offset][0]) + 1
        first_offset = int(positions[has_offset][0]) + 1
        raise ValueError(
            f"{what} at {path}, column '{column}' mixes times without a UTC offset "
            f"(first at data row {first_naive}) and times with one (first at data row "
            f"{first_offset}). Use one convention per column: all naive (taken as the "
            "catalog's time scale) or all with an explicit offset (converted to UTC)."
        )
    try:
        parsed = pd.to_datetime(series, utc=True) if has_offset.any() else pd.to_datetime(series)
        return _to_naive_utc_series(parsed)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{what} at {path} has an unparseable value in column '{column}': {exc}"
        ) from None


def _reject_numeric_time_column(series: pd.Series, column: str, what: str, path) -> None:
    """
    Raise ValueError when a time column holds numbers rather than
    date-time text. pandas would read 20240101100000 or 1700000000 as
    nanoseconds since 1970, placing every row in 1970 and matching
    nothing, so the value is ambiguous and is refused. An all-blank
    column (float64 of NaN) is not numeric data and passes.
    """
    if pd.api.types.is_bool_dtype(series) or (
        pd.api.types.is_numeric_dtype(series) and series.notna().any()
    ):
        position = int(np.flatnonzero(series.notna().to_numpy())[0])
        value = series.iloc[position]
        value = value.item() if hasattr(value, "item") else value
        raise ValueError(
            f"{what} at {path}, column '{column}', data row {position + 1}: the value "
            f"{value!r} is a number, not date-time text. Write full "
            f"date-times such as 2024-01-01 10:00:00; numbers are ambiguous "
            f"(yyyymmddHHMMSS, epoch seconds, ...) and are not interpreted."
        )


def _parse_full_timestamp_column(df: pd.DataFrame, column: str, path, what: str = "Flare catalog") -> pd.Series:
    """
    Parse a column of full date-time strings (e.g. "2010-05-01 01:39:00").
    A blank cell becomes NaT, so that flare can never match through this
    column. Errors name the file, column and value.
    """
    return _parse_time_text_column(df[column], column, what, path)


# Column names a user may map their own columns to, per input file.
INDEX_COLUMNS = ("timestamp", "active_region")
CATALOG_COLUMNS = ("start_time", "peak_time", "end_time", "class", "active_region", "rxfi")
XRAY_FLUX_COLUMNS = ("time", "flux")


def _rename_columns(df: pd.DataFrame, columns, allowed: tuple, path, what: str, setting: str) -> pd.DataFrame:
    """
    Rename a file's own column names to the package's, from a
    {their name: package name} mapping given as `setting`. Nothing is
    guessed: every name must exist in the file, every target must be one
    of `allowed`, two columns cannot get the same name, and an existing
    column is never overwritten.
    """
    if columns is None:
        return df
    if not isinstance(columns, Mapping):
        raise TypeError(f"{setting} must be a dict {{your column name: package column name}}, got {columns!r}")
    for theirs, ours in columns.items():
        if not isinstance(theirs, str) or not isinstance(ours, str):
            raise TypeError(f"{setting} must map column names (text) to column names, got {theirs!r}: {ours!r}")
        if theirs not in df.columns:
            raise ValueError(
                f"{setting} names column {theirs!r}, but {what} at {path} has no such column. "
                f"Its columns: {', '.join(map(str, df.columns))}"
            )
        if ours not in allowed:
            raise ValueError(
                f"{setting} maps {theirs!r} to {ours!r}, which is not a column name the package uses here. "
                f"Allowed: {', '.join(allowed)}"
            )
    targets = list(columns.values())
    repeated = sorted({ours for ours in targets if targets.count(ours) > 1})
    if repeated:
        raise ValueError(f"{setting} maps more than one column to {', '.join(map(repr, repeated))}.")
    renamed_away = {theirs for theirs, ours in columns.items() if theirs != ours}
    for theirs, ours in columns.items():
        if theirs != ours and ours in df.columns and ours not in renamed_away:
            raise ValueError(
                f"{setting} maps {theirs!r} to {ours!r}, but {what} at {path} already has a column {ours!r}; "
                "remove or rename it first so it is not overwritten."
            )
    return df.rename(columns=dict(columns))


def _validate_rxfi_column(df: pd.DataFrame, path) -> None:
    """
    The optional rxfi column (relative X-ray flux increase: peak flux /
    background flux) must hold positive finite numbers; a blank cell means
    the value is unknown for that flare.
    """
    values = pd.to_numeric(df["rxfi"], errors="coerce")
    bad = (df["rxfi"].notna() & values.isna()) | ~(values.isna() | (np.isfinite(values) & (values > 0)))
    if bad.any():
        position = int(np.flatnonzero(bad.to_numpy())[0])
        raise ValueError(
            f"Flare catalog at {path}, column 'rxfi', data row {position + 1}: expected a positive "
            f"number or a blank cell, got {df['rxfi'].iloc[position]!r}."
        )
    df["rxfi"] = values


def _validate_time_order(df: pd.DataFrame, path) -> None:
    """
    Reject rows whose available times violate start <= peak <= end.

    Equal times are allowed. A missing time (NaT) is skipped, and the
    remaining times are still compared with each other. Must run before
    the catalog is re-sorted, so the reported data row matches the CSV
    (row 1 = first row after the header).
    """
    columns = ["start_time", "peak_time", "end_time"]
    for earlier, later in [("start_time", "peak_time"), ("peak_time", "end_time"), ("start_time", "end_time")]:
        bad = df[earlier].notna() & df[later].notna() & (df[earlier] > df[later])
        if not bad.any():
            continue
        position = int(np.flatnonzero(bad.to_numpy())[0])
        row = df.iloc[position]
        times = ", ".join(f"{c}={row[c]}" for c in columns)
        raise ValueError(
            f"Flare catalog at {path}, data row {position + 1}: times must satisfy "
            f"start <= peak <= end, but {earlier} > {later} ({times})."
        )


class EventMatcher:
    """
    Loads a flare catalog and finds the flares in the prediction window of
    an instance time.

    The catalog is a CSV with the columns start_time, peak_time, end_time,
    class, active_region (optionally fl_lon, fl_lat, rxfi); every time cell
    is a full date-time.
    """

    REQUIRED_CATALOG_COLUMNS = {"start_time", "peak_time", "end_time", "class", "active_region"}

    def __init__(self, catalog_path: str | Path, columns=None):
        # columns: optional {your column name: package column name}.
        self._catalog = self._load_catalog(catalog_path, columns)

    def _load_catalog(self, path: str | Path, columns=None) -> pd.DataFrame:
        """
        Load and validate the flare catalog CSV.

        Raises ValueError if the file has no columns at all (completely
        empty file) or lacks a required column. A catalog with a header
        row but zero data rows is valid: it produces an empty catalog, so
        every query() call against it simply returns no flares.
        """
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            raise ValueError(
                f"Flare catalog at {path} is empty: no header row / columns found."
            ) from None
        df = _rename_columns(df, columns, CATALOG_COLUMNS, path, "Flare catalog", "catalog_columns")

        missing = self.REQUIRED_CATALOG_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Flare catalog at {path} is missing required column(s): "
                f"{', '.join(sorted(missing))}"
            )
        if "goes_class" in df.columns:
            raise ValueError(
                f"Flare catalog at {path} has both a 'class' and a 'goes_class' column; "
                "keep only 'class' (the column the catalog schema uses)."
            )

        df = df.rename(columns={"class": "goes_class"})
        df["start_time"] = _parse_full_timestamp_column(df, "start_time", path)
        df["peak_time"] = _parse_full_timestamp_column(df, "peak_time", path)
        df["end_time"] = _parse_full_timestamp_column(df, "end_time", path)
        _validate_time_order(df, path)
        validate_goes_class_column(df["goes_class"], path)
        if "rxfi" in df.columns:
            _validate_rxfi_column(df, path)

        return df.sort_values("peak_time").reset_index(drop=True)

    def _sorted_starts(self) -> tuple[np.ndarray, np.ndarray]:
        """
        The catalog's start times, sorted, and their row positions. Built
        once and reused by every query(), so labeling tens of thousands of
        instances does not re-sort the catalog each time.
        """
        # getattr, not an attribute set in __init__: tests build an
        # EventMatcher with __new__() and set _catalog directly.
        if getattr(self, "_start_index", None) is None:
            values = self._catalog["start_time"].to_numpy()
            order = np.argsort(values, kind="stable")
            self._start_index = (values[order], order)
        return self._start_index

    def query(
        self,
        reference_time: pd.Timestamp,
        prediction_window,
        active_region: str | None = None,
    ) -> list[FlareEvent]:
        """
        Returns all flares whose start time falls inside the prediction
        window.

        The window is [reference_time, reference_time + prediction_window): it
        includes reference_time itself and excludes the window end, so a 24 h
        window means "less than 24 hours after the instance".

        prediction_window is either a number of hours (e.g. 24) or a
        pandas-compatible duration (e.g. "23h59min59s", pd.Timedelta).

        Example: instance at 10:00, window = 24 hours
        → returns all flares with start_time in [10:00, 10:00 next day)

        If active_region is given, only flares attributed to that exact
        active region are returned; flares with a missing active_region
        never match, and flares from any other active region are excluded
        regardless of strength. The value is normalized like the catalog
        side (13700, 13700.0, "13700" and "13700.0" all mean "13700"); a
        blank or missing value is an error, not full-disk matching.
        Several regions can be given as comma-separated text, e.g.
        "11160,11161,11162" for a HARP patch that covers three NOAA
        regions: a flare in any of them is returned. If
        active_region is None (the default), matching is full-disk: any
        flare in the window counts.

        reference_time must be a pandas Timestamp (not NaT). A Timestamp with
        a UTC offset is converted to UTC before matching; a naive one is
        assumed to be on the catalog's time scale.
        """
        if isinstance(reference_time, (datetime.datetime, np.datetime64)) and not isinstance(reference_time, pd.Timestamp):
            reference_time = pd.Timestamp(reference_time)
        if not isinstance(reference_time, pd.Timestamp) or pd.isna(reference_time):
            raise ValueError(
                f"reference_time must be a pandas Timestamp (not missing/NaT), got {reference_time!r} "
                f"({type(reference_time).__name__})."
            )
        if active_region is not None:
            normalized_query_ar = _normalize_active_region(active_region)
            if normalized_query_ar is None:
                raise ValueError(
                    f"active_region must be a region identifier such as '13700', got {active_region!r}; "
                    "pass None for full-disk matching."
                )
            active_region = normalized_query_ar

        reference_time = _to_naive_utc(reference_time)
        window_end = reference_time + _parse_prediction_window(prediction_window)

        sorted_values, sort_positions = self._sorted_starts()
        # to_datetime64 keeps nanoseconds; np.datetime64(Timestamp) would cut them.
        reference_time_np = reference_time.to_datetime64()
        window_end_np = window_end.to_datetime64()

        # [reference_time, window_end): include reference_time, exclude window_end.
        lo = np.searchsorted(sorted_values, reference_time_np, side="left")
        hi = np.searchsorted(sorted_values, window_end_np, side="left")

        candidate_positions = sort_positions[lo:hi]
        candidates = self._catalog.iloc[candidate_positions]

        if active_region is not None:
            wanted = set(active_region.split(","))
            in_region = candidates["active_region"].apply(
                lambda value: _shares_region(_normalize_active_region(value), wanted)
            )
            candidates = candidates[in_region.to_numpy(dtype=bool)]

        return [
            FlareEvent(
                peak_time=row["peak_time"],
                goes_class=row["goes_class"],
                start_time=row["start_time"],
                active_region=_normalize_active_region(row.get("active_region")),
                end_time=row.get("end_time"),
                fl_lon=row.get("fl_lon"),
                fl_lat=row.get("fl_lat"),
                rxfi=None if pd.isna(row.get("rxfi")) else float(row["rxfi"]),
            )
            for _, row in candidates.iterrows()
        ]
