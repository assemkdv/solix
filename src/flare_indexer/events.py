import datetime
import numbers
import re

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from .classifier import validate_goes_class_column

# A bare number written as text ("24", "1.5", "2.4e1"): pandas would read
# it as nanoseconds, so such a duration string is rejected and a unit asked for.
_UNITLESS_NUMBER_TEXT = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")

# Region text that is an integer, possibly with a ".0" tail as pandas writes
# a float column ("13700", "13700.0", "013700", "+13700").
_INTEGER_REGION_TEXT = re.compile(r"^[+-]?\d+(?:\.0*)?$")

# A date-time text that ends in an explicit UTC offset: "Z", "+00:00",
# "+0500", "-07:00". Used to tell offset-aware cells from naive ones.
_OFFSET_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$", re.IGNORECASE)


# A simple container that holds the details of one flare event
# Think of it like a row from the NOAA catalog, but as a Python object
@dataclass
class FlareEvent:
    peak_time: pd.Timestamp          # when the flare was at its strongest
    goes_class: str                  # flare class letter + number, e.g. "M2.3"
    start_time: pd.Timestamp         # when the flare started
    active_region: str | None        # which region of the sun it came from (can be empty)
    end_time: pd.Timestamp | None = None   # when the flare ended (not all catalogs carry this)
    fl_lon: float | None = None            # heliographic longitude, when the source catalog has it
    fl_lat: float | None = None            # heliographic latitude, when the source catalog has it
    rxfi: float | None = None              # relative X-ray flux increase, when the catalog has it


VALID_EVENT_TIMES = {"start", "peak"}


def _parse_prediction_window(value) -> pd.Timedelta:
    """
    Normalize a prediction window into a strictly positive pd.Timedelta.

    A plain number (int or float) is interpreted as hours, preserving the
    original `prediction_window=24` behavior. Anything else is parsed as a
    pandas-compatible duration, e.g. "24h", "23h59min59s", or a
    pd.Timedelta.
    """
    if isinstance(value, bool):
        raise ValueError(f"prediction_window must be a number of hours or a duration, got {value!r}")
    _reject_unitless_duration_text(value, "prediction_window")
    try:
        if isinstance(value, numbers.Real):
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
    """
    if value is None or isinstance(value, bool):
        return None if value is None else str(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
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
    Parse a column of full datetime strings (e.g. "2010-05-01 01:39:00"),
    raising a clear ValueError naming the file/column/value on failure.

    Each cell carries its own complete calendar date, so a flare that
    crosses midnight (start_time on one date, peak_time/end_time on the
    next) is represented as written. A blank cell becomes NaT: the flare
    stays in the catalog, but it can never be matched through that
    column (a flare with a blank start_time is invisible to
    event_time="start", one with a blank peak_time to event_time="peak";
    a blank end_time never affects matching). Numeric columns, time-zone
    offsets and mixed conventions are handled by _parse_time_text_column.
    """
    return _parse_time_text_column(df[column], column, what, path)


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
    Given an image timestamp, finds all flares that happened
    within a certain number of hours after that image was taken.

    The catalog is a CSV with the columns start_time, peak_time, end_time,
    class, active_region (optionally fl_lon, fl_lat, rxfi). Every time cell is a
    complete date-time string, so a flare whose start, peak and end fall on
    different calendar dates is written as it happened. This is the schema
    `flare_indexer.loaders.adapt_goes_catalog` produces from an integrated
    GOES catalog.
    """

    REQUIRED_CATALOG_COLUMNS = {"start_time", "peak_time", "end_time", "class", "active_region"}

    def __init__(self, catalog_path: str | Path):
        # Load the catalog once when the object is created
        # so we don't re-read the file every time we query
        self._catalog = self._load_catalog(catalog_path)
        # Lazily built, cached {event_time: (sorted_values, original_positions)}
        # views used by query()'s searchsorted fast path -- see _sorted_view().
        self._sorted_view_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _load_catalog(self, path: str | Path) -> pd.DataFrame:
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

    def _sorted_view(self, event_time: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Cached, sorted (values, original_positions) for the given event-time
        basis, built once per basis and reused across query() calls so a
        DatasetBuilder run over tens of thousands of images doesn't re-sort
        the catalog every time -- see query()'s searchsorted fast path.
        """
        # Tests (and any other caller) sometimes build an EventMatcher via
        # __new__() and set _catalog directly, bypassing __init__ -- fall
        # back to an empty cache rather than requiring that pattern to also
        # know about this private cache attribute.
        cache = self.__dict__.setdefault("_sorted_view_cache", {})
        if event_time not in cache:
            column = "start_time" if event_time == "start" else "peak_time"
            values = self._catalog[column].to_numpy()
            order = np.argsort(values, kind="stable")
            cache[event_time] = (values[order], order)
        return cache[event_time]

    def query(
        self,
        image_time: pd.Timestamp,
        prediction_window_hours: int,
        active_region: str | None = None,
        event_time: str = "start",
    ) -> list[FlareEvent]:
        """
        Returns all flares whose event time falls inside the prediction
        window.

        event_time selects which of a flare's timestamps is checked against
        the window:
        - "start" (default): the flare's start_time.
        - "peak": the flare's peak_time.

        The window is [image_time, image_time + prediction_window): it
        includes image_time itself and excludes the window end, so a 24 h
        window means "less than 24 hours after the image".

        prediction_window_hours is either a number of hours (e.g. 24) or a
        pandas-compatible duration (e.g. "23h59min59s", pd.Timedelta).

        Example: image taken at 10:00, window = 24 hours, defaults
        → returns all flares with start_time in [10:00, 10:00 next day)

        If active_region is given, only flares attributed to that exact
        active region are returned; flares with a missing active_region
        never match, and flares from any other active region are excluded
        regardless of strength. The value is normalized like the catalog
        side (13700, 13700.0, "13700" and "13700.0" all mean "13700"); a
        blank or missing value is an error, not full-disk matching. If
        active_region is None (the default), matching is full-disk: any
        flare in the window counts.

        image_time must be a pandas Timestamp (not NaT). A Timestamp with
        a UTC offset is converted to UTC before matching; a naive one is
        assumed to be on the catalog's time scale.
        """
        if event_time not in VALID_EVENT_TIMES:
            raise ValueError(f"event_time must be one of {sorted(VALID_EVENT_TIMES)}, got {event_time!r}")
        if isinstance(image_time, (datetime.datetime, np.datetime64)) and not isinstance(image_time, pd.Timestamp):
            image_time = pd.Timestamp(image_time)
        if not isinstance(image_time, pd.Timestamp) or pd.isna(image_time):
            raise ValueError(
                f"image_time must be a pandas Timestamp (not missing/NaT), got {image_time!r} "
                f"({type(image_time).__name__})."
            )
        if active_region is not None:
            normalized_query_ar = _normalize_active_region(active_region)
            if normalized_query_ar is None:
                raise ValueError(
                    f"active_region must be a region identifier such as '13700', got {active_region!r}; "
                    "pass None for full-disk matching."
                )
            active_region = normalized_query_ar

        image_time = _to_naive_utc(image_time)
        window_end = image_time + _parse_prediction_window(prediction_window_hours)

        sorted_values, sort_positions = self._sorted_view(event_time)
        image_time_np = np.datetime64(image_time)
        window_end_np = np.datetime64(window_end)

        # [image_time, window_end): include image_time, exclude window_end.
        lo = np.searchsorted(sorted_values, image_time_np, side="left")
        hi = np.searchsorted(sorted_values, window_end_np, side="left")

        candidate_positions = sort_positions[lo:hi]
        candidates = self._catalog.iloc[candidate_positions]

        if active_region is not None:
            normalized_catalog_ar = candidates["active_region"].apply(_normalize_active_region)
            candidates = candidates[normalized_catalog_ar.to_numpy() == active_region]

        # Convert the matching rows into FlareEvent objects and return them
        # If no flares match, this returns an empty list → label will be 0
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
