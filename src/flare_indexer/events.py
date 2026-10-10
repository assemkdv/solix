import numbers

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from .classifier import validate_goes_class_column


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


VALID_EVENT_TIMES = {"start", "peak"}
VALID_INTERVAL_MODES = {"left_closed", "right_closed"}


def _parse_prediction_window(value) -> pd.Timedelta:
    """
    Normalize a prediction window into a strictly positive pd.Timedelta.

    A plain number (int or float) is interpreted as hours, preserving the
    original `prediction_window=24` behavior. Anything else is parsed as a
    pandas-compatible duration, e.g. "24h", "23h59min59s", or a
    pd.Timedelta -- which is what allows reproducing a legacy window such
    as (t, t + 23:59:59] exactly.
    """
    if isinstance(value, bool):
        raise ValueError(f"prediction_window must be a number of hours or a duration, got {value!r}")
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


def _normalize_active_region(value) -> str | None:
    # A catalog with any missing active_region cells gets read by pandas as
    # float64 (NaN for missing), so a present value like 4456 arrives as the
    # Python float 4456.0, not a string. Normalize both cases here so
    # FlareEvent.active_region actually matches its str | None type.
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _parse_full_timestamp_column(df: pd.DataFrame, column: str, path) -> pd.Series:
    """
    Parse a column of full datetime strings (e.g. "2010-05-01 01:39:00"),
    raising a clear ValueError naming the file/column/value on failure.

    Each cell carries its own complete calendar date, so a flare that
    crosses midnight (start_time on one date, peak_time/end_time on the
    next) is represented as written. A blank cell becomes NaT.
    """
    try:
        return pd.to_datetime(df[column])
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Flare catalog at {path} has an unparseable value in column "
            f"'{column}': {exc}"
        ) from None


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
    class, active_region (optionally fl_lon, fl_lat). Every time cell is a
    complete date-time string, so a flare whose start, peak and end fall on
    different calendar dates is written as it happened. This is the schema
    `flare_indexer.loaders.adapt_goes_catalog` produces from the lab's
    integrated GOES catalog.
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

        df = df.rename(columns={"class": "goes_class"})
        df["start_time"] = _parse_full_timestamp_column(df, "start_time", path)
        df["peak_time"] = _parse_full_timestamp_column(df, "peak_time", path)
        df["end_time"] = _parse_full_timestamp_column(df, "end_time", path)
        _validate_time_order(df, path)
        validate_goes_class_column(df["goes_class"], path)

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
        event_time: str = "peak",
        interval_mode: str = "left_closed",
    ) -> list[FlareEvent]:
        """
        Returns all flares whose event time falls inside the prediction
        window.

        event_time selects which of a flare's timestamps is checked against
        the window:
        - "peak" (default, preserves original behavior): the flare's
          peak_time.
        - "start": the flare's start_time. This matches how Dr. Pandey's
          legacy full-disk labeling script defines its prediction window.

        interval_mode selects the window's boundary convention, given
        window_end = image_time + prediction_window_hours:
        - "left_closed" (default, preserves original behavior):
          [image_time, window_end) -- includes image_time itself, excludes
          window_end exactly.
        - "right_closed": (image_time, window_end] -- excludes image_time
          itself, includes window_end exactly. This matches the legacy
          full-disk script's convention.

        prediction_window_hours is either a number of hours (e.g. 24) or a
        pandas-compatible duration (e.g. "23h59min59s", pd.Timedelta).

        Example: image taken at 10:00, window = 24 hours, defaults
        → returns all flares with peak_time in [10:00, 10:00 next day)

        If active_region is given, only flares attributed to that exact
        active region are returned; flares with a missing active_region
        never match, and flares from any other active region are excluded
        regardless of strength. If active_region is None (the default),
        matching is full-disk: any flare in the window counts.
        """
        if event_time not in VALID_EVENT_TIMES:
            raise ValueError(f"event_time must be one of {sorted(VALID_EVENT_TIMES)}, got {event_time!r}")
        if interval_mode not in VALID_INTERVAL_MODES:
            raise ValueError(f"interval_mode must be one of {sorted(VALID_INTERVAL_MODES)}, got {interval_mode!r}")

        window_end = image_time + _parse_prediction_window(prediction_window_hours)

        sorted_values, sort_positions = self._sorted_view(event_time)
        image_time_np = np.datetime64(image_time)
        window_end_np = np.datetime64(window_end)

        if interval_mode == "left_closed":
            lo = np.searchsorted(sorted_values, image_time_np, side="left")
            hi = np.searchsorted(sorted_values, window_end_np, side="left")
        else:
            lo = np.searchsorted(sorted_values, image_time_np, side="right")
            hi = np.searchsorted(sorted_values, window_end_np, side="right")

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
            )
            for _, row in candidates.iterrows()
        ]
