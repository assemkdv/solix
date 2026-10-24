import numbers
import warnings

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from .events import (
    EventMatcher, _normalize_active_region, _parse_prediction_window, _parse_time_text_column,
    _reject_unitless_duration_text, _to_naive_utc,
    VALID_EVENT_TIMES,
)

_TIME_BASED_PARAM_NAMES = ("start", "cadence", "observation_window", "sliding_window")

# Cap on how many skipped-sequence-starts / missing-timestamps a
# SequenceBuildReport keeps verbatim, so a very long-running build doesn't
# blow up memory just to report on itself. Counts are always exact.
_MAX_SEQUENCE_REPORT_SAMPLES = 50

VALID_MISSING_ACTIVE_REGION_POLICIES = {"negative", "drop", "error"}


@dataclass
class SequenceBuildReport:
    """
    Validation/bookkeeping report for DatasetBuilder.build(return_report=True)
    in time-based sequence mode.

    inferred_source_cadence : the base cadence detected from the image
        index's own timestamps (see DatasetBuilder's "Source cadence
        inference" docs).
    requested_cadence, observation_window, sliding_window : the
        (parsed, pd.Timedelta) values this build used.
    effective_start : the actual starting timestamp used (either the
        explicit `start`, or the dataset's earliest timestamp).
    total_candidate_sequences : every candidate start considered, whether
        or not it was ultimately emitted.
    emitted_sequences / skipped_sequences : how many candidates were
        emitted vs. skipped because a required timestamp was missing.
    skipped_sequence_starts : up to the first `_MAX_SEQUENCE_REPORT_SAMPLES`
        candidate start timestamps that were skipped.
    missing_timestamps : up to the first `_MAX_SEQUENCE_REPORT_SAMPLES`
        specific timestamps that were missing and caused a skip.
    skipped_sequence_regions : target="active_region" only -- the region
        of each entry in skipped_sequence_starts (None for the group of
        images without a region); empty for full-disk builds.
    region_count : target="active_region" only -- how many region groups
        sequences were built for (the missing-region group counts as one);
        None for full-disk builds.
    """

    inferred_source_cadence: pd.Timedelta
    requested_cadence: pd.Timedelta
    observation_window: pd.Timedelta
    sliding_window: pd.Timedelta
    effective_start: pd.Timestamp | None
    total_candidate_sequences: int
    emitted_sequences: int
    skipped_sequences: int
    skipped_sequence_starts: list = field(default_factory=list)
    missing_timestamps: list = field(default_factory=list)
    skipped_sequence_regions: list = field(default_factory=list)
    region_count: int | None = None


def _parse_duration(value, name: str) -> pd.Timedelta:
    """
    Normalize a time-based duration parameter (cadence, observation_window,
    sliding_window): duration text with a unit ("1h", "30min"), a
    pd.Timedelta, a datetime.timedelta or a numpy.timedelta64, strictly
    positive. A bare number is rejected: pandas would read it as
    nanoseconds, and unlike prediction_window (hours) these parameters
    give a number no unit.
    """
    if isinstance(value, bool) or (isinstance(value, numbers.Real) and not isinstance(value, np.timedelta64)):
        raise ValueError(
            f"{name} must be a duration with a unit (e.g. '1h', '30min') or a Timedelta, got the bare "
            f"number {value!r}. A number has no unit here; only prediction_window reads a number as "
            "hours."
        )
    _reject_unitless_duration_text(value, name)
    try:
        duration = pd.Timedelta(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a valid duration (e.g. '1h', '30min'), got {value!r}: {exc}") from None
    if pd.isna(duration):
        raise ValueError(f"{name} must be a valid duration (e.g. '1h', '30min'), got {value!r}")
    if duration <= pd.Timedelta(0):
        raise ValueError(f"{name} must be strictly positive, got {value!r} (parsed as {duration})")
    return duration


def _parse_timestamp_param(value, name: str) -> pd.Timestamp:
    """
    Normalize a pandas-compatible timestamp value. A value with a UTC
    offset is converted to UTC and stored naive, like every other
    timestamp the package compares (see events._to_naive_utc).
    """
    try:
        timestamp = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a valid timestamp, got {value!r}: {exc}") from None
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid timestamp, got {value!r}")
    return _to_naive_utc(timestamp)


def _parse_image_timestamps(column: pd.Series, path: Path) -> pd.Series:
    """
    Parse the image index's timestamp column, refusing anything that
    would otherwise produce samples silently: a numeric column (read as
    nanoseconds since 1970), an unparseable cell, or a blank cell (NaT,
    which single-image mode would crash on and sequence mode would drop
    without a trace).
    Values with a UTC offset are converted to UTC and stored naive, as in
    the catalog (see events._to_naive_utc).
    """
    parsed = _parse_time_text_column(column, "timestamp", "Image index", path)
    missing = parsed.isna()
    if missing.any():
        rows = [int(i) + 1 for i in np.flatnonzero(missing.to_numpy())]
        shown = ", ".join(str(r) for r in rows[:10]) + (", ..." if len(rows) > 10 else "")
        raise ValueError(
            f"Image index at {path} has {len(rows)} blank/missing timestamp(s) "
            f"(data row(s) {shown}; row 1 is the first row after the header). "
            "Fill or remove those rows; samples are never built from an image without a time."
        )
    return parsed


def _infer_source_cadence(timestamps: pd.Series) -> pd.Timedelta:
    """
    Infer an image index's base cadence as the mode (most common value) of
    its sorted-unique timestamps' positive consecutive differences.

    Using the mode rather than, say, the minimum or largest gap makes this
    robust to a dataset with occasional missing images (larger, rarer gaps
    don't affect the result as long as the "normal" spacing is still the
    most frequent one).
    """
    unique_sorted = pd.Series(timestamps.unique()).sort_values().reset_index(drop=True)
    if len(unique_sorted) < 2:
        raise ValueError(
            "Cannot infer source cadence: the image index has fewer than 2 unique "
            "timestamps, so no consecutive difference can be measured."
        )
    diffs = unique_sorted.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        raise ValueError("Cannot infer source cadence: no positive consecutive timestamp differences found.")
    return diffs.mode().iloc[0]


def _validate_requested_cadence(requested: pd.Timedelta, inferred: pd.Timedelta) -> None:
    if requested < inferred:
        raise ValueError(
            f"cadence {requested} is smaller than the inferred source cadence {inferred}; "
            "cannot request a cadence finer than the image index actually provides."
        )
    if requested % inferred != pd.Timedelta(0):
        raise ValueError(
            f"cadence {requested} is not an exact multiple of the inferred source cadence {inferred}."
        )


class DatasetBuilder:
    """
    Maps image timestamps to flare labels using a configurable strategy.

    Parameters
    ----------
    prediction_window : int | float | str | pd.Timedelta
        How far after each image timestamp (or, for sequences, after the
        final image in the sequence) to search for flares. A plain number
        is hours (e.g. 24); a string/Timedelta is a pandas duration with
        a unit (e.g. "24h", "23h59min59s"). Bare numeric text ("24") is
        rejected. This hours convention is specific to prediction_window:
        the sequence durations below accept no bare number at all.
    strategy : BinaryThresholdStrategy | MaxFlareStrategy
        Labeling strategy that converts a list of FlareEvents to a label.
    target : str
        "full_disk" (default) matches any flare in the window, preserving
        original behavior. "active_region" matches only flares from the
        specific active region each image is assigned to (requires an
        active_region column in the image index). In sequence mode the
        index is first grouped by region and each region's sequences are
        built from that region's images only, so several regions may be
        observed at the same timestamps; see "Active-region sequences".
    event_time : str
        Which of a flare's timestamps EventMatcher checks against the
        prediction window: "start" (default) or "peak". See EventMatcher.query.
    missing_active_region : str
        Only used when target="active_region". What to do with images
        whose image-side active_region is missing:
        "negative" (default, preserves original behavior) labels them as
        if no flare occurred and emits a UserWarning with the count, since
        such a label is unknown, not a confirmed negative; "drop" leaves
        them out of the output; "error" raises ValueError. In sequence
        mode such images never join a region's sequence; with "negative"
        they form one extra group whose sequences get active_region=None.
    include_metadata : bool
        Single-image mode only. False (default, preserves original
        behavior) returns just (timestamp, label) -- or, for
        target="active_region", the same. True also returns every other
        image-index column (e.g. image_path); for target="active_region"
        the normalized active_region is placed right before label, so
        several regions observed at the same timestamp stay distinguishable.
    include_flares : bool
        False (default) returns only the label. True adds a "flares" column
        right after it: the GOES classes of the flares in each window, in
        order of event time (e.g. ["C3.2", "M1.4"]; [] for an empty
        window), so every label can be checked by eye. All modes.
    verbose : bool
        False (default) prints nothing. True prints a short summary of each
        build: images and flares loaded, the settings used, and the result
        (row count and label counts or range). The returned table is the
        same either way.

    Sequence parameters (giving any of them builds sequences instead of
    single images)
    ----------------------------------------------------------------------
    start : pandas-compatible timestamp | None
        The first candidate sequence's starting timestamp. Inclusive: an
        image must exist at exactly this timestamp, or build() raises
        ValueError -- it never silently moves to the next available image.
        If omitted, defaults to the image index's earliest timestamp.
    cadence : duration text with a unit, or Timedelta | None
        The spacing between consecutive images *within* a sequence (e.g.
        "1h", "30min", pd.Timedelta(hours=1)). A bare number is rejected
        as unit-less (it is not hours and not minutes). Required whenever
        any sequence parameter is used. Must be an exact multiple of
        the image index's own inferred cadence (see "Source cadence
        inference"). observation_window and sliding_window take the same
        forms.
    observation_window : pandas-compatible duration | None
        The half-open span [sequence_start, sequence_start +
        observation_window) covered by one sequence. Required in
        sequence mode. Must be an exact multiple of `cadence`; the
        number of images per sequence is observation_window / cadence
        (e.g. cadence="1h", observation_window="6h" -> 6 images at
        t, t+1h, ..., t+5h -- note the *last* image is at
        t + observation_window - cadence, since the interval excludes its
        own right endpoint).
    sliding_window : pandas-compatible duration | None
        How far forward the next candidate sequence's start moves relative
        to the current one. Required in sequence mode. Must be an exact
        multiple of `cadence`.

    Active-region sequences
    ----------------------------------------------------------------------
    With target="active_region", sequences are built per region: rows
    are grouped by normalized active_region, and the uniqueness check and
    the time grid apply within each region. Duplicate timestamps within
    one region raise ValueError; the same timestamp in different regions
    is fine. The source cadence is inferred from all image timestamps, `start`
    must be an image time somewhere in the index, and every region uses
    the grid start + k * sliding_window (or, without `start`, a grid
    beginning at the region's own first image). Output rows are sorted
    by sequence_start, then active_region (missing region last).

    Sequence mode selection
    ----------------------------------------------------------------------
    Supplying any of start/cadence/observation_window/sliding_window
    builds sequences, in which cadence, observation_window, and
    sliding_window are all required (start remains optional).

    Source cadence inference
    ----------------------------------------------------------------------
    In sequence mode, build() infers the image index's own base cadence
    as the mode (most common value) of its sorted-unique timestamps'
    positive consecutive differences -- robust to occasional missing
    images/larger gaps, unlike using the timestamps' minimum or maximum
    gap. The requested `cadence` must be greater than or equal to this
    inferred cadence and an exact integer multiple of it (e.g. a 30-minute
    source image index accepts cadence="1h", producing a compatible
    downsampled sequence -- but rejects cadence="1min", since that would
    require images finer than the source actually provides).
    """

    VALID_TARGETS = {"full_disk", "active_region"}

    def __init__(
        self,
        prediction_window,
        strategy,
        target="full_disk",
        event_time="start",
        start=None,
        cadence=None,
        observation_window=None,
        sliding_window=None,
        missing_active_region="negative",
        include_metadata=False,
        include_flares=False,
        verbose=False,
    ):
        parsed_prediction_window = _parse_prediction_window(prediction_window)
        if missing_active_region not in VALID_MISSING_ACTIVE_REGION_POLICIES:
            raise ValueError(
                f"missing_active_region must be one of {sorted(VALID_MISSING_ACTIVE_REGION_POLICIES)}, "
                f"got {missing_active_region!r}"
            )
        if target not in self.VALID_TARGETS:
            raise ValueError(f"target must be one of {sorted(self.VALID_TARGETS)}, got {target!r}")
        if event_time not in VALID_EVENT_TIMES:
            raise ValueError(f"event_time must be one of {sorted(VALID_EVENT_TIMES)}, got {event_time!r}")

        time_based_given = [
            name for name, value in zip(_TIME_BASED_PARAM_NAMES, (start, cadence, observation_window, sliding_window))
            if value is not None
        ]

        reducer = getattr(strategy, "reducer", None)
        if getattr(reducer, "single_image_only", False) and time_based_given:
            raise ValueError(
                f"{type(reducer).__name__} labels single images only; "
                "it cannot be used with sequence parameters."
            )

        self.prediction_window = prediction_window
        self._window = parsed_prediction_window
        self.strategy = strategy
        self.target = target
        self.event_time = event_time
        self.missing_active_region = missing_active_region
        self.include_metadata = include_metadata
        if not isinstance(include_flares, bool):
            raise ValueError(f"include_flares must be True or False, got {include_flares!r}")
        self.include_flares = include_flares
        if not isinstance(verbose, bool):
            raise ValueError(f"verbose must be True or False, got {verbose!r}")
        self.verbose = verbose
        self.time_based = bool(time_based_given)

        if self.time_based:
            missing_required = [
                name for name, value in (
                    ("cadence", cadence),
                    ("observation_window", observation_window),
                    ("sliding_window", sliding_window),
                )
                if value is None
            ]
            if missing_required:
                raise ValueError(
                    f"Sequence mode also requires: {', '.join(missing_required)} "
                    "(start is the only optional sequence parameter)."
                )

            self.cadence = _parse_duration(cadence, "cadence")
            self.observation_window = _parse_duration(observation_window, "observation_window")
            self.sliding_window = _parse_duration(sliding_window, "sliding_window")
            self.start = _parse_timestamp_param(start, "start") if start is not None else None

            if self.observation_window % self.cadence != pd.Timedelta(0):
                raise ValueError(
                    f"observation_window ({observation_window}) must be an exact multiple "
                    f"of cadence ({cadence})."
                )
            if self.sliding_window % self.cadence != pd.Timedelta(0):
                raise ValueError(
                    f"sliding_window ({sliding_window}) must be an exact multiple of cadence ({cadence})."
                )

            self.number_of_images = round(self.observation_window / self.cadence)
            self.slide_steps = round(self.sliding_window / self.cadence)

    def build(
        self, image_index_path: str | Path, event_catalog_path: str | Path, return_report: bool = False,
    ) -> pd.DataFrame:
        """
        Build a labeled dataset.

        Single images (no sequence parameters): returns a DataFrame with
        columns (timestamp, label). An image index with a header row but
        zero data rows is valid and produces an empty result with the same
        columns.

        Sequences (see the class docstring): returns one row per sequence
        with columns (sequence_start, sequence_end, timestamps, n_images,
        label) -- or, when target="active_region", (sequence_start,
        sequence_end, timestamps, n_images, active_region, label) -- plus
        one list-column per other image-index column (e.g. an image path),
        holding that column's values for each image in the sequence.
        Sequences are built on an exact timestamp grid (candidate_start +
        i * cadence), which also allows downsampling a finer-cadence image
        index. A candidate whose grid is missing any timestamp is skipped,
        not interpolated or substituted. An image-index column whose name
        equals an output column (e.g. "label") raises ValueError instead of
        overwriting it.

        return_report=True is only supported for sequences, where it
        returns (DataFrame, SequenceBuildReport) instead of just the
        DataFrame; passing it for single images raises ValueError.

        A completely empty image index (no header) raises ValueError, as
        does an image index missing the required timestamp column.
        """
        if return_report and not self.time_based:
            raise ValueError(
                "return_report=True is only supported for sequences "
                "(i.e. when start/cadence/observation_window/sliding_window were used)."
            )

        image_index_path = Path(image_index_path)

        try:
            index_df = pd.read_csv(image_index_path)
        except pd.errors.EmptyDataError:
            raise ValueError(
                f"Image index at {image_index_path} is empty: no header row / columns found."
            ) from None

        if "timestamp" not in index_df.columns:
            raise ValueError(
                f"Image index at {image_index_path} is missing required column: timestamp"
            )

        if self.target == "active_region" and "active_region" not in index_df.columns:
            raise ValueError(
                f"Image index at {image_index_path} is missing required column: "
                "active_region (required when target='active_region')"
            )

        index_df = index_df.reset_index(drop=True)
        index_df["timestamp"] = _parse_image_timestamps(index_df["timestamp"], image_index_path)

        matcher = EventMatcher(event_catalog_path)
        self._check_required_catalog_column(matcher, event_catalog_path)
        self._warn_catalog_missing_event_time(matcher, event_catalog_path)
        if self.target == "active_region":
            self._warn_catalog_missing_active_region(matcher, event_catalog_path)

        if self.verbose:
            self._print_inputs(index_df, image_index_path, matcher, event_catalog_path)

        report = None
        if self.time_based:
            if self.target == "active_region":
                result, report = self._build_time_based_sequences_active_region(index_df, matcher, image_index_path)
            else:
                result, report = self._build_time_based_sequences(index_df, matcher, image_index_path)
        elif self.target == "active_region":
            result = self._build_single_image_active_region(index_df, matcher, image_index_path)
        else:
            result = self._build_single_image(index_df, matcher, image_index_path)

        if self.verbose:
            self._print_result(result, report)
        return (result, report) if return_report else result

    # ── verbose summary ─────────────────────────────────────────────────

    def _print_inputs(self, index_df: pd.DataFrame, index_path: Path, matcher: EventMatcher, catalog_path) -> None:
        images = f"{len(index_df):,} images"
        if self.target == "active_region":
            regions = index_df["active_region"].map(_normalize_active_region).dropna().nunique()
            images += f", {regions:,} regions"
        print(f"Image index: {images} ({index_path})")
        print(f"Flare catalog: {len(matcher._catalog):,} flares ({catalog_path})")

        target = "full disk" if self.target == "full_disk" else "active region"
        if self.time_based:
            mode = (f"sequences of {self.number_of_images} images {_format_duration(self.cadence)} apart, "
                    f"new sequence every {_format_duration(self.sliding_window)}")
        else:
            mode = "single images"
        print(f"Settings: window [t, t + {_format_duration(self._window)}) on {self.event_time}_time, {target}, {mode}")

    def _print_result(self, result: pd.DataFrame, report) -> None:
        unit = "sequences" if self.time_based else "rows"
        line = f"Result: {len(result):,} {unit}"
        if report is not None and report.skipped_sequences:
            line += f" ({report.skipped_sequences:,} skipped: missing images)"
        summary = _label_summary(result["label"])
        print(line + (f"; {summary}" if summary else ""))

    def _build_single_image(self, index_df: pd.DataFrame, matcher: EventMatcher, path: Path) -> pd.DataFrame:
        if self.include_metadata:
            extra_columns = _extra_columns(index_df, ("timestamp",), tuple(self._label_columns()), path)
        records = []
        for ts in index_df["timestamp"]:
            flares = matcher.query(
                ts, self._window,
                event_time=self.event_time,
            )
            records.append({"timestamp": ts, **self._label_fields(flares)})

        columns = ["timestamp"] + self._label_columns()
        if not records:
            result = pd.DataFrame(columns=columns)
        else:
            result = pd.DataFrame(records)
        if self.include_metadata:
            for col in extra_columns:
                result[col] = index_df[col].to_numpy()
        return result

    def _build_single_image_active_region(
        self, index_df: pd.DataFrame, matcher: EventMatcher, path: Path,
    ) -> pd.DataFrame:
        if self.include_metadata:
            extra_columns = _extra_columns(index_df, ("timestamp", "active_region"), tuple(self._label_columns()), path)
        else:
            extra_columns = []
        if self.include_metadata:
            columns = ["timestamp", "active_region"] + self._label_columns() + extra_columns
        else:
            columns = ["timestamp"] + self._label_columns()

        records = []
        missing_rows = []
        for position, (ts, raw_active_region) in enumerate(zip(index_df["timestamp"], index_df["active_region"])):
            active_region = _normalize_active_region(raw_active_region)
            if active_region is None:
                # Missing image-side active_region must not silently match
                # anything -- skip the query entirely rather than falling
                # back to full-disk matching. What happens to the row is
                # governed by missing_active_region.
                missing_rows.append(f"row {position} ({ts})")
                if self.missing_active_region == "drop":
                    continue
                flares = []
            else:
                flares = matcher.query(
                    ts, self._window, active_region=active_region,
                    event_time=self.event_time,
                )
            record = {"timestamp": ts, "active_region": active_region, **self._label_fields(flares)}
            for col in extra_columns:
                record[col] = index_df[col].iloc[position]
            records.append(record)

        self._report_missing_active_regions(missing_rows, "image(s)")

        if not records:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(records)[columns]

    def _report_missing_active_regions(self, missing: list, unit: str) -> None:
        if not missing:
            return
        sample = ", ".join(missing[:5]) + (", ..." if len(missing) > 5 else "")
        if self.missing_active_region == "error":
            raise ValueError(
                f"{len(missing)} {unit} have a missing active_region (e.g. {sample}). "
                "Their label is unknown, not negative. Fill in active_region, or pass "
                "missing_active_region='drop' or 'negative' to DatasetBuilder."
            )
        if self.missing_active_region == "negative":
            warnings.warn(
                f"{len(missing)} {unit} have a missing active_region (e.g. {sample}) and were "
                "labeled as if no flare occurred. Such labels are unknown, not confirmed "
                "negatives. Pass missing_active_region='drop' to leave them out, or 'error' "
                "to stop.",
                UserWarning,
                stacklevel=4,
            )

    def _check_required_catalog_column(self, matcher: EventMatcher, path) -> None:
        """
        A reducer that reads an optional catalog column (e.g. rxfi) names it
        in requires_catalog_column. Without that column every window would
        silently reduce to 0, so the build stops instead; blank cells are
        counted in a warning.
        """
        column = getattr(getattr(self.strategy, "reducer", None), "requires_catalog_column", None)
        if column is None:
            return
        catalog = matcher._catalog
        if column not in catalog.columns:
            raise ValueError(
                f"Flare catalog at {path} has no '{column}' column, which "
                f"{type(self.strategy.reducer).__name__} needs."
            )
        missing = int(catalog[column].isna().sum())
        if missing:
            warnings.warn(
                f"{missing} of {len(catalog)} flare(s) in the catalog at {path} have no {column}; "
                "they are skipped when the label is computed.",
                UserWarning,
                stacklevel=3,
            )

    def _label_fields(self, flares: list) -> dict:
        """The label of one window, plus its flare list when include_flares is on."""
        fields = {"label": self.strategy.label(flares)}
        if self.include_flares:
            fields["flares"] = [flare.goes_class for flare in flares]
        return fields

    def _label_columns(self) -> list:
        return ["label", "flares"] if self.include_flares else ["label"]

    def _sequence_reserved(self) -> tuple:
        return _SEQUENCE_OUTPUT_COLUMNS + (("flares",) if self.include_flares else ())

    def _warn_catalog_missing_event_time(self, matcher: EventMatcher, path) -> None:
        """
        A catalog row may have a blank start_time or peak_time (the
        catalog still loads). Such a flare can never be matched through
        the blank column, so when it is the selected event_time column
        the build says how many flares are invisible to it.
        """
        catalog = matcher._catalog
        column = "start_time" if self.event_time == "start" else "peak_time"
        if catalog.empty or column not in catalog.columns:
            return
        missing = int(catalog[column].isna().sum())
        if missing:
            warnings.warn(
                f"{missing} of {len(catalog)} flare(s) in the catalog at {path} have no {column}. "
                f"With event_time={self.event_time!r} they can never match any image, so a window "
                "that contains one of them is labeled as if it were absent.",
                UserWarning,
                stacklevel=3,
            )

    def _warn_catalog_missing_active_region(self, matcher: EventMatcher, path) -> None:
        catalog = matcher._catalog
        if "active_region" not in catalog.columns or catalog.empty:
            return
        missing = int(catalog["active_region"].apply(_normalize_active_region).isna().sum())
        if missing:
            warnings.warn(
                f"{missing} of {len(catalog)} flare(s) in the catalog at {path} have no "
                "active_region. With target='active_region' they can never match any image, "
                "so a region that produced one of them may be labeled negative.",
                UserWarning,
                stacklevel=3,
            )

    def _validate_unique_timestamps(self, timestamps: pd.Series, path: Path, group: str | None = None) -> None:
        duplicated = timestamps[timestamps.duplicated()]
        if not duplicated.empty:
            dupe_list = ", ".join(str(t) for t in duplicated.unique()[:10])
            more = ", ..." if duplicated.nunique() > 10 else ""
            within = f" within {group}" if group else ""
            raise ValueError(
                f"Image index at {path} contains duplicate timestamp(s){within}: {dupe_list}{more}"
            )

    # ── Active-region sequences: one group of images per region ──────────

    def _region_groups(self, index_df: pd.DataFrame, path: Path) -> list[tuple[str | None, pd.DataFrame]]:
        """
        Split the image index into one group per normalized active region,
        in order of first appearance. Rows without a region never join a
        region's group; missing_active_region decides what happens to them:
        "error" raises, "drop" leaves them out, "negative" puts them in one
        extra group (key None) whose sequences are labeled as if no flare
        occurred -- the closest equivalent of the pre-grouping behavior.
        """
        regions = index_df["active_region"].map(_normalize_active_region)
        missing = regions.isna()
        if missing.any() and self.missing_active_region == "error":
            examples = ", ".join(
                f"row {i} ({index_df['timestamp'].iloc[i]})" for i in index_df.index[missing][:5]
            )
            raise ValueError(
                f"{int(missing.sum())} image(s) in {path} have a missing active_region "
                f"(e.g. {examples}{', ...' if missing.sum() > 5 else ''}). They cannot be assigned "
                "to any region's sequence. Fill in active_region, or pass "
                "missing_active_region='drop' or 'negative' to DatasetBuilder."
            )

        groups = [
            (region, index_df.loc[rows.index])
            for region, rows in regions[~missing].groupby(regions[~missing], sort=False)
        ]
        if missing.any() and self.missing_active_region == "negative":
            groups.append((None, index_df.loc[missing]))
        return groups

    def _warn_missing_region_sequences(self, index_df: pd.DataFrame, missing_sequences: int) -> None:
        if self.missing_active_region != "negative":
            return
        missing_rows = int(index_df["active_region"].map(_normalize_active_region).isna().sum())
        if not missing_rows:
            return
        warnings.warn(
            f"{missing_rows} image(s) have a missing active_region. They are not part of any "
            f"region's sequence; {missing_sequences} sequence(s) built only from such images were "
            "labeled as if no flare occurred. Such labels are unknown, not confirmed negatives. "
            "Pass missing_active_region='drop' to leave them out, or 'error' to stop.",
            UserWarning,
            stacklevel=4,
        )

    @staticmethod
    def _group_name(region: str | None) -> str:
        return "rows with a missing active_region" if region is None else f"active_region {region}"

    def _query_region(self, matcher: EventMatcher, reference_time, region: str | None) -> list:
        if region is None:
            # Never fall back to full-disk matching for an unknown region.
            return []
        return matcher.query(
            reference_time, self._window, active_region=region,
            event_time=self.event_time,
        )

    # ── Time-based sequence mode ─────────────────────────────────────────

    def _grid_candidates(self, ts_to_position: dict, first_start: pd.Timestamp, max_ts: pd.Timestamp):
        """
        Walk candidate starts first_start, first_start + sliding_window, ...
        while the last required image (start + observation_window - cadence)
        is not after max_ts. Returns (emitted, total, skipped_starts,
        missing_timestamps): emitted is a list of (window_timestamps,
        row_positions) for candidates with every grid time present;
        skipped samples are capped at _MAX_SEQUENCE_REPORT_SAMPLES.
        """
        offsets = [i * self.cadence for i in range(self.number_of_images)]

        emitted = []
        total_candidates = 0
        skipped_starts = []
        missing_timestamps = []

        candidate_start = first_start
        while candidate_start + offsets[-1] <= max_ts:
            total_candidates += 1
            required_timestamps = [candidate_start + offset for offset in offsets]

            positions = []
            missing = None
            for required_ts in required_timestamps:
                position = ts_to_position.get(required_ts)
                if position is None:
                    missing = required_ts
                    break
                positions.append(position)

            if missing is not None:
                if len(skipped_starts) < _MAX_SEQUENCE_REPORT_SAMPLES:
                    skipped_starts.append(candidate_start)
                if len(missing_timestamps) < _MAX_SEQUENCE_REPORT_SAMPLES:
                    missing_timestamps.append(missing)
            else:
                emitted.append((required_timestamps, positions))

            candidate_start = candidate_start + self.sliding_window

        return emitted, total_candidates, skipped_starts, missing_timestamps

    def _validated_start(self, timestamps: pd.Series, path: Path) -> pd.Timestamp:
        if self.start is None:
            return timestamps.min()
        if not (timestamps == self.start).any():
            raise ValueError(
                f"start={self.start} has no exact matching image timestamp in "
                f"the image index at {path}."
            )
        return self.start

    def _time_based_candidates(self, index_df: pd.DataFrame, path: Path) -> tuple[list, SequenceBuildReport]:
        """
        Full-disk candidate generation: one grid over the whole index.

        Returns (emitted, report) where `emitted` is a list of
        (window_timestamps, row_positions) -- row_positions are positions
        into index_df, suitable for .iloc, used to fetch extra columns.
        """
        timestamps = index_df["timestamp"]
        self._validate_unique_timestamps(timestamps, path)

        inferred_cadence = _infer_source_cadence(timestamps)
        _validate_requested_cadence(self.cadence, inferred_cadence)

        effective_start = self._validated_start(timestamps, path)
        ts_to_position = {ts: i for i, ts in enumerate(timestamps)}
        emitted, total, skipped_starts, missing_timestamps = self._grid_candidates(
            ts_to_position, effective_start, timestamps.max(),
        )

        report = SequenceBuildReport(
            inferred_source_cadence=inferred_cadence,
            requested_cadence=self.cadence,
            observation_window=self.observation_window,
            sliding_window=self.sliding_window,
            effective_start=effective_start,
            total_candidate_sequences=total,
            emitted_sequences=len(emitted),
            skipped_sequences=total - len(emitted),
            skipped_sequence_starts=skipped_starts,
            missing_timestamps=missing_timestamps,
        )
        return emitted, report

    def _build_time_based_sequences(
        self, index_df: pd.DataFrame, matcher: EventMatcher, path: Path,
    ) -> tuple[pd.DataFrame, SequenceBuildReport]:
        extra_columns = _extra_columns(index_df, ("timestamp",), self._sequence_reserved(), path)
        candidates, report = self._time_based_candidates(index_df, path)
        columns = ["sequence_start", "sequence_end", "timestamps", "n_images"] + self._label_columns() + extra_columns

        records = []
        for window_timestamps, positions in candidates:
            sequence_end = window_timestamps[-1]
            flares = matcher.query(
                sequence_end, self._window,
                event_time=self.event_time,
            )
            record = {
                "sequence_start": window_timestamps[0],
                "sequence_end": sequence_end,
                "timestamps": list(window_timestamps),
                "n_images": len(window_timestamps),
                **self._label_fields(flares),
            }
            for col in extra_columns:
                record[col] = index_df[col].iloc[positions].tolist()
            records.append(record)

        if not records:
            return pd.DataFrame(columns=columns), report
        return pd.DataFrame(records)[columns], report

    def _build_time_based_sequences_active_region(
        self, index_df: pd.DataFrame, matcher: EventMatcher, path: Path,
    ) -> tuple[pd.DataFrame, SequenceBuildReport]:
        """
        One timestamp grid per region. The source cadence is inferred from
        all image timestamps (the instrument cadence), and an explicit
        `start` must be an image time somewhere in the index. Each region's
        grid begins at its first image time, or -- when `start` is given --
        at the first grid point start + k * sliding_window (k >= 0) that is
        not before the region's first image, so all regions share one grid.
        """
        extra_columns = _extra_columns(index_df, ("timestamp", "active_region"), self._sequence_reserved(), path)
        columns = ["sequence_start", "sequence_end", "timestamps", "n_images", "active_region"] + self._label_columns() + extra_columns

        groups = self._region_groups(index_df, path)
        for region, rows in groups:
            self._validate_unique_timestamps(rows["timestamp"], path, self._group_name(region))

        inferred_cadence = _infer_source_cadence(index_df["timestamp"])
        _validate_requested_cadence(self.cadence, inferred_cadence)
        effective_start = self._validated_start(index_df["timestamp"], path)

        records = []
        missing_sequences = 0
        total = 0
        skipped_starts, skipped_regions, missing_timestamps = [], [], []
        for region, rows in groups:
            first, last = rows["timestamp"].min(), rows["timestamp"].max()
            if self.start is not None:
                if last < self.start:
                    continue
                steps = max(0, -(-(first - self.start) // self.sliding_window))  # ceiling division
                first = self.start + steps * self.sliding_window
            ts_to_position = dict(zip(rows["timestamp"], rows.index))
            emitted, group_total, group_skipped, group_missing = self._grid_candidates(ts_to_position, first, last)

            total += group_total
            room = _MAX_SEQUENCE_REPORT_SAMPLES - len(skipped_starts)
            skipped_starts += group_skipped[:room]
            skipped_regions += [region] * len(group_skipped[:room])
            missing_timestamps += group_missing[:_MAX_SEQUENCE_REPORT_SAMPLES - len(missing_timestamps)]

            for window_timestamps, positions in emitted:
                flares = self._query_region(matcher, window_timestamps[-1], region)
                missing_sequences += region is None
                record = {
                    "sequence_start": window_timestamps[0],
                    "sequence_end": window_timestamps[-1],
                    "timestamps": list(window_timestamps),
                    "n_images": len(window_timestamps),
                    "active_region": region,
                    **self._label_fields(flares),
                }
                for col in extra_columns:
                    record[col] = index_df[col].iloc[positions].tolist()
                records.append(record)

        self._warn_missing_region_sequences(index_df, missing_sequences)

        report = SequenceBuildReport(
            inferred_source_cadence=inferred_cadence,
            requested_cadence=self.cadence,
            observation_window=self.observation_window,
            sliding_window=self.sliding_window,
            effective_start=effective_start,
            total_candidate_sequences=total,
            emitted_sequences=len(records),
            skipped_sequences=total - len(records),
            skipped_sequence_starts=skipped_starts,
            missing_timestamps=missing_timestamps,
            skipped_sequence_regions=skipped_regions,
            region_count=len(groups),
        )
        return _sequence_frame(records, columns), report


_SEQUENCE_OUTPUT_COLUMNS = ("sequence_start", "sequence_end", "timestamps", "n_images", "label")


def _format_duration(value: pd.Timedelta) -> str:
    """24h, 90min, 23h59min59s -- the shortest exact form of a duration."""
    seconds = int(value.total_seconds())
    if seconds != value.total_seconds():
        return str(value)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    parts = [f"{hours}h" if hours else "", f"{minutes}min" if minutes else "", f"{secs}s" if secs else ""]
    return "".join(parts) or "0s"


def _label_summary(labels: pd.Series) -> str:
    """Counts for 0/1 or class labels, the range for numbers, nothing for dicts."""
    values = labels.dropna()
    if values.empty:
        return ""
    if values.map(lambda v: isinstance(v, (dict, list))).any():
        return ""
    if values.map(lambda v: isinstance(v, str)).all() or set(values.unique()) <= {0, 1}:
        counts = values.value_counts()
        return ", ".join(f"label {key}: {int(count):,}" for key, count in sorted(counts.items(), key=lambda kv: str(kv[0])))
    return f"values from {values.min():g} to {values.max():g}"


def _extra_columns(index_df: pd.DataFrame, excluded: tuple, reserved: tuple, path: Path) -> list:
    """
    Image-index columns copied into the output, refusing names that would
    overwrite a column the builder itself produces (e.g. an index column
    called "label").
    """
    extra = [c for c in index_df.columns if c not in excluded]
    clashes = [c for c in extra if c in reserved]
    if clashes:
        raise ValueError(
            f"Image index at {path} has column(s) {clashes} that would overwrite output column(s) of "
            "the same name. Rename them before building."
        )
    return extra


def _sequence_frame(records: list, columns: list) -> pd.DataFrame:
    """Active-region sequence output, ordered by sequence_start, then region (missing region last)."""
    if not records:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(records)[columns]
    order = sorted(
        range(len(records)),
        key=lambda i: (records[i]["sequence_start"], records[i]["active_region"] is None, records[i]["active_region"] or ""),
    )
    return frame.iloc[order].reset_index(drop=True)
