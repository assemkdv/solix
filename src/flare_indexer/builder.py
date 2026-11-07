import numbers
import warnings

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from .events import (
    EventMatcher, _none_for_missing, _normalize_active_region, _parse_prediction_window, _parse_time_text_column,
    _reject_unitless_duration_text, _rename_columns, _to_naive_utc,
    IMAGE_INDEX_COLUMNS,
)


# How many skipped starts / missing timestamps a SequenceBuildReport lists
# (its counts are always exact).
_MAX_SEQUENCE_REPORT_SAMPLES = 50

VALID_MISSING_ACTIVE_REGION_POLICIES = {"negative", "drop", "error"}


@dataclass
class SequenceBuildReport:
    """
    Validation/bookkeeping report returned by FullDiskSequence /
    ActiveRegionSequence .build(return_report=True).

    inferred_source_cadence : the most common gap between the image
        index's timestamps.
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
    skipped_sequence_regions : ActiveRegionSequence only -- the region
        of each entry in skipped_sequence_starts (None for the group of
        images without a region); empty for full-disk builds.
    region_count : ActiveRegionSequence only -- how many region groups
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
    Parse the image index's timestamp column. A numeric column, an
    unparseable cell or a blank cell raises ValueError. UTC offsets are
    converted to UTC and dropped, as in the catalog.
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
    """The most common positive gap between the index's unique timestamps (robust to missing images)."""
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


def _validate_sliding_window(sliding_window: pd.Timedelta, inferred: pd.Timedelta) -> None:
    """The step between sequence starts may be smaller than cadence, but must land on image times."""
    if sliding_window % inferred != pd.Timedelta(0):
        raise ValueError(
            f"sliding_window {sliding_window} is not an exact multiple of the inferred source cadence "
            f"{inferred}, so sequences would start between images."
        )


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
    The engine behind the four dataset classes in datasets.py: maps image
    timestamps to flare labels with a reducer and a labeler.

    Parameters
    ----------
    prediction_window : hours (number) or a duration with a unit ("24h").
        Flares that start in [t, t + prediction_window) count, where t is
        the image time or, for a sequence, the time of its last image.
    reducer : turns the flares of one window into one value; any object
        with reduce(flares), e.g. MaxFluxReducer().
    labeler : turns that value into the label; any object with
        assign(value), e.g. BinaryThresholdLabeler("M").
    target : "full_disk" (all flares count) or "active_region" (only the
        flares of the image's own region; the index needs active_region).
    missing_active_region : "negative", "drop" or "error", for
        active-region images without a region.
    include_metadata : single images only; keep the other index columns.
    include_flares : add a "flares" column listing each window's classes.
    verbose : print a short summary of each build.
    start, cadence, observation_window, sliding_window : observation_window
        or sliding_window makes sequences, and then cadence,
        observation_window and sliding_window are required; each sequence
        holds observation_window / cadence images. cadence (and start)
        alone keep only the single images at start, start + cadence, ....
        cadence and sliding_window must be multiples of the index's own
        cadence (the most common gap between its timestamps); sliding_window
        may be smaller than cadence. start must be an image time.
        Active-region sequences are built per region.
    """

    VALID_TARGETS = {"full_disk", "active_region"}

    def __init__(
        self,
        prediction_window,
        *,
        reducer,
        labeler,
        target="full_disk",
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
        if not callable(getattr(reducer, "reduce", None)):
            raise TypeError(
                f"reducer={reducer!r} must have a reduce(flares) method, like MaxFluxReducer()."
            )
        if not callable(getattr(labeler, "assign", None)):
            raise TypeError(
                f"labeler={labeler!r} must have an assign(value) method, like BinaryThresholdLabeler('M')."
            )
        if missing_active_region not in VALID_MISSING_ACTIVE_REGION_POLICIES:
            raise ValueError(
                f"missing_active_region must be one of {sorted(VALID_MISSING_ACTIVE_REGION_POLICIES)}, "
                f"got {missing_active_region!r}"
            )
        if target not in self.VALID_TARGETS:
            raise ValueError(f"target must be one of {sorted(self.VALID_TARGETS)}, got {target!r}")

        # observation_window or sliding_window makes sequences; cadence (and
        # start) alone thin out single images to a time grid.
        sequence_given = observation_window is not None or sliding_window is not None

        if getattr(reducer, "single_image_only", False) and sequence_given:
            raise ValueError(
                f"{type(reducer).__name__} labels single images only; "
                "it cannot be used with sequence parameters."
            )

        self.prediction_window = prediction_window
        self._window = parsed_prediction_window
        self.reducer = reducer
        self.labeler = labeler
        self.target = target
        self.missing_active_region = missing_active_region
        if not isinstance(include_metadata, bool):
            raise ValueError(f"include_metadata must be True or False, got {include_metadata!r}")
        self.include_metadata = include_metadata
        if not isinstance(include_flares, bool):
            raise ValueError(f"include_flares must be True or False, got {include_flares!r}")
        self.include_flares = include_flares
        if not isinstance(verbose, bool):
            raise ValueError(f"verbose must be True or False, got {verbose!r}")
        self.verbose = verbose
        self.time_based = sequence_given
        self.image_cadence = None

        if not self.time_based and (cadence is not None or start is not None):
            if cadence is None:
                raise ValueError("start needs a cadence: it is the first point of the cadence grid.")
            self.image_cadence = _parse_duration(cadence, "cadence")
            self.start = _parse_timestamp_param(start, "start") if start is not None else None

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

            self.number_of_images = round(self.observation_window / self.cadence)

    def build(
        self, image_index_path: str | Path, event_catalog_path: str | Path, return_report: bool = False,
        *, index_columns=None, catalog_columns=None,
    ) -> pd.DataFrame:
        """
        Build a labeled dataset.

        Single images (no sequence parameters): returns a DataFrame with
        columns (timestamp, label) -- or, for active regions,
        (timestamp, active_region, label). An image index with a header row but
        zero data rows is valid and produces an empty result with the same
        columns.

        Sequences (see the class docstring): returns one row per sequence
        with columns (sequence_start, sequence_end, timestamps, n_images,
        label) -- or, for active regions, (sequence_start,
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

        index_columns and catalog_columns let the files use their own
        column names: {your column name: package column name}, e.g.
        index_columns={"obs_time": "timestamp"} or
        catalog_columns={"goes_class": "class"}. The columns are renamed
        right after each file is read, so the output uses the package's
        names. A name missing from the file, a target the package does not
        use, two columns mapped to one name, or a target that already
        exists raises ValueError.
        """
        if not isinstance(return_report, bool):
            raise ValueError(f"return_report must be True or False, got {return_report!r}")
        if return_report and not self.time_based:
            raise ValueError(
                "return_report=True is only supported for sequences "
                "(when observation_window and sliding_window are given)."
            )

        image_index_path = Path(image_index_path)

        try:
            index_df = pd.read_csv(image_index_path)
        except pd.errors.EmptyDataError:
            raise ValueError(
                f"Image index at {image_index_path} is empty: no header row / columns found."
            ) from None
        index_df = _rename_columns(
            index_df, index_columns, IMAGE_INDEX_COLUMNS, image_index_path, "Image index", "index_columns",
        )

        if "timestamp" not in index_df.columns:
            raise ValueError(
                f"Image index at {image_index_path} is missing required column: timestamp"
            )

        if self.target == "active_region" and "active_region" not in index_df.columns:
            raise ValueError(
                f"Image index at {image_index_path} is missing required column: "
                "active_region (required for active-region labels)"
            )

        index_df = index_df.reset_index(drop=True)
        index_df["timestamp"] = _parse_image_timestamps(index_df["timestamp"], image_index_path)

        matcher = EventMatcher(event_catalog_path, columns=catalog_columns)
        self._check_required_catalog_column(matcher, event_catalog_path)
        self._warn_catalog_missing_start_time(matcher, event_catalog_path)
        if self.target == "active_region":
            self._warn_catalog_missing_active_region(matcher, event_catalog_path)

        if self.verbose:
            self._print_inputs(index_df, image_index_path, matcher, event_catalog_path)

        if self.image_cadence is not None:
            index_df = self._images_on_cadence_grid(index_df, image_index_path)

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
        elif self.image_cadence is not None:
            mode = f"single images every {_format_duration(self.image_cadence)}"
        else:
            mode = "single images"
        print(f"Settings: window [t, t + {_format_duration(self._window)}) on start_time, {target}, {mode}")

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
            flares = matcher.query(ts, self._window)
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
        # active_region is always kept: several regions can share a timestamp.
        columns = ["timestamp", "active_region"] + self._label_columns() + extra_columns

        records = []
        missing_rows = []
        for position, (ts, raw_active_region) in enumerate(zip(index_df["timestamp"], index_df["active_region"])):
            active_region = _normalize_active_region(raw_active_region)
            if active_region is None:
                # Missing image-side active_region must not silently match
                # anything -- skip the query entirely rather than falling
                # back to full-disk matching. What happens to the row is
                # governed by missing_active_region.
                missing_rows.append(f"data row {index_df.index[position] + 1} ({ts})")
                if self.missing_active_region == "drop":
                    continue
                flares = []
            else:
                flares = matcher.query(ts, self._window, active_region=active_region)
            record = {"timestamp": ts, "active_region": active_region, **self._label_fields(flares)}
            for col in extra_columns:
                record[col] = index_df[col].iloc[position]
            records.append(record)

        self._report_missing_active_regions(missing_rows, "image(s)")

        if not records:
            return pd.DataFrame(columns=columns)
        return _missing_regions_as_none(pd.DataFrame(records)[columns])

    def _report_missing_active_regions(self, missing: list, unit: str) -> None:
        if not missing:
            return
        sample = ", ".join(missing[:5]) + (", ..." if len(missing) > 5 else "")
        if self.missing_active_region == "error":
            raise ValueError(
                f"{len(missing)} {unit} have a missing active_region (e.g. {sample}). "
                "Their label is unknown, not negative. Fill in active_region, or pass "
                "missing_active_region='drop' or 'negative'."
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
        column = getattr(self.reducer, "requires_catalog_column", None)
        if column is None:
            return
        catalog = matcher._catalog
        if column not in catalog.columns:
            raise ValueError(
                f"Flare catalog at {path} has no '{column}' column, which "
                f"{type(self.reducer).__name__} needs."
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
        # A copy, so a reducer can never change the list the window found.
        fields = {"label": self.labeler.assign(self.reducer.reduce(list(flares)))}
        if self.include_flares:
            fields["flares"] = [flare.goes_class for flare in flares]
        return fields

    def _label_columns(self) -> list:
        return ["label", "flares"] if self.include_flares else ["label"]

    def _sequence_reserved(self) -> tuple:
        return _SEQUENCE_OUTPUT_COLUMNS + (("flares",) if self.include_flares else ())

    def _warn_catalog_missing_start_time(self, matcher: EventMatcher, path) -> None:
        """
        A catalog row may have a blank start_time (the catalog still
        loads). Such a flare can never be matched, so the build says how
        many flares are invisible to it.
        """
        catalog = matcher._catalog
        missing = int(catalog["start_time"].isna().sum())
        if missing:
            warnings.warn(
                f"{missing} of {len(catalog)} flare(s) in the catalog at {path} have no start_time. "
                "They can never match any image, so a window that contains one of them is labeled "
                "as if it were absent.",
                UserWarning,
                stacklevel=3,
            )

    def _warn_catalog_missing_active_region(self, matcher: EventMatcher, path) -> None:
        catalog = matcher._catalog
        missing = int(catalog["active_region"].apply(_normalize_active_region).isna().sum())
        if missing:
            warnings.warn(
                f"{missing} of {len(catalog)} flare(s) in the catalog at {path} have no "
                "active_region. In active-region labels they can never match any image, "
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
        occurred.
        """
        regions = index_df["active_region"].map(_normalize_active_region)
        missing = regions.isna()
        if missing.any() and self.missing_active_region == "error":
            examples = ", ".join(
                f"data row {i + 1} ({index_df['timestamp'].loc[i]})" for i in index_df.index[missing][:5]
            )
            raise ValueError(
                f"{int(missing.sum())} image(s) in {path} have a missing active_region "
                f"(e.g. {examples}{', ...' if missing.sum() > 5 else ''}). They cannot be assigned "
                "to any region's sequence. Fill in active_region, or pass "
                "missing_active_region='drop' or 'negative'."
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
        return matcher.query(reference_time, self._window, active_region=region)

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

    def _images_on_cadence_grid(self, index_df: pd.DataFrame, path: Path) -> pd.DataFrame:
        """
        Single images with a cadence: keep only the images at start,
        start + cadence, start + 2 * cadence, ... (start defaults to the
        earliest image; all regions share one grid). The cadence must be a
        multiple of the image index's own cadence, as for sequences.
        """
        timestamps = index_df["timestamp"]
        if timestamps.nunique() >= 2:   # one image (or none) has no cadence to check against
            _validate_requested_cadence(self.image_cadence, _infer_source_cadence(timestamps))
        grid_start = self._validated_start(timestamps, path)
        offset = timestamps - grid_start
        on_grid = (offset >= pd.Timedelta(0)) & (offset % self.image_cadence == pd.Timedelta(0))
        return index_df[on_grid]   # the index keeps each image's row in the file, for messages

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
        _validate_sliding_window(self.sliding_window, inferred_cadence)

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
            flares = matcher.query(sequence_end, self._window)
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
            if region is None and rows["timestamp"].duplicated().any():
                raise ValueError(
                    f"Image index at {path} has several images without an active_region at the same time "
                    f"(e.g. {rows['timestamp'][rows['timestamp'].duplicated()].iloc[0]}). They cannot form one "
                    "sequence; pass missing_active_region='drop' to leave them out, or fill in active_region."
                )
            self._validate_unique_timestamps(rows["timestamp"], path, self._group_name(region))

        inferred_cadence = _infer_source_cadence(index_df["timestamp"])
        _validate_requested_cadence(self.cadence, inferred_cadence)
        _validate_sliding_window(self.sliding_window, inferred_cadence)
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
    return _missing_regions_as_none(frame.iloc[order].reset_index(drop=True))


def _missing_regions_as_none(frame: pd.DataFrame) -> pd.DataFrame:
    frame["active_region"] = _none_for_missing(frame["active_region"])
    return frame
