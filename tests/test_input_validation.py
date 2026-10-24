"""Input validation: durations, time columns, region ids, blank timestamps, time zones, class columns, labeler input."""

import datetime
import warnings
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from flare_indexer import (
    BinaryThresholdLabeler,
    BinaryThresholdStrategy,
    EventMatcher,
    adapt_goes_catalog,
)
from flare_indexer.events import _normalize_active_region, _parse_prediction_window
from flare_indexer.builder import DatasetBuilder

CATALOG_HEADER = "start_time,peak_time,end_time,class,active_region\n"
FLARE_ROW = "2024-01-01 10:00:00,2024-01-01 10:10:00,2024-01-01 10:30:00,M1.0,100\n"
GOES_HEADER = "start_time,peak_time,end_time,goes_class,noaa_active_region\n"
T0 = pd.Timestamp("2024-01-01 09:00:00")


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


@pytest.fixture
def catalog(tmp_path):
    return _write(tmp_path, "catalog.csv", CATALOG_HEADER + FLARE_ROW)


def _labels(builder, index_path, catalog_path):
    return builder.build(index_path, catalog_path)["label"].tolist()


# --- 1. unit-less duration text -------------------------------------------

@pytest.mark.parametrize("text", ["24", " 24 ", "1.5", "2.4e1", "1E3", ".5", "+24", "24."])
def test_unitless_duration_text_is_rejected(text):
    with pytest.raises(ValueError, match="has no unit"):
        _parse_prediction_window(text)
    with pytest.raises(ValueError, match="has no unit"):
        DatasetBuilder(text, BinaryThresholdStrategy())


def test_numbers_still_mean_hours_and_unit_text_still_works():
    assert _parse_prediction_window(24) == pd.Timedelta(hours=24)
    assert _parse_prediction_window(1.5) == pd.Timedelta(hours=1.5)
    assert _parse_prediction_window(np.float64(2)) == pd.Timedelta(hours=2)
    assert _parse_prediction_window(np.int64(6)) == pd.Timedelta(hours=6)
    assert _parse_prediction_window("24h") == pd.Timedelta(hours=24)
    assert _parse_prediction_window("23h59min59s") == pd.Timedelta("23:59:59")
    assert _parse_prediction_window(pd.Timedelta(minutes=90)) == pd.Timedelta(minutes=90)


@pytest.mark.parametrize("name", ["cadence", "observation_window", "sliding_window"])
def test_unitless_time_based_duration_text_is_rejected(name):
    params = {"cadence": "1h", "observation_window": "2h", "sliding_window": "1h", name: "3600"}
    with pytest.raises(ValueError, match=f"{name} '3600' has no unit"):
        DatasetBuilder(24, BinaryThresholdStrategy(), **params)


def test_text_window_with_unit_gives_the_numeric_result(tmp_path, catalog):
    index = _write(tmp_path, "index.csv", "timestamp\n2024-01-01 09:00:00\n")
    assert _labels(DatasetBuilder(24, BinaryThresholdStrategy()), index, catalog) == [1]
    assert _labels(DatasetBuilder("24h", BinaryThresholdStrategy()), index, catalog) == [1]


# --- 2. numeric time columns ------------------------------------------------

@pytest.mark.parametrize("value", ["20240101100000", "1704067200", "45000.5"])
def test_numeric_catalog_time_column_is_rejected_with_context(tmp_path, value):
    path = _write(
        tmp_path, "catalog.csv",
        CATALOG_HEADER + f"{value},2024-01-01 10:10:00,2024-01-01 10:30:00,M1.0,100\n",
    )
    with pytest.raises(ValueError) as excinfo:
        EventMatcher(path)
    message = str(excinfo.value)
    assert str(path) in message
    assert "column 'start_time'" in message
    assert "data row 1" in message
    assert "not date-time text" in message


def test_numeric_image_timestamp_column_is_rejected_with_context(tmp_path, catalog):
    index = _write(tmp_path, "index.csv", "timestamp\n20240101000000\n1704067200\n")
    with pytest.raises(ValueError, match=r"Image index at .*column 'timestamp', data row 1: the value 20240101000000"):
        DatasetBuilder(24, BinaryThresholdStrategy()).build(index, catalog)


def test_numeric_goes_catalog_time_is_rejected_by_the_adapter(tmp_path):
    path = _write(tmp_path, "goes.csv", GOES_HEADER + "20240101100000,20240101101000,20240101103000,M1.0,100\n")
    with pytest.raises(ValueError, match=r"GOES catalog at .*column 'start_time', data row 1"):
        adapt_goes_catalog(path)


def test_all_blank_time_column_is_not_treated_as_numeric(tmp_path):
    path = _write(tmp_path, "catalog.csv", CATALOG_HEADER + "2024-01-01 10:00:00,2024-01-01 10:10:00,,M1.0,100\n")
    matcher = EventMatcher(path)
    assert matcher._catalog["end_time"].isna().all()
    assert len(matcher.query(T0, 24)) == 1


# --- 3. region identifier normalization -----------------------------------

@pytest.mark.parametrize("value", [
    13700, 13700.0, np.int64(13700), np.float64(13700.0), np.float32(13700),
    "13700", "13700.0", "13700.00", " 13700 ", "013700", "+13700",
])
def test_equivalent_region_identifiers_normalize_identically(value):
    assert _normalize_active_region(value) == "13700"


def test_integer_region_text_is_normalized_exactly_without_float_rounding():
    big = "123456789012345678901"
    assert _normalize_active_region(big) == big
    assert _normalize_active_region(big + ".0") == big


@pytest.mark.parametrize("value, expected", [
    ("AR13700", "AR13700"), (" AR13700 ", "AR13700"), ("HARP 7115", "HARP 7115"),
    ("13700.5", "13700.5"), (13700.5, "13700.5"), ("1e3", "1e3"), ("13700a", "13700a"),
])
def test_nonnumeric_region_text_is_kept_with_whitespace_stripped(value, expected):
    assert _normalize_active_region(value) == expected


@pytest.mark.parametrize("value", [None, float("nan"), np.nan, pd.NA, pd.NaT, "", "   "])
def test_missing_region_values_become_none(value):
    assert _normalize_active_region(value) is None


def test_text_typed_image_region_column_matches_the_catalog(tmp_path, catalog):
    # "AR100" makes pandas read the whole column as text; the numeric
    # spellings must still match catalog region 100 (M1.0 at 10:00).
    rows = ["100.0", " 100", "0100", "AR100", "100"]
    index = _write(
        tmp_path, "index.csv",
        "timestamp,active_region\n" + "".join(f"2024-01-01 09:00:00,{r}\n" for r in rows),
    )
    builder = DatasetBuilder(24, BinaryThresholdStrategy(), target="active_region", include_metadata=True)
    out = builder.build(index, catalog)
    assert out["label"].tolist() == [1, 1, 1, 0, 1]
    assert out["active_region"].tolist() == ["100", "100", "100", "AR100", "100"]


def test_text_typed_catalog_region_column_matches_an_integer_image_region(tmp_path):
    catalog = _write(
        tmp_path, "catalog.csv",
        CATALOG_HEADER
        + "2024-01-01 10:00:00,2024-01-01 10:10:00,2024-01-01 10:30:00,M1.0,100.0\n"
        + "2024-01-02 10:00:00,2024-01-02 10:10:00,2024-01-02 10:30:00,C1.0,AR5\n",
    )
    index = _write(tmp_path, "index.csv", "timestamp,active_region\n2024-01-01 09:00:00,100\n")
    builder = DatasetBuilder(24, BinaryThresholdStrategy(), target="active_region")
    assert _labels(builder, index, catalog) == [1]


@pytest.mark.parametrize("region", [100, 100.0, np.int64(100), "100", "100.0", " 100 ", "0100"])
def test_query_normalizes_its_active_region_argument(catalog, region):
    assert len(EventMatcher(catalog).query(T0, 24, active_region=region)) == 1


def test_query_rejects_a_blank_region_instead_of_matching_the_full_disk(catalog):
    matcher = EventMatcher(catalog)
    for blank in ["", "   ", float("nan")]:
        with pytest.raises(ValueError, match="active_region must be a region identifier"):
            matcher.query(T0, 24, active_region=blank)
    assert len(matcher.query(T0, 24, active_region=None)) == 1
    assert matcher.query(T0, 24, active_region="101") == []


def test_whitespace_only_image_region_counts_as_missing(tmp_path, catalog):
    index = _write(tmp_path, "index.csv", 'timestamp,active_region\n2024-01-01 09:00:00,"   "\n2024-01-01 09:00:00,AR1\n')
    builder = DatasetBuilder(24, BinaryThresholdStrategy(), target="active_region", missing_active_region="error")
    with pytest.raises(ValueError, match="missing active_region"):
        builder.build(index, catalog)


# --- 4. blank image timestamps --------------------------------------------

def _builders_for_every_mode(**common):
    return [
        DatasetBuilder(24, BinaryThresholdStrategy(), **common),
        DatasetBuilder(24, BinaryThresholdStrategy(), cadence="1h", observation_window="2h", sliding_window="1h", **common),
    ]


def test_blank_image_timestamp_is_rejected_before_any_sample_is_built(tmp_path, catalog):
    index = _write(tmp_path, "index.csv", "timestamp,image_path\n2024-01-01 00:00:00,a\n,b\n2024-01-01 01:00:00,c\n")
    for builder in _builders_for_every_mode():
        with pytest.raises(ValueError, match=r"Image index at .* has 1 blank/missing timestamp\(s\) \(data row\(s\) 2;"):
            builder.build(index, catalog)


def test_blank_image_timestamp_is_rejected_in_active_region_modes(tmp_path, catalog):
    index = _write(
        tmp_path, "index.csv",
        "timestamp,active_region\n2024-01-01 00:00:00,100\n,100\n2024-01-01 01:00:00,100\n,100\n",
    )
    for builder in _builders_for_every_mode(target="active_region"):
        with pytest.raises(ValueError, match=r"has 2 blank/missing timestamp\(s\) \(data row\(s\) 2, 4;"):
            builder.build(index, catalog)


def test_unparseable_image_timestamp_names_the_index_file(tmp_path, catalog):
    index = _write(tmp_path, "index.csv", "timestamp\n2024-01-01 00:00:00\nnot a date\n")
    with pytest.raises(ValueError, match=r"Image index at .*index\.csv has an unparseable value in column 'timestamp'"):
        DatasetBuilder(24, BinaryThresholdStrategy()).build(index, catalog)


def test_header_only_image_index_still_builds_an_empty_result(tmp_path, catalog):
    index = _write(tmp_path, "index.csv", "timestamp\n")
    out = DatasetBuilder(24, BinaryThresholdStrategy()).build(index, catalog)
    assert out.empty and list(out.columns) == ["timestamp", "label"]


def test_query_rejects_a_missing_or_non_timestamp_image_time(catalog):
    matcher = EventMatcher(catalog)
    for bad in [pd.NaT, None, "2024-01-01 09:00", 1700000000]:
        with pytest.raises(ValueError, match="image_time must be a pandas Timestamp"):
            matcher.query(bad, 24)
    assert len(matcher.query(datetime.datetime(2024, 1, 1, 9), 24)) == 1
    assert len(matcher.query(np.datetime64("2024-01-01T09:00"), 24)) == 1


# --- 5. time-zone policy ----------------------------------------------------

def test_catalog_with_utc_offsets_is_converted_to_naive_utc(tmp_path):
    zulu = _write(
        tmp_path, "zulu.csv",
        CATALOG_HEADER + "2024-01-01T10:00:00Z,2024-01-01T10:10:00Z,2024-01-01T10:30:00Z,M1.0,100\n",
    )
    matcher = EventMatcher(zulu)
    assert matcher._catalog["start_time"].dt.tz is None
    assert matcher._catalog["start_time"].iloc[0] == pd.Timestamp("2024-01-01 10:00:00")
    assert len(matcher.query(T0, 24)) == 1

    plus_five = _write(
        tmp_path, "plus_five.csv",
        CATALOG_HEADER + "2024-01-01T15:00:00+05:00,2024-01-01T15:10:00+05:00,2024-01-01T15:30:00+05:00,M1.0,100\n",
    )
    matcher = EventMatcher(plus_five)
    assert matcher._catalog["peak_time"].iloc[0] == pd.Timestamp("2024-01-01 10:10:00")
    assert len(matcher.query(T0, 24)) == 1


def test_adapted_goes_catalog_with_offsets_is_naive_utc_and_loadable(tmp_path):
    goes = _write(tmp_path, "goes.csv", GOES_HEADER + "2024-01-01T10:00:00Z,2024-01-01T10:10:00Z,2024-01-01T10:30:00Z,M1.0,13100\n")
    output = tmp_path / "adapted.csv"
    adapted = adapt_goes_catalog(goes, output)
    assert adapted["start_time"].dt.tz is None
    assert adapted["start_time"].iloc[0] == pd.Timestamp("2024-01-01 10:00:00")
    assert len(EventMatcher(output).query(T0, 24)) == 1


def test_image_index_with_offsets_is_converted_to_utc_without_warnings(tmp_path, catalog):
    index = _write(tmp_path, "index.csv", "timestamp\n2024-01-01T14:00:00+05:00\n2024-01-02T14:00:00+05:00\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = DatasetBuilder(24, BinaryThresholdStrategy()).build(index, catalog)
    assert out["timestamp"].tolist() == [pd.Timestamp("2024-01-01 09:00:00"), pd.Timestamp("2024-01-02 09:00:00")]
    assert out["label"].tolist() == [1, 0]


def test_query_with_an_offset_aware_image_time(catalog):
    matcher = EventMatcher(catalog)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # 14:00+05:00 is 09:00 UTC: [09:00, 10:00) excludes the 10:00 flare.
        assert matcher.query(pd.Timestamp("2024-01-01 14:00+05:00"), 1, event_time="peak") == []
        # 14:30+05:00 is 09:30 UTC: [09:30, 10:30) includes it.
        assert len(matcher.query(pd.Timestamp("2024-01-01 14:30+05:00"), 1, event_time="peak")) == 1
        # The same instant written in UTC gives the same answer.
        assert len(matcher.query(pd.Timestamp("2024-01-01 09:30Z"), 1, event_time="peak")) == 1


def test_time_based_start_with_an_offset_is_converted_to_utc(tmp_path, catalog):
    index = _write(tmp_path, "index.csv", "timestamp\n2024-01-01 00:00:00\n2024-01-01 01:00:00\n2024-01-01 02:00:00\n")
    builder = DatasetBuilder(
        24, BinaryThresholdStrategy(),
        start="2024-01-01T05:00:00+05:00", cadence="1h", observation_window="2h", sliding_window="1h",
    )
    assert builder.start == pd.Timestamp("2024-01-01 00:00:00")
    assert len(builder.build(index, catalog)) == 2


def test_mixed_naive_and_offset_cells_in_one_column_are_rejected_clearly(tmp_path, catalog):
    path = _write(
        tmp_path, "mixed.csv",
        CATALOG_HEADER
        + "2024-01-01 10:00:00,2024-01-01 10:10:00,2024-01-01 10:30:00,M1.0,100\n"
        + "2024-01-02T10:00:00Z,2024-01-02 10:10:00,2024-01-02 10:30:00,M1.0,100\n",
    )
    expected = r"column 'start_time' mixes times without a UTC offset \(first at data row 1\) and times with one \(first at data row 2\)"
    with pytest.raises(ValueError, match=expected):
        EventMatcher(path)
    # Offset first, naive second: the rows in the message swap accordingly.
    reversed_path = _write(
        tmp_path, "mixed2.csv",
        CATALOG_HEADER
        + "2024-01-02T10:00:00+00:00,2024-01-02 10:10:00,2024-01-02 10:30:00,M1.0,100\n"
        + "2024-01-01 10:00:00,2024-01-01 10:10:00,2024-01-01 10:30:00,M1.0,100\n",
    )
    with pytest.raises(ValueError, match=r"\(first at data row 2\) and times with one \(first at data row 1\)"):
        EventMatcher(reversed_path)
    # The same rule for the image index and the adapter input.
    index = _write(tmp_path, "index.csv", "timestamp\n2024-01-01 09:00:00\n2024-01-01T09:00:00+05:00\n")
    with pytest.raises(ValueError, match=r"Image index at .*column 'timestamp' mixes times without a UTC offset"):
        DatasetBuilder(24, BinaryThresholdStrategy()).build(index, catalog)
    goes = _write(
        tmp_path, "goes.csv",
        GOES_HEADER
        + "2024-01-01T10:00:00Z,2024-01-01T10:10:00Z,2024-01-01T10:30:00Z,M1.0,100\n"
        + "2024-01-02T10:00:00Z,2024-01-02 10:10:00,2024-01-02T10:30:00Z,M1.0,100\n",
    )
    with pytest.raises(ValueError, match=r"GOES catalog at .*column 'peak_time' mixes"):
        adapt_goes_catalog(goes)


def test_same_instant_in_z_zero_and_nonzero_offsets_parses_to_one_value(tmp_path):
    path = _write(
        tmp_path, "catalog.csv",
        CATALOG_HEADER
        + "2024-01-01T10:00:00Z,2024-01-01T10:10:00Z,2024-01-01T10:30:00Z,M1.0,100\n"
        + "2024-01-01T10:00:00+00:00,2024-01-01T10:10:00+00:00,2024-01-01T10:30:00+00:00,M2.0,100\n"
        + "2024-01-01T15:00:00+05:00,2024-01-01T15:10:00+05:00,2024-01-01T15:30:00+05:00,M3.0,100\n"
        + "2024-01-01T03:00:00-07:00,2024-01-01T03:10:00-07:00,2024-01-01T03:30:00-07:00,M4.0,100\n",
    )
    matcher = EventMatcher(path)
    assert matcher._catalog["start_time"].dt.tz is None
    assert matcher._catalog["start_time"].nunique() == 1
    assert matcher._catalog["start_time"].iloc[0] == pd.Timestamp("2024-01-01 10:00:00")
    assert len(matcher.query(T0, 24, event_time="start")) == 4
    assert matcher.query(pd.Timestamp("2024-01-01 10:00:01"), 24, event_time="start") == []


def test_mixed_offsets_across_a_daylight_saving_change_convert_to_utc(tmp_path):
    # Central European Time switches from +01:00 to +02:00 on 2024-03-31 at 02:00 local.
    path = _write(
        tmp_path, "catalog.csv",
        CATALOG_HEADER
        + "2024-03-31T01:30:00+01:00,2024-03-31T01:40:00+01:00,2024-03-31T01:50:00+01:00,C1.0,100\n"
        + "2024-03-31T03:30:00+02:00,2024-03-31T03:40:00+02:00,2024-03-31T03:50:00+02:00,C2.0,100\n",
    )
    matcher = EventMatcher(path)
    assert matcher._catalog["start_time"].tolist() == [
        pd.Timestamp("2024-03-31 00:30:00"), pd.Timestamp("2024-03-31 01:30:00"),
    ]
    assert matcher._catalog["start_time"].dt.tz is None
    # The two flares are one hour apart in UTC, not two: [00:00, 02:00) holds both,
    # [00:00, 01:00) only the first.
    assert len(matcher.query(pd.Timestamp("2024-03-31 00:00"), "2h", event_time="start")) == 2
    assert len(matcher.query(pd.Timestamp("2024-03-31 00:00"), "1h", event_time="start")) == 1


def test_offset_column_with_missing_cells_keeps_them_missing(tmp_path):
    path = _write(
        tmp_path, "catalog.csv",
        CATALOG_HEADER
        + "2024-01-01T10:00:00Z,2024-01-01T10:10:00Z,,M1.0,100\n"
        + "2024-01-02T10:00:00Z,,2024-01-02T10:30:00Z,M1.0,100\n"
        + ",2024-01-03T10:10:00Z,2024-01-03T10:30:00Z,M1.0,100\n",
    )
    matcher = EventMatcher(path)
    for column in ("start_time", "peak_time", "end_time"):
        assert matcher._catalog[column].dt.tz is None
        assert int(matcher._catalog[column].isna().sum()) == 1
    assert len(matcher.query(pd.Timestamp("2024-01-01 09:00"), 24, event_time="start")) == 1
    assert len(matcher.query(pd.Timestamp("2024-01-02 09:00"), 24, event_time="peak")) == 0


def test_offset_normalization_is_not_a_time_scale_correction():
    from flare_indexer.events import _to_naive_utc
    # A naive value is kept exactly as written: no TAI-to-UTC (leap-second) shift.
    naive = pd.Timestamp("2024-01-01 10:00:37")
    assert _to_naive_utc(naive) is naive
    # An offset value only changes the calendar reading of the same instant.
    assert _to_naive_utc(pd.Timestamp("2024-01-01 15:00:37+05:00")) == pd.Timestamp("2024-01-01 10:00:37")


# --- 1b. a number means hours only for prediction_window -------------------

def test_prediction_window_number_means_hours_in_builder_and_query(tmp_path, catalog):
    index = _write(tmp_path, "index.csv", "timestamp\n2024-01-01 09:00:00\n")
    # Flare starts/peaks at 10:00: a 1-hour window [09:00, 10:00) misses it, a 2-hour one has it.
    assert _labels(DatasetBuilder(1, BinaryThresholdStrategy(), event_time="peak"), index, catalog) == [0]
    assert _labels(DatasetBuilder(2, BinaryThresholdStrategy(), event_time="peak"), index, catalog) == [1]
    assert _labels(DatasetBuilder("2h", BinaryThresholdStrategy(), event_time="peak"), index, catalog) == [1]
    matcher = EventMatcher(catalog)
    assert matcher.query(T0, 1, event_time="peak") == []
    assert len(matcher.query(T0, 2, event_time="peak")) == 1
    assert len(matcher.query(T0, 1.5, event_time="peak")) == 1


@pytest.mark.parametrize("name", ["cadence", "observation_window", "sliding_window"])
@pytest.mark.parametrize("number", [60, 60.0, np.int64(60), 3600])
def test_time_based_durations_reject_bare_numbers(name, number):
    params = {"cadence": "1h", "observation_window": "2h", "sliding_window": "1h", name: number}
    with pytest.raises(ValueError, match=f"{name} must be a duration with a unit .* got the bare number"):
        DatasetBuilder(24, BinaryThresholdStrategy(), **params)


def test_time_based_durations_accept_text_and_timedelta_forms():
    builder = DatasetBuilder(
        24, BinaryThresholdStrategy(),
        cadence="60min", observation_window=pd.Timedelta(hours=2),
        sliding_window=np.timedelta64(1, "h"),
    )
    assert builder.cadence == pd.Timedelta(hours=1)
    assert builder.observation_window == pd.Timedelta(hours=2)
    assert builder.sliding_window == pd.Timedelta(hours=1)
    assert builder.number_of_images == 2
    builder = DatasetBuilder(
        24, BinaryThresholdStrategy(),
        cadence=datetime.timedelta(hours=1), observation_window="2h", sliding_window="1h",
    )
    assert builder.cadence == pd.Timedelta(hours=1)

# --- 2b. all-blank timestamp column; catalog missing-time policy -----------

def test_all_blank_image_timestamp_column_raises_in_every_mode(tmp_path, catalog):
    full_disk_index = _write(tmp_path, "fd.csv", "timestamp,image_path\n,a\n,b\n")
    region_index = _write(tmp_path, "ar.csv", "timestamp,active_region\n,100\n,100\n")
    for builder in _builders_for_every_mode():
        with pytest.raises(ValueError, match=r"has 2 blank/missing timestamp\(s\) \(data row\(s\) 1, 2;"):
            builder.build(full_disk_index, catalog)
    for builder in _builders_for_every_mode(target="active_region"):
        with pytest.raises(ValueError, match=r"has 2 blank/missing timestamp\(s\) \(data row\(s\) 1, 2;"):
            builder.build(region_index, catalog)


def test_catalog_flare_with_blank_selected_time_never_matches_and_build_warns(tmp_path):
    path = _write(
        tmp_path, "catalog.csv",
        CATALOG_HEADER
        + ",2024-01-01 10:10:00,2024-01-01 10:30:00,X1.0,100\n"
        + "2024-01-05 10:00:00,2024-01-05 10:10:00,2024-01-05 10:30:00,C1.0,100\n",
    )
    index = _write(tmp_path, "index.csv", "timestamp\n2024-01-01 09:00:00\n")
    matcher = EventMatcher(path)
    assert matcher.query(T0, 24, event_time="start") == []
    assert len(matcher.query(T0, 24, event_time="peak")) == 1

    with pytest.warns(UserWarning, match="1 of 2 flare\\(s\\) in the catalog at .* have no start_time"):
        out = DatasetBuilder(24, BinaryThresholdStrategy(), event_time="start").build(index, path)
    assert out["label"].tolist() == [0]

    with warnings.catch_warnings():
        warnings.simplefilter("error")   # the blank start_time is irrelevant to peak matching
        out = DatasetBuilder(24, BinaryThresholdStrategy(), event_time="peak").build(index, path)
    assert out["label"].tolist() == [1]


def test_blank_end_time_does_not_affect_matching_or_warn(tmp_path):
    path = _write(tmp_path, "catalog.csv", CATALOG_HEADER + "2024-01-01 10:00:00,2024-01-01 10:10:00,,M1.0,100\n")
    index = _write(tmp_path, "index.csv", "timestamp\n2024-01-01 09:00:00\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for event_time in ("start", "peak"):
            out = DatasetBuilder(24, BinaryThresholdStrategy(), event_time=event_time).build(index, path)
            assert out["label"].tolist() == [1]


# --- 6. conflicting class columns ------------------------------------------

def test_catalog_with_both_class_and_goes_class_is_rejected_clearly(tmp_path):
    path = _write(
        tmp_path, "catalog.csv",
        "start_time,peak_time,end_time,class,goes_class,active_region\n"
        "2024-01-01 10:00:00,2024-01-01 10:10:00,2024-01-01 10:30:00,M1.0,M1.0,100\n",
    )
    with pytest.raises(ValueError, match="both a 'class' and a 'goes_class' column"):
        EventMatcher(path)


# --- 7. BinaryThresholdLabeler input contract ------------------------------

@pytest.mark.parametrize("value, expected", [
    (1e-5, 1),                    # exactly the M threshold
    (np.float64(1e-5), 1),
    (Fraction(1, 100000), 1),
    (9.999e-6, 0),
    (1.2e-4, 1),
    (0, 0),
    (0.0, 0),
    (np.int64(0), 0),
    (1, 1),
    (np.float32(1e-4), 1),
    (np.float32(1e-5), 0),        # float32(1e-5) is 9.99999974737875e-06 after float()
])
def test_binary_labeler_numeric_contract_and_threshold_equality(value, expected):
    assert BinaryThresholdLabeler("M").assign(value) == expected


def test_binary_labeler_threshold_equality_for_a_full_class():
    assert BinaryThresholdLabeler("C4.2").assign(4.2e-6) == 1
    assert BinaryThresholdLabeler("C4.2").assign(4.1999e-6) == 0


@pytest.mark.parametrize("value", [
    True, False, np.bool_(True), Decimal("1e-5"), "1e-5", None, [1e-5], pd.Timestamp("2024-01-01"),
])
def test_binary_labeler_rejects_unsupported_types(value):
    with pytest.raises(TypeError, match="expects a flux number"):
        BinaryThresholdLabeler("M").assign(value)


@pytest.mark.parametrize("value", [float("nan"), np.nan, float("inf"), -float("inf"), np.float64("inf")])
def test_binary_labeler_rejects_non_finite_values(value):
    with pytest.raises(ValueError, match="non-finite"):
        BinaryThresholdLabeler("M").assign(value)
