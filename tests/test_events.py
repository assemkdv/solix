import pandas as pd
import pytest
from solix.events import EventMatcher


def test_query_normalizes_missing_active_region_to_none():
    # pandas represents a missing active_region cell as float NaN once the
    # column contains any missing values — this must come out as None.
    matcher = EventMatcher.__new__(EventMatcher)
    matcher._catalog = pd.DataFrame({
        "peak_time": pd.to_datetime(["2024-01-01 00:30"]),
        "start_time": pd.to_datetime(["2024-01-01 00:20"]),
        "goes_class": ["M2.3"],
        "active_region": [float("nan")],
    })

    results = matcher.query(
        reference_time=pd.Timestamp("2024-01-01 00:00"),
        prediction_window=1,
    )

    assert results[0].active_region is None


def test_query_preserves_non_empty_active_region():
    # A present active_region value in a column that also has missing cells
    # arrives from pandas as a float (e.g. 4456.0) — it must be preserved
    # as a proper string ("4456"), not dropped or left as a float.
    matcher = EventMatcher.__new__(EventMatcher)
    matcher._catalog = pd.DataFrame({
        "peak_time": pd.to_datetime(["2024-01-01 00:30"]),
        "start_time": pd.to_datetime(["2024-01-01 00:20"]),
        "goes_class": ["M2.3"],
        "active_region": [4456.0],
    })

    results = matcher.query(
        reference_time=pd.Timestamp("2024-01-01 00:00"),
        prediction_window=1,
    )

    assert results[0].active_region == "4456"
    assert isinstance(results[0].active_region, str)


def test_load_catalog_does_not_print(tmp_path, capsys):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "start_time,peak_time,end_time,class,active_region\n"
        "2024-01-01 00:20:00,2024-01-01 00:30:00,2024-01-01 00:40:00,M2.3,4456\n"
    )

    EventMatcher(catalog_path)

    captured = capsys.readouterr()
    assert captured.out == ""


def test_load_catalog_parses_midnight(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "start_time,peak_time,end_time,class,active_region\n"
        "2024-01-23 00:00:00,2024-01-23 00:00:00,2024-01-23 00:05:00,C1.0,3559\n"
    )

    matcher = EventMatcher(catalog_path)

    assert matcher._catalog.loc[0, "peak_time"] == pd.Timestamp("2024-01-23 00:00:00")


def test_load_catalog_rejects_unparseable_time(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "start_time,peak_time,end_time,class,active_region\n"
        "2024-01-23 03:09:00,abc,2024-01-23 03:38:00,M5.1,3559\n"
    )

    with pytest.raises(ValueError) as excinfo:
        EventMatcher(catalog_path)

    assert "abc" in str(excinfo.value)


def test_query_active_region_stronger_flare_from_other_region_ignored():
    matcher = EventMatcher.__new__(EventMatcher)
    matcher._catalog = pd.DataFrame({
        "peak_time": pd.to_datetime(["2024-01-01 00:30", "2024-01-01 00:31"]),
        "start_time": pd.to_datetime(["2024-01-01 00:20", "2024-01-01 00:21"]),
        "goes_class": ["M5.1", "X9.0"],
        "active_region": ["3559", "9999"],
    })

    results = matcher.query(
        reference_time=pd.Timestamp("2024-01-01 00:00"),
        prediction_window=1,
        active_region="3559",
    )

    assert len(results) == 1
    assert results[0].goes_class == "M5.1"


def test_query_active_region_missing_catalog_active_region_does_not_match():
    matcher = EventMatcher.__new__(EventMatcher)
    matcher._catalog = pd.DataFrame({
        "peak_time": pd.to_datetime(["2024-01-01 00:30"]),
        "start_time": pd.to_datetime(["2024-01-01 00:20"]),
        "goes_class": ["X1.0"],
        "active_region": [float("nan")],
    })

    results = matcher.query(
        reference_time=pd.Timestamp("2024-01-01 00:00"),
        prediction_window=1,
        active_region="3559",
    )

    assert results == []


def test_load_catalog_missing_one_required_column(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "start_time,peak_time,end_time,class\n"
        "2024-01-01 00:20:00,2024-01-01 00:30:00,2024-01-01 00:40:00,M2.3\n"
    )

    with pytest.raises(ValueError) as excinfo:
        EventMatcher(catalog_path)

    message = str(excinfo.value)
    assert "active_region" in message
    assert str(catalog_path) in message


def test_load_catalog_missing_multiple_required_columns(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "date,peak,class\n"
        "2024-01-01,2024-01-01T00:30:00,M2.3\n"
    )

    with pytest.raises(ValueError) as excinfo:
        EventMatcher(catalog_path)

    message = str(excinfo.value)
    assert "start" in message
    assert "end" in message
    assert "active_region" in message


def test_load_catalog_completely_empty_file_raises(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text("")

    with pytest.raises(ValueError) as excinfo:
        EventMatcher(catalog_path)

    assert "empty" in str(excinfo.value).lower()


def test_load_catalog_header_only_is_valid_and_empty(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text("start_time,peak_time,end_time,class,active_region\n")

    matcher = EventMatcher(catalog_path)
    results = matcher.query(pd.Timestamp("2024-01-01"), prediction_window=24)

    assert results == []


# ── Catalog schema details ───────────────────────────────────────────────


def test_load_catalog_parses_full_timestamps(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "start_time,peak_time,end_time,class,active_region\n"
        "2024-01-23 03:09:00,2024-01-23 03:31:00,2024-01-23 03:38:00,M5.1,3559\n"
    )

    matcher = EventMatcher(catalog_path)

    assert matcher._catalog.loc[0, "peak_time"] == pd.Timestamp("2024-01-23 03:31:00")
    assert matcher._catalog.loc[0, "start_time"] == pd.Timestamp("2024-01-23 03:09:00")
    assert matcher._catalog.loc[0, "end_time"] == pd.Timestamp("2024-01-23 03:38:00")
    assert matcher._catalog.loc[0, "goes_class"] == "M5.1"


def test_load_catalog_preserves_dates_crossing_midnight(tmp_path):
    # A flare that starts just before midnight and peaks/ends just after it
    # keeps its different calendar dates.
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "start_time,peak_time,end_time,class,active_region\n"
        "2020-01-01 23:58:00,2020-01-02 00:04:00,2020-01-02 00:10:00,C1.0,4456\n"
    )

    matcher = EventMatcher(catalog_path)
    row = matcher._catalog.iloc[0]

    assert row["start_time"] == pd.Timestamp("2020-01-01 23:58:00")
    assert row["peak_time"] == pd.Timestamp("2020-01-02 00:04:00")
    assert row["end_time"] == pd.Timestamp("2020-01-02 00:10:00")
    assert row["start_time"].date() == pd.Timestamp("2020-01-01").date()
    assert row["peak_time"].date() == pd.Timestamp("2020-01-02").date()
    assert row["end_time"].date() == pd.Timestamp("2020-01-02").date()

    results = matcher.query(
        pd.Timestamp("2020-01-01 23:00:00"), prediction_window=2
    )
    assert len(results) == 1
    assert results[0].start_time == pd.Timestamp("2020-01-01 23:58:00")
    assert results[0].peak_time == pd.Timestamp("2020-01-02 00:04:00")
    assert results[0].end_time == pd.Timestamp("2020-01-02 00:10:00")


FULL_HEADER = "start_time,peak_time,end_time,class,active_region\n"
VALID_FULL_ROW = "2030-01-11 23:50:00,2030-01-12 00:05:00,2030-01-12 00:20:00,X1.5,90002\n"


@pytest.mark.parametrize("row, earlier, later", [
    ("2030-01-12 00:10:00,2030-01-12 00:05:00,2030-01-12 00:20:00,M1.0,90002\n", "start_time", "peak_time"),
    ("2030-01-11 23:50:00,2030-01-12 00:05:00,2030-01-11 00:20:00,M1.0,90002\n", "peak_time", "end_time"),
    # Blank peak: start and end are still compared.
    ("2030-01-12 00:30:00,,2030-01-12 00:20:00,M1.0,90002\n", "start_time", "end_time"),
])
def test_load_catalog_full_timestamp_rejects_times_out_of_order(tmp_path, row, earlier, later):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(FULL_HEADER + VALID_FULL_ROW + row)

    with pytest.raises(ValueError) as excinfo:
        EventMatcher(catalog_path)

    message = str(excinfo.value)
    assert "data row 2" in message
    assert f"{earlier} > {later}" in message
    assert "start <= peak <= end" in message


def test_load_catalog_full_timestamp_allows_equal_and_missing_times(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        FULL_HEADER
        + "2030-01-12 12:00:00,2030-01-12 12:00:00,2030-01-12 12:00:00,C3.0,90003\n"
        + "2030-01-13 01:00:00,2030-01-13 01:10:00,,C1.0,90004\n"
        + ",2030-01-14 02:10:00,2030-01-14 02:20:00,C2.0,90005\n"
    )

    catalog = EventMatcher(catalog_path)._catalog

    assert catalog["goes_class"].tolist() == ["C3.0", "C1.0", "C2.0"]
    assert catalog["end_time"].isna().tolist() == [False, True, False]
    assert catalog["start_time"].isna().tolist() == [False, False, True]


# ── flares are matched by start time ────────────────────────────────


def test_query_matches_flares_by_start_time():
    matcher = EventMatcher.__new__(EventMatcher)
    matcher._catalog = pd.DataFrame({
        "peak_time": pd.to_datetime(["2024-01-01 01:00"]),
        "start_time": pd.to_datetime(["2024-01-01 00:30"]),
        "goes_class": ["M2.3"],
        "active_region": [None],
    })

    # Window [00:00, 00:45) contains the start (00:30), not the peak (01:00).
    results = matcher.query(pd.Timestamp("2024-01-01 00:00"), prediction_window=0.75)
    assert [flare.goes_class for flare in results] == ["M2.3"]


# ── window boundaries: [t, t + w) ──────────────────────────


def _boundary_matcher():
    matcher = EventMatcher.__new__(EventMatcher)
    matcher._catalog = pd.DataFrame({
        "peak_time": pd.to_datetime([
            "2024-01-01 00:00:00",
            "2024-01-01 00:00:01",
            "2024-01-01 00:59:59",
            "2024-01-01 01:00:00",
        ]),
        "start_time": pd.to_datetime([
            "2024-01-01 00:00:00",  # exactly at t
            "2024-01-01 00:00:01",  # one second after t
            "2024-01-01 00:59:59",  # one second before t + window
            "2024-01-01 01:00:00",  # exactly at t + window
        ]),
        "goes_class": ["A1.0", "A1.0", "A1.0", "A1.0"],
        "active_region": [None, None, None, None],
    })
    return matcher


def test_window_includes_t_and_excludes_window_end():
    matcher = _boundary_matcher()
    window_start = pd.Timestamp("2024-01-01 00:00:00")

    results = matcher.query(window_start, prediction_window=1)
    start_times = sorted(r.start_time for r in results)

    # [t, t+1h): includes t itself and t+1h-1s, excludes t+1h exactly.
    assert start_times == [
        pd.Timestamp("2024-01-01 00:00:00"),
        pd.Timestamp("2024-01-01 00:00:01"),
        pd.Timestamp("2024-01-01 00:59:59"),
    ]


def test_query_returns_flares_within_window():
    # A small in-memory catalog, no CSV needed.
    matcher = EventMatcher.__new__(EventMatcher)  # create object without calling __init__
    matcher._catalog = pd.DataFrame({
        "peak_time": pd.to_datetime([
            "2024-01-01 00:30",
            "2024-01-01 01:30",
            "2024-01-01 00:59",
        ]),
        "start_time": pd.to_datetime([  # what matching uses
            "2024-01-01 00:20",  # inside a 1-hour window starting at 00:00
            "2024-01-01 01:20",  # outside — too late
            "2024-01-01 00:50",  # inside
        ]),
        "goes_class": ["M2.3", "X1.0", "C4.5"],
        "active_region": [None, None, None],
    }).sort_values("peak_time").reset_index(drop=True)

    # Query with a 1-hour prediction window starting at midnight
    results = matcher.query(
        reference_time=pd.Timestamp("2024-01-01 00:00"),
        prediction_window=1,
    )

    # Should find 2 flares (M2.3 and C4.5) — X1.0 is outside the window
    assert len(results) == 2
    assert results[0].goes_class == "M2.3"
    assert results[1].goes_class == "C4.5"
