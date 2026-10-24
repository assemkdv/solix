import pandas as pd
import pytest

from flare_indexer.loaders import ImageIndexReport, adapt_goes_catalog, build_image_index_from_filenames

# ── build_image_index_from_filenames ────────────────────────────────────


def test_build_image_index_from_filename_list():
    paths = [
        "/data/hmi/2010/12/21/HMI.m2010.12.21_21.00.00.jpg",
        "/data/hmi/2010/12/21/HMI.m2010.12.21_20.00.00.jpg",
        "/data/hmi/2010/12/22/HMI.m2010.12.22_00.30.15.jpg",
    ]

    result = build_image_index_from_filenames(paths)

    assert list(result.columns) == ["timestamp", "image_path"]
    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
    # Sorted ascending by time.
    assert result["timestamp"].is_monotonic_increasing
    assert result["timestamp"].tolist() == [
        pd.Timestamp("2010-12-21 20:00:00"),
        pd.Timestamp("2010-12-21 21:00:00"),
        pd.Timestamp("2010-12-22 00:30:15"),
    ]
    # Original path preserved, associated with the right row after sorting.
    assert result.loc[result["timestamp"] == pd.Timestamp("2010-12-22 00:30:15"), "image_path"].iloc[0] == (
        "/data/hmi/2010/12/22/HMI.m2010.12.22_00.30.15.jpg"
    )


def test_build_image_index_varying_directory_depth():
    # Directory depth must not matter -- only the basename is parsed.
    paths = [
        "HMI.m2010.01.01_00.00.00.jpg",
        "/a/b/c/d/e/HMI.m2010.01.02_00.00.00.jpg",
    ]
    result = build_image_index_from_filenames(paths)
    assert len(result) == 2


def test_build_image_index_malformed_filename_raises():
    paths = [
        "/data/hmi/2010/12/21/HMI.m2010.12.21_21.00.00.jpg",
        "/data/hmi/not_a_valid_filename.jpg",
    ]

    with pytest.raises(ValueError) as excinfo:
        build_image_index_from_filenames(paths)

    assert "not_a_valid_filename.jpg" in str(excinfo.value)


def test_build_image_index_from_headerless_paths_file(tmp_path):
    paths_file = tmp_path / "paths.txt"
    paths_file.write_text(
        "/data/HMI.m2010.12.21_21.00.00.jpg\n"
        "/data/HMI.m2010.12.21_20.00.00.jpg\n"
    )

    result = build_image_index_from_filenames(paths_file)

    assert len(result) == 2
    assert result["timestamp"].is_monotonic_increasing


def test_build_image_index_strips_utf8_bom(tmp_path):
    paths_file = tmp_path / "paths.txt"
    # Raw bytes, so the BOM is attached to the first path (some exported lists have one).
    paths_file.write_bytes(
        "﻿/data/HMI.m2010.12.21_21.00.00.jpg\n/data/HMI.m2010.12.21_20.00.00.jpg\n".encode("utf-8")
    )

    result = build_image_index_from_filenames(paths_file)

    assert len(result) == 2
    # The BOM must not leak into the basename match or the preserved path.
    assert result["image_path"].iloc[0] == "/data/HMI.m2010.12.21_20.00.00.jpg"
    assert result["image_path"].iloc[1] == "/data/HMI.m2010.12.21_21.00.00.jpg"


def test_build_image_index_unsorted_paths_become_sorted():
    paths = [
        "HMI.m2010.12.21_23.00.00.jpg",
        "HMI.m2010.12.21_00.00.00.jpg",
        "HMI.m2010.12.21_12.00.00.jpg",
    ]

    result = build_image_index_from_filenames(paths)

    assert result["timestamp"].tolist() == [
        pd.Timestamp("2010-12-21 00:00:00"),
        pd.Timestamp("2010-12-21 12:00:00"),
        pd.Timestamp("2010-12-21 23:00:00"),
    ]


def test_build_image_index_writes_output_csv(tmp_path):
    paths = ["/data/HMI.m2010.12.21_21.00.00.jpg"]
    output_path = tmp_path / "image_index.csv"

    result = build_image_index_from_filenames(paths, output_path=output_path)

    assert output_path.exists()
    on_disk = pd.read_csv(output_path)
    assert list(on_disk.columns) == ["timestamp", "image_path"]
    assert len(on_disk) == 1
    assert result["image_path"].iloc[0] == paths[0]


# ── strict vs. tolerant malformed-row handling ──────────────────────────


def test_build_image_index_strict_invalid_date_in_correctly_shaped_filename_raises():
    # Shape matches (2-digit groups) but month=13 is not a real calendar month.
    paths = ["HMI.m2010.13.21_21.00.00.jpg"]

    with pytest.raises(ValueError) as excinfo:
        build_image_index_from_filenames(paths)

    assert "HMI.m2010.13.21_21.00.00.jpg" in str(excinfo.value)


def test_build_image_index_tolerant_excludes_and_reports_malformed_rows():
    paths = [
        "/data/HMI.m2010.12.21_21.00.00.jpg",
        "/data/notes/README.csv",  # not an image
        "HMI.m2010.13.21_21.00.00.jpg",  # shape-valid, invalid month
        "/data/HMI.m2010.12.21_20.00.00.jpg",
    ]

    df, report = build_image_index_from_filenames(paths, strict=False, return_report=True)

    assert len(df) == 2
    assert isinstance(report, ImageIndexReport)
    assert report.total_input_rows == 4
    assert report.valid_image_rows_before_dedup == 2
    assert report.excluded_row_count == 2
    assert "/data/notes/README.csv" in report.excluded_rows
    assert "HMI.m2010.13.21_21.00.00.jpg" in report.excluded_rows
    assert report.final_row_count == 2


# ── duplicate_policy ─────────────────────────────────────────────────────


_DUPLICATE_PATHS = [
    "HMI.m2010.12.21_20.00.00.jpg",
    "/other/dir/HMI.m2010.12.21_20.00.00.jpg",  # same timestamp, different path
    "HMI.m2010.12.21_21.00.00.jpg",
]


def test_build_image_index_duplicate_policy_error_raises():
    with pytest.raises(ValueError) as excinfo:
        build_image_index_from_filenames(_DUPLICATE_PATHS)  # default duplicate_policy="error"

    assert "duplicate" in str(excinfo.value).lower()


def test_build_image_index_duplicate_policy_first_keeps_first_encountered():
    result = build_image_index_from_filenames(_DUPLICATE_PATHS, duplicate_policy="first")

    assert len(result) == 2
    kept = result.loc[result["timestamp"] == pd.Timestamp("2010-12-21 20:00:00"), "image_path"].iloc[0]
    assert kept == "HMI.m2010.12.21_20.00.00.jpg"


def test_build_image_index_duplicate_policy_last_keeps_last_encountered():
    result = build_image_index_from_filenames(_DUPLICATE_PATHS, duplicate_policy="last")

    assert len(result) == 2
    kept = result.loc[result["timestamp"] == pd.Timestamp("2010-12-21 20:00:00"), "image_path"].iloc[0]
    assert kept == "/other/dir/HMI.m2010.12.21_20.00.00.jpg"


def test_build_image_index_duplicate_policy_report_fields():
    df, report = build_image_index_from_filenames(
        _DUPLICATE_PATHS, duplicate_policy="first", return_report=True
    )

    assert report.duplicate_timestamp_count == 1
    assert report.duplicate_policy == "first"
    assert report.final_row_count == 2
    assert len(report.duplicate_rows) == 2  # both rows sharing the duplicated timestamp


def test_build_image_index_invalid_duplicate_policy_raises():
    with pytest.raises(ValueError) as excinfo:
        build_image_index_from_filenames(["HMI.m2010.12.21_20.00.00.jpg"], duplicate_policy="skip")

    assert "duplicate_policy" in str(excinfo.value)


# ── adapt_goes_catalog ───────────────────────────────────────────────────


def _write_source_catalog(tmp_path, rows_csv: str):
    path = tmp_path / "goes_catalog.csv"
    path.write_text(rows_csv)
    return path


def test_adapt_goes_catalog_basic_schema(tmp_path):
    source = _write_source_catalog(
        tmp_path,
        "start_time,peak_time,end_time,goes_class,noaa_active_region,other_col\n"
        "2010-05-01 01:39:00,2010-05-01 01:45:00,2010-05-01 01:51:00,C5.7,11067.0,ignored\n",
    )

    result = adapt_goes_catalog(source)

    assert list(result.columns) == ["start_time", "peak_time", "end_time", "class", "active_region"]
    row = result.iloc[0]
    assert row["start_time"] == pd.Timestamp("2010-05-01 01:39:00")
    assert row["peak_time"] == pd.Timestamp("2010-05-01 01:45:00")
    assert row["end_time"] == pd.Timestamp("2010-05-01 01:51:00")
    assert row["class"] == "C5.7"
    assert row["active_region"] == "11067"


def test_adapt_goes_catalog_preserves_dates_crossing_midnight(tmp_path):
    # start_time and peak_time/end_time on different calendar dates must
    # each keep their own date, not get flattened onto peak's date.
    source = _write_source_catalog(
        tmp_path,
        "start_time,peak_time,end_time,goes_class,noaa_active_region\n"
        "2020-01-01 23:58:00,2020-01-02 00:04:00,2020-01-02 00:10:00,C1.0,14456.0\n",
    )

    result = adapt_goes_catalog(source)
    row = result.iloc[0]

    assert row["start_time"] == pd.Timestamp("2020-01-01 23:58:00")
    assert row["peak_time"] == pd.Timestamp("2020-01-02 00:04:00")
    assert row["end_time"] == pd.Timestamp("2020-01-02 00:10:00")
    assert row["start_time"].date() != row["peak_time"].date()


_REGION_HEADER = "start_time,peak_time,end_time,goes_class,noaa_active_region\n"


def test_adapt_goes_catalog_adds_leading_one_after_june_2002(tmp_path):
    source = _write_source_catalog(
        tmp_path,
        _REGION_HEADER
        + "2001-04-02 21:32:00,2001-04-02 21:51:00,2001-04-02 22:03:00,X20,9393.0\n"   # real 4-digit region
        + "2002-06-13 10:00:00,2002-06-13 10:10:00,2002-06-13 10:20:00,C1.0,9997.0\n"  # the day before
        + "2002-06-14 00:00:00,2002-06-14 00:10:00,2002-06-14 00:20:00,C2.0,1.0\n"     # from that day on
        + "2010-07-22 17:19:00,2010-07-22 17:23:00,2010-07-22 17:30:00,B3.8,1089.0\n"
        + "2023-06-01 12:00:00,2023-06-01 12:10:00,2023-06-01 12:20:00,M1.0,13324.0\n"
        + "2023-06-02 12:00:00,2023-06-02 12:10:00,2023-06-02 12:20:00,M2.0,\n",
    )
    with pytest.warns(UserWarning, match="2 flare\\(s\\) .* got the leading 1 back .* 0 of them"):
        result = adapt_goes_catalog(source)
    regions = result["active_region"].tolist()
    assert regions[:5] == ["9393", "9997", "10001", "11089", "13324"] and pd.isna(regions[5])


def test_adapt_goes_catalog_drops_the_copies_the_leading_one_creates(tmp_path):
    source = _write_source_catalog(
        tmp_path,
        _REGION_HEADER
        # The same flare twice, once with a four-digit number (as in some real flare lists).
        + "2010-07-22 17:19:00,2010-07-22 17:23:00,2010-07-22 17:30:00,B3.8,1089.0\n"
        + "2010-07-22 17:19:00,2010-07-22 17:23:00,2010-07-22 17:30:00,B3.8,11089.0\n"
        # Same start and class, different peak: a different flare, kept.
        + "2010-08-05 22:23:00,2010-08-05 22:27:00,2010-08-05 22:30:00,B5.7,1094.0\n"
        + "2010-08-05 22:23:00,2010-08-05 22:29:00,2010-08-05 22:30:00,B5.7,11094.0\n"
        # Same flare in two different regions: kept in both.
        + "2010-12-12 18:43:00,2010-12-12 18:49:00,2010-12-12 18:54:00,B5.9,1133.0\n"
        + "2010-12-12 18:43:00,2010-12-12 18:49:00,2010-12-12 18:54:00,B5.9,11135.0\n",
    )
    with pytest.warns(UserWarning, match="3 flare\\(s\\) .* 1 of them then equaled another row"):
        result = adapt_goes_catalog(source)
    assert result["active_region"].tolist() == ["11089", "11094", "11094", "11133", "11135"]
    assert result["peak_time"].tolist()[1:3] == [pd.Timestamp("2010-08-05 22:27"), pd.Timestamp("2010-08-05 22:29")]


def test_adapt_goes_catalog_without_four_digit_regions_does_not_warn(tmp_path, recwarn):
    source = _write_source_catalog(
        tmp_path,
        _REGION_HEADER + "2010-05-01 01:39:00,2010-05-01 01:45:00,2010-05-01 01:51:00,C5.7,11067.0\n",
    )
    adapt_goes_catalog(source)
    assert not [w for w in recwarn if "leading 1" in str(w.message)]


def test_adapt_goes_catalog_includes_fl_lon_fl_lat_when_present(tmp_path):
    source = _write_source_catalog(
        tmp_path,
        "start_time,peak_time,end_time,goes_class,noaa_active_region,fl_lon,fl_lat\n"
        "2010-05-01 01:39:00,2010-05-01 01:45:00,2010-05-01 01:51:00,C5.7,11067.0,-73.0,23.0\n",
    )

    result = adapt_goes_catalog(source)

    assert "fl_lon" in result.columns
    assert "fl_lat" in result.columns
    assert result.iloc[0]["fl_lon"] == -73.0
    assert result.iloc[0]["fl_lat"] == 23.0


def test_adapt_goes_catalog_omits_fl_lon_fl_lat_when_absent(tmp_path):
    source = _write_source_catalog(
        tmp_path,
        "start_time,peak_time,end_time,goes_class,noaa_active_region\n"
        "2010-05-01 01:39:00,2010-05-01 01:45:00,2010-05-01 01:51:00,C5.7,11067.0\n",
    )

    result = adapt_goes_catalog(source)

    assert "fl_lon" not in result.columns
    assert "fl_lat" not in result.columns


def test_adapt_goes_catalog_nan_active_region_becomes_none(tmp_path):
    source = _write_source_catalog(
        tmp_path,
        "start_time,peak_time,end_time,goes_class,noaa_active_region\n"
        "2010-05-01 01:39:00,2010-05-01 01:45:00,2010-05-01 01:51:00,C5.7,\n",
    )

    result = adapt_goes_catalog(source)

    assert result.iloc[0]["active_region"] is None


def test_adapt_goes_catalog_single_digit_vs_multi_digit_active_region(tmp_path):
    source = _write_source_catalog(
        tmp_path,
        "start_time,peak_time,end_time,goes_class,noaa_active_region\n"
        "2001-05-01 01:39:00,2001-05-01 01:45:00,2001-05-01 01:51:00,C5.7,7.0\n"
        "2010-05-02 01:39:00,2010-05-02 01:45:00,2010-05-02 01:51:00,M2.3,11067.0\n",
    )

    result = adapt_goes_catalog(source)

    assert result.iloc[0]["active_region"] == "7"
    assert result.iloc[1]["active_region"] == "11067"


def test_adapt_goes_catalog_missing_required_column_raises(tmp_path):
    # No noaa_active_region column at all.
    source = _write_source_catalog(
        tmp_path,
        "start_time,peak_time,end_time,goes_class\n"
        "2010-05-01 01:39:00,2010-05-01 01:45:00,2010-05-01 01:51:00,C5.7\n",
    )

    with pytest.raises(ValueError) as excinfo:
        adapt_goes_catalog(source)

    assert "noaa_active_region" in str(excinfo.value)


def test_adapt_goes_catalog_writes_output_csv(tmp_path):
    source = _write_source_catalog(
        tmp_path,
        "start_time,peak_time,end_time,goes_class,noaa_active_region\n"
        "2010-05-01 01:39:00,2010-05-01 01:45:00,2010-05-01 01:51:00,C5.7,11067.0\n",
    )
    output_path = tmp_path / "flare_catalog.csv"

    result = adapt_goes_catalog(source, output_path=output_path)

    assert output_path.exists()
    on_disk = pd.read_csv(output_path)
    assert list(on_disk.columns) == ["start_time", "peak_time", "end_time", "class", "active_region"]
    assert on_disk.iloc[0]["start_time"] == "2010-05-01 01:39:00"
    assert result.iloc[0]["start_time"] == pd.Timestamp("2010-05-01 01:39:00")


def test_adapt_goes_catalog_output_is_directly_loadable_by_event_matcher(tmp_path):
    # EventMatcher accepts the adapter's output as-is.
    from flare_indexer.events import EventMatcher

    source = _write_source_catalog(
        tmp_path,
        "start_time,peak_time,end_time,goes_class,noaa_active_region\n"
        "2020-01-01 23:58:00,2020-01-02 00:04:00,2020-01-02 00:10:00,M2.3,14456.0\n",
    )
    catalog_path = tmp_path / "flare_catalog.csv"
    adapt_goes_catalog(source, output_path=catalog_path)

    matcher = EventMatcher(catalog_path)
    results = matcher.query(
        pd.Timestamp("2020-01-01 23:00:00"), prediction_window=2
    )

    assert len(results) == 1
    assert results[0].peak_time == pd.Timestamp("2020-01-02 00:04:00")


def test_a_list_of_paths_follows_the_file_rules(tmp_path):
    listed = ["", "  ", " HMI.m2010.12.21_21.00.00.jpg "]
    text = tmp_path / "paths.txt"
    text.write_text("\n".join(listed) + "\n")
    from_list, report = build_image_index_from_filenames(listed, return_report=True)
    from_file = build_image_index_from_filenames(text)
    pd.testing.assert_frame_equal(from_list, from_file)
    assert report.total_input_rows == 1 and report.excluded_row_count == 0


def test_adapted_catalog_gives_none_for_a_missing_region(tmp_path):
    source = tmp_path / "goes.csv"
    source.write_text(
        "start_time,peak_time,end_time,goes_class,noaa_active_region\n"
        "2010-01-01 01:00:00,2010-01-01 01:10:00,2010-01-01 01:20:00,C1.0,11000\n"
        "2010-01-01 02:00:00,2010-01-01 02:10:00,2010-01-01 02:20:00,C2.0,\n"
    )
    assert adapt_goes_catalog(source)["active_region"].tolist() == ["11000", None]

