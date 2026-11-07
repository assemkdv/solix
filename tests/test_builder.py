import pandas as pd
import pytest
from flare_indexer.builder import DatasetBuilder
from flare_indexer import MaxFluxReducer, BinaryThresholdLabeler, RegressionLabeler



def _write_valid_catalog(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "start_time,peak_time,end_time,class,active_region\n"
        "2024-01-01 00:20:00,2024-01-01 00:30:00,2024-01-01 00:40:00,M2.3,4456\n"
    )
    return catalog_path


def _write_two_flare_catalog(tmp_path):
    # Flare A: M5.0 starts 2024-01-01T01:50 (qualifies M threshold, flux=5e-5)
    # Flare B: C3.0 starts 2024-01-03T01:50 (does not qualify, flux=3e-6)
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "start_time,peak_time,end_time,class,active_region\n"
        "2024-01-01 01:50:00,2024-01-01 02:00:00,2024-01-01 02:10:00,M5.0,4456\n"
        "2024-01-03 01:50:00,2024-01-03 02:00:00,2024-01-03 02:10:00,C3.0,4456\n"
    )
    return catalog_path


def test_build_missing_timestamp_column_raises(tmp_path):
    index_path = tmp_path / "index.csv"
    index_path.write_text("not_timestamp\n2024-01-01T00:00:00\n")
    catalog_path = _write_valid_catalog(tmp_path)

    builder = DatasetBuilder(prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"))

    with pytest.raises(ValueError) as excinfo:
        builder.build(index_path, catalog_path)

    message = str(excinfo.value)
    assert "timestamp" in message
    assert str(index_path) in message


def test_build_completely_empty_image_index_raises(tmp_path):
    index_path = tmp_path / "index.csv"
    index_path.write_text("")
    catalog_path = _write_valid_catalog(tmp_path)

    builder = DatasetBuilder(prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"))

    with pytest.raises(ValueError) as excinfo:
        builder.build(index_path, catalog_path)

    assert "empty" in str(excinfo.value).lower()


def test_build_produces_expected_timestamp_and_label_columns(tmp_path):
    catalog_path = _write_two_flare_catalog(tmp_path)
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "timestamp\n"
        "2024-01-01T00:00:00\n"
        "2024-01-03T00:00:00\n"
    )

    builder = DatasetBuilder(prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"))
    result = builder.build(index_path, catalog_path)

    assert list(result.columns) == ["timestamp", "label"]
    assert result["timestamp"].tolist() == [
        pd.Timestamp("2024-01-01T00:00:00"),
        pd.Timestamp("2024-01-03T00:00:00"),
    ]
    # Jan 1 window catches Flare A (M5.0, qualifies); Jan 3 window catches
    # only Flare B (C3.0, does not qualify).
    assert result["label"].tolist() == [1, 0]


def test_build_preserves_image_timestamp_order(tmp_path):
    catalog_path = _write_two_flare_catalog(tmp_path)
    index_path = tmp_path / "index.csv"
    # Deliberately out of chronological order.
    index_path.write_text(
        "timestamp\n"
        "2024-01-03T00:00:00\n"
        "2024-01-01T00:00:00\n"
    )

    builder = DatasetBuilder(prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"))
    result = builder.build(index_path, catalog_path)

    assert result["timestamp"].tolist() == [
        pd.Timestamp("2024-01-03T00:00:00"),
        pd.Timestamp("2024-01-01T00:00:00"),
    ]
    assert result["label"].tolist() == [0, 1]


def test_build_works_with_max_flux_labels(tmp_path):
    catalog_path = _write_two_flare_catalog(tmp_path)
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "timestamp\n"
        "2024-01-01T00:00:00\n"
        "2024-01-03T00:00:00\n"
    )

    builder = DatasetBuilder(prediction_window=24, reducer=MaxFluxReducer(), labeler=RegressionLabeler())
    result = builder.build(index_path, catalog_path)

    assert result["label"].tolist() == [5e-5, 3e-6]


# ── Active-region single-image labeling tests ───────────────────────────


def test_active_region_works_with_max_flux_labels(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "start_time,peak_time,end_time,class,active_region\n"
        "2024-01-23 02:50:00,2024-01-23 03:00:00,2024-01-23 03:10:00,M5.1,3559\n"
        "2024-01-23 02:50:00,2024-01-23 03:00:00,2024-01-23 03:10:00,X9.0,9999\n"
    )
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "timestamp,active_region\n"
        "2024-01-23T00:00:00,3559\n"
    )

    builder = DatasetBuilder(prediction_window=24, reducer=MaxFluxReducer(), labeler=RegressionLabeler(), target="active_region")
    result = builder.build(index_path, catalog_path)

    # The stronger X9.0 flare belongs to a different region and must be
    # excluded; the label should reflect only the same-region M5.1 flare.
    assert result["label"].tolist() == [5.1e-5]


def test_active_region_image_side_values_normalize_to_match_catalog(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "start_time,peak_time,end_time,class,active_region\n"
        "2024-01-23 02:50:00,2024-01-23 03:00:00,2024-01-23 03:10:00,M5.1,3559\n"
    )
    index_path = tmp_path / "index.csv"
    # Mixing a real value with a later missing row forces pandas to read
    # this column as float64 (3559 -> 3559.0) -- the same normalization
    # challenge already solved for the catalog side in _normalize_active_region.
    index_path.write_text(
        "timestamp,active_region\n"
        "2024-01-23T00:00:00,3559\n"
        "2024-01-25T00:00:00,\n"
    )

    builder = DatasetBuilder(prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), target="active_region")
    with pytest.warns(UserWarning, match="missing active_region"):
        result = builder.build(index_path, catalog_path)

    assert result["label"].tolist()[0] == 1


def test_active_region_missing_image_side_value_matches_nothing(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "start_time,peak_time,end_time,class,active_region\n"
        "2024-01-23 02:50:00,2024-01-23 03:00:00,2024-01-23 03:10:00,X1.0,3559\n"
    )
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "timestamp,active_region\n"
        "2024-01-23T00:00:00,3559\n"
        "2024-01-23T01:00:00,\n"
    )

    builder = DatasetBuilder(prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), target="active_region")
    with pytest.warns(UserWarning, match="missing active_region"):
        result = builder.build(index_path, catalog_path)

    # First image has a real matching region -> catches the X1.0 flare ->
    # label 1. Second image's active_region is missing -> must not silently
    # match anything, even though the same strong flare for region 3559 is
    # still inside its window too.
    assert result["label"].tolist() == [1, 0]


def test_active_region_requires_active_region_column(tmp_path):
    catalog_path = _write_valid_catalog(tmp_path)
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "timestamp\n"
        "2024-01-23T00:00:00\n"
    )

    builder = DatasetBuilder(prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), target="active_region")

    with pytest.raises(ValueError) as excinfo:
        builder.build(index_path, catalog_path)

    message = str(excinfo.value)
    assert "active_region" in message
    assert str(index_path) in message


def test_invalid_target_raises():
    with pytest.raises(ValueError) as excinfo:
        DatasetBuilder(prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), target="something_else")

    assert "target" in str(excinfo.value)


def test_full_disk_target_explicit_matches_default(tmp_path):
    catalog_path = _write_two_flare_catalog(tmp_path)
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "timestamp\n"
        "2024-01-01T00:00:00\n"
        "2024-01-03T00:00:00\n"
    )

    default_builder = DatasetBuilder(prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"))
    explicit_builder = DatasetBuilder(
        prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), target="full_disk"
    )

    default_result = default_builder.build(index_path, catalog_path)
    explicit_result = explicit_builder.build(index_path, catalog_path)

    pd.testing.assert_frame_equal(default_result, explicit_result)
    assert explicit_result["label"].tolist() == [1, 0]


# ── Active-region sequence labeling tests ────────────────────────────────

def test_active_region_sequence_stronger_flare_from_other_region_ignored(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "start_time,peak_time,end_time,class,active_region\n"
        "2024-01-01 08:40:00,2024-01-01 08:40:00,2024-01-01 08:45:00,M5.1,3559\n"
        "2024-01-01 08:40:00,2024-01-01 08:40:00,2024-01-01 08:45:00,X9.0,9999\n"
    )
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "timestamp,active_region\n"
        "2024-01-01T08:00:00,3559\n"
        "2024-01-01T08:12:00,3559\n"
        "2024-01-01T08:24:00,3559\n"
    )

    builder = DatasetBuilder(
        prediction_window=1,
        reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("X"),
        target="active_region",
        cadence="12min", observation_window="36min", sliding_window="12min",
    )
    result = builder.build(index_path, catalog_path)

    # The X9.0 flare belongs to a different region and must be excluded --
    # only the M5.1 (same region) is a candidate, which doesn't meet the
    # X-class threshold, so the label must be 0, not 1.
    assert result.iloc[0]["label"] == 0

def test_active_region_sequence_requires_active_region_column(tmp_path):
    catalog_path = _write_valid_catalog(tmp_path)
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "timestamp\n"
        "2024-01-01T08:00:00\n"
        "2024-01-01T08:12:00\n"
    )

    builder = DatasetBuilder(
        prediction_window=24,
        reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"),
        target="active_region",
        cadence="12min", observation_window="24min", sliding_window="12min",
    )

    with pytest.raises(ValueError) as excinfo:
        builder.build(index_path, catalog_path)

    message = str(excinfo.value)
    assert "active_region" in message
    assert str(index_path) in message

# ── Flares are matched by their start time ─────────────────────────────────────


def test_a_flare_counts_by_its_start_time(tmp_path):
    # A flare that starts inside the window but peaks just after it closes.
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "start_time,peak_time,end_time,class,active_region\n"
        "2024-01-01 08:30:00,2024-01-01 09:01:00,2024-01-01 09:05:00,M1.0,4456\n"
    )
    index_path = tmp_path / "index.csv"
    index_path.write_text("timestamp\n2024-01-01T08:00:00\n")

    result = DatasetBuilder(prediction_window=1, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M")).build(index_path, catalog_path)

    # start_time (08:30) falls inside [08:00, 09:00), so it counts although
    # peak_time (09:01) is after the window.
    assert result["label"].tolist() == [1]
