import pandas as pd
import pytest
from solix.builder import DatasetBuilder
from solix import MaxFluxReducer, BinaryThresholdLabeler, RegressionLabeler



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


def test_build_completely_empty_index_raises(tmp_path):
    index_path = tmp_path / "index.csv"
    index_path.write_text("")
    catalog_path = _write_valid_catalog(tmp_path)

    builder = DatasetBuilder(prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"))

    with pytest.raises(ValueError) as excinfo:
        builder.build(index_path, catalog_path)

    assert "empty" in str(excinfo.value).lower()


def test_build_preserves_instance_timestamp_order(tmp_path):
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


# ── Active-region single-instance labeling tests ───────────────────────────


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


def test_active_region_missing_instance_side_value_matches_nothing(tmp_path):
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

    # First instance has a real matching region -> catches the X1.0 flare ->
    # label 1. Second instance's active_region is missing -> must not silently
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


