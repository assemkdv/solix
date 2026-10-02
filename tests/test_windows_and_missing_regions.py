import warnings

import pandas as pd
import pytest

from flare_indexer import BinaryThresholdStrategy, DatasetBuilder, EventMatcher, MaxFlareStrategy


def _write(path, text):
    path.write_text(text)
    return path


@pytest.fixture
def boundary_catalog(tmp_path):
    # M1.0 starts exactly 24h after the 2024-01-01T00:00 image.
    return _write(
        tmp_path / "catalog.csv",
        "start_time,peak_time,end_time,class,active_region\n"
        "2024-01-02 00:00:00,2024-01-02 00:10:00,2024-01-02 00:20:00,M1.0,100\n",
    )


@pytest.fixture
def ar_catalog(tmp_path):
    return _write(
        tmp_path / "ar_catalog.csv",
        "start_time,peak_time,end_time,class,active_region\n"
        "2024-01-01 05:00:00,2024-01-01 05:10:00,2024-01-01 05:20:00,M2.0,100\n"
        "2024-01-01 06:00:00,2024-01-01 06:10:00,2024-01-01 06:20:00,X1.0,\n",
    )


@pytest.fixture
def multi_region_index(tmp_path):
    return _write(
        tmp_path / "index.csv",
        "timestamp,active_region,image_path\n"
        "2024-01-01 00:00:00,100,a0.jpg\n"
        "2024-01-01 00:00:00,200,b0.jpg\n"
        "2024-01-01 00:00:00,,c0.jpg\n",
    )


# ── prediction_window as a duration ──────────────────────────────────────


@pytest.mark.parametrize("window", [24, 24.0, "24h", "1 day", pd.Timedelta(hours=24)])
def test_prediction_window_accepts_hours_and_durations(tmp_path, boundary_catalog, window):
    index_path = _write(tmp_path / "index.csv", "timestamp\n2024-01-01 00:00:00\n")
    builder = DatasetBuilder(
        prediction_window=window, strategy=BinaryThresholdStrategy(),
        event_time="start", interval_mode="right_closed",
    )
    assert builder.build(index_path, boundary_catalog)["label"].tolist() == [1]


def test_legacy_23h59m59s_window_excludes_flare_at_exactly_24h(tmp_path, boundary_catalog):
    index_path = _write(tmp_path / "index.csv", "timestamp\n2024-01-01 00:00:00\n")
    builder = DatasetBuilder(
        prediction_window="23h59min59s", strategy=BinaryThresholdStrategy(),
        event_time="start", interval_mode="right_closed",
    )
    assert builder.build(index_path, boundary_catalog)["label"].tolist() == [0]


def test_event_matcher_query_accepts_duration(boundary_catalog):
    matcher = EventMatcher(boundary_catalog)
    t = pd.Timestamp("2024-01-01 00:00:00")
    assert len(matcher.query(t, "24h", event_time="start", interval_mode="right_closed")) == 1
    assert len(matcher.query(t, "23h59min59s", event_time="start", interval_mode="right_closed")) == 0


@pytest.mark.parametrize("window", [0, -1, "0h", "-2h", "soon", None, True])
def test_invalid_prediction_window_raises(window):
    with pytest.raises(ValueError, match="prediction_window"):
        DatasetBuilder(prediction_window=window, strategy=BinaryThresholdStrategy())


# ── missing_active_region policy ─────────────────────────────────────────


def test_missing_active_region_default_labels_negative_and_warns(multi_region_index, ar_catalog):
    builder = DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy(), target="active_region")
    with pytest.warns(UserWarning, match="1 image\\(s\\) have a missing active_region"):
        result = builder.build(multi_region_index, ar_catalog)
    assert result["label"].tolist() == [1, 0, 0]


def test_missing_active_region_drop_removes_rows(multi_region_index, ar_catalog):
    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(), target="active_region",
        missing_active_region="drop", include_metadata=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no missing-image warning in drop mode
        warnings.filterwarnings("ignore", message=".*in the catalog.*")
        result = builder.build(multi_region_index, ar_catalog)
    assert result["image_path"].tolist() == ["a0.jpg", "b0.jpg"]
    assert result["label"].tolist() == [1, 0]


def test_missing_active_region_error_raises(multi_region_index, ar_catalog):
    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(), target="active_region",
        missing_active_region="error",
    )
    with pytest.raises(ValueError, match="missing active_region"):
        builder.build(multi_region_index, ar_catalog)


def test_invalid_missing_active_region_policy_raises():
    with pytest.raises(ValueError, match="missing_active_region"):
        DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy(), missing_active_region="skip")


def test_catalog_flares_without_active_region_warn_in_active_region_mode(tmp_path, ar_catalog):
    index_path = _write(tmp_path / "index.csv", "timestamp,active_region\n2024-01-01 00:00:00,100\n")
    builder = DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy(), target="active_region")
    with pytest.warns(UserWarning, match="1 of 2 flare\\(s\\) in the catalog .* have no active_region"):
        builder.build(index_path, ar_catalog)


def test_catalog_missing_active_region_does_not_warn_in_full_disk_mode(tmp_path, ar_catalog):
    index_path = _write(tmp_path / "index.csv", "timestamp\n2024-01-01 00:00:00\n")
    builder = DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert builder.build(index_path, ar_catalog)["label"].tolist() == [1]


def _ar_sequence_index(tmp_path):
    return _write(
        tmp_path / "seq_index.csv",
        "timestamp,active_region\n"
        "2024-01-01 00:00:00,100\n"
        "2024-01-01 01:00:00,100\n"
        "2024-01-01 02:00:00,\n"
        "2024-01-01 03:00:00,100\n",
    )


@pytest.mark.parametrize("mode_kwargs", [
    {"sequence_length": 2},
    {"cadence": "1h", "observation_window": "2h", "sliding_window": "1h"},
])
def test_missing_active_region_policies_for_sequences(tmp_path, ar_catalog, mode_kwargs):
    index_path = _ar_sequence_index(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = DatasetBuilder(
            prediction_window=24, strategy=BinaryThresholdStrategy(), target="active_region", **mode_kwargs,
        ).build(index_path, ar_catalog)
    assert len(default) == 3
    assert default["active_region"].iloc[0] == "100"
    assert default["active_region"].iloc[1:].isna().all()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dropped = DatasetBuilder(
            prediction_window=24, strategy=BinaryThresholdStrategy(), target="active_region",
            missing_active_region="drop", **mode_kwargs,
        ).build(index_path, ar_catalog)
    assert dropped["active_region"].tolist() == ["100"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="2 sequence\\(s\\) have a missing active_region"):
            DatasetBuilder(
                prediction_window=24, strategy=BinaryThresholdStrategy(), target="active_region",
                missing_active_region="error", **mode_kwargs,
            ).build(index_path, ar_catalog)


# ── include_metadata ─────────────────────────────────────────────────────


def test_include_metadata_full_disk_single_image(tmp_path, ar_catalog):
    index_path = _write(
        tmp_path / "index.csv",
        "timestamp,image_path\n2024-01-01 00:00:00,a.jpg\n2024-01-03 00:00:00,b.jpg\n",
    )
    result = DatasetBuilder(
        prediction_window=24, strategy=MaxFlareStrategy(), include_metadata=True,
    ).build(index_path, ar_catalog)
    assert list(result.columns) == ["timestamp", "label", "image_path"]
    assert result["image_path"].tolist() == ["a.jpg", "b.jpg"]
    assert result["label"].tolist() == [pytest.approx(1e-4), 0.0]


def test_include_metadata_keeps_regions_at_same_timestamp_distinguishable(multi_region_index, ar_catalog):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = DatasetBuilder(
            prediction_window=24, strategy=BinaryThresholdStrategy(), target="active_region",
            include_metadata=True,
        ).build(multi_region_index, ar_catalog)
    assert list(result.columns) == ["timestamp", "active_region", "label", "image_path"]
    assert result["active_region"].iloc[:2].tolist() == ["100", "200"]
    assert pd.isna(result["active_region"].iloc[2])
    assert result["image_path"].tolist() == ["a0.jpg", "b0.jpg", "c0.jpg"]
    assert result["label"].tolist() == [1, 0, 0]


def test_include_metadata_default_keeps_original_schema(multi_region_index, ar_catalog):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = DatasetBuilder(
            prediction_window=24, strategy=BinaryThresholdStrategy(), target="active_region",
        ).build(multi_region_index, ar_catalog)
    assert list(result.columns) == ["timestamp", "label"]


def test_include_metadata_empty_index_keeps_columns(tmp_path, ar_catalog):
    index_path = _write(tmp_path / "index.csv", "timestamp,active_region,image_path\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = DatasetBuilder(
            prediction_window=24, strategy=BinaryThresholdStrategy(), target="active_region",
            include_metadata=True,
        ).build(index_path, ar_catalog)
    assert list(result.columns) == ["timestamp", "active_region", "label", "image_path"]
    assert result.empty
