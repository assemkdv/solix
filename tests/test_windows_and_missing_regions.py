import warnings

import pandas as pd
import pytest

from flare_indexer import EventMatcher
from flare_indexer.builder import DatasetBuilder
from flare_indexer import MaxFluxReducer, BinaryThresholdLabeler, RegressionLabeler


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
        prediction_window=window, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"),
    )
    # Every spelling means the same 24 h window; the M1.0 starts exactly at
    # t + 24h, so [t, t + 24h) excludes it.
    assert builder.build(index_path, boundary_catalog)["label"].tolist() == [0]
    longer = DatasetBuilder(prediction_window="24h1min", reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"))
    assert longer.build(index_path, boundary_catalog)["label"].tolist() == [1]


def test_event_matcher_query_accepts_duration(boundary_catalog):
    matcher = EventMatcher(boundary_catalog)
    t = pd.Timestamp("2024-01-01 00:00:00")
    assert len(matcher.query(t, "24h1min")) == 1
    assert len(matcher.query(t, "24h")) == 0       # exactly t + 24h is excluded
    assert len(matcher.query(t, "23h59min59s")) == 0


@pytest.mark.parametrize("window", [0, -1, "0h", "-2h", "soon", None, True, float("inf"), float("nan")])
def test_invalid_prediction_window_raises(window):
    with pytest.raises(ValueError, match="prediction_window"):
        DatasetBuilder(prediction_window=window, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"))


# ── missing_active_region policy ─────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:.*flare\\(s\\) in the catalog:UserWarning")
def test_missing_active_region_default_labels_negative_and_warns(multi_region_index, ar_catalog):
    builder = DatasetBuilder(prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), target="active_region")
    with pytest.warns(UserWarning, match="1 image\\(s\\) have a missing active_region"):
        result = builder.build(multi_region_index, ar_catalog)
    assert result["label"].tolist() == [1, 0, 0]


def test_missing_active_region_drop_removes_rows(multi_region_index, ar_catalog):
    builder = DatasetBuilder(
        prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), target="active_region",
        missing_active_region="drop", include_metadata=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no missing-image warning in drop mode
        warnings.filterwarnings("ignore", message=".*in the catalog.*")
        result = builder.build(multi_region_index, ar_catalog)
    assert result["image_path"].tolist() == ["a0.jpg", "b0.jpg"]
    assert result["label"].tolist() == [1, 0]


@pytest.mark.filterwarnings("ignore:.*flare\\(s\\) in the catalog:UserWarning")
def test_missing_active_region_error_raises(multi_region_index, ar_catalog):
    builder = DatasetBuilder(
        prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), target="active_region",
        missing_active_region="error",
    )
    with pytest.raises(ValueError, match="missing active_region"):
        builder.build(multi_region_index, ar_catalog)


def test_invalid_missing_active_region_policy_raises():
    with pytest.raises(ValueError, match="missing_active_region"):
        DatasetBuilder(prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), missing_active_region="skip")


def test_catalog_flares_without_active_region_warn_in_active_region_mode(tmp_path, ar_catalog):
    index_path = _write(tmp_path / "index.csv", "timestamp,active_region\n2024-01-01 00:00:00,100\n")
    builder = DatasetBuilder(prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), target="active_region")
    with pytest.warns(UserWarning, match="1 of 2 flare\\(s\\) in the catalog .* have no active_region"):
        builder.build(index_path, ar_catalog)


def test_catalog_missing_active_region_does_not_warn_in_full_disk_mode(tmp_path, ar_catalog):
    index_path = _write(tmp_path / "index.csv", "timestamp\n2024-01-01 00:00:00\n")
    builder = DatasetBuilder(prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"))
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


@pytest.mark.filterwarnings("ignore:.*flare\\(s\\) in the catalog:UserWarning")
def test_missing_active_region_policies_for_sequences(tmp_path, ar_catalog):
    # Region 100 has 00, 01, 03 (02 has no region), so on a 1-hour grid only
    # (00, 01) is a complete sequence of region 100; the 02 image without a
    # region never fills region 100's gap.
    index_path = _ar_sequence_index(tmp_path)
    kwargs = dict(prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), target="active_region",
                  cadence="1h", observation_window="2h", sliding_window="1h")

    with pytest.warns(UserWarning, match="0 sequence\\(s\\) built only from such images"):
        default = DatasetBuilder(**kwargs).build(index_path, ar_catalog)
    assert default["active_region"].tolist() == ["100"]
    assert [ts[-1].hour for ts in default["timestamps"]] == [1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dropped = DatasetBuilder(missing_active_region="drop", **kwargs).build(index_path, ar_catalog)
    assert dropped.equals(default)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="1 image\\(s\\) .* have a missing active_region"):
            DatasetBuilder(missing_active_region="error", **kwargs).build(index_path, ar_catalog)


def test_missing_active_region_group_is_labeled_negative(tmp_path, ar_catalog):
    index_path = _write(
        tmp_path / "index.csv",
        "timestamp,active_region\n"
        "2024-01-01 00:00:00,\n"
        "2024-01-01 01:00:00,\n"
        "2024-01-01 02:00:00,100\n",
    )
    kwargs = dict(prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), target="active_region",
                  cadence="1h", observation_window="2h", sliding_window="1h")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = DatasetBuilder(**kwargs).build(index_path, ar_catalog)
    assert len(result) == 1
    assert pd.isna(result.iloc[0]["active_region"])
    assert result.iloc[0]["label"] == 0  # the M2.0 of region 100 is never matched
    assert any("1 sequence(s) built only from such images" in str(w.message) for w in caught)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert DatasetBuilder(missing_active_region="drop", **kwargs).build(index_path, ar_catalog).empty


# ── include_metadata ─────────────────────────────────────────────────────


def test_include_metadata_full_disk_single_image(tmp_path, ar_catalog):
    index_path = _write(
        tmp_path / "index.csv",
        "timestamp,image_path\n2024-01-01 00:00:00,a.jpg\n2024-01-03 00:00:00,b.jpg\n",
    )
    result = DatasetBuilder(
        prediction_window=24, reducer=MaxFluxReducer(), labeler=RegressionLabeler(), include_metadata=True,
    ).build(index_path, ar_catalog)
    assert list(result.columns) == ["timestamp", "label", "image_path"]
    assert result["image_path"].tolist() == ["a.jpg", "b.jpg"]
    assert result["label"].tolist() == [pytest.approx(1e-4), 0.0]


def test_include_metadata_keeps_regions_at_same_timestamp_distinguishable(multi_region_index, ar_catalog):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = DatasetBuilder(
            prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), target="active_region",
            include_metadata=True,
        ).build(multi_region_index, ar_catalog)
    assert list(result.columns) == ["timestamp", "active_region", "label", "image_path"]
    assert result["active_region"].iloc[:2].tolist() == ["100", "200"]
    assert pd.isna(result["active_region"].iloc[2])
    assert result["image_path"].tolist() == ["a0.jpg", "b0.jpg", "c0.jpg"]
    assert result["label"].tolist() == [1, 0, 0]


def test_active_region_column_is_kept_without_include_metadata(multi_region_index, ar_catalog):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = DatasetBuilder(
            prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), target="active_region",
        ).build(multi_region_index, ar_catalog)
    assert list(result.columns) == ["timestamp", "active_region", "label"]
    # Three patches share one timestamp; the region column tells them apart.
    regions = result["active_region"].tolist()
    assert regions[:2] == ["100", "200"] and pd.isna(regions[2])


def test_include_metadata_empty_index_keeps_columns(tmp_path, ar_catalog):
    index_path = _write(tmp_path / "index.csv", "timestamp,active_region,image_path\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = DatasetBuilder(
            prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), target="active_region",
            include_metadata=True,
        ).build(index_path, ar_catalog)
    assert list(result.columns) == ["timestamp", "active_region", "label", "image_path"]
    assert result.empty


# ── Other checks ─────────────────────────────────────────────────────────


def test_prediction_window_attribute_keeps_given_value():
    assert DatasetBuilder(prediction_window="24h", reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M")).prediction_window == "24h"


def test_missing_region_warning_points_at_caller(multi_region_index, ar_catalog):
    builder = DatasetBuilder(prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), target="active_region")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        builder.build(multi_region_index, ar_catalog)
    assert len(caught) == 2
    assert all(w.filename == __file__ for w in caught)


@pytest.mark.parametrize("kwargs", [
    {"include_metadata": True},
    {"include_metadata": True, "target": "active_region"},
    {"cadence": "1h", "observation_window": "2h", "sliding_window": "1h"},
    {"cadence": "1h", "observation_window": "2h", "sliding_window": "1h", "target": "active_region"},
])
def test_index_column_named_like_an_output_column_raises(tmp_path, ar_catalog, kwargs):
    index_path = _write(
        tmp_path / "index.csv",
        "timestamp,active_region,label\n2024-01-01 00:00:00,100,x\n2024-01-01 01:00:00,100,y\n",
    )
    with pytest.raises(ValueError, match="would overwrite output column"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            DatasetBuilder(prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), **kwargs).build(index_path, ar_catalog)


def test_label_column_in_index_is_fine_without_metadata(tmp_path, ar_catalog):
    index_path = _write(tmp_path / "index.csv", "timestamp,label\n2024-01-01 00:00:00,x\n")
    result = DatasetBuilder(prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M")).build(index_path, ar_catalog)
    assert result["label"].tolist() == [1]


# ── Missing regions: None in the output, file row numbers in messages ────

def _region_files(tmp_path, regions):
    index = tmp_path / "index.csv"
    pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=len(regions), freq="1h"),
                  "active_region": regions}).to_csv(index, index=False)
    catalog = tmp_path / "catalog.csv"
    catalog.write_text("start_time,peak_time,end_time,class,active_region\n"
                       "2024-01-01 00:30:00,2024-01-01 00:40:00,2024-01-01 00:50:00,M1.0,100\n")
    return index, catalog


@pytest.mark.filterwarnings("ignore:.*missing active_region:UserWarning")
@pytest.mark.parametrize("sequences", [False, True], ids=["single images", "sequences"])
def test_a_missing_region_is_none_on_every_pandas_version(tmp_path, sequences):
    index, catalog = _region_files(tmp_path, [100, None, 100])
    settings = dict(cadence="1h", observation_window="1h", sliding_window="1h") if sequences else {}
    result = DatasetBuilder(24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), target="active_region", **settings).build(index, catalog)
    assert result["active_region"].tolist() == ["100", None, "100"]


def test_missing_region_message_gives_the_row_in_the_file_after_cadence(tmp_path):
    # Data row 5 (04:00) has no region; cadence 2h keeps rows 1, 3 and 5.
    index, catalog = _region_files(tmp_path, [100, 100, 100, 100, None])
    builder = DatasetBuilder(24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), target="active_region", cadence="2h",
                             missing_active_region="error")
    with pytest.raises(ValueError, match="e.g. data row 5 \\(2024-01-01 04:00:00\\)"):
        builder.build(index, catalog)

