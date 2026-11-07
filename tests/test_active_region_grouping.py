"""
Active-region sequences are built per region: several regions may share
timestamps, and a region's sequence only ever contains that region's instances.
"""

import warnings

import pandas as pd
import pytest


from solix.builder import DatasetBuilder
from solix import MaxFluxReducer, BinaryThresholdLabeler, RegressionLabeler


def _write(path, text):
    path.write_text(text)
    return path


@pytest.fixture
def catalog(tmp_path):
    # Region 100: M2.0 starting 05:00. Region 200: C5.0 starting 04:00.
    return _write(
        tmp_path / "catalog.csv",
        "start_time,peak_time,end_time,class,active_region\n"
        "2024-01-01 05:00:00,2024-01-01 05:10:00,2024-01-01 05:20:00,M2.0,100\n"
        "2024-01-01 04:00:00,2024-01-01 04:10:00,2024-01-01 04:20:00,C5.0,200\n",
    )


def _hourly_index(tmp_path, regions_by_hour, name="index.csv"):
    """regions_by_hour: {hour: [region or None, ...]} -> CSV with image_path."""
    lines = ["timestamp,active_region,image_path"]
    for hour, regions in regions_by_hour.items():
        for region in regions:
            label = "na" if region is None else region
            lines.append(f"2024-01-01 {hour:02d}:00:00,{'' if region is None else region},p_{label}_{hour:02d}.jpg")
    return _write(tmp_path / name, "\n".join(lines) + "\n")


def _build(index_path, catalog_path, **kwargs):
    kwargs.setdefault("prediction_window", 24)
    kwargs.setdefault("reducer", MaxFluxReducer())
    kwargs.setdefault("labeler", BinaryThresholdLabeler("M"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DatasetBuilder(target="active_region", **kwargs).build(index_path, catalog_path)


TIME_BASED = {"cadence": "1h", "observation_window": "3h", "sliding_window": "1h"}


def test_simultaneous_regions_get_separate_sequences(tmp_path, catalog):
    index = _hourly_index(tmp_path, {h: [100, 200] for h in range(4)})
    result = _build(index, catalog, **TIME_BASED)

    assert result["active_region"].tolist() == ["100", "200", "100", "200"]
    assert result["sequence_start"].tolist() == [pd.Timestamp("2024-01-01 00:00")] * 2 + [pd.Timestamp("2024-01-01 01:00")] * 2
    # Each sequence holds only its own region's instances.
    for region, paths in zip(result["active_region"], result["image_path"]):
        assert all(path.startswith(f"p_{region}_") for path in paths)
    # Region 100's M2.0 counts only for region 100; region 200's flare is C-class.
    assert result["label"].tolist() == [1, 0, 1, 0]


def test_missing_frame_in_one_region_does_not_affect_another(tmp_path, catalog):
    # Region 200 has no instance at 02:00.
    index = _hourly_index(tmp_path, {0: [100, 200], 1: [100, 200], 2: [100], 3: [100, 200], 4: [100, 200]})
    result = _build(index, catalog, **TIME_BASED)

    by_region = result.groupby("active_region")["sequence_start"].apply(list).to_dict()
    assert by_region["100"] == [pd.Timestamp(f"2024-01-01 {h:02d}:00") for h in (0, 1, 2)]
    # Every 3-hour window of region 200 needs 02:00, so none is emitted.
    assert "200" not in by_region


def test_time_based_report_lists_skips_per_region(tmp_path, catalog):
    index = _hourly_index(tmp_path, {0: [100, 200], 1: [100, 200], 2: [100], 3: [100, 200], 4: [100, 200]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, report = DatasetBuilder(
            prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), target="active_region", **TIME_BASED,
        ).build(index, catalog, return_report=True)

    assert report.region_count == 2
    assert report.total_candidate_sequences == 6
    assert report.emitted_sequences == 3
    assert report.skipped_sequences == 3
    assert report.skipped_sequence_regions == ["200", "200", "200"]
    assert report.missing_timestamps == [pd.Timestamp("2024-01-01 02:00")] * 3
    assert report.inferred_source_cadence == pd.Timedelta("1h")


def test_explicit_start_gives_all_regions_one_grid(tmp_path, catalog):
    # Region 200 first appears at 01:00; with start=00:00 and a 2-hour slide
    # its grid points are 02:00, 04:00, ... (never 01:00).
    index = _hourly_index(tmp_path, {0: [100], 1: [100, 200], 2: [100, 200], 3: [100, 200], 4: [100, 200]})
    result = _build(index, catalog, start="2024-01-01 00:00", cadence="1h", observation_window="2h", sliding_window="2h")

    by_region = result.groupby("active_region")["sequence_start"].apply(list).to_dict()
    assert by_region["100"] == [pd.Timestamp("2024-01-01 00:00"), pd.Timestamp("2024-01-01 02:00")]
    assert by_region["200"] == [pd.Timestamp("2024-01-01 02:00")]


def test_without_start_each_region_starts_at_its_first_instance(tmp_path, catalog):
    index = _hourly_index(tmp_path, {0: [100], 1: [100, 200], 2: [100, 200], 3: [100, 200]})
    result = _build(index, catalog, cadence="1h", observation_window="2h", sliding_window="2h")

    by_region = result.groupby("active_region")["sequence_start"].apply(list).to_dict()
    assert by_region["100"] == [pd.Timestamp("2024-01-01 00:00"), pd.Timestamp("2024-01-01 02:00")]
    assert by_region["200"] == [pd.Timestamp("2024-01-01 01:00")]


def test_explicit_start_must_be_an_instance_time(tmp_path, catalog):
    index = _hourly_index(tmp_path, {h: [100, 200] for h in range(3)})
    with pytest.raises(ValueError, match="no exact matching instance timestamp"):
        _build(index, catalog, start="2024-01-01 00:30", **TIME_BASED)


def test_duplicate_timestamp_within_a_region_raises(tmp_path, catalog):
    index = _hourly_index(tmp_path, {0: [100, 200], 1: [100, 100, 200], 2: [100, 200]})
    with pytest.raises(ValueError, match="duplicate timestamp\\(s\\) within active_region 100: 2024-01-01 01:00:00"):
        _build(index, catalog, **TIME_BASED)


def test_region_index_sorted_by_region_then_time_is_accepted(tmp_path, catalog):
    lines = ["timestamp,active_region"]
    lines += [f"2024-01-01 {h:02d}:00:00,200" for h in range(3)]
    lines += [f"2024-01-01 {h:02d}:00:00,100" for h in range(3)]
    index = _write(tmp_path / "index.csv", "\n".join(lines) + "\n")
    result = _build(index, catalog, **TIME_BASED)
    assert result["active_region"].tolist() == ["100", "200"]


def test_missing_region_rows_never_join_a_region_sequence(tmp_path, catalog):
    # At 01:00 one instance has no region. Region 100 is complete without it.
    index = _hourly_index(tmp_path, {0: [100], 1: [100, None], 2: [100]})
    with pytest.warns(UserWarning, match="1 instance\\(s\\) have a missing active_region"):
        result = DatasetBuilder(
            prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), target="active_region", **TIME_BASED,
        ).build(index, catalog)
    assert result["active_region"].tolist() == ["100"]
    assert "p_na_01.jpg" not in result["image_path"].iloc[0]


def test_several_unknown_instances_at_one_time_need_drop(tmp_path, catalog):
    index = _hourly_index(tmp_path, {0: [100, None, None], 1: [100, None, None], 2: [100, None, None]})
    with pytest.raises(ValueError, match="several instances without an active_region at the same time.*missing_active_region='drop'"):
        _build(index, catalog, **TIME_BASED)
    dropped = _build(index, catalog, missing_active_region="drop", **TIME_BASED)
    assert dropped["active_region"].tolist() == ["100"]


def test_missing_region_error_policy_names_rows(tmp_path, catalog):
    index = _hourly_index(tmp_path, {0: [100], 1: [100, None], 2: [100]})
    with pytest.raises(ValueError, match="1 instance\\(s\\) in .* have a missing active_region \\(e.g. data row 3 "):
        _build(index, catalog, missing_active_region="error", **TIME_BASED)


def test_max_flux_label_per_region(tmp_path, catalog):
    index = _hourly_index(tmp_path, {h: [100, 200] for h in range(3)})
    result = _build(index, catalog, reducer=MaxFluxReducer(), labeler=RegressionLabeler(), **TIME_BASED)
    assert result.set_index("active_region")["label"].to_dict() == {"100": pytest.approx(2e-5), "200": pytest.approx(5e-6)}


def test_single_region_gives_one_sequence_per_start(tmp_path, catalog):
    # One region with unique timestamps gives one sequence per start time.
    index = _hourly_index(tmp_path, {h: [100] for h in range(5)})
    result = _build(index, catalog, **TIME_BASED)
    assert result["sequence_start"].tolist() == [pd.Timestamp(f"2024-01-01 {h:02d}:00") for h in range(3)]
    assert result["label"].tolist() == [1, 1, 1]


def test_full_disk_sequences_still_reject_duplicate_timestamps(tmp_path, catalog):
    index = _hourly_index(tmp_path, {h: [100, 200] for h in range(3)})
    with pytest.raises(ValueError, match="contains duplicate timestamp\\(s\\): "):
        DatasetBuilder(prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), **TIME_BASED).build(index, catalog)
