"""
include_flares=True adds a "flares" column next to the label: the GOES
classes of the flares in each window, in order of start time.
"""

import pandas as pd
import pytest


from solix.builder import DatasetBuilder
from solix import MaxFluxReducer, BinaryThresholdLabeler, RegressionLabeler

CATALOG = (
    "start_time,peak_time,end_time,class,active_region\n"
    "2024-01-01 03:10:00,2024-01-01 03:20:00,2024-01-01 03:30:00,C2.0,13324\n"
    "2024-01-01 05:40:00,2024-01-01 05:50:00,2024-01-01 06:00:00,M1.0,13324\n"
    "2024-01-01 09:00:00,2024-01-01 09:05:00,2024-01-01 09:10:00,C5.0,13325\n"
)
HOURS = [f"2024-01-01 {h:02d}:00:00" for h in range(0, 12)]


def _files(tmp_path, timestamps, regions=None):
    catalog = tmp_path / "catalog.csv"
    catalog.write_text(CATALOG)
    index = tmp_path / "index.csv"
    columns = {"timestamp": timestamps}
    if regions is not None:
        columns["active_region"] = regions
    pd.DataFrame(columns).to_csv(index, index=False)
    return index, catalog


def test_default_output_is_unchanged(tmp_path):
    index, catalog = _files(tmp_path, HOURS[:2])
    result = DatasetBuilder("6h", reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M")).build(index, catalog)
    assert list(result.columns) == ["timestamp", "label"]


def test_single_instance_full_disk(tmp_path):
    index, catalog = _files(tmp_path, ["2024-01-01 00:00:00", "2024-01-01 05:00:00", "2024-01-01 10:00:00"])
    result = DatasetBuilder("6h", reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), include_flares=True).build(index, catalog)
    assert list(result.columns) == ["timestamp", "label", "flares"]
    assert result["label"].tolist() == [1, 1, 0]
    assert result["flares"].tolist() == [["C2.0", "M1.0"], ["M1.0", "C5.0"], []]


def test_window_edges_follow_the_label(tmp_path):
    # Instance at 03:10: the C2.0 starting at t counts; a 2h30 window ends at 05:40, which is excluded.
    index, catalog = _files(tmp_path, ["2024-01-01 03:10:00"])
    result = DatasetBuilder("2h30min", reducer=MaxFluxReducer(), labeler=RegressionLabeler(), include_flares=True).build(index, catalog)
    assert result["flares"].tolist() == [["C2.0"]]
    assert result["label"].tolist() == [2e-6]


def test_single_instance_active_region_lists_only_that_region(tmp_path):
    index, catalog = _files(tmp_path, ["2024-01-01 00:00:00"] * 2, regions=["13324", "13325"])
    result = DatasetBuilder(
        "12h", reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), target="active_region", include_flares=True, include_metadata=True,
    ).build(index, catalog)
    assert list(result.columns) == ["timestamp", "active_region", "label", "flares"]
    assert result["flares"].tolist() == [["C2.0", "M1.0"], ["C5.0"]]
    assert result["label"].tolist() == [1, 0]

def test_time_based_sequences(tmp_path):
    index, catalog = _files(tmp_path, HOURS)
    result = DatasetBuilder(
        "4h", reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), cadence="1h", observation_window="3h", sliding_window="3h",
        include_flares=True,
    ).build(index, catalog)
    assert result["sequence_end"].tolist()[:3] == [pd.Timestamp(f"2024-01-01 0{h}:00:00") for h in (2, 5, 8)]
    # 05:00 -> [05:00, 09:00): the C5.0 starting at 09:00 is at the excluded end.
    assert result["flares"].tolist()[:3] == [["C2.0", "M1.0"], ["M1.0"], ["C5.0"]]


def test_active_region_sequences(tmp_path):
    index, catalog = _files(tmp_path, HOURS[:3] * 2, regions=["13324"] * 3 + ["13325"] * 3)
    result = DatasetBuilder(
        "8h", reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), target="active_region", include_flares=True,
        cadence="1h", observation_window="3h", sliding_window="3h",
    ).build(index, catalog)
    assert result["active_region"].tolist() == ["13324", "13325"]
    assert result["flares"].tolist() == [["C2.0", "M1.0"], ["C5.0"]]


def test_index_column_named_flares_is_refused(tmp_path):
    index, catalog = _files(tmp_path, HOURS[:3])
    frame = pd.read_csv(index)
    frame["flares"] = "x"
    frame.to_csv(index, index=False)
    with pytest.raises(ValueError, match="flares"):
        DatasetBuilder("6h", reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), include_flares=True, include_metadata=True).build(index, catalog)
    with pytest.raises(ValueError, match="flares"):
        DatasetBuilder("6h", reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), cadence="1h", observation_window="2h", sliding_window="1h",
                       include_flares=True).build(index, catalog)


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_include_flares_must_be_a_bool(value):
    with pytest.raises(ValueError, match="include_flares"):
        DatasetBuilder("6h", reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), include_flares=value)
