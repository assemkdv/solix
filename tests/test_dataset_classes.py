"""
FullDiskSingleInstance, FullDiskSequence, ActiveRegionSingleInstance and
ActiveRegionSequence: each gives exactly the table of the matching engine
settings, and accepts only the settings of its own case.
"""

from pathlib import Path

import pandas as pd
import pytest

import flare_indexer as f
from flare_indexer.builder import DatasetBuilder

DATA = Path(__file__).resolve().parent.parent / "examples" / "data"
FULL_DISK = DATA / "full_disk_index.csv"
ACTIVE_REGION = DATA / "active_region_index.csv"
CATALOG = DATA / "flare_catalog.csv"
GRID = dict(cadence="1h", observation_window="3h", sliding_window="1h")
AR_GRID = dict(cadence="6h", observation_window="12h", sliding_window="6h")

pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.skipif(not DATA.is_dir(), reason="examples/ not available (installed package)"),
]

CASES = [
    (f.FullDiskSingleInstance, {}, dict(target="full_disk"), FULL_DISK),
    (f.FullDiskSingleInstance, dict(include_metadata=True),
     dict(target="full_disk", include_metadata=True), FULL_DISK),
    (f.FullDiskSequence, GRID, dict(target="full_disk", **GRID), FULL_DISK),
    (f.FullDiskSequence, dict(GRID, start="2024-03-01 02:00:00"),
     dict(target="full_disk", start="2024-03-01 02:00:00", **GRID), FULL_DISK),
    (f.ActiveRegionSingleInstance, dict(include_metadata=True),
     dict(target="active_region", include_metadata=True), ACTIVE_REGION),
    (f.ActiveRegionSingleInstance, dict(missing_active_region="drop"),
     dict(target="active_region", missing_active_region="drop"), ACTIVE_REGION),
    (f.ActiveRegionSequence, AR_GRID, dict(target="active_region", **AR_GRID), ACTIVE_REGION),
]


LABELINGS = {
    "binary": dict(reducer=f.MaxFluxReducer(), labeler=f.BinaryThresholdLabeler("M")),
    "max": dict(reducer=f.MaxFluxReducer(), labeler=f.RegressionLabeler()),
}


@pytest.mark.parametrize("cls, kwargs, engine_kwargs, index", CASES, ids=lambda v: getattr(v, "__name__", None))
@pytest.mark.parametrize("labeling", list(LABELINGS.values()), ids=list(LABELINGS))
def test_same_table_as_the_engine(cls, kwargs, engine_kwargs, index, labeling):
    expected = DatasetBuilder("24h", **labeling, **engine_kwargs).build(index, CATALOG)
    result = cls("24h", **labeling, **kwargs).build(index, CATALOG)
    pd.testing.assert_frame_equal(result, expected)


def test_sequence_classes_return_the_report():
    result, report = f.FullDiskSequence("24h", reducer=f.MaxFluxReducer(), labeler=f.BinaryThresholdLabeler("M"), **GRID).build(
        FULL_DISK, CATALOG, return_report=True)
    assert report.emitted_sequences == len(result)


def test_options_reach_the_engine():
    labels = f.FullDiskSingleInstance("24h", reducer=f.MaxFluxReducer(), labeler=f.BinaryThresholdLabeler("M"), include_flares=True).build(FULL_DISK, CATALOG)
    assert list(labels.columns) == ["timestamp", "label", "flares"]
    sequences = f.ActiveRegionSequence("24h", reducer=f.MaxFluxReducer(), labeler=f.RegressionLabeler(), include_flares=True, **AR_GRID).build(
        ACTIVE_REGION, CATALOG)
    assert "flares" in sequences.columns and set(sequences["active_region"].dropna()) == {"13600", "13601"}


@pytest.mark.parametrize("cls, kwargs", [
    (f.FullDiskSingleInstance, dict(observation_window="3h")),
    (f.FullDiskSingleInstance, dict(target="active_region")),
    (f.FullDiskSingleInstance, dict(missing_active_region="drop")),
    (f.FullDiskSequence, dict(GRID, target="active_region")),
    (f.FullDiskSequence, dict(GRID, include_metadata=True)),
    (f.ActiveRegionSingleInstance, dict(observation_window="3h")),
    (f.ActiveRegionSequence, dict(AR_GRID, target="full_disk")),
])
def test_settings_of_another_case_are_refused(cls, kwargs):
    with pytest.raises(TypeError):
        cls("24h", reducer=f.MaxFluxReducer(), labeler=f.BinaryThresholdLabeler("M"), **kwargs)


@pytest.mark.parametrize("cls", [f.FullDiskSequence, f.ActiveRegionSequence])
def test_sequence_classes_require_the_grid(cls):
    with pytest.raises(TypeError, match="observation_window"):
        cls("24h", reducer=f.MaxFluxReducer(), labeler=f.BinaryThresholdLabeler("M"), cadence="1h", sliding_window="1h")


@pytest.mark.parametrize("cls, grid", [(f.FullDiskSequence, GRID), (f.ActiveRegionSequence, AR_GRID)])
def test_relative_flux_labels_stay_single_instance_only(cls, grid):
    labeling = dict(reducer=f.MaxRelativeFluxReducer(), labeler=f.RegressionLabeler())
    with pytest.raises(ValueError, match="single instances only"):
        cls("24h", **labeling, **grid)
    f.FullDiskSingleInstance("24h", **labeling)
    f.ActiveRegionSingleInstance("24h", **labeling)


@pytest.mark.parametrize("setting", ["include_metadata", "include_flares", "verbose"])
def test_on_off_settings_must_be_true_or_false(setting):
    with pytest.raises(ValueError, match=f"{setting} must be True or False"):
        f.FullDiskSingleInstance("24h", reducer=f.MaxFluxReducer(), labeler=f.BinaryThresholdLabeler("M"), **{setting: "yes"})
