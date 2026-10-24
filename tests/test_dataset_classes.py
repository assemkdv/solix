"""
FullDiskSingleImage, FullDiskSequence, ActiveRegionSingleImage and
ActiveRegionSequence: each gives exactly the table of the matching engine
settings, and accepts only the settings of its own case.
"""

from pathlib import Path

import pandas as pd
import pytest

import flare_indexer as f
from flare_indexer.builder import DatasetBuilder

DATA = Path(__file__).resolve().parent.parent / "examples" / "data"
FULL_DISK = DATA / "full_disk_image_index.csv"
ACTIVE_REGION = DATA / "active_region_image_index.csv"
CATALOG = DATA / "flare_catalog.csv"
GRID = dict(cadence="1h", observation_window="3h", sliding_window="1h")
AR_GRID = dict(cadence="6h", observation_window="12h", sliding_window="6h")

pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.skipif(not DATA.is_dir(), reason="examples/ not available (installed package)"),
]

CASES = [
    (f.FullDiskSingleImage, {}, dict(target="full_disk"), FULL_DISK),
    (f.FullDiskSingleImage, dict(include_metadata=True, event_time="peak"),
     dict(target="full_disk", include_metadata=True, event_time="peak"), FULL_DISK),
    (f.FullDiskSequence, GRID, dict(target="full_disk", **GRID), FULL_DISK),
    (f.FullDiskSequence, dict(GRID, start="2024-03-01 02:00:00"),
     dict(target="full_disk", start="2024-03-01 02:00:00", **GRID), FULL_DISK),
    (f.ActiveRegionSingleImage, dict(include_metadata=True),
     dict(target="active_region", include_metadata=True), ACTIVE_REGION),
    (f.ActiveRegionSingleImage, dict(missing_active_region="drop"),
     dict(target="active_region", missing_active_region="drop"), ACTIVE_REGION),
    (f.ActiveRegionSequence, AR_GRID, dict(target="active_region", **AR_GRID), ACTIVE_REGION),
]


@pytest.mark.parametrize("cls, kwargs, engine_kwargs, index", CASES, ids=lambda v: getattr(v, "__name__", None))
@pytest.mark.parametrize("strategy", [f.BinaryThresholdStrategy("M"), f.MaxFlareStrategy()], ids=["binary", "max"])
def test_same_table_as_the_engine(cls, kwargs, engine_kwargs, index, strategy):
    expected = DatasetBuilder("24h", strategy, **engine_kwargs).build(index, CATALOG)
    result = cls("24h", strategy, **kwargs).build(index, CATALOG)
    pd.testing.assert_frame_equal(result, expected)


def test_sequence_classes_return_the_report():
    result, report = f.FullDiskSequence("24h", f.BinaryThresholdStrategy("M"), **GRID).build(
        FULL_DISK, CATALOG, return_report=True)
    assert report.emitted_sequences == len(result)


def test_options_reach_the_engine():
    labels = f.FullDiskSingleImage("24h", f.BinaryThresholdStrategy("M"), include_flares=True).build(FULL_DISK, CATALOG)
    assert list(labels.columns) == ["timestamp", "label", "flares"]
    sequences = f.ActiveRegionSequence("24h", f.MaxFlareStrategy(), include_flares=True, **AR_GRID).build(
        ACTIVE_REGION, CATALOG)
    assert "flares" in sequences.columns and set(sequences["active_region"].dropna()) == {"13600", "13601"}


@pytest.mark.parametrize("cls, kwargs", [
    (f.FullDiskSingleImage, dict(observation_window="3h")),
    (f.FullDiskSingleImage, dict(target="active_region")),
    (f.FullDiskSingleImage, dict(missing_active_region="drop")),
    (f.FullDiskSequence, dict(GRID, target="active_region")),
    (f.FullDiskSequence, dict(GRID, include_metadata=True)),
    (f.ActiveRegionSingleImage, dict(cadence="1h")),
    (f.ActiveRegionSequence, dict(AR_GRID, target="full_disk")),
])
def test_settings_of_another_case_are_refused(cls, kwargs):
    with pytest.raises(TypeError):
        cls("24h", f.BinaryThresholdStrategy("M"), **kwargs)


@pytest.mark.parametrize("cls", [f.FullDiskSequence, f.ActiveRegionSequence])
def test_sequence_classes_require_the_grid(cls):
    with pytest.raises(TypeError, match="observation_window"):
        cls("24h", f.BinaryThresholdStrategy("M"), cadence="1h", sliding_window="1h")


@pytest.mark.parametrize("cls, grid", [(f.FullDiskSequence, GRID), (f.ActiveRegionSequence, AR_GRID)])
def test_relative_flux_labels_stay_single_image_only(cls, grid):
    pipeline = f.DatasetBuildingPipeline(f.MaxRelativeFluxReducer(), f.RegressionLabeler())
    with pytest.raises(ValueError, match="single images only"):
        cls("24h", pipeline, **grid)
    f.FullDiskSingleImage("24h", pipeline)
    f.ActiveRegionSingleImage("24h", pipeline)
