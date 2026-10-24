"""
Input files with their own column names: build(index_columns=...,
catalog_columns=...) and add_relative_flux(catalog_columns=...,
xray_columns=...) rename them to the package's names. Nothing is guessed.
"""

import warnings
from pathlib import Path

import pandas as pd
import pytest

import flare_indexer as f

DATA = Path(__file__).resolve().parent.parent / "examples" / "data"
FULL_DISK = DATA / "full_disk_image_index.csv"
ACTIVE_REGION = DATA / "active_region_image_index.csv"
CATALOG = DATA / "flare_catalog.csv"
XRAY = DATA / "xray_flux.csv"

INDEX_NAMES = {"timestamp": "obs_time", "active_region": "noaa_ar"}
CATALOG_NAMES = {"start_time": "begin", "peak_time": "peak", "end_time": "end", "class": "goes_class",
                 "active_region": "noaa_ar"}


def _renamed(source, names, path):
    pd.read_csv(source).rename(columns=names).to_csv(path, index=False)
    return path


def _reverse(names):
    return {theirs: ours for ours, theirs in names.items()}


@pytest.fixture
def renamed_catalog(tmp_path):
    return _renamed(CATALOG, CATALOG_NAMES, tmp_path / "catalog.csv")


SEQUENCE = dict(cadence="6h", observation_window="12h", sliding_window="6h")


@pytest.mark.parametrize("make, index", [
    (lambda: f.FullDiskSingleImage("24h", f.BinaryThresholdStrategy("M"), include_metadata=True), FULL_DISK),
    (lambda: f.FullDiskSequence("24h", f.MaxFlareStrategy(), **SEQUENCE), FULL_DISK),
    (lambda: f.ActiveRegionSingleImage("24h", f.BinaryThresholdStrategy("M"), include_flares=True), ACTIVE_REGION),
    (lambda: f.ActiveRegionSequence("24h", f.BinaryThresholdStrategy("C"), **SEQUENCE), ACTIVE_REGION),
], ids=["full disk", "full disk sequence", "active region", "active region sequence"])
def test_own_column_names_give_the_same_labels(tmp_path, renamed_catalog, make, index):
    names = INDEX_NAMES if index == ACTIVE_REGION else {"timestamp": "obs_time"}
    own_index = _renamed(index, names, tmp_path / "index.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = make().build(index, CATALOG)
        result = make().build(own_index, renamed_catalog,
                              index_columns=_reverse(names), catalog_columns=_reverse(CATALOG_NAMES))
    pd.testing.assert_frame_equal(result, expected)


def test_other_columns_keep_their_own_names(tmp_path):
    index = _renamed(FULL_DISK, {"timestamp": "obs_time", "image_path": "file"}, tmp_path / "index.csv")
    result = f.FullDiskSingleImage("24h", f.BinaryThresholdStrategy("M"), include_metadata=True).build(
        index, CATALOG, index_columns={"obs_time": "timestamp"})
    assert list(result.columns) == ["timestamp", "label", "file"]


def test_without_the_settings_the_standard_names_are_required(renamed_catalog):
    builder = f.FullDiskSingleImage("24h", f.BinaryThresholdStrategy("M"))
    with pytest.raises(ValueError, match="missing required column\\(s\\): active_region, class, end_time"):
        builder.build(FULL_DISK, renamed_catalog)


def test_a_mapping_to_the_same_name_changes_nothing():
    builder = f.FullDiskSingleImage("24h", f.BinaryThresholdStrategy("M"))
    pd.testing.assert_frame_equal(builder.build(FULL_DISK, CATALOG, index_columns={"timestamp": "timestamp"}),
                                  builder.build(FULL_DISK, CATALOG))


@pytest.mark.parametrize("index_columns, message", [
    ({"obs_tme": "timestamp"}, "index_columns names column 'obs_tme', but Image index at .* has no such column. "
                               "Its columns: obs_time, image_path"),
    ({"obs_time": "time"}, "index_columns maps 'obs_time' to 'time', which is not a column name the package uses "
                           "here. Allowed: timestamp, active_region"),
    ({"obs_time": "timestamp", "image_path": "timestamp"}, "index_columns maps more than one column to 'timestamp'"),
])
def test_index_column_mistakes_raise(tmp_path, index_columns, message):
    index = _renamed(FULL_DISK, {"timestamp": "obs_time"}, tmp_path / "index.csv")
    with pytest.raises(ValueError, match=message):
        f.FullDiskSingleImage("24h", f.BinaryThresholdStrategy("M")).build(index, CATALOG, index_columns=index_columns)


def test_an_existing_column_is_never_overwritten(tmp_path):
    index = tmp_path / "index.csv"
    pd.read_csv(FULL_DISK).assign(obs_time=lambda d: d["timestamp"]).to_csv(index, index=False)
    with pytest.raises(ValueError, match="already has a column 'timestamp'"):
        f.FullDiskSingleImage("24h", f.BinaryThresholdStrategy("M")).build(
            index, CATALOG, index_columns={"obs_time": "timestamp"})


def test_swapping_two_columns_is_allowed(tmp_path):
    # Renaming a away and b onto a's name does not overwrite anything.
    catalog = _renamed(CATALOG, {"start_time": "peak_time", "peak_time": "start_time"}, tmp_path / "catalog.csv")
    builder = f.FullDiskSingleImage("24h", f.BinaryThresholdStrategy("M"))
    result = builder.build(FULL_DISK, catalog, catalog_columns={"peak_time": "start_time", "start_time": "peak_time"})
    pd.testing.assert_frame_equal(result, builder.build(FULL_DISK, CATALOG))


def test_catalog_column_mistake_names_the_catalog(renamed_catalog):
    with pytest.raises(ValueError, match="catalog_columns names column 'Class', but Flare catalog at .* has no such"):
        f.FullDiskSingleImage("24h", f.BinaryThresholdStrategy("M")).build(
            FULL_DISK, renamed_catalog, catalog_columns={"Class": "class"})


@pytest.mark.parametrize("bad", [["obs_time"], "obs_time", {"obs_time": 1}])
def test_the_setting_must_be_a_dict_of_names(bad):
    with pytest.raises(TypeError, match="index_columns must"):
        f.FullDiskSingleImage("24h", f.BinaryThresholdStrategy("M")).build(FULL_DISK, CATALOG, index_columns=bad)


def test_add_relative_flux_with_own_column_names(tmp_path, renamed_catalog):
    xray = _renamed(XRAY, {"time": "time_tag", "flux": "xrsb_flux"}, tmp_path / "xray.csv")
    expected = f.add_relative_flux(CATALOG, XRAY)
    result = f.add_relative_flux(renamed_catalog, xray, catalog_columns=_reverse(CATALOG_NAMES),
                                 xray_columns={"time_tag": "time", "xrsb_flux": "flux"})
    pd.testing.assert_series_equal(result["rxfi"], expected["rxfi"])
    # The result uses the package's names, so it goes straight to build().
    assert {"start_time", "peak_time", "end_time", "class", "active_region", "rxfi"} <= set(result.columns)


def test_add_relative_flux_xray_column_mistake_raises(tmp_path):
    xray = _renamed(XRAY, {"time": "time_tag"}, tmp_path / "xray.csv")
    with pytest.raises(ValueError, match="xray_columns maps 'time_tag' to 'timestamp', .* Allowed: time, flux"):
        f.add_relative_flux(CATALOG, xray, xray_columns={"time_tag": "timestamp"})
