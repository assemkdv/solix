"""
cadence for single images keeps only the images at start, start + cadence,
start + 2 * cadence, ... and labels them like any single image.
"""

import pandas as pd
import pytest

import flare_indexer as f

CATALOG = (
    "start_time,peak_time,end_time,class,active_region\n"
    "2024-01-01 01:10:00,2024-01-01 01:20:00,2024-01-01 01:30:00,M1.0,100\n"
    "2024-01-01 02:40:00,2024-01-01 02:50:00,2024-01-01 03:00:00,C2.0,200\n"
)


def _files(tmp_path, timestamps, regions=None):
    catalog = tmp_path / "catalog.csv"
    catalog.write_text(CATALOG)
    columns = {"timestamp": timestamps, "image_path": [f"img{i}.jpg" for i in range(len(timestamps))]}
    if regions is not None:
        columns["active_region"] = regions
    index = tmp_path / "index.csv"
    pd.DataFrame(columns).to_csv(index, index=False)
    return index, catalog


def _every(minutes, count, start="2024-01-01 00:00:00"):
    return [str(pd.Timestamp(start) + pd.Timedelta(minutes=minutes * i)) for i in range(count)]


def test_hourly_images_from_12_minute_list(tmp_path):
    index, catalog = _files(tmp_path, _every(12, 20))   # 00:00 .. 03:48
    result = f.FullDiskSingleImage("1h", f.BinaryThresholdStrategy("M"), cadence="1h",
                                   include_metadata=True).build(index, catalog)
    assert result["timestamp"].astype(str).tolist() == [f"2024-01-01 0{h}:00:00" for h in range(4)]
    assert result["image_path"].tolist() == ["img0.jpg", "img5.jpg", "img10.jpg", "img15.jpg"]
    assert result["label"].tolist() == [0, 1, 0, 0]   # M1.0 starts 01:10, inside [01:00, 02:00)
    assert list(result.columns) == ["timestamp", "label", "image_path"]


def test_labels_equal_the_same_images_without_cadence(tmp_path):
    index, catalog = _files(tmp_path, _every(12, 20))
    thinned = f.FullDiskSingleImage("3h", f.MaxFlareStrategy(), cadence="36min").build(index, catalog)
    everything = f.FullDiskSingleImage("3h", f.MaxFlareStrategy()).build(index, catalog)
    expected = everything[everything["timestamp"].isin(thinned["timestamp"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(thinned, expected)
    assert len(thinned) == 7   # 00:00, 00:36, ..., 03:36


def test_start_sets_the_grid(tmp_path):
    index, catalog = _files(tmp_path, _every(12, 20))
    result = f.FullDiskSingleImage("1h", f.BinaryThresholdStrategy("M"), cadence="1h",
                                   start="2024-01-01 00:36:00").build(index, catalog)
    assert result["timestamp"].astype(str).tolist() == ["2024-01-01 00:36:00", "2024-01-01 01:36:00",
                                                       "2024-01-01 02:36:00", "2024-01-01 03:36:00"]
    assert result["label"].tolist() == [1, 0, 0, 0]


def test_missing_grid_images_are_skipped(tmp_path):
    times = [t for t in _every(12, 20) if t != "2024-01-01 02:00:00"]
    index, catalog = _files(tmp_path, times)
    result = f.FullDiskSingleImage("1h", f.BinaryThresholdStrategy("M"), cadence="1h").build(index, catalog)
    assert result["timestamp"].astype(str).tolist() == ["2024-01-01 00:00:00", "2024-01-01 01:00:00",
                                                       "2024-01-01 03:00:00"]


def test_active_region_regions_share_one_grid(tmp_path):
    times = _every(30, 6) * 2
    index, catalog = _files(tmp_path, times, regions=["100"] * 6 + ["200"] * 6)
    result = f.ActiveRegionSingleImage("2h", f.BinaryThresholdStrategy("C"), cadence="1h",
                                       include_metadata=True).build(index, catalog)
    assert sorted(zip(result["timestamp"].astype(str), result["active_region"])) == [
        (f"2024-01-01 0{h}:00:00", region) for h in range(3) for region in ("100", "200")]
    labels = dict(zip(zip(result["timestamp"].dt.hour, result["active_region"]), result["label"]))
    assert labels[(0, "100")] == 1 and labels[(0, "200")] == 0 and labels[(1, "200")] == 1


@pytest.mark.parametrize("cadence, message", [
    ("6min", "finer than the image index"),
    ("30min", "not an exact multiple"),
])
def test_cadence_must_fit_the_images(tmp_path, cadence, message):
    index, catalog = _files(tmp_path, _every(12, 20))
    with pytest.raises(ValueError, match=message):
        f.FullDiskSingleImage("1h", f.BinaryThresholdStrategy("M"), cadence=cadence).build(index, catalog)


@pytest.mark.parametrize("times", [["2024-01-01 01:00:00"], []], ids=["one image", "no images"])
def test_one_image_or_none_with_a_cadence(tmp_path, times):
    index, catalog = _files(tmp_path, times)
    result = f.FullDiskSingleImage("1h", f.BinaryThresholdStrategy("M"), cadence="1h").build(index, catalog)
    assert result["label"].tolist() == [1] * len(times)


def test_start_must_be_an_image_time_and_needs_a_cadence(tmp_path):
    index, catalog = _files(tmp_path, _every(12, 20))
    with pytest.raises(ValueError, match="no exact matching image timestamp"):
        f.FullDiskSingleImage("1h", f.BinaryThresholdStrategy("M"), cadence="1h",
                              start="2024-01-01 00:05:00").build(index, catalog)
    with pytest.raises(ValueError, match="start needs a cadence"):
        f.FullDiskSingleImage("1h", f.BinaryThresholdStrategy("M"), start="2024-01-01 00:00:00")


def test_relative_flux_labels_allow_cadence():
    # cadence alone keeps single images, so the RXF labels stay allowed.
    pipeline = f.DatasetBuildingPipeline(f.MaxRelativeFluxReducer(), f.RegressionLabeler())
    assert not f.FullDiskSingleImage("24h", pipeline, cadence="1h").time_based


def test_verbose_names_the_cadence(tmp_path, capsys):
    index, catalog = _files(tmp_path, _every(12, 20))
    f.FullDiskSingleImage("1h", f.BinaryThresholdStrategy("M"), cadence="1h", verbose=True).build(index, catalog)
    lines = capsys.readouterr().out.splitlines()
    assert lines[2].endswith("full disk, single images every 1h")
    assert lines[3] == "Result: 4 rows; label 0: 3, label 1: 1"
