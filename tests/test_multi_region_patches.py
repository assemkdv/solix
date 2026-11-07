"""
One instance can cover several NOAA regions (a HARP patch that lists more
than one region). Its active_region cell lists them with commas, and a
flare in any of them counts for the instance.
"""

import warnings

import pandas as pd
import pytest

from solix import (
    ActiveRegionSequence, ActiveRegionSingleInstance, BinaryThresholdLabeler, EventMatcher, MaxFluxReducer,
)
from solix.events import _normalize_active_region


def _write(path, text):
    path.write_text(text)
    return path


@pytest.fixture
def catalog(tmp_path):
    # The case from the lab's AR labels: HARP 384 covers NOAA 11160, 11161
    # and 11162, and has an M1.1 flare in 11161 on 2011-02-16 at 07:35.
    return _write(
        tmp_path / "catalog.csv",
        "start_time,peak_time,end_time,class,active_region\n"
        "2011-02-16 07:35:00,2011-02-16 07:44:00,2011-02-16 07:55:00,M1.1,11161\n"
        "2011-02-16 09:00:00,2011-02-16 09:10:00,2011-02-16 09:20:00,C2.0,11162\n"
        "2011-02-16 10:00:00,2011-02-16 10:10:00,2011-02-16 10:20:00,X1.0,11170\n",
    )


def _index(tmp_path, regions, hours=range(3)):
    lines = ["timestamp,active_region,file"]
    for hour in hours:
        for region in regions:
            lines.append(f'2011-02-16 {hour:02d}:00:00,"{region}",f_{hour}.jpg')
    return _write(tmp_path / "index.csv", "\n".join(lines) + "\n")


def _single(index, catalog, **kwargs):
    builder = ActiveRegionSingleInstance(
        "24h", reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"), **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return builder.build(index, catalog)


@pytest.mark.parametrize("text, expected", [
    ("11160,11161,11162", "11160,11161,11162"),
    (" 11160 , 11161.0,11162 ", "11160,11161,11162"),
    ("11160,11160,11161", "11160,11161"),
    ("11162,11160,11161", "11160,11161,11162"),
    ("9999,10001", "9999,10001"),                  # sorted by value, not as text
    ("11160,,11161,", "11160,11161"),
    ("11161,", "11161"),
    (",", None),
    (" , ", None),
])
def test_region_lists_are_normalized(text, expected):
    assert _normalize_active_region(text) == expected


def test_a_flare_in_any_listed_region_counts(tmp_path, catalog):
    result = _single(_index(tmp_path, ["11160,11161,11162"]), catalog)
    assert result["active_region"].tolist() == ["11160,11161,11162"] * 3
    assert result["label"].tolist() == [1, 1, 1]


def test_a_single_region_still_counts_only_its_own_flares(tmp_path, catalog):
    result = _single(_index(tmp_path, ["11160", "11161", "11162"]), catalog)
    by_region = result.groupby("active_region")["label"].max().to_dict()
    assert by_region == {"11160": 0, "11161": 1, "11162": 0}


def test_regions_outside_the_list_do_not_count(tmp_path, catalog):
    # The X1.0 flare is in 11170, which the patch does not cover.
    result = _single(_index(tmp_path, ["11160,11162"]), catalog)
    assert result["label"].tolist() == [0, 0, 0]


def test_query_returns_the_flares_of_every_listed_region(catalog):
    matcher = EventMatcher(catalog)
    flares = matcher.query(pd.Timestamp("2011-02-16 00:00"), "24h", active_region="11160, 11161,11162")
    assert sorted(f.active_region for f in flares) == ["11161", "11162"]
    assert matcher.query(pd.Timestamp("2011-02-16 00:00"), "24h", active_region="11161")[0].goes_class == "M1.1"


def test_a_blank_region_list_is_an_error_in_query(catalog):
    with pytest.raises(ValueError, match="active_region must be a region identifier"):
        EventMatcher(catalog).query(pd.Timestamp("2011-02-16 00:00"), "24h", active_region=" , ")


def test_a_blank_region_list_counts_as_a_missing_region(tmp_path, catalog):
    result = _single(_index(tmp_path, [","]), catalog, missing_active_region="drop")
    assert result.empty


def test_sequences_of_a_multi_region_patch(tmp_path, catalog):
    builder = ActiveRegionSequence(
        "24h", reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"),
        cadence="1h", observation_window="3h", sliding_window="3h",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = builder.build(_index(tmp_path, ["11160,11161,11162", "11170"]), catalog)
    assert result["active_region"].tolist() == ["11160,11161,11162", "11170"]
    assert result["label"].tolist() == [1, 1]
    assert result["n_instances"].tolist() == [3, 3]


def test_the_same_regions_in_another_order_form_one_sequence_group(tmp_path, catalog):
    lines = ["timestamp,active_region",
             '2011-02-16 00:00:00,"11160,11161"',
             '2011-02-16 01:00:00,"11161,11160"',
             '2011-02-16 02:00:00,"11160,11161"']
    index = _write(tmp_path / "index.csv", "\n".join(lines) + "\n")
    builder = ActiveRegionSequence(
        "24h", reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"),
        cadence="1h", observation_window="2h", sliding_window="1h",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = builder.build(index, catalog)
    assert result["active_region"].tolist() == ["11160,11161"] * 2
    assert result["label"].tolist() == [1, 1]
