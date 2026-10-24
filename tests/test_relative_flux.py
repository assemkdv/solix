"""
Relative X-ray flux increase labels (MaxRelativeFluxReducer,
SumRelativeFluxReducer): rxfi = peak flux / background flux, read from the
catalog's rxfi column (Hong, Ji, Pandey, Aydin, DaWaK 2023).
"""

import pandas as pd
import pytest

from flare_indexer import (
    DatasetBuilder,
    DatasetBuildingPipeline,
    EventMatcher,
    FlareEvent,
    MaxRelativeFluxReducer,
    NumericThresholdLabeler,
    RegressionLabeler,
    SumRelativeFluxReducer,
    adapt_goes_catalog,
)

CATALOG = (
    "start_time,peak_time,end_time,class,active_region,rxfi\n"
    "2024-01-01 02:00:00,2024-01-01 02:10:00,2024-01-01 02:20:00,C2.0,100,4.5\n"
    "2024-01-01 05:00:00,2024-01-01 05:10:00,2024-01-01 05:20:00,M1.0,100,12.0\n"
    "2024-01-01 09:00:00,2024-01-01 09:10:00,2024-01-01 09:20:00,B5.0,200,\n"
    "2024-01-02 03:00:00,2024-01-02 03:10:00,2024-01-02 03:20:00,C1.0,200,2.5\n"
)


def _events(values):
    return [FlareEvent(peak_time=None, goes_class="C1.0", start_time=None, active_region=None, rxfi=v)
            for v in values]


def _files(tmp_path, catalog=CATALOG, times=("2024-01-01 00:00:00", "2024-01-01 06:00:00", "2024-01-02 06:00:00")):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(catalog)
    index_path = tmp_path / "index.csv"
    pd.DataFrame({"timestamp": list(times)}).to_csv(index_path, index=False)
    return index_path, catalog_path


def _build(index_path, catalog_path, reducer, labeler=None, **kwargs):
    pipeline = DatasetBuildingPipeline(reducer, labeler or RegressionLabeler())
    return DatasetBuilder(prediction_window="24h", strategy=pipeline, **kwargs).build(index_path, catalog_path)


@pytest.mark.parametrize("reducer, values, expected", [
    (MaxRelativeFluxReducer(), [4.5, 12.0, 2.5], 12.0),
    (SumRelativeFluxReducer(), [4.5, 12.0, 2.5], 19.0),
    (MaxRelativeFluxReducer(), [], 0.0),
    (SumRelativeFluxReducer(), [], 0.0),
    (MaxRelativeFluxReducer(), [None, 3.0], 3.0),
    (SumRelativeFluxReducer(), [None, 3.0], 3.0),
    (MaxRelativeFluxReducer(), [None], 0.0),
    (SumRelativeFluxReducer(), [None], 0.0),
])
def test_reducers(reducer, values, expected):
    assert reducer.reduce(_events(values)) == expected


def test_catalog_rxfi_reaches_events(tmp_path):
    _, catalog_path = _files(tmp_path)
    events = EventMatcher(catalog_path).query(pd.Timestamp("2024-01-01 00:00:00"), "24h")
    assert [e.rxfi for e in events] == [4.5, 12.0, None]


def test_raw_and_thresholded_labels(tmp_path):
    index_path, catalog_path = _files(tmp_path)
    with pytest.warns(UserWarning, match="1 of 4 flare"):
        maximum = _build(index_path, catalog_path, MaxRelativeFluxReducer())
    with pytest.warns(UserWarning):
        total = _build(index_path, catalog_path, SumRelativeFluxReducer())
    with pytest.warns(UserWarning):
        binary = _build(index_path, catalog_path, MaxRelativeFluxReducer(), NumericThresholdLabeler(10))
    # 00:00 -> 4.5, 12.0 and a blank; 06:00 -> blank, 2.5; 2024-01-02 06:00 -> nothing.
    assert maximum["label"].tolist() == [12.0, 2.5, 0.0]
    assert total["label"].tolist() == [16.5, 2.5, 0.0]
    assert binary["label"].tolist() == [1, 0, 0]


def test_window_excludes_image_time(tmp_path):
    index_path, catalog_path = _files(tmp_path, times=("2024-01-01 05:00:00",))
    with pytest.warns(UserWarning):
        result = _build(index_path, catalog_path, SumRelativeFluxReducer())
    # The M1.0 starting exactly at t does not count; B5.0 (blank) and C1.0 (2.5) do.
    assert result["label"].tolist() == [2.5]


def test_active_region(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(CATALOG)
    index_path = tmp_path / "index.csv"
    pd.DataFrame({"timestamp": ["2024-01-01 00:00:00"] * 2, "active_region": ["100", "200"]}).to_csv(index_path, index=False)
    with pytest.warns(UserWarning):
        result = _build(index_path, catalog_path, SumRelativeFluxReducer(), target="active_region")
    assert result["label"].tolist() == [16.5, 0.0]


def test_catalog_without_rxfi_column_is_refused(tmp_path):
    catalog = "\n".join(",".join(line.split(",")[:5]) for line in CATALOG.strip().splitlines()) + "\n"
    index_path, catalog_path = _files(tmp_path, catalog=catalog)
    with pytest.raises(ValueError, match="no 'rxfi' column"):
        _build(index_path, catalog_path, MaxRelativeFluxReducer())


@pytest.mark.parametrize("bad", ["abc", "-1", "0", "inf"])
def test_invalid_rxfi_is_refused(tmp_path, bad):
    _, catalog_path = _files(tmp_path, catalog=CATALOG.replace(",12.0\n", f",{bad}\n"))
    with pytest.raises(ValueError, match="column 'rxfi', data row 2"):
        EventMatcher(catalog_path)


@pytest.mark.parametrize("sequence_kwargs", [
    dict(sequence_length=3),
    dict(cadence="1h", observation_window="3h", sliding_window="1h"),
])
@pytest.mark.parametrize("reducer", [MaxRelativeFluxReducer(), SumRelativeFluxReducer()])
def test_sequences_are_refused(reducer, sequence_kwargs):
    with pytest.raises(ValueError, match="single images only"):
        DatasetBuilder(prediction_window="24h", strategy=DatasetBuildingPipeline(reducer, RegressionLabeler()),
                       **sequence_kwargs)


def test_other_labels_ignore_rxfi_column(tmp_path):
    index_path, catalog_path = _files(tmp_path)
    from flare_indexer import BinaryThresholdStrategy
    result = DatasetBuilder(prediction_window="24h", strategy=BinaryThresholdStrategy("M")).build(index_path, catalog_path)
    assert result["label"].tolist() == [1, 0, 0]


def test_adapt_goes_catalog_carries_rxfi(tmp_path):
    source = tmp_path / "goes.csv"
    source.write_text(
        "start_time,peak_time,end_time,goes_class,noaa_active_region,relative_X-ray_flux_increase\n"
        "2024-01-01 02:00:00,2024-01-01 02:10:00,2024-01-01 02:20:00,C2.0,13600.0,4.5\n"
        "2024-01-01 05:00:00,2024-01-01 05:10:00,2024-01-01 05:20:00,M1.0,13600.0,\n"
    )
    adapted = adapt_goes_catalog(source)
    assert adapted["rxfi"].tolist()[0] == 4.5
    assert pd.isna(adapted["rxfi"].tolist()[1])
