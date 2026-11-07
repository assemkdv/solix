"""
Relative X-ray flux increase labels (MaxRelativeFluxReducer,
SumRelativeFluxReducer): rxfi = peak flux / background flux, read from the
catalog's rxfi column (Hong, Ji, Pandey, Aydin, DaWaK 2023).
"""

import pandas as pd
import pytest

from solix import (
    EventMatcher,
    FlareEvent,
    MaxRelativeFluxReducer,
    NumericThresholdLabeler,
    RegressionLabeler,
    SumRelativeFluxReducer,
    adapt_goes_catalog,
)
from solix.builder import DatasetBuilder

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
    return DatasetBuilder(
        prediction_window="24h", reducer=reducer, labeler=labeler or RegressionLabeler(), **kwargs,
    ).build(index_path, catalog_path)


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


def test_window_includes_instance_time(tmp_path):
    index_path, catalog_path = _files(tmp_path, times=("2024-01-01 05:00:00",))
    with pytest.warns(UserWarning):
        result = _build(index_path, catalog_path, SumRelativeFluxReducer())
    # The M1.0 starting exactly at t counts (12.0); B5.0 (blank) adds nothing; C1.0 adds 2.5.
    assert result["label"].tolist() == [14.5]


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


@pytest.mark.parametrize("reducer", [MaxRelativeFluxReducer(), SumRelativeFluxReducer()])
def test_sequences_are_refused(reducer):
    with pytest.raises(ValueError, match="single instances only"):
        DatasetBuilder(prediction_window="24h", reducer=reducer, labeler=RegressionLabeler(),
                       cadence="1h", observation_window="3h", sliding_window="1h")


def test_other_labels_ignore_rxfi_column(tmp_path):
    index_path, catalog_path = _files(tmp_path)
    result = DatasetBuilder(prediction_window="24h", reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M")).build(index_path, catalog_path)
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


# ── add_relative_flux: background flux and rxfi from 1-minute X-ray flux ──

from solix import add_relative_flux  # noqa: E402
from solix import MaxFluxReducer, BinaryThresholdLabeler

FLARE = "2024-01-02 00:00:00,2024-01-02 00:05:00,2024-01-02 00:10:00,{cls},100\n"
HEADER = "start_time,peak_time,end_time,class,active_region\n"


def _flux_file(tmp_path, values, start="2024-01-01 00:00:00"):
    """values: {minutes after start: flux}."""
    path = tmp_path / "xray.csv"
    t0 = pd.Timestamp(start)
    pd.DataFrame({
        "time": [t0 + pd.Timedelta(minutes=m) for m in values],
        "flux": list(values.values()),
    }).to_csv(path, index=False)
    return path


def _catalog_file(tmp_path, rows):
    path = tmp_path / "flares.csv"
    path.write_text(HEADER + "".join(rows))
    return path


def test_background_is_mean_of_quiet_minutes_below_start_flux(tmp_path):
    # Flare starts 2024-01-02 00:00 (minute 1440). Window: minute 0 .. 1440.
    flux = {0: 2e-7, 600: 4e-7, 1200: 6e-7, 1439: 9e-7, 1440: 8e-7, 1450: 5e-6}
    result = add_relative_flux(_catalog_file(tmp_path, [FLARE.format(cls="M1.0")]), _flux_file(tmp_path, flux))
    # Minute 1440 is inside the flare (dropped); 9e-7 is not below the start flux 8e-7.
    assert result["background_flux"].iloc[0] == pytest.approx((2e-7 + 4e-7 + 6e-7) / 3)
    assert result["rxfi"].iloc[0] == pytest.approx(1e-5 / 4e-7)


def test_minutes_before_window_and_bad_values_are_ignored(tmp_path):
    flux = {-1: 1e-8, 0: 3e-7, 5: 1e-9, 6: 0.0, 1440: 8e-7}
    result = add_relative_flux(_catalog_file(tmp_path, [FLARE.format(cls="C2.0")]),
                               _flux_file(tmp_path, flux, start="2024-01-01 00:00:00"))
    assert result["background_flux"].iloc[0] == pytest.approx(3e-7)
    assert result["rxfi"].iloc[0] == pytest.approx(2e-6 / 3e-7)


def test_other_flares_are_removed_from_the_background(tmp_path):
    other = "2024-01-01 10:00:00,2024-01-01 10:05:00,2024-01-01 10:30:00,C5.0,200\n"
    flux = {0: 2e-7, 600: 7e-7, 620: 7e-7, 640: 4e-7, 1440: 8e-7}
    result = add_relative_flux(_catalog_file(tmp_path, [FLARE.format(cls="M1.0"), other]),
                               _flux_file(tmp_path, flux))
    # Minutes 600 and 620 (10:00, 10:20) fall inside the C5.0 flare.
    assert result["background_flux"].iloc[0] == pytest.approx((2e-7 + 4e-7) / 2)


def test_flare_without_flux_at_start_is_left_blank_with_warning(tmp_path):
    flux = {0: 2e-7, 1439: 3e-7}
    with pytest.warns(UserWarning, match="1 of 1 flare"):
        result = add_relative_flux(_catalog_file(tmp_path, [FLARE.format(cls="M1.0")]), _flux_file(tmp_path, flux))
    assert pd.isna(result["background_flux"].iloc[0]) and pd.isna(result["rxfi"].iloc[0])


def test_flare_with_no_value_below_start_flux_is_left_blank(tmp_path):
    flux = {0: 9e-7, 1440: 8e-7}
    with pytest.warns(UserWarning, match="no rxfi"):
        result = add_relative_flux(_catalog_file(tmp_path, [FLARE.format(cls="M1.0")]), _flux_file(tmp_path, flux))
    assert pd.isna(result["rxfi"].iloc[0])


def test_output_keeps_rows_and_columns_and_feeds_the_labels(tmp_path):
    second = "2024-01-02 06:00:00,2024-01-02 06:05:00,2024-01-02 06:10:00,C1.0,200\n"
    flux = {0: 2e-7, 600: 1e-7, 1440: 8e-7, 1800: 3e-7}
    catalog_path = _catalog_file(tmp_path, [FLARE.format(cls="M1.0"), second])
    out = tmp_path / "flares_rxfi.csv"
    result = add_relative_flux(catalog_path, _flux_file(tmp_path, flux), output_path=out)
    assert list(result.columns) == ["start_time", "peak_time", "end_time", "class", "active_region",
                                    "background_flux", "rxfi"]
    assert result["class"].tolist() == ["M1.0", "C1.0"]
    assert pd.read_csv(out).shape == (2, 7)

    index_path = tmp_path / "index.csv"
    pd.DataFrame({"timestamp": ["2024-01-01 12:00:00"]}).to_csv(index_path, index=False)
    labels = _build(index_path, out, MaxRelativeFluxReducer())
    assert labels["label"].iloc[0] == pytest.approx(result["rxfi"].iloc[0])


def test_catalog_that_already_has_rxfi_is_refused(tmp_path):
    path = tmp_path / "flares.csv"
    path.write_text(CATALOG)
    with pytest.raises(ValueError, match="already has rxfi"):
        add_relative_flux(path, _flux_file(tmp_path, {0: 1e-7}))


def test_flare_that_ends_before_it_starts_is_refused(tmp_path):
    # The same check as build(), so a catalog add_relative_flux accepts also loads there.
    bad = "2024-01-02 00:00:00,2024-01-02 00:05:00,2024-01-01 23:00:00,M1.0,100\n"
    with pytest.raises(ValueError, match="data row 1: times must satisfy start <= peak <= end"):
        add_relative_flux(_catalog_file(tmp_path, [bad]), _flux_file(tmp_path, {0: 1e-7}))


@pytest.mark.parametrize("content, message", [
    ("when,flux\n2024-01-01 00:00:00,1e-7\n", "missing required column"),
    ("time,flux\n2024-01-01 00:00:00,abc\n", "column 'flux', data row 1"),
    ("time,flux\n2024-01-01 00:00:00,1e-7\n2024-01-01 00:00:00,2e-7\n", "appears more than once"),
    ("time,flux\n,1e-7\n", "the time is blank"),
])
def test_bad_xray_file_is_refused(tmp_path, content, message):
    xray = tmp_path / "xray.csv"
    xray.write_text(content)
    with pytest.raises(ValueError, match=message):
        add_relative_flux(_catalog_file(tmp_path, [FLARE.format(cls="M1.0")]), xray)


def test_unsorted_xray_file_gives_same_result(tmp_path):
    flux = {0: 2e-7, 600: 4e-7, 1440: 8e-7}
    catalog_path = _catalog_file(tmp_path, [FLARE.format(cls="M1.0")])
    sorted_result = add_relative_flux(catalog_path, _flux_file(tmp_path, flux))
    shuffled = _flux_file(tmp_path, dict(reversed(list(flux.items()))))
    assert add_relative_flux(catalog_path, shuffled)["rxfi"].iloc[0] == sorted_result["rxfi"].iloc[0]
