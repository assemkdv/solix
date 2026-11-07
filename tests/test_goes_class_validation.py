import pandas as pd
import pytest

from flare_indexer import EventMatcher, FluxConverter, InvalidGOESClassError, adapt_goes_catalog
from flare_indexer.goes_class import parse_goes_class
from flare_indexer.builder import DatasetBuilder
from flare_indexer import MaxFluxReducer, BinaryThresholdLabeler


@pytest.mark.parametrize("value, expected", [
    ("M2.3", ("M", 2.3)), ("M", ("M", 1.0)), ("X10", ("X", 10.0)), ("m2.3", ("M", 2.3)),
    (" C4.0 ", ("C", 4.0)), ("A1.0", ("A", 1.0)),
])
def test_valid_goes_classes(value, expected):
    assert parse_goes_class(value) == expected


@pytest.mark.parametrize("value", ["", "   ", None, float("nan"), "Z5.0", "M2.3.4", "M-1", "M0", "M0.0", "2.3", "MX", 5.0])
def test_invalid_goes_classes_raise_clear_error(value):
    with pytest.raises(InvalidGOESClassError) as excinfo:
        FluxConverter().to_flux(value)
    message = str(excinfo.value)
    assert "GOES class" in message


def test_error_is_a_value_error():
    assert issubclass(InvalidGOESClassError, ValueError)


def test_catalog_with_invalid_classes_fails_at_load(tmp_path):
    header = "start_time,peak_time,end_time,class,active_region"
    row = "2024-01-01 05:00:00,2024-01-01 05:10:00,2024-01-01 05:20:00,{cls},100"
    lines = [header, row.format(cls="M2.0"), row.format(cls=""), row.format(cls="Q3"), row.format(cls="C1.0")]
    path = tmp_path / "catalog.csv"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(InvalidGOESClassError) as excinfo:
        EventMatcher(path)
    message = str(excinfo.value)
    assert "2 missing or invalid GOES class value(s)" in message
    assert "data row 2: <missing>" in message
    assert "data row 3: 'Q3'" in message


def test_invalid_class_outside_any_window_is_still_rejected(tmp_path):
    # A bad class fails at load, even when no window would reach it.
    catalog = tmp_path / "catalog.csv"
    catalog.write_text(
        "start_time,peak_time,end_time,class,active_region\n"
        "2030-01-01 05:00:00,2030-01-01 05:10:00,2030-01-01 05:20:00,bad,100\n"
    )
    index = tmp_path / "index.csv"
    index.write_text("timestamp\n2024-01-01 00:00:00\n")
    with pytest.raises(InvalidGOESClassError, match="'bad'"):
        DatasetBuilder(prediction_window=24, reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M")).build(index, catalog)


def test_adapt_goes_catalog_validates_classes(tmp_path):
    source = tmp_path / "goes.csv"
    source.write_text(
        "start_time,peak_time,end_time,goes_class,noaa_active_region\n"
        "2024-01-01 05:00:00,2024-01-01 05:10:00,2024-01-01 05:20:00,M2.0,13600\n"
        "2024-01-01 06:00:00,2024-01-01 06:10:00,2024-01-01 06:20:00,,13600\n"
    )
    with pytest.raises(InvalidGOESClassError, match="column 'goes_class' \\(data row 2: <missing>\\)"):
        adapt_goes_catalog(source)


def test_valid_catalog_unchanged(tmp_path):
    path = tmp_path / "catalog.csv"
    path.write_text(
        "start_time,peak_time,end_time,class,active_region\n"
        "2024-01-01 05:00:00,2024-01-01 05:10:00,2024-01-01 05:20:00,m2.0,100\n"
    )
    flares = EventMatcher(path).query(pd.Timestamp("2024-01-01 00:00"), 24)
    assert [f.goes_class for f in flares] == ["m2.0"]
    assert BinaryThresholdLabeler("M").assign(MaxFluxReducer().reduce(flares)) == 1
