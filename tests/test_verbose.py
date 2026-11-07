"""
verbose=True prints a short summary; it never changes the result, and
nothing is printed by default.
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
XRAY = DATA / "xray_flux.csv"

pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.skipif(not DATA.is_dir(), reason="examples/ not available (installed package)"),
]

SETTINGS = {
    "single full disk": (FULL_DISK, dict(reducer=f.MaxFluxReducer(), labeler=f.BinaryThresholdLabeler("M"))),
    "single active region": (ACTIVE_REGION, dict(reducer=f.MaxFluxReducer(), labeler=f.BinaryThresholdLabeler("M"), target="active_region")),
    "time sequences": (FULL_DISK, dict(reducer=f.MaxFluxReducer(), labeler=f.BinaryThresholdLabeler("M"), cadence="1h",
                                       observation_window="3h", sliding_window="1h")),
    "active region time sequences": (ACTIVE_REGION, dict(reducer=f.MaxFluxReducer(), labeler=f.RegressionLabeler(), target="active_region",
                                                         cadence="6h", observation_window="12h", sliding_window="6h")),
}


@pytest.mark.parametrize("name", SETTINGS)
def test_result_is_identical_and_default_is_silent(name, capsys):
    index, kwargs = SETTINGS[name]
    quiet = DatasetBuilder("24h", **kwargs).build(index, CATALOG)
    assert capsys.readouterr().out == ""
    loud = DatasetBuilder("24h", verbose=True, **kwargs).build(index, CATALOG)
    assert capsys.readouterr().out.startswith("Index: ")
    pd.testing.assert_frame_equal(quiet, loud)


def test_single_instance_summary(capsys):
    DatasetBuilder("24h", reducer=f.MaxFluxReducer(), labeler=f.BinaryThresholdLabeler("M"), verbose=True).build(FULL_DISK, CATALOG)
    assert capsys.readouterr().out.splitlines() == [
        f"Index: 118 instances ({FULL_DISK})",
        f"Flare catalog: 8 flares ({CATALOG})",
        "Settings: window [t, t + 24h) on start_time, full disk, single instances",
        "Result: 118 rows; label 0: 41, label 1: 77",
    ]


def test_active_region_time_sequence_summary(capsys):
    index, kwargs = SETTINGS["active region time sequences"]
    DatasetBuilder("24h", verbose=True, **kwargs).build(index, CATALOG)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"Index: 41 instances, 2 regions ({ACTIVE_REGION})"
    assert lines[2] == ("Settings: window [t, t + 24h) on start_time, active region, "
                        "sequences of 2 instances 6h apart, new sequence every 6h")
    assert lines[3] == "Result: 38 sequences; values from 0 to 0.00011"


def test_skipped_sequences_and_class_labels(capsys):
    DatasetBuilder("24h", reducer=f.MaxFluxReducer(), labeler=f.MulticlassLabeler(), cadence="1h", observation_window="3h", sliding_window="1h",
                   verbose=True).build(FULL_DISK, CATALOG)
    lines = capsys.readouterr().out.splitlines()
    assert "window [t, t + 24h) on start_time" in lines[2]
    assert "skipped: missing instances" in lines[3]
    assert "label NF: " in lines[3]


def test_return_report_still_works(capsys):
    result, report = DatasetBuilder("24h", reducer=f.MaxFluxReducer(), labeler=f.BinaryThresholdLabeler("M"), cadence="1h", observation_window="3h",
                                      sliding_window="1h", verbose=True).build(FULL_DISK, CATALOG, return_report=True)
    assert len(result) == report.emitted_sequences
    assert f"Result: {len(result)} sequences ({report.skipped_sequences} skipped: missing instances)" in capsys.readouterr().out


def test_add_relative_flux(capsys):
    quiet = f.add_relative_flux(CATALOG, XRAY)
    assert capsys.readouterr().out == ""
    loud = f.add_relative_flux(CATALOG, XRAY, verbose=True)
    assert capsys.readouterr().out.splitlines() == [
        f"Flare catalog: 8 flares ({CATALOG})",
        f"X-ray flux: 10,080 minutes, 2024-02-29 to 2024-03-06 ({XRAY})",
        "Result: rxfi computed for 8 flares; 0 without (no flux at start or no usable minutes before it)",
    ]
    pd.testing.assert_frame_equal(quiet, loud)


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_verbose_must_be_a_bool(value):
    with pytest.raises(ValueError, match="verbose"):
        DatasetBuilder("24h", reducer=f.MaxFluxReducer(), labeler=f.BinaryThresholdLabeler("M"), verbose=value)
    with pytest.raises(ValueError, match="verbose"):
        f.add_relative_flux(CATALOG, XRAY, verbose=value)
