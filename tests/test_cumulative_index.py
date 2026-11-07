"""
Cumulative normalized GOES subclass index (CumulativeClassIndexReducer)
and its numeric threshold labeler (NumericThresholdLabeler).

Reference: Hong, Ji, Pandey, Aydin, "Beyond Traditional Flare
Forecasting: A Data-driven Labeling Approach for High-fidelity
Predictions", DaWaK 2023, Sec. 2.2:  sum(C) + 10*sum(M) + 100*sum(X).
A and B flares contribute 0.
"""

import math
from pathlib import Path

import pandas as pd
import pytest

from flare_indexer import (
    CumulativeClassIndexReducer,
    DatasetBuildingPipeline,
    EventMatcher,
    FlareEvent,
    InvalidGOESClassError,
    NumericThresholdLabeler,
    RegressionLabeler,
)
from flare_indexer.builder import DatasetBuilder

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# The paper's worked prediction window (2015-09-20T04:55 to 2015-09-21T04:55).
PAPER_CLASSES = ["M1.5", "C2.6", "C1.4", "M2.1", "C2.8", "C4.2", "C1.5"]
PAPER_INDEX = 48.5  # (1.5 + 2.1) * 10 + (2.6 + 1.4 + 2.8 + 4.2 + 1.5)


def _flare(goes_class) -> FlareEvent:
    return FlareEvent(
        peak_time=pd.Timestamp("2024-01-01 00:10"),
        goes_class=goes_class,
        start_time=pd.Timestamp("2024-01-01 00:00"),
        active_region=None,
    )


def _events(classes):
    return [_flare(c) for c in classes]


# --- reducer -----------------------------------------------------------------

def test_paper_worked_example_is_exactly_48_5():
    assert CumulativeClassIndexReducer().reduce(_events(PAPER_CLASSES)) == PAPER_INDEX


def test_result_is_a_plain_float():
    value = CumulativeClassIndexReducer().reduce(_events(PAPER_CLASSES))
    assert type(value) is float


def test_order_independent():
    reducer = CumulativeClassIndexReducer()
    assert reducer.reduce(_events(reversed(PAPER_CLASSES))) == PAPER_INDEX
    assert reducer.reduce(_events(sorted(PAPER_CLASSES))) == PAPER_INDEX


def test_empty_window_is_zero():
    assert CumulativeClassIndexReducer().reduce([]) == 0.0


@pytest.mark.parametrize("classes", [["A1.2"], ["B5.0"], ["A1.0", "B9.9", "B15.0"]])
def test_a_and_b_only_windows_are_zero(classes):
    # The formula lists only C, M, X. B15.0 has the flux of C1.5 but
    # still counts 0 under this letter-based rule.
    assert CumulativeClassIndexReducer().reduce(_events(classes)) == 0.0


def test_a_and_b_do_not_change_a_mixed_sum():
    reducer = CumulativeClassIndexReducer()
    assert reducer.reduce(_events(PAPER_CLASSES + ["B5.0", "A1.2"])) == PAPER_INDEX


@pytest.mark.parametrize(
    "goes_class, expected",
    [
        ("C1.0", 1.0), ("C2.6", 2.6), ("C10.0", 10.0),
        ("M1.0", 10.0), ("M1.5", 15.0), ("M10.0", 100.0),
        ("X1.0", 100.0), ("X3.5", 350.0),
        ("C", 1.0), ("M", 10.0), ("X", 100.0),       # bare letters mean 1.0
        ("m2.0", 20.0),                               # case-insensitive, like the rest of the package
    ],
)
def test_single_event_weights(goes_class, expected):
    assert CumulativeClassIndexReducer().reduce(_events([goes_class])) == expected


def test_equal_strength_written_differently_gives_identical_value():
    reducer = CumulativeClassIndexReducer()
    assert reducer.reduce(_events(["C10.0"])) == reducer.reduce(_events(["M1.0"])) == 10.0
    assert reducer.reduce(_events(["M10.0"])) == reducer.reduce(_events(["X1.0"])) == 100.0


def test_decimal_sum_is_exact_for_decimal_class_text():
    # Ten C0.1 flares: a naive float sum gives 0.9999999999999999.
    assert CumulativeClassIndexReducer().reduce(_events(["C0.1"] * 10)) == 1.0
    assert NumericThresholdLabeler(1).assign(CumulativeClassIndexReducer().reduce(_events(["C0.1"] * 10))) == 1


def test_weights_are_the_documented_constants():
    assert CumulativeClassIndexReducer.WEIGHTS == {"C": 1, "M": 10, "X": 100}


@pytest.mark.parametrize("bad", ["Z5.0", "", None, "M2.3.4", "M0", 5.0])
def test_invalid_class_raises(bad):
    with pytest.raises(InvalidGOESClassError):
        CumulativeClassIndexReducer().reduce(_events(["C1.0", bad]))


def test_reducer_does_not_mutate_input():
    events = _events(PAPER_CLASSES)
    snapshot = list(events)
    CumulativeClassIndexReducer().reduce(events)
    assert events == snapshot


# --- labeler -----------------------------------------------------------------

@pytest.mark.parametrize(
    "threshold, value, expected",
    [
        (10, 10.0, 1), (10, 9.999, 0), (10, 0.0, 0), (10, 48.5, 1),
        (20, 48.5, 1), (48.5, 48.5, 1), (48.6, 48.5, 0),
        (0, 0.0, 1), (0, -1.0, 0), (-5, -1.0, 1),
        (10, 10, 1), (10.0, 10, 1),
    ],
)
def test_numeric_threshold_boundaries(threshold, value, expected):
    assert NumericThresholdLabeler(threshold).assign(value) == expected


def test_labels_are_plain_ints():
    label = NumericThresholdLabeler(10).assign(48.5)
    assert type(label) is int


def test_threshold_is_stored_as_float():
    assert NumericThresholdLabeler(10).threshold == 10.0
    assert type(NumericThresholdLabeler(10).threshold) is float


@pytest.mark.parametrize("bad", ["10", "M", True, None, [10]])
def test_non_numeric_threshold_is_rejected(bad):
    with pytest.raises(TypeError):
        NumericThresholdLabeler(bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_threshold_is_rejected(bad):
    with pytest.raises(ValueError):
        NumericThresholdLabeler(bad)


@pytest.mark.parametrize("bad", ["10", "M2.0", True, None, {"C": 1}])
def test_assign_rejects_non_numbers(bad):
    with pytest.raises(TypeError):
        NumericThresholdLabeler(10).assign(bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_assign_rejects_non_finite_values(bad):
    with pytest.raises(ValueError):
        NumericThresholdLabeler(10).assign(bad)


# --- pipeline integration ---------------------------------------------------------

def test_pipeline_label_from_events():
    raw = DatasetBuildingPipeline(CumulativeClassIndexReducer(), RegressionLabeler())
    binary = DatasetBuildingPipeline(CumulativeClassIndexReducer(), NumericThresholdLabeler(10))
    events = _events(PAPER_CLASSES)

    def label(pipeline, events):
        return pipeline.assign_label(pipeline.reduce_events(events))

    assert label(raw, events) == PAPER_INDEX
    assert label(binary, events) == 1
    assert label(raw, []) == 0.0
    assert label(binary, []) == 0


def test_pipeline_run_one_with_fixture_catalog():
    # tests/fixtures/catalog.csv: C3.0 starts 2024-02-01 11:50, X1.0 starts
    # 2024-02-02 00:00 (exactly 24 h after the instance), C1.0 and M4.0 on
    # 2024-02-05. Start time, window [t, t + w).
    matcher = EventMatcher(FIXTURES_DIR / "catalog.csv")
    pipeline = DatasetBuildingPipeline(CumulativeClassIndexReducer(), NumericThresholdLabeler(10))
    result = pipeline.run_one(
        matcher, pd.Timestamp("2024-02-01 00:00"), "24h",
    )
    assert [e.goes_class for e in result.events] == ["C3.0"]
    assert result.reduced_value == 3.0
    assert result.label == 0

    # A window longer than 24 h includes the X1.0 starting exactly 24 h later: 3 + 100.
    wide = pipeline.run_one(
        matcher, pd.Timestamp("2024-02-01 00:00"), "24h1min",
    )
    assert wide.reduced_value == 103.0
    assert wide.label == 1


def test_dataset_builder_single_instances_raw_and_binary():
    raw = DatasetBuilder(
        prediction_window="24h",
        reducer=CumulativeClassIndexReducer(), labeler=RegressionLabeler(),
    ).build(FIXTURES_DIR / "index.csv", FIXTURES_DIR / "catalog.csv")
    binary = DatasetBuilder(
        prediction_window="24h",
        reducer=CumulativeClassIndexReducer(), labeler=NumericThresholdLabeler(10),
    ).build(FIXTURES_DIR / "index.csv", FIXTURES_DIR / "catalog.csv")
    # 2024-02-01: C3.0 -> 3.0; 2024-02-05: C1.0 + M4.0 -> 41.0; 2024-02-10: empty -> 0.0
    assert raw["label"].tolist() == [3.0, 41.0, 0.0]
    assert binary["label"].tolist() == [0, 1, 0]

