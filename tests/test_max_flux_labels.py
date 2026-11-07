"""
The most common label, written as a reducer and a labeler: a 0/1 label
from the strongest flare (MaxFluxReducer + BinaryThresholdLabeler). The
strongest flare's flux itself is tested in test_reducers.py.
"""

import pandas as pd

from solix.events import FlareEvent
from solix.labelers import BinaryThresholdLabeler
from solix.reducers import MaxFluxReducer


def _flare(goes_class: str) -> FlareEvent:
    return FlareEvent(
        peak_time=pd.Timestamp("2024-01-01 00:00"),
        goes_class=goes_class,
        start_time=pd.Timestamp("2024-01-01 00:00"),
        active_region=None,
    )


def _binary(flares, threshold="M"):
    return BinaryThresholdLabeler(threshold).assign(MaxFluxReducer().reduce(flares))


def test_binary_threshold_default_is_m_class():
    assert BinaryThresholdLabeler().threshold == "M"
    assert _binary([_flare("M2.3")]) == 1
    assert _binary([_flare("C4.0")]) == 0


def test_binary_threshold_custom_c_class():
    assert _binary([_flare("C1.0")], "C") == 1
    assert _binary([_flare("B9.0")], "C") == 0


def test_binary_threshold_any_qualifying_flare_gives_one():
    assert _binary([_flare("B1.0"), _flare("C4.0"), _flare("X1.0")]) == 1


def test_binary_threshold_no_qualifying_flares_gives_zero():
    assert _binary([_flare("B1.0"), _flare("C4.0")]) == 0
    assert _binary([]) == 0


