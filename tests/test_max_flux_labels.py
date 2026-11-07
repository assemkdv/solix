"""
The two most common labels, written as a reducer and a labeler: a 0/1
label from the strongest flare (MaxFluxReducer + BinaryThresholdLabeler)
and the strongest flare's flux (MaxFluxReducer + RegressionLabeler).
"""

import pandas as pd

from solix.events import FlareEvent
from solix.labelers import BinaryThresholdLabeler, RegressionLabeler
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


def _max_flux(flares):
    return RegressionLabeler().assign(MaxFluxReducer().reduce(flares))


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


def test_max_flux_empty_list_returns_zero():
    assert _max_flux([]) == 0.0


def test_max_flux_single_flare_returns_its_flux():
    assert _max_flux([_flare("M2.3")]) == 2.3e-5


def test_max_flux_multiple_flares_returns_max_flux():
    assert _max_flux([_flare("C4.0"), _flare("X1.0"), _flare("M2.3")]) == 1e-4


def test_max_flux_uses_numeric_not_lexical_goes_ordering():
    # A text sort of goes_class would pick the wrong "strongest" flare:
    # "M9.0" > "M10.0" (since '9' > '1' character-wise). The reducer must
    # pick M10.0 as stronger, matching its actual numeric flux.
    flares = [_flare("M9.0"), _flare("M10.0")]
    assert _max_flux(flares) == 1e-4  # M10.0's flux, not M9.0's 9e-5
    assert _max_flux(flares) > _max_flux([_flare("M9.0")])
