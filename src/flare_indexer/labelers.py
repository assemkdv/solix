import math
import numbers
from typing import Protocol

from .classifier import FluxConverter

_flux_converter = FluxConverter()


class LabelAssigner(Protocol):
    """
    Structural contract for turning a reducer's output into a final label.
    Any object with a matching `assign` method satisfies this -- no base
    class or registration required.
    """

    def assign(self, reduced_value):
        ...


class BinaryThresholdLabeler:
    """
    Returns 1 if a reduced flux value meets or exceeds a GOES class
    threshold, else 0.

    threshold may be a plain letter ("M") or a full GOES class with a
    number ("C4.2") -- both are converted via FluxConverter.to_flux(), so
    the comparison is always numeric.
    """

    def __init__(self, threshold: str = "M"):
        self.threshold = threshold
        try:
            self._threshold_flux = _flux_converter.to_flux(threshold)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(f"Invalid GOES threshold {threshold!r}: {exc}") from exc

    def assign(self, reduced_value) -> int:
        return int(reduced_value >= self._threshold_flux)


class RegressionLabeler:
    """
    Passes a reduced numeric flux value through unchanged (0.0 for no
    flare), normalizing scalar numeric types (e.g. a numpy float64) to
    plain Python int/float so callers always get a standard scalar back.
    """

    def assign(self, reduced_value):
        if isinstance(reduced_value, numbers.Integral):
            return int(reduced_value)
        if isinstance(reduced_value, numbers.Real):
            return float(reduced_value)
        return reduced_value


class MulticlassLabeler:
    """
    Maps a reduced flux value (W/m^2) to a GOES class letter:
    "NF", "A", "B", "C", "M" or "X".

    Intended for the output of MaxFluxReducer, so the label describes the
    single representative flux the reducer selected (the strongest flare
    in the window) -- not which classes occurred or how many flares there
    were. A window with one X1.0 flare and ten C flares is labeled "X".

    Classes are decided numerically by the lower boundary of each class
    (FluxConverter.FLUX_MAP): A >= 1e-8, B >= 1e-7, C >= 1e-6, M >= 1e-5,
    X >= 1e-4 (no upper bound), each boundary belonging to the higher
    class. The comparison is exact: a float even one step below a boundary
    (math.nextafter(1e-5, 0)) is in the lower class. FluxConverter.to_flux
    returns exactly the boundary for equivalent class text, so "C10.0" and
    "M1.0" are both "M", and "M10.0" and "X1.0" both "X".

    Exactly 0.0 -- MaxFluxReducer's value for an empty window -- is "NF"
    (no flare). Any other input that is not a GOES class flux is rejected:
    - a positive flux below the A-class boundary (e.g. from "A0.5") raises
      ValueError: the GOES scale has no letter for it, and it is a flare,
      so it is not "NF";
    - a negative, NaN or infinite value raises ValueError;
    - a non-numeric value (including text such as "M2.0" and booleans)
      raises TypeError.
    """

    CLASSES = ("NF", "A", "B", "C", "M", "X")

    def assign(self, reduced_flux) -> str:
        if isinstance(reduced_flux, bool) or not isinstance(reduced_flux, numbers.Real):
            raise TypeError(
                "MulticlassLabeler.assign expects a flux number in W/m^2 (e.g. the output of "
                f"MaxFluxReducer), got {reduced_flux!r} ({type(reduced_flux).__name__})."
            )
        flux = float(reduced_flux)
        if not math.isfinite(flux):
            raise ValueError(f"MulticlassLabeler.assign got a non-finite flux {reduced_flux!r}.")
        if flux < 0:
            raise ValueError(f"MulticlassLabeler.assign got a negative flux {flux!r}; X-ray flux cannot be negative.")
        if flux == 0:
            return "NF"
        for letter in reversed(self.CLASSES[1:]):
            boundary = _flux_converter.FLUX_MAP[letter]
            if flux >= boundary:
                return letter
        raise ValueError(
            f"MulticlassLabeler.assign got a positive flux {flux!r} W/m^2 below the A-class boundary "
            f"{_flux_converter.FLUX_MAP['A']!r} W/m^2. The GOES scale has no class for it, and it is not "
            "'NF' (no flare). Check the catalog for classes below A1.0 (e.g. 'A0.5')."
        )
