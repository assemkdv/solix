import math
import numbers
from collections.abc import Mapping
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

    assign() accepts a real number (Python int/float, numpy integer or
    floating scalar, fractions.Fraction); it is converted with float()
    and compared with the threshold flux, so a value exactly equal to the
    threshold (1e-5 for "M") gives 1. Booleans (bool, numpy.bool_),
    Decimal, text, None and other non-real inputs raise TypeError; NaN
    and infinities raise ValueError. A label is never guessed for them.
    """

    def __init__(self, threshold: str = "M"):
        self.threshold = threshold
        try:
            self._threshold_flux = _flux_converter.to_flux(threshold)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(f"Invalid GOES threshold {threshold!r}: {exc}") from exc

    def assign(self, reduced_value) -> int:
        if isinstance(reduced_value, bool) or not isinstance(reduced_value, numbers.Real):
            raise TypeError(
                "BinaryThresholdLabeler.assign expects a flux number in W/m^2 (e.g. the output of "
                f"MaxFluxReducer), got {reduced_value!r} ({type(reduced_value).__name__})."
            )
        value = float(reduced_value)
        if not math.isfinite(value):
            raise ValueError(f"BinaryThresholdLabeler.assign got a non-finite flux {reduced_value!r}.")
        return int(value >= self._threshold_flux)


class NumericThresholdLabeler:
    """
    Returns 1 if a reduced *numeric index* meets or exceeds a numeric
    threshold, else 0.

    For dimensionless indices such as the cumulative class index
    (CumulativeClassIndexReducer), e.g. 10 or 20. The threshold is a
    plain number compared directly with the reduced value; it is not a
    GOES class and is not converted to flux. For flux values in W/m^2
    with a GOES-class threshold use BinaryThresholdLabeler instead.

    The threshold must be a finite real number (not a bool, not text).
    assign() rejects booleans and non-numbers (TypeError) and NaN or
    infinite values (ValueError); it never guesses a label for them.
    """

    def __init__(self, threshold: float):
        if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
            raise TypeError(
                f"NumericThresholdLabeler threshold must be a real number, got {threshold!r} "
                f"({type(threshold).__name__}). GOES-class thresholds belong to BinaryThresholdLabeler."
            )
        if not math.isfinite(threshold):
            raise ValueError(f"NumericThresholdLabeler threshold must be finite, got {threshold!r}.")
        self.threshold = float(threshold)

    def assign(self, reduced_value) -> int:
        if isinstance(reduced_value, bool) or not isinstance(reduced_value, numbers.Real):
            raise TypeError(
                "NumericThresholdLabeler.assign expects a number (e.g. the output of "
                f"CumulativeClassIndexReducer), got {reduced_value!r} ({type(reduced_value).__name__})."
            )
        value = float(reduced_value)
        if not math.isfinite(value):
            raise ValueError(f"NumericThresholdLabeler.assign got a non-finite value {reduced_value!r}.")
        return int(value >= self.threshold)


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

    CLASSES = ("NF",) + FluxConverter.CLASS_LETTERS

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
        if flux < _flux_converter.FLUX_MAP["A"]:
            raise ValueError(
                f"MulticlassLabeler.assign got a positive flux {flux!r} W/m^2 below the A-class boundary "
                f"{_flux_converter.FLUX_MAP['A']!r} W/m^2. The GOES scale has no class for it, and it is not "
                "'NF' (no flare). Check the catalog for classes below A1.0 (e.g. 'A0.5')."
            )
        return _flux_converter.to_class(flux)   # the boundary rule lives in classifier.py


class ClassCountLabeler:
    """Return a validated copy of A/B/C/M/X event counts.

    Expects exactly those five keys and nonnegative integer values. No
    NF bin: an empty window is represented by five zero counts.
    """

    def assign(self, counts) -> dict[str, int]:
        if not isinstance(counts, Mapping):
            raise TypeError("Class counts must be a mapping, not a reduced scalar flux.")
        classes = FluxConverter.CLASS_LETTERS
        if set(counts) != set(classes):
            raise ValueError("Class counts must contain exactly A, B, C, M, X.")
        result = {}
        for letter in classes:
            value = counts[letter]
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(f"Count for {letter} must be an integer.")
            if value < 0:
                raise ValueError(f"Count for {letter} must be nonnegative.")
            result[letter] = int(value)
        return result


class ClassPresenceLabeler(ClassCountLabeler):
    """Convert each validated class count to 0/1 presence; all zeros if empty."""

    def assign(self, counts) -> dict[str, int]:
        return {letter: int(count > 0) for letter, count in super().assign(counts).items()}
