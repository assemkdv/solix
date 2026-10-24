from decimal import Decimal
from typing import Protocol

from .goes_class import FluxConverter, _parse_goes_class_decimal
from .events import FlareEvent

_flux_converter = FluxConverter()


class EventReducer(Protocol):
    """
    Structural contract for reducing a list of FlareEvents to a single
    numeric (or other) value. Any object with a matching `reduce` method
    satisfies this -- no base class or registration required.
    """

    def reduce(self, events: list[FlareEvent]):
        ...


class MaxFluxReducer:
    """
    Reduces a list of FlareEvents to the numeric maximum physical X-ray
    flux among them, via FluxConverter -- never by comparing goes_class
    strings lexically (see FluxConverter's docstring for why that matters:
    "M9.0" < "M10.0" numerically, despite sorting the other way as text).

    Returns 0.0 for an empty event list (no flare in the window).
    """

    def reduce(self, events: list[FlareEvent]) -> float:
        if not events:
            return 0.0
        return max(_flux_converter.to_flux(event.goes_class) for event in events)


class MinFluxReducer:
    """Minimum event peak flux in W/m²; 0.0 for an empty window.

    This selects the weakest recorded event, not the minimum of a
    continuous background-flux time series.
    """

    def reduce(self, events: list[FlareEvent]) -> float:
        return min((_flux_converter.to_flux(e.goes_class) for e in events), default=0.0)


class MaxRelativeFluxReducer:
    """Largest relative X-ray flux increase (rxfi) among the events.

    rxfi = peak X-ray flux / background X-ray flux, read from the catalog's
    rxfi column (Hong, Ji, Pandey, Aydin, DaWaK 2023). Events without a
    value are skipped; 0.0 if no event has one. Single images only.
    """

    requires_catalog_column = "rxfi"
    single_image_only = True

    def reduce(self, events: list[FlareEvent]) -> float:
        return max((e.rxfi for e in events if e.rxfi is not None), default=0.0)


class SumRelativeFluxReducer:
    """Sum of the relative X-ray flux increase (rxfi) of all events.

    Same rxfi values as MaxRelativeFluxReducer. Events without a value are
    skipped; 0.0 if no event has one. Single images only.
    """

    requires_catalog_column = "rxfi"
    single_image_only = True

    def reduce(self, events: list[FlareEvent]) -> float:
        return float(sum(e.rxfi for e in events if e.rxfi is not None))


class MeanFluxReducer:
    """Mean peak flux of all events in the window, in W/m²; 0.0 if empty.

    Each event's GOES class is converted to flux and the values are
    averaged, so C1.0, C3.0 and M2.0 give (1e-6 + 3e-6 + 2e-5) / 3.
    """

    def reduce(self, events: list[FlareEvent]) -> float:
        if not events:
            return 0.0
        return sum(_flux_converter.to_flux(e.goes_class) for e in events) / len(events)


class ClassCountReducer:
    """Count every matched event by numeric GOES class, in A/B/C/M/X order.

    Empty windows return all zeros. Equivalent strengths (C10.0/M1.0)
    share a bin (FluxConverter.to_class). A class below A1.0 (e.g. "A0.5")
    raises ValueError there, as MulticlassLabeler does. Each input event
    is counted once; this component does not deduplicate.
    """

    def reduce(self, events: list[FlareEvent]) -> dict[str, int]:
        counts = dict.fromkeys(FluxConverter.CLASS_LETTERS, 0)
        for event in events:
            counts[_flux_converter.to_class(_flux_converter.to_flux(event.goes_class))] += 1
        return counts


class CumulativeClassIndexReducer:
    """Cumulative normalized GOES subclass index of all events in the window.

    Formula (Hong, Ji, Pandey, Aydin, DaWaK 2023, Sec. 2.2):

        index = sum(C_i) + 10 * sum(M_j) + 100 * sum(X_k)

    where C_i, M_j, X_k are the numbers written in the class text
    (C2.6 -> 2.6, M1.5 -> 15, X1.0 -> 100). The unit is "one C1.0 flare";
    the value is dimensionless. The paper's worked window (M1.5, C2.6,
    C1.4, M2.1, C2.8, C4.2, C1.5) gives 48.5.

    A- and B-class events contribute 0, as in the paper's formula.
    Consequences: "C10.0" and "M1.0" both add 10; "B15.0" adds 0
    although its flux equals C1.5. The weights are per letter, in
    WEIGHTS; a letter missing from WEIGHTS weighs 0.

    The sum is formed in decimal arithmetic from the class text and
    rounded to a float once, so values written differently but equal
    (C10.0 and M1.0) give the identical result and the paper's 48.5 is
    exact. Returns 0.0 for an empty window. An invalid class raises
    InvalidGOESClassError (only possible for hand-built events; loaded
    catalogs are already checked).

    This reducer only aggregates; it does not choose the prediction
    window.
    """

    WEIGHTS = {"C": 1, "M": 10, "X": 100}

    def reduce(self, events: list[FlareEvent]) -> float:
        total = Decimal(0)
        for event in events:
            letter, multiplier = _parse_goes_class_decimal(event.goes_class)
            total += multiplier * self.WEIGHTS.get(letter, 0)
        return float(total)
