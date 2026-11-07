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
    The largest peak X-ray flux (W/m^2) among the events, compared as
    numbers, not as class text. Returns 0.0 for an empty window.
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
    rxfi column. Events without a
    value are skipped; 0.0 if no event has one. Single instances only.
    """

    requires_catalog_column = "rxfi"
    single_instance_only = True

    def reduce(self, events: list[FlareEvent]) -> float:
        return max((e.rxfi for e in events if e.rxfi is not None), default=0.0)


class SumRelativeFluxReducer:
    """Sum of the relative X-ray flux increase (rxfi) of all events.

    Same rxfi values as MaxRelativeFluxReducer. Events without a value are
    skipped; 0.0 if no event has one. Single instances only.
    """

    requires_catalog_column = "rxfi"
    single_instance_only = True

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

        index = sum(C_i) + 10 * sum(M_j) + 100 * sum(X_k)

    where C_i, M_j, X_k are the numbers written in the class text
    (C2.6 -> 2.6, M1.5 -> 15, X1.0 -> 100), so the unit is one C1.0
    flare. A- and B-class events add 0, so "B15.0" adds 0 although its
    flux equals C1.5; the weights per letter are in WEIGHTS.

    The sum is formed in decimal arithmetic and rounded to a float once,
    so equal values written differently (C10.0 and M1.0) give the same
    result. Returns 0.0 for an empty window.
    """

    WEIGHTS = {"C": 1, "M": 10, "X": 100}

    def reduce(self, events: list[FlareEvent]) -> float:
        total = Decimal(0)
        for event in events:
            letter, multiplier = _parse_goes_class_decimal(event.goes_class)
            total += multiplier * self.WEIGHTS.get(letter, 0)
        return float(total)
