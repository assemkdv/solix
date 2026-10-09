from typing import Protocol

from .classifier import FluxConverter
from .events import FlareEvent
from .labelers import MulticlassLabeler

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


class ClassCountReducer:
    """Count every matched event by numeric GOES class, in A/B/C/M/X order.

    Empty windows return all zeros. Equivalent strengths (C10.0/M1.0)
    share a bin. Below-A flux follows MulticlassLabeler's rejection policy.
    Each input event is counted once; this component does not deduplicate.
    """

    def reduce(self, events: list[FlareEvent]) -> dict[str, int]:
        counts = dict.fromkeys(MulticlassLabeler.CLASSES[1:], 0)
        classifier = MulticlassLabeler()
        for event in events:
            counts[classifier.assign(_flux_converter.to_flux(event.goes_class))] += 1
        return counts
