from dataclasses import dataclass
from typing import Any

from .events import EventMatcher, FlareEvent


@dataclass
class PipelineResult:
    """The full set of intermediate values a run_one() call produced."""

    events: list[FlareEvent]
    reduced_value: Any
    label: Any


def _require_method(component: object, method_name: str, role: str) -> None:
    method = getattr(component, method_name, None)
    if method is None or not callable(method):
        raise TypeError(
            f"{role}={component!r} must provide a callable {method_name!r} method "
            f"(e.g. `def {method_name}(self, ...): ...`)."
        )


class DatasetBuildingPipeline:
    """
    A label in two steps: a reducer turns the flares of a window into one
    value (reduce), and a labeler turns that value into the label (assign).
    Any objects with those methods work. extract_events, reduce_events and
    assign_label run the steps one at a time; label(flares) runs both
    steps, so a pipeline can be passed to a dataset class as strategy.
    """

    def __init__(self, reducer, labeler):
        _require_method(reducer, "reduce", "reducer")
        _require_method(labeler, "assign", "labeler")
        self.reducer = reducer
        self.labeler = labeler

    def extract_events(
        self,
        matcher: EventMatcher,
        reference_time,
        prediction_window,
        active_region=None,
        event_time: str = "start",
    ) -> list[FlareEvent]:
        """Stage 1: query EventMatcher and return the matched FlareEvents."""
        events = matcher.query(
            reference_time, prediction_window, active_region=active_region,
            event_time=event_time,
        )
        return list(events)

    def reduce_events(self, events: list[FlareEvent]):
        """Stage 2: reduce a list of FlareEvents to a single value via self.reducer."""
        return self.reducer.reduce(list(events))   # a copy, so the caller's list is never changed

    def assign_label(self, reduced_value):
        """Stage 3: turn a reduced value into a final label via self.labeler."""
        return self.labeler.assign(reduced_value)

    def run_one(
        self,
        matcher: EventMatcher,
        reference_time,
        prediction_window,
        active_region=None,
        event_time: str = "start",
    ) -> PipelineResult:
        """Run all three stages for one reference time and return every intermediate value."""
        events = self.extract_events(
            matcher, reference_time, prediction_window,
            active_region=active_region, event_time=event_time,
        )
        reduced_value = self.reduce_events(events)
        label = self.assign_label(reduced_value)
        return PipelineResult(events=events, reduced_value=reduced_value, label=label)

    def label(self, events: list[FlareEvent]):
        """Reduce, then assign: the method the dataset classes call on their strategy."""
        reduced_value = self.reduce_events(events)
        return self.assign_label(reduced_value)
