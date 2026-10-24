from .events import FlareEvent
from .labelers import BinaryThresholdLabeler, RegressionLabeler
from .reducers import MaxFluxReducer

_max_flux_reducer = MaxFluxReducer()


class BinaryThresholdStrategy:
    """
    Returns 1 if any flare in the window meets or exceeds the threshold
    class, else 0 (MaxFluxReducer followed by BinaryThresholdLabeler).
    """

    def __init__(self, threshold: str = "M"):
        self.threshold = threshold
        self._labeler = BinaryThresholdLabeler(threshold)

    def label(self, flares: list[FlareEvent]) -> int:
        reduced_value = _max_flux_reducer.reduce(flares)
        return self._labeler.assign(reduced_value)


class MaxFlareStrategy:
    """
    Returns the peak flux (W/m^2) of the strongest flare in the window, or
    0.0 (MaxFluxReducer followed by RegressionLabeler).
    """

    def __init__(self):
        self._labeler = RegressionLabeler()

    def label(self, flares: list[FlareEvent]) -> float:
        reduced_value = _max_flux_reducer.reduce(flares)
        return self._labeler.assign(reduced_value)
