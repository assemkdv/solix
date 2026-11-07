"""
The four kinds of labeled dataset, one class each:

    FullDiskSingleInstance       one full-disk instance -> one label
    FullDiskSequence             a sequence of full-disk instances -> one label
    ActiveRegionSingleInstance   one active-region instance -> one label
    ActiveRegionSequence         a sequence of instances of one region -> one label

An instance is one row of the index: one point in time, with whatever
file belongs to it (an image, a .npy array, ...). The files are never
opened, so any file type works.

Each class takes only the settings that apply to its case and runs the
shared DatasetBuilder engine. All four are used the same way:

    labels = FullDiskSingleInstance(
        "24h", reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"),
    ).build("index.csv", "flare_catalog.csv")

The reducer turns the flares of one window into one value and the
labeler turns that value into the label; see reducers.py and labelers.py.

Sequences are built on a time grid: `cadence` is the
time between instances, `observation_window` the time a sequence
covers, `sliding_window` the step between sequence starts, and `start`
the optional first sequence start. Each sequence is labeled from its
last instance.
"""

from .builder import DatasetBuilder


class FullDiskSingleInstance(DatasetBuilder):
    """One label per full-disk instance, from all flares in its window.

    cadence (optional) keeps only the instances at start, start + cadence,
    ... (e.g. one per hour from a 12-minute index); start defaults to the
    earliest instance.
    """

    def __init__(
        self,
        prediction_window,
        *,
        reducer,
        labeler,
        cadence=None,
        start=None,
        include_metadata=False,
        include_flares=False,
        verbose=False,
    ):
        super().__init__(
            prediction_window, reducer=reducer, labeler=labeler, target="full_disk",
            cadence=cadence, start=start,
            include_metadata=include_metadata, include_flares=include_flares, verbose=verbose,
        )


class FullDiskSequence(DatasetBuilder):
    """One label per sequence of full-disk instances, from the window after its last instance."""

    def __init__(
        self,
        prediction_window,
        *,
        reducer,
        labeler,
        cadence,
        observation_window,
        sliding_window,
        start=None,
        include_flares=False,
        verbose=False,
    ):
        super().__init__(
            prediction_window, reducer=reducer, labeler=labeler, target="full_disk",
            cadence=cadence, observation_window=observation_window, sliding_window=sliding_window, start=start,
            include_flares=include_flares, verbose=verbose,
        )


class ActiveRegionSingleInstance(DatasetBuilder):
    """One label per active-region instance, from that region's flares only.

    cadence (optional) keeps only the instances at start, start + cadence,
    ...; all regions share this grid, and start defaults to the earliest
    instance.
    """

    def __init__(
        self,
        prediction_window,
        *,
        reducer,
        labeler,
        cadence=None,
        start=None,
        missing_active_region="negative",
        include_metadata=False,
        include_flares=False,
        verbose=False,
    ):
        super().__init__(
            prediction_window, reducer=reducer, labeler=labeler, target="active_region",
            cadence=cadence, start=start,
            missing_active_region=missing_active_region, include_metadata=include_metadata,
            include_flares=include_flares, verbose=verbose,
        )


class ActiveRegionSequence(DatasetBuilder):
    """One label per sequence of instances of one region, from that region's flares after its last instance."""

    def __init__(
        self,
        prediction_window,
        *,
        reducer,
        labeler,
        cadence,
        observation_window,
        sliding_window,
        start=None,
        missing_active_region="negative",
        include_flares=False,
        verbose=False,
    ):
        super().__init__(
            prediction_window, reducer=reducer, labeler=labeler, target="active_region",
            cadence=cadence, observation_window=observation_window, sliding_window=sliding_window, start=start,
            missing_active_region=missing_active_region, include_flares=include_flares, verbose=verbose,
        )
