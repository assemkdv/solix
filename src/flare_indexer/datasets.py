"""
The four kinds of labeled dataset, one class each:

    FullDiskSingleImage       one full-disk image   -> one label
    FullDiskSequence          a sequence of full-disk images -> one label
    ActiveRegionSingleImage   one active-region patch -> one label
    ActiveRegionSequence      a sequence of patches of one region -> one label

Each class takes only the settings that apply to its case and runs the
shared DatasetBuilder engine. All four are used the same way:

    labels = FullDiskSingleImage(
        "24h", reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"),
    ).build("image_index.csv", "flare_catalog.csv")

The reducer turns the flares of one window into one value and the
labeler turns that value into the label; see reducers.py and labelers.py.

Sequences are built on a time grid: `cadence` is the
time between images, `observation_window` the time a sequence covers,
`sliding_window` the step between sequence starts, and `start` the
optional first sequence start. Each sequence is labeled from its last
image.
"""

from .builder import DatasetBuilder


class FullDiskSingleImage(DatasetBuilder):
    """One label per full-disk image, from all flares in its window.

    cadence (optional) keeps only the images at start, start + cadence, ...
    (e.g. one image per hour from a 12-minute image list); start defaults
    to the earliest image.
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
    """One label per sequence of full-disk images, from the window after its last image."""

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


class ActiveRegionSingleImage(DatasetBuilder):
    """One label per active-region patch, from that region's flares only.

    cadence (optional) keeps only the patches at start, start + cadence, ...;
    all regions share this grid, and start defaults to the earliest patch.
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
    """One label per sequence of patches of one region, from that region's flares after its last image."""

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
