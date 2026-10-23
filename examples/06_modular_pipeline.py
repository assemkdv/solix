"""
Example 6 -- looking inside the labeling pipeline.

Every label is computed in three stages:
    EventMatcher (find flares) -> reducer (one number) -> labeler (final label)

DatasetBuildingPipeline exposes each stage, which is useful for checking
a single surprising label, and lets you plug in your own reducer.

Run:  python examples/06_modular_pipeline.py
"""

from pathlib import Path

import pandas as pd

import flare_indexer as fidx

DATA = Path(__file__).parent / "data"


class FlareCountReducer:
    """A custom reducer: how many flares are in the window (any class)."""

    def reduce(self, events):
        return len(events)


class AtLeastTwoFlares:
    """A custom labeler: 1 if the window holds two or more flares."""

    def assign(self, reduced_value):
        return int(reduced_value >= 2)


def main():
    matcher = fidx.EventMatcher(DATA / "flare_catalog.csv")
    image_time = pd.Timestamp("2024-03-01 12:00:00")

    pipeline = fidx.DatasetBuildingPipeline(fidx.MaxFluxReducer(), fidx.BinaryThresholdLabeler("M"))
    result = pipeline.run_one(matcher, image_time, 24, event_time="start")
    print(f"Image at {image_time}, window (t, t+24h) on start_time")
    for event in result.events:
        print(f"  matched {event.goes_class:<5} start={event.start_time}  region={event.active_region}")
    print(f"  reduced value (max flux): {result.reduced_value:.1e} W/m^2")
    print(f"  label (>= M1.0):          {result.label}")

    custom = fidx.DatasetBuildingPipeline(FlareCountReducer(), AtLeastTwoFlares())
    counted = custom.run_one(matcher, image_time, 24, event_time="start")
    print(f"\nCustom pipeline: {counted.reduced_value} flares in the window -> label {counted.label}")

    # A pipeline is also a drop-in strategy for DatasetBuilder.
    labels = fidx.DatasetBuilder(prediction_window=24, strategy=custom,
                                 event_time="start") \
        .build(DATA / "full_disk_image_index.csv", DATA / "flare_catalog.csv")
    print(f"Custom pipeline over the whole sample index: {int(labels['label'].sum())} of "
          f"{len(labels)} images have >= 2 flares in their window")


if __name__ == "__main__":
    main()
