"""Inspect all-event counts/presence and minimum flux without losing events."""
import flare_indexer as f
import pandas as pd

events = [f.FlareEvent(pd.Timestamp('2024-01-01 01:05'), c,
                      pd.Timestamp('2024-01-01 01:00'), '100')
          for c in ['C2', 'C2', 'X1', 'C10']]
minimum = f.DatasetBuildingPipeline(f.MinFluxReducer(), f.RegressionLabeler())
counts = f.DatasetBuildingPipeline(f.ClassCountReducer(), f.ClassCountLabeler())
presence = f.DatasetBuildingPipeline(f.ClassCountReducer(), f.ClassPresenceLabeler())

if __name__ == '__main__':
    for name, pipeline in [('minimum W/m²', minimum), ('counts', counts), ('presence', presence)]:
        print(name, pipeline.label(events))
        print(name, 'empty:', pipeline.label([]))
