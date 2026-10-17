"""Cumulative normalized GOES subclass index: raw value and thresholded label.

Formula (Hong, Ji, Pandey, Aydin, DaWaK 2023, Sec. 2.2; also coded in
NASA-IMPACT/SuryaBench flare/flare_index_creator.py):

    index = sum(C_i) + 10 * sum(M_j) + 100 * sum(X_k)

over the flares in the prediction window, using the numbers written in the
class text. A and B flares contribute 0 (selected convention, pending
Dr. Pandey's confirmation). SuryaBench binarizes at 10; the paper's
active-region study used 20.
"""
from pathlib import Path

import flare_indexer as f
import pandas as pd

DATA = Path(__file__).parent / "data"

# Part 1: the paper's worked window, 2015-09-20T04:55 to 2015-09-21T04:55.
paper_classes = ["M1.5", "C2.6", "C1.4", "M2.1", "C2.8", "C4.2", "C1.5"]
paper_events = [
    f.FlareEvent(peak_time=None, goes_class=c, start_time=None, active_region=None)
    for c in paper_classes
]
reducer = f.CumulativeClassIndexReducer()

# Part 2: the same reducer inside DatasetBuilder on the synthetic data,
# with the reference convention of the lab notebooks and SuryaBench:
# start time, (t, t+23:59:59]. Any other window is a parameter change.
raw_pipeline = f.DatasetBuildingPipeline(f.CumulativeClassIndexReducer(), f.RegressionLabeler())
binary_pipeline = f.DatasetBuildingPipeline(f.CumulativeClassIndexReducer(), f.NumericThresholdLabeler(10))
settings = dict(prediction_window="23h59min59s", event_time="start", interval_mode="right_closed")

if __name__ == "__main__":
    print("paper window:", ", ".join(paper_classes))
    print("cumulative index:", reducer.reduce(paper_events))
    for threshold in (10, 20, 50):
        label = f.NumericThresholdLabeler(threshold).assign(reducer.reduce(paper_events))
        print(f"label (index >= {threshold}): {label}")
    print("empty window:", reducer.reduce([]))
    print("A/B-only window:", reducer.reduce([
        f.FlareEvent(peak_time=None, goes_class=c, start_time=None, active_region=None)
        for c in ("B5.0", "A1.2")
    ]))

    raw = f.DatasetBuilder(strategy=raw_pipeline, **settings).build(
        DATA / "full_disk_image_index.csv", DATA / "flare_catalog.csv")
    binary = f.DatasetBuilder(strategy=binary_pipeline, **settings).build(
        DATA / "full_disk_image_index.csv", DATA / "flare_catalog.csv")
    table = pd.DataFrame({
        "timestamp": raw["timestamp"],
        "cumulative_index": raw["label"],
        "label_cum_ge_10": binary["label"],
    })
    print()
    print("full-disk images, start time, (t, t+23:59:59], first 12 rows:")
    print(table.head(12).to_string(index=False))
    print("images:", len(table), "| label_cum_ge_10 = 1:", int(table["label_cum_ge_10"].sum()),
          "| max index:", table["cumulative_index"].max())
