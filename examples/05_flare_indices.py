"""
Example 5 -- the four flare indices of Hong, Ji, Pandey, Aydin (DaWaK 2023).

For the flares in the prediction window of each image:

    GC max   strongest GOES class (peak flux)
    GC sum   sum(C_i) + 10 * sum(M_j) + 100 * sum(X_k)  (A and B add 0)
    RXF max  largest relative X-ray flux increase
    RXF sum  sum of the relative X-ray flux increases

RXF = a flare's peak flux / the background X-ray flux of the 24 hours
before it. add_relative_flux computes it once from 1-minute X-ray flux
and writes it into the catalog. The RXF indices label single images only.

Run:  python examples/05_flare_indices.py
"""
from pathlib import Path

import flare_indexer as f
import pandas as pd

DATA = Path(__file__).parent / "data"
OUTPUT = Path(__file__).parent / "output"

# The worked example of Hong et al., 2015-09-20T04:55 to 2015-09-21T04:55.
paper_classes = ["M1.5", "C2.6", "C1.4", "M2.1", "C2.8", "C4.2", "C1.5"]
paper_events = [
    f.FlareEvent(peak_time=None, goes_class=c, start_time=None, active_region=None)
    for c in paper_classes
]
reducer = f.CumulativeClassIndexReducer()

INDICES = {
    "gc_max": f.MaxFluxReducer(),
    "gc_sum": f.CumulativeClassIndexReducer(),
    "rxf_max": f.MaxRelativeFluxReducer(),
    "rxf_sum": f.SumRelativeFluxReducer(),
}


def main():
    print("worked example:", ", ".join(paper_classes))
    print("cumulative index:", reducer.reduce(paper_events))
    for threshold in (10, 20, 50):
        label = f.NumericThresholdLabeler(threshold).assign(reducer.reduce(paper_events))
        print(f"label (index >= {threshold}): {label}")
    print("empty window:", reducer.reduce([]))
    print("A/B-only window:", reducer.reduce([
        f.FlareEvent(peak_time=None, goes_class=c, start_time=None, active_region=None)
        for c in ("B5.0", "A1.2")
    ]))

    OUTPUT.mkdir(exist_ok=True)
    catalog = f.add_relative_flux(DATA / "flare_catalog.csv", DATA / "xray_flux.csv",
                                  output_path=OUTPUT / "05_flare_catalog_rxfi.csv")
    print("\nflares with background flux and RXF:")
    print(catalog[["start_time", "class", "background_flux", "rxfi"]].to_string(index=False, float_format="%.3g"))

    table = None
    for name, index_reducer in INDICES.items():
        labels = f.DatasetBuilder(
            prediction_window="24h",
            strategy=f.DatasetBuildingPipeline(index_reducer, f.RegressionLabeler()),
        ).build(DATA / "full_disk_image_index.csv", OUTPUT / "05_flare_catalog_rxfi.csv")
        table = labels[["timestamp"]].copy() if table is None else table
        table[name] = labels["label"]
    table["rxf_max_ge_10"] = (table["rxf_max"] >= 10).astype(int)
    print("\nfull-disk images, window (t, t+24h), every 12th shown:")
    print(table.iloc[::12].to_string(index=False, float_format="%.3g"))


if __name__ == "__main__":
    main()
