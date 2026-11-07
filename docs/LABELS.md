# Labels

A label describes the flares in one prediction window. Every label below
is made by a reducer and a labeler, passed to a dataset class as
`reducer` and `labeler`:

```python
import solix

builder = solix.FullDiskSingleInstance(
    "24h", reducer=solix.MaxFluxReducer(), labeler=solix.BinaryThresholdLabeler("M"),
)
```

## One window, every label

In `examples/data/`, the 24-hour window of the instance at
2024-03-01 00:00 holds three flares:

| Flare | Start | Peak flux (W/m²) | RXF |
|---|---|---|---|
| C3.2 | 03-01 04:00 | 3.2e-6 | 10.02 |
| M1.4 | 03-01 13:20 | 1.4e-5 | 50.58 |
| M2.7 | 03-01 23:52 | 2.7e-5 | 86.20 |

This is the label each reducer and labeler give that instance:

| Label | Reducer | Labeler | Value |
|---|---|---|---|
| 0/1: strongest flare ≥ M1.0 | `MaxFluxReducer()` | `BinaryThresholdLabeler("M")` | `1` |
| 0/1: strongest flare ≥ X1.0 | `MaxFluxReducer()` | `BinaryThresholdLabeler("X")` | `0` |
| Peak flux of the strongest flare | `MaxFluxReducer()` | `RegressionLabeler()` | `2.7e-05` |
| Peak flux of the weakest flare | `MinFluxReducer()` | `RegressionLabeler()` | `3.2e-06` |
| Mean peak flux | `MeanFluxReducer()` | `RegressionLabeler()` | `1.47e-05` |
| Class of the strongest flare, one-hot | `MaxFluxReducer()` | `MulticlassLabeler()` | `{"NF": 0, "A": 0, "B": 0, "C": 0, "M": 1, "X": 0}` |
| Number of flares per class | `ClassCountReducer()` | `ClassCountLabeler()` | `{"A": 0, "B": 0, "C": 1, "M": 2, "X": 0}` |
| Which classes occur (0/1 each) | `ClassCountReducer()` | `ClassPresenceLabeler()` | `{"A": 0, "B": 0, "C": 1, "M": 1, "X": 0}` |
| Cumulative index ΣC + 10ΣM + 100ΣX | `CumulativeClassIndexReducer()` | `RegressionLabeler()` | `44.2` (3.2 + 14 + 27) |
| Cumulative index ≥ 50, as 0/1 | `CumulativeClassIndexReducer()` | `NumericThresholdLabeler(50)` | `0` |
| Largest RXF | `MaxRelativeFluxReducer()` | `RegressionLabeler()` | `86.2` |
| Sum of RXF | `SumRelativeFluxReducer()` | `RegressionLabeler()` | `146.8` |
| Largest RXF ≥ 100, as 0/1 | `MaxRelativeFluxReducer()` | `NumericThresholdLabeler(100)` | `0` |

The threshold class can carry a number: `BinaryThresholdLabeler("C4.2")`.

A window with no flare gives 0, `0.0`, NF = 1 in the one-hot label (no flare) or all-zero counts.

## How a label is made

A label is made in two steps:

1. A **reducer** turns the window's flares into one value, for example the
   largest peak flux.
2. A **labeler** turns that value into the label, for example 1 if the
   value is at least M1.0.

The dataset classes take both as `reducer=` and `labeler=`. To inspect the
steps for one time outside a dataset class,
`DatasetBuildingPipeline(reducer, labeler).run_one(...)` returns the
flares, the reduced value and the label.

| Reducers | Labelers |
|---|---|
| `MaxFluxReducer`, `MinFluxReducer`, `MeanFluxReducer` | `BinaryThresholdLabeler(class)`: 0/1 against a GOES class |
| `ClassCountReducer` | `NumericThresholdLabeler(number)`: 0/1 against a number |
| `CumulativeClassIndexReducer` (single instances only) | `RegressionLabeler()`: the value itself |
| `MaxRelativeFluxReducer`, `SumRelativeFluxReducer` | `MulticlassLabeler()`: one-hot NF, A, B, C, M, X |
| | `ClassCountLabeler()`, `ClassPresenceLabeler()`: per-class counts or 0/1 |

Any object with a `reduce(flares)` method can be a reducer, and any object
with an `assign(value)` method can be a labeler. A reducer receives the
window's flares as a list of `FlareEvent` objects (`goes_class`,
`start_time`, `peak_time`, `end_time`, `active_region`, `fl_lon`, `fl_lat`,
`rxfi`; the last three are `None` if the catalog has no such column):

```python
class StrongFlareCount:
    def reduce(self, flares):
        return sum(f.goes_class[0] in "MX" for f in flares)

builder = solix.FullDiskSingleInstance(
    "24h", reducer=StrongFlareCount(), labeler=solix.RegressionLabeler(),
)
```

## Relative X-ray flux (RXF)

A flare's RXF is its peak X-ray flux divided by the background X-ray flux
of the 24 hours before it starts. The RXF labels read it from the
catalog's `rxfi` column and are for single instances only.

If the catalog has no `rxfi` column, `add_relative_flux` computes it once
from GOES 1–8 Å X-ray flux, a CSV with the columns `time, flux` and one
value per minute:

```python
solix.add_relative_flux("flare_catalog.csv", "xray_flux.csv",
                       output_path="flare_catalog_rxfi.csv")
```

If the files use other column names, add `catalog_columns=` and
`xray_columns=`, e.g. `xray_columns={"time_tag": "time", "xrsb_flux": "flux"}`
(see [DATASETS.md](DATASETS.md#your-own-column-names)).

It adds two columns, `background_flux` and `rxfi`. For each flare, the
background is the mean of the flux values in the 24 hours before its start
that are

- above 1e-9 W/m²,
- outside every catalog flare, and
- below the flux at the flare's start.

### Missing minutes

X-ray files have gaps where no value was recorded.
Missing minutes are skipped, not filled in: the background is the mean of
the minutes that exist. The one minute that must exist is the flare's start
minute, because the background is compared with the flux at that minute.
If it is missing, or none of the 24 hours remain, the flare gets no RXF and
a warning says how many flares were left without one. To fill gaps
(for example from another GOES satellite), fill them in the X-ray file
before calling `add_relative_flux`.
