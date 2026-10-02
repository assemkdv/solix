# Active-region workflow

How to label active-region (AR) image patches with `flare-indexer`, what
real data is needed, and what is still missing. Run
`examples/04_active_region.py` for a small synthetic version of
everything on this page.

## 1. What "active-region labeling" means here

A full-disk image shows the whole Sun, so any flare counts. An AR patch
(e.g. an HMI SHARP cutout) shows one region, so only flares *from that
region* should count. `target="active_region"` does exactly that: a flare
counts only if its region key equals the patch's region key.

## 2. The data you need

| Table | Columns | Where it comes from |
|---|---|---|
| Patch index | `timestamp`, `active_region`, `image_path` | SHARP filenames contain HARP number and time, e.g. `hmi.sharp_cea_720s.362.20110209_020000_TAI…` → HARP 362, 2011-02-09 02:00 (TAI) |
| Flare catalog | `start_time, peak_time, end_time, class, active_region` | `goes_flares_integrated.csv` via `adapt_goes_catalog()` |
| Region mapping | `DEF_HARPNUM`, `NOAA_ARS` | `allharp2noaa.csv` (arpatch-sfp) |

**The key problem:** patches are identified by **HARP** number, flares by
**NOAA** number, and the relation is many-to-many (one HARP covers up to
6 NOAA regions; one NOAA region is in up to 2 HARPs). Both tables must use
the same key. Two options:

- **HARP key (used by the legacy pipeline):** map each flare's NOAA number
  to its HARP(s) and write `active_region = HARP` in the catalog. A flare
  in a NOAA region shared by two HARPs appears once per HARP.
- **NOAA key:** map each patch to one NOAA number. Not possible for HARPs
  covering several NOAA regions without an extra rule.

`scripts/validate_active_region.py` (`harp_keyed_catalog()`) implements
the HARP-key option with pandas.

## 3. Step by step (HARP key)

```python
import warnings
import pandas as pd
import flare_indexer as fidx

# 1. Patch index: one row per patch.
patches = pd.read_csv("my_sharp_patches.csv")      # has harpnumber, timestamp, SHARP_FILE
index = pd.DataFrame({
    "timestamp": pd.to_datetime(patches["timestamp"]),
    "active_region": patches["harpnumber"].astype(int).astype(str),
    "image_path": patches["SHARP_FILE"],
})
index.to_csv("ar_index.csv", index=False)

# 2. Catalog keyed by HARP: see harp_keyed_catalog() in scripts/validate_active_region.py
#    -> ar_catalog.csv with active_region = HARP number

# 3. Label.
builder = fidx.DatasetBuilder(
    prediction_window=24,
    strategy=fidx.BinaryThresholdStrategy("M"),
    target="active_region",
    event_time="start",
    interval_mode="right_closed",
    missing_active_region="error",   # see decision D3
    include_metadata=True,           # keeps active_region and image_path
)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    labels = builder.build("ar_index.csv", "ar_catalog.csv")
for w in caught:
    print(w.message)                 # e.g. catalog flares without a region
```

Use `include_metadata=True`: without it the output is only
`timestamp, label`, and several patches observed at the same time cannot
be told apart.

## 4. Sequences of AR patches

Real AR data has many patches per timestamp (up to 22 in the published
files). `DatasetBuilder` groups the index by region before building
sequences, so no splitting is needed:

```python
seqs, report = fidx.DatasetBuilder(
    prediction_window=24, strategy=fidx.BinaryThresholdStrategy("M"),
    target="active_region", event_time="start", interval_mode="right_closed",
    missing_active_region="drop",
    cadence="1h", observation_window="6h", sliding_window="1h",
).build("ar_index.csv", "ar_catalog.csv", return_report=True)
print(report.region_count, report.emitted_sequences, report.skipped_sequences)
```

Rules (same for row-based and time-based modes unless noted):

- A sequence only contains images of one region. Different regions may
  share timestamps.
- Within a region, timestamps must be unique (else `ValueError` naming the
  region); row-based mode also needs them sorted within the region.
- A missing frame affects only its own region: row-based mode skips the
  sequence only if `cadence_minutes` is set (otherwise neighboring rows of
  that region are used, as in full-disk mode); time-based mode always
  skips an incomplete grid and records it in
  `report.skipped_sequence_starts` / `skipped_sequence_regions`.
- Time-based mode infers the source cadence from all timestamps. Without
  `start`, each region's grid starts at its first image; with `start`, all
  regions use the grid `start + k·sliding_window`.
- Output is sorted by `sequence_start`, then `active_region`.

## 5. Missing information – what can go wrong

All three cases below can produce **false negatives** (a 0 where the true
label would be 1). None of them proves that any particular row is
positive, and their effect on model training or skill scores is not
known in advance.

1. **Patch without a region.** Its label is unknown. It never matches a
   flare and never joins a region's sequence. Default
   (`missing_active_region="negative"`): labeled 0 with a warning;
   `"drop"`: removed; `"error"`: stop. In sequence modes, with the default
   such patches form one separate group (several unknown patches at the
   same time raise `ValueError`; use `"drop"`).
2. **Flare without a region.** It cannot be assigned to a patch. In the lab
   catalog: 2,343 of 14,401 flares (33 of 792 M/X), plus 108 flares whose
   NOAA region has no HARP. The package warns once per build.
3. **Times the catalog may not cover.** A catalog lists flares, not its
   observation periods, so the package cannot tell "no flare" from "not
   covered". It does not guess coverage from the first or last recorded
   flare. See [decisions.md](decisions.md) D4.

See [decisions.md](decisions.md) D3, D4 and D6.

## 6. Status

- **Reproduction of the published labels: done.** All 511,421 rows of
  the published `-90_to_90` fold files are reproduced exactly (label and
  class string) by the legacy look-back rule `[t − 1 day, t]`. The
  package's forward window agrees on 97.573 %. Details, sources and
  commands: [validation/active_region.md](validation/active_region.md).
- **Validation of the complete original patch dataset: not possible
  yet.** The unfiltered SHARP patch table (`meta_data_updated.csv`, read
  by `step2_data_filtering.py`) is not public.
- **Open:** which window direction is intended (decision D2).
