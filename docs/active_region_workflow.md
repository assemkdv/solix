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

Sequence modes need unique timestamps, and real AR data has many patches
per timestamp (up to 21 in the published files). Build sequences one
region at a time:

```python
for harp, rows in index.groupby("active_region"):
    rows.to_csv(f"index_{harp}.csv", index=False)
    seqs = fidx.DatasetBuilder(
        prediction_window=24, strategy=fidx.BinaryThresholdStrategy("M"),
        target="active_region", event_time="start", interval_mode="right_closed",
        cadence="1h", observation_window="6h", sliding_window="1h",
    ).build(f"index_{harp}.csv", "ar_catalog.csv")
```

Grouping by region inside `DatasetBuilder` is a planned improvement.

## 5. Missing regions – what can silently go wrong

1. **Patch without a region.** Its label is unknown. The default
   (`missing_active_region="negative"`) labels it 0 and warns; `"drop"`
   removes it; `"error"` stops. Previously it was labeled 0 silently.
2. **Flare without a region.** It can never match a patch, so the patch
   that produced it gets 0 for that flare. In the lab catalog: 2,343 of
   14,401 flares, including 33 M/X flares, plus 108 flares whose NOAA
   region has no HARP. The package warns once per build.
3. **Time after the catalog ends.** Treated as "no flare". 286 published
   patches are after the last catalog flare.

In all three cases the error goes in one direction: positives become
negatives. Models trained on such labels look worse on recall, and
evaluation on such labels can over- or under-state skill.
See [decisions.md](decisions.md) D3, D4 and D6.

## 6. Status of real AR validation

- **Done:** the package labels the 294,286 published arpatch-sfp patches
  using the HARP-keyed catalog ([validation/active_region.md](validation/active_region.md)).
- **Blocked:** the published labels use a **look-back** window
  `[t−24h, t]`; the package looks forward. Agreement is 96.567 % forward
  vs. 100 % for the re-implemented look-back rule. Needs decision D2.
- **Missing:** the unfiltered SHARP patch table (`meta_data_updated.csv`
  used by `step2_data_filtering.py`) – needed to validate labels before
  the legacy width/longitude filters and for location-based assignment.
