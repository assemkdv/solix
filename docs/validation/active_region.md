# Active-region labels: reproduction of the published arpatch-sfp labels

This page reports two different things, which must not be confused:

| | Question | Status |
|---|---|---|
| **A. Reproduction of the published labels** | Given the patch list and labels published in `arpatch-sfp`, which rule produces exactly those labels? | Done: a look-back rule reproduces all 511,421 rows. |
| **B. Validation of the complete original patch dataset** | Are labels correct for *every* SHARP patch that existed before the legacy filters? | Not done: the unfiltered patch table is not public. |

A finding in A says nothing about patches that were filtered out before
publication, and it does not show that the labels are physically correct;
it only shows how the published labels were computed.

---

## A. Reproduction of the published labels

### Sources

All files come from `github.com/chetrajpandey/arpatch-sfp` at commit
`ee20210dd6c6aa0396f1f4b8575ad05e9d0142b6` (2024-06-14), under
`data_process/data_labeling/`. SHA-256 (first 12 hex digits; full values
in `active_region_summary.json`):

| File | Rows | Positive (`label`=1) | HARPs | `harp_start` months | SHA-256 |
|---|---|---|---|---|---|
| `data_labels/-90_to_90/Fold1_test.csv` | 120,047 | 2,646 | 950 | 1–3 | `65a1d7d88671` |
| `data_labels/-90_to_90/Fold2_test.csv` | 130,961 | 2,193 | 1,010 | 4–6 | `46535fa33d02` |
| `data_labels/-90_to_90/Fold3_test.csv` | 127,450 | 2,378 | 956 | 7–9 | `104e56deb78a` |
| `data_labels/-90_to_90/Fold4_test.csv` | 132,963 | 3,098 | 973 | 10–12 | `760e327104f3` |
| **Total (denominator)** | **511,421** | **10,315** | 3,889 | | |
| `allharp2noaa.csv` (HARP → NOAA) | | | | | `575c40caf42e` |
| `goes_flares_integrated.csv` (GOES catalog) | 14,401 flares | | | | `2a1c76ff6ba5` |

Why these four files: `step5_CV_lon_filter.py` splits one filtered
dataset into four folds by the month of `harp_start`, so the four
`Fold*_test.csv` files together are the complete published −90°…90°
dataset. No `(harpnumber, timestamp)` pair appears twice within or across
them.

How they relate to `train/val/test.csv` (checked by hashing and by
matching `(harpnumber, timestamp)` keys):

- `val.csv` is byte-identical to `Fold3_test.csv`, and `test.csv` to `Fold4_test.csv`;
- `train.csv` (33,873 rows) is a subset of Fold1 + Fold2 (251,008 rows)
  that keeps all 4,839 positives and samples the rest, as
  `step6_undersample.py` does. It has a `log_scale` column that the
  `step6_undersample.py` in this commit does not write, so the exact script
  version that produced it is not in the repository.

The earlier version of this page used train + val + test (294,286 rows).
Because `train.csv` is a sample, the four fold files are the better
denominator.

### Rule tested

Re-implemented from `step1_harp_to_noaa.py`, `step3_labeling.py` and
`step4_with_lon_filtering.py`:

1. Map each GOES flare's `noaa_active_region` to HARP numbers with
   `allharp2noaa.csv` (a HARP can list several NOAA regions; a flare
   whose region is in two HARPs counts for both).
2. For a patch of HARP *h* at time *t*: take flares of *h* with
   `start_time >= t − 1 day` and `start_time <= t` (both ends included).
3. `goes_class` = the alphabetically largest class string among them
   (pandas `.max()`); empty if none. The published files write `FQ`
   (421,927 rows) where the class is empty.
4. `label` = 1 if `goes_class >= "M1.0"` alphabetically and not empty.

### Step by step, with real rows

What `scripts/validate_active_region.py` does, shown on a few real rows.
The rows are copied from the actual files and run (full files are not
committed).

**Input 1 – published labels** (`Fold1_test.csv`, first rows):

```text
harpnumber,timestamp,SHARP_FILE,goes_class,label,harp_start,LON_FWT,log_scale
330,2011-01-02 22:00:00,hmi.sharp_cea_720s.330.20110102_220000_TAI.magnetogram.jpg,FQ,0,2011-01-02 17:00:00,-48.937923,0.1
330,2011-01-02 23:00:00,hmi.sharp_cea_720s.330.20110102_230000_TAI.magnetogram.jpg,FQ,0,2011-01-02 17:00:00,-48.385422,0.1
```

`FQ` means "no flare in the window". The script turns each row into an
image-index row: `timestamp`, `active_region` = `harpnumber`,
`image_path` = `SHARP_FILE`.

**Input 2 – HARP → NOAA mapping** (`allharp2noaa.csv`). One HARP can list
several NOAA regions:

```text
DEF_HARPNUM,NOAA_ARS,NRT_HARPNUM
1.0,11067,
2.0,11064,
115.0,"11093,11099",        (line 33)
```

**Step 1 – flare catalog keyed by HARP.** The GOES catalog's
`noaa_active_region` is replaced by the HARP number(s) that contain it
(`harp_keyed_catalog()` in the script, same join as `step1_harp_to_noaa.py`).
Output, first rows (`active_region` is now a HARP number):

```text
start_time,peak_time,end_time,class,active_region
2010-05-01 01:34:00,2010-05-01 01:39:00,2010-05-01 01:43:00,C5.7,1
2010-05-01 05:23:00,2010-05-01 05:27:00,2010-05-01 05:31:00,B1.6,2
```

**Step 2 – compute labels two ways.**

```python
# (a) the legacy rule, re-implemented in the script (NOT a package feature):
#     same HARP, start_time >= t - 1 day and start_time <= t,
#     goes_class = alphabetical max, label = goes_class >= "M1.0"
lookback = legacy_lookback(index, catalog)

# (b) the package, which only looks forward:
from flare_indexer import BinaryThresholdStrategy, DatasetBuilder
forward = DatasetBuilder(
    prediction_window="24h", strategy=BinaryThresholdStrategy("M"),
    target="active_region", event_time="start", interval_mode="right_closed",
    missing_active_region="error", include_metadata=True,
).build("ar_image_index.csv", "ar_flare_catalog_harp.csv")
```

**Output – HARP 362 around its M1.9 flare.** Flares of HARP 362 in the
catalog include C1.0 at 2011-02-08 18:46, C4.0 at 21:06, C1.1 at
2011-02-09 00:08, and **M1.9 at 2011-02-09 01:23**:

```text
          timestamp harp published_class  published_label lookback_class  lookback_label  forward_label
2011-02-08 20:00:00  362            C1.0                0           C1.0               0              1
2011-02-08 21:00:00  362            C1.0                0           C1.0               0              1
2011-02-08 22:00:00  362            C4.0                0           C4.0               0              1
2011-02-08 23:00:00  362            C4.0                0           C4.0               0              1
2011-02-09 00:00:00  362            C4.0                0           C4.0               0              1
2011-02-09 01:00:00  362            C4.0                0           C4.0               0              1
2011-02-09 02:00:00  362            M1.9                1           M1.9               1              0
2011-02-09 03:00:00  362            M1.9                1           M1.9               1              0
```

How to read it:

- **Published** labels become 1 only from 02:00, *after* the M1.9 started
  (01:23). The look-back rule gives exactly the same class and label on
  every row.
- **Forward** (the package) gives 1 from 20:00 to 01:00, *before* the
  flare, because the M1.9 lies in the next 24 hours of those images; and
  0 from 02:00, because no M-class flare of HARP 362 follows within 24
  hours.

This is the whole finding in eight rows: the published labels describe
the past 24 hours, the package predicts the next 24 hours.

### Results

| Rule | `label` agreement | `goes_class` agreement | Positives |
|---|---|---|---|
| Look-back `[t − 1 day, t]` (legacy rule above) | **511,421 / 511,421** | **511,421 / 511,421** | 10,315 |
| `flare_indexer` forward `(t, t + 24h]`, `start_time`, same HARP, ≥ M1.0 | 499,009 / 511,421 (97.573 %) | – | 10,141 |

Per file, the look-back rule agrees on every row of every file. The
same check on `train.csv` + `val.csv` + `test.csv` (294,286 rows, second
command below) also gives 294,286 / 294,286 for both columns, and 96.567 %
for the forward window. The
forward window differs on 12,412 rows (6,119 package = 1 / published = 0;
6,293 package = 0 / published = 1). In these data the alphabetical
maximum class never differs from the numeric maximum, so the known
alphabetical-ordering issue does not affect these labels.

Two independent columns (`label` and the full `goes_class` string) are
reproduced exactly on every row. This is strong evidence that the
published labels were computed with the look-back window as written in
`step3_labeling.py`. It does not tell us whether that window was
*intended*; that is decision [D2](../decisions.md#d2-active-region-window-direction).

Example: HARP 362 (NOAA 11153). An M1.9 flare starts 2011-02-09 01:23.
In `train.csv` the patches from 2011-02-09 02:00 to 2011-02-10 01:00 are
labeled `M1.9` / 1, and the patch at 2011-02-08 21:00, before the flare,
is labeled `C1.0` / 0.

### Command

```bash
pip install -e .
python scripts/validate_active_region.py --summary docs/validation/active_region_summary.json
# the same check on the training split files:
python scripts/validate_active_region.py --labels \
  https://raw.githubusercontent.com/chetrajpandey/arpatch-sfp/ee20210dd6c6aa0396f1f4b8575ad05e9d0142b6/data_process/data_labeling/data_labels/train.csv \
  https://raw.githubusercontent.com/chetrajpandey/arpatch-sfp/ee20210dd6c6aa0396f1f4b8575ad05e9d0142b6/data_process/data_labeling/data_labels/val.csv \
  https://raw.githubusercontent.com/chetrajpandey/arpatch-sfp/ee20210dd6c6aa0396f1f4b8575ad05e9d0142b6/data_process/data_labeling/data_labels/test.csv
```

The first run downloads about 65 MB of fold CSVs plus the mapping and
catalog into `scripts/.cache/` (ignored by git); no images are
downloaded. The console output is in `active_region_log.txt`.

### Limitations of A

- The comparison uses the published patch list as given. Patch times come
  from SHARP filenames (TAI); GOES times are UTC; like the legacy code, no
  conversion is applied.
- `last_flare_start` of the HARP-mapped catalog is 2018-12-15; the full
  catalog's latest recorded flare starts 2018-12-29 13:16. 1,244 patches
  are later than that, and 8 are earlier than the first recorded flare.
  The catalog does not state its observation coverage, so whether those
  patches had no flares or were simply not covered is unknown
  ([D4](../decisions.md#d4-catalog-coverage)).

---

## B. Validation of the complete original patch dataset

Not possible with public data. What is missing:

| Needed | Where the legacy code expects it | Status |
|---|---|---|
| Full SHARP patch table (all patches, widths, `LON_FWT`, `USFLUX_RATIO`, …) | `../../fits_ar_processor/meta_data_updated.csv`, read by `step2_data_filtering.py` | not in the repository |
| Intermediate files `intermediates/harp2noaaAR_mappedGOES.csv`, `intermediates/stride/*.csv` | written by steps 1–4 | not in the repository |

Known filters between the full table and the published files:
`step2` keeps on-the-hour patches; `step4` keeps `Cropped_Width >= 70` and
`|LON_FWT| <= 30/60/90`; the published `-90_to_90` folds have `LON_FWT`
between −89.8° and 89.8°. Patches removed by these filters were never
labeled in the published data, so nothing here validates their labels or
the effect of the filters on class balance.

With the full patch table, the same script could be pointed at it
(`--labels`) after converting it to the published column layout.
