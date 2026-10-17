# Cumulative class index: reference calculation and published SuryaBench labels

**Status:** reproduced on 2026-10-16 from this repository
(`scripts/validate_cumulative_index.py`). Full numbers, hashes, revisions
and every modified input row are in
[cumulative_index_summary.json](cumulative_index_summary.json).

This page answers two questions, which are kept apart:

| | Question | Result |
|---|---|---|
| **A. Reference calculation** | Does `CumulativeClassIndexReducer` compute what SuryaBench's `flare_index_creator.py` computes, on the same catalog? | Yes: every index within 2.3 × 10⁻¹³, every ≥ 10 label identical; 9 windows differ at ≥ 20 only because the reference sums in floating point (19.999999999999996 vs exactly 20). |
| **B. Published labels** | Does the package reproduce the public SuryaBench `cumulative_index` / `label_cum`? | With SuryaBench's own catalog: all 128,328 rows (index within 4.6 × 10⁻¹³, labels identical). With the lab's catalog: all 75,638 ≥ 10 labels within the comparison bounds identical; 158 raw indices differ, each explained by a flare record that exists only in SuryaBench's catalog. |

Agreement shows how the published numbers were computed. It does not show
that the index is the right scientific target, that A/B flares should be
ignored, or that the window should be `(t, t+23:59:59]`; those are open
([decisions.md](../decisions.md), Q1).

## Inputs

| Input | Source | Revision | SHA-256 |
|---|---|---|---|
| Lab GOES catalog | `explainingFullDisk/labeling/data_source/goes_flares_integrated.csv` | commit `e7d6481` | `2a1c76ff…cf48` |
| SuryaBench catalog | `NASA-IMPACT/SuryaBench/flare/flare_catalog_2010-2024.csv` | commit `bbf4f88c` (2025-05-24); `flare_index_creator.py` last changed in `579fc5f9` | `2c75afd7…126e64` |
| Published labels | `huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-flare-forecasting`, `data.csv` (CC BY 4.0) | dataset revision `bf474bc1` (2025-12-16) | in the JSON |

| Catalog | Flares used | Recorded-event bounds (first → last start) | A / B / C / M / X |
|---|---|---|---|
| Lab | 14,401 | 2010-05-01 01:34 → 2018-12-29 13:16 | 38 / 5,817 / 7,754 / 742 / 50 |
| SuryaBench | 27,210 of 27,211 rows | 2010-05-01 01:34 → 2024-12-31 22:46 | 96 / 9,033 / 15,773 / 2,180 / 128 |

**Coverage is unknown.** The bounds above are the first and last recorded
event start times. They say nothing about when the instrument was
observing or whether every flare in between was recorded; neither file
carries coverage information, and none was established independently.
Everything below that is called "within bounds" means only "the whole
window lies between the first and the last recorded event".

Published labels: 128,328 hourly rows, 2010-05-13 00:00 → 2024-12-31 23:00,
no duplicated timestamps, no non-finite `cumulative_index`, `label_cum`
only 0/1, `label_cum = 1` on 41,967 rows, and
`label_cum == (cumulative_index >= 10)` on every row. The file is used as
downloaded; `max_goes_class` and `label_max` are not compared.

### Benchmark-only input modifications

All of these exist only inside this script's cache copies; no source file
is edited, and no start time or class of a kept row is changed. The
summary JSON lists every affected row.

| Input | Modification | Rows | Reason |
|---|---|---|---|
| Lab catalog | The seven documented `peak_time` corrections of `scripts/catalog_corrections.py` ([full_disk.md](full_disk.md), Step 2a) | 7 | The package rejects a catalog whose peak precedes its start. Peaks only; start-time matching is unaffected. |
| SuryaBench catalog | Columns renamed to the lab names (`event_starttime` → `start_time`, `event_peaktime` → `peak_time`, `event_endtime` → `end_time`, `fl_goescls` → `goes_class`, `ar_noaanum` → `noaa_active_region`), then `adapt_goes_catalog` | all | Same adapter path as the lab catalog. |
| SuryaBench catalog | `peak_time` blanked where it precedes `start_time` | 13 | Same upstream HEK error as the lab file (7 of the 13 are the same flares as the lab corrections; the other 6, from 2021 on, carry peak timestamps days away from the flare). The package rejects such rows; a blank peak is accepted and cannot matter for start-time matching. |
| SuryaBench catalog | `peak_time` blanked where it follows `end_time` | 11 | Same reason (2020–2024 rows; 9 B, 1 C, 1 M class). |
| SuryaBench catalog | `end_time` blanked where it precedes `start_time` | 0 | – |
| SuryaBench catalog | Row dropped: class `A0.0`, start 2019-08-18 18:56 | 1 | Zero multiplier, rejected by the package's class parser. SuryaBench's regex accepts it, but an A value contributes `None` (nothing) to its sum, so dropping it cannot change any index. |

Both catalogs contain repeated records (same start time and class; the
lab file has 242 such rows, mostly one with a 4-digit and one with a
5-digit NOAA number). Neither SuryaBench nor the package removes them, so
a repeated C/M/X record counts twice. This is a property of the inputs
and is left as is.

## Settings

| Setting | Value | Why |
|---|---|---|
| `event_time` | `start` | SuryaBench filters on `event_starttime` |
| `interval_mode` | `right_closed` | `event_starttime > ws` and `<= ws + windowsize` |
| `prediction_window` | `23h59min59s` | `windowsize={"hours": 23, "minutes": 59, "seconds": 59}` |
| grid | hourly, 2010-05-01 00:00 → 2024-12-31 23:00 (128,616 timestamps, unique by construction) | `cadence={"hours": 1}` |
| thresholds | 10 (SuryaBench `thres_cum`) and 20 (DaWaK 2023 AR study) | |
| tolerance | 10⁻⁹ absolute on the raw index; labels compared exactly | |

These are the reference code's conventions, chosen to reproduce its
numbers. The paper's text writes the window as `[t_i, t_i + 24h]`, closed
at both ends. That interval is not one of the two `interval_mode` options
(`[t, t+w)` and `(t, t+w]`), so it cannot be run here; a flare starting
exactly at `t` or exactly at `t + 24h` is the only case where the
conventions differ. The package defaults (`peak`, `[t, t+w)`) are not the
reference convention and were not changed.

## Steps

1. **Grid.** `pd.date_range(first_day, last_day, freq="1h")` over the
   union of the catalog's event bounds and the published labels, written
   to `scripts/.cache/cumulative_grid_<catalog>.csv` as an image index.
2. **Package.** `DatasetBuilder(prediction_window="23h59min59s",
   event_time="start", interval_mode="right_closed",
   strategy=DatasetBuildingPipeline(CumulativeClassIndexReducer(),
   RegressionLabeler())).build(grid, catalog)`. The reducer sums in decimal
   arithmetic and rounds once.
3. **Reference.** A literal re-implementation of SuryaBench's
   `rolling_window()`: `np.select` gives C → n, M → 10 n, X → 100 n and
   `None` for A/B; flares with `t < start <= t + 23:59:59` are selected by
   `searchsorted`; `pandas.Series.sum()` adds the values in floating point
   (skipping `None`). Labels: `index >= threshold`.
4. **Compare A.** Raw index exactly and within tolerance; labels exactly;
   every threshold flip between the decimal and the float sum is listed.
5. **Compare B.** Inner join with `data.csv` on `timestamp`. Rows are
   split into *within bounds* (`first recorded start <= t` and
   `t + window <= last recorded start`) and *extending past the last
   recorded event*. The second group is reported separately. Within it, a
   window that begins before the final recorded event can still contain
   that event; a window that begins at or after it contains no recorded
   event and is 0. The number of non-zero values on each side is reported.
   Raw-index and binary-label agreement are reported separately for both
   groups, with the published `label_cum` compared as published (not
   recomputed).
6. **Guards.** The script stops if the published file has duplicated
   timestamps or `label_cum` values outside {0, 1}, if the package output
   is not aligned with the grid, or if the package produces a non-finite
   value. Non-finite values on either side of a comparison are excluded
   and counted (`nonfinite_excluded`) instead of being folded into
   "different" or "label 0" counts. In this run every count is 0.
7. **Investigate.** For every within-bounds disagreement the script prints
   the flares the package matched; the attribution below was then made by
   diffing the two catalogs. The rule was not changed.

## Results

### A. Package vs reference calculation (same catalog)

| Catalog | Rows | Raw index exactly equal | Within 10⁻⁹ | Max difference | ≥ 10 agree | ≥ 20 agree |
|---|---|---|---|---|---|---|
| Lab | 128,616 | 119,927 | 128,616 | 2.3 × 10⁻¹³ | 128,616 (100 %) | 128,608 (8 flips) |
| SuryaBench | 128,616 | 111,547 | 128,616 | 2.3 × 10⁻¹³ | 128,616 (100 %) | 128,607 (9 flips) |

The non-identical raw values are floating-point rounding of the reference
sum. All flips are of one kind: the package gives exactly `20.0`, the
float reference `19.999999999999996`, so package = 1 and reference = 0 at
threshold 20 (2011-09-27 03:00 to 07:00, 2014-10-03 04:00 to 06:00, and,
with the SuryaBench catalog, 2023-10-10 14:00). There is no flip at 10.
A user who needs the float behavior can sum the multipliers in floating
point in a custom reducer; the package keeps the exact sum.

### B. Package vs published SuryaBench labels

**SuryaBench catalog as input** (event bounds to 2024-12-31 22:46):

| Rows | Raw index exactly equal | Within 10⁻⁹ | Beyond | Max difference | ≥ 10 recomputed agree | published `label_cum` vs package ≥ 10 | ≥ 20 recomputed agree |
|---|---|---|---|---|---|---|---|
| 128,328 matched | 117,373 (91.46 %) | 128,328 | 0 | 4.6 × 10⁻¹³ | 128,328 (100 %) | 128,328 (100 %) | 128,319 (the 9 flips above) |

Unmatched timestamps: 288 on the package side only (the grid starts
2010-05-01, the published file 2010-05-13); none on the published side.
25 matched windows (2024-12-30 23:00 → 2024-12-31 23:00) extend past the
last recorded event (2024-12-31 22:46); they agree as well. The 24 that
begin before that event contain it and are non-zero on both sides; the
one that begins after it (2024-12-31 23:00) contains no recorded event
and is 0.

**Lab catalog as input** (event bounds to 2018-12-29 13:16):

| Scope | Rows | Raw index exactly equal | Within 10⁻⁹ | Beyond | Max difference | ≥ 10 recomputed agree | published `label_cum` vs package ≥ 10 | ≥ 20 recomputed agree |
|---|---|---|---|---|---|---|---|---|
| Within bounds (2010-05-13 00:00 → 2018-12-28 13:00) | 75,638 | 69,940 (92.47 %) | 75,480 | 158 | 15.9 | 75,638 (100 %) | 75,638 (100 %) | 75,630 (the 8 flips) |
| All matched | 128,328 | 93,552 | 99,092 | 29,236 | 1,913 | 106,670 (83.12 %) | 106,670 | 111,030 |

The 52,690 windows that extend past the lab file's last recorded event
(2018-12-28 14:00 → 2024-12-31 23:00) are reported, not hidden. The 24
that begin before the final recorded event (2018-12-29 13:16) contain it,
but it is an A1.1 flare and the records before it in December 2018 are A
and B class, so the package's index is 0 on them; the windows that begin
after it contain no recorded event and are 0 by construction. The
published values are non-zero on 29,078 of these windows, which measures
the difference in recorded events between the two files, not the rule.

**The 158 within-bounds raw differences, all investigated.** Diffing the
two catalogs between the lab file's bounds finds 18 flare records only in
SuryaBench's file (16 B-class, which add nothing) and 8 C/M/X records
that SuryaBench has once more than the lab file:

| Record (start, class) | In SuryaBench | In lab | Hourly windows affected | Difference |
|---|---|---|---|---|
| 2010-10-26 01:06 C1.1 | 2 | 1 | 24 | 1.1 |
| 2011-01-14 12:52 C1.0 | 2 | 1 | 24 | 1.0 |
| 2014-01-04 09:48 C2.9 and 10:16 M1.3 | 2 each | 1 each | 25 | 2.9, 13.0 or 15.9 |
| 2016-07-07 07:49 C5.1 | 1 | 0 | 24 | 5.1 |
| 2016-08-30 09:58 C1.0 and 22:41 C1.5 | 2 each | 1 each | 37 | 1.0, 1.5 or 2.5 |
| 2017-02-23 20:36 C1.3 | 1 | 0 | 24 | 1.3 |

Every one of the 158 windows contains at least one of these records, and
recomputing the index from SuryaBench's catalog restricted to the same
75,638 rows reproduces the published values exactly. No difference
remains that the rule would have to explain. Whether repeated records
should count twice is a question about the catalogs, not about the
reducer; it is listed under the open questions.

## Reproduce

```bash
python scripts/validate_cumulative_index.py --summary docs/validation/cumulative_index_summary.json
python scripts/validate_cumulative_index.py --catalog lab --no-published      # reference check only
python scripts/validate_cumulative_index.py --prediction-window 24h            # another window convention
```

Downloads (about 13 MB) are cached in `scripts/.cache/`. The SuryaBench
catalog URL is pinned to the last commit that changed the file, resolved
through the GitHub API and recorded in the summary; without network access
to the API the script falls back to `main` and says so.
