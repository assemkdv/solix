# Active-region check against published arpatch-sfp labels

**Status:** partial. The package runs on the real active-region data, but
it cannot reproduce the published labels until decision
[D2](../decisions.md#d2-active-region-window-direction) is made.

## Inputs (from `chetrajpandey/arpatch-sfp`, commit `ee20210`, 2024-06-14)

| Input | Path in that repository |
|---|---|
| Published AR labels (patch list) | `data_process/data_labeling/data_labels/{train,val,test}.csv` |
| HARP → NOAA mapping | `data_process/data_labeling/allharp2noaa.csv` |
| GOES catalog | `data_process/data_labeling/goes_flares_integrated.csv` (byte-identical to the full-disk one) |

Hashes are in `active_region_summary.json`.

## How the data was mapped

- **Image → region:** each published row has `harpnumber` and the SHARP
  filename, so `active_region` = HARP number.
- **Flare → region:** the GOES catalog gives a NOAA number; it is joined to
  HARP numbers with `allharp2noaa.csv` (exploding HARPs that cover several
  NOAA regions), exactly as `step1_harp_to_noaa.py` does.

## Results

| Quantity | Value |
|---|---|
| Published patches (train + val + test) | 294,286 |
| HARPs | 3,702 |
| Time range | 2010-05-01 05:00 → 2019-03-10 20:00 |
| Most patches at one timestamp | 21 |
| GOES flares without a NOAA region | 2,343 of 14,401 |
| Flares with a NOAA region but no HARP | 108 |
| Flare–HARP pairs | 12,026 (11,950 distinct flares) |
| Published positive labels | 10,315 |

| Labeling rule | Agreement with published labels | Positives |
|---|---|---|
| `flare_indexer`, `(t, t+24h]` on `start_time`, same HARP | 284,184 / 294,286 (96.567 %) | 7,831 |
| Legacy rule re-implemented: `[t−24h, t]` on `start_time`, same HARP | **294,286 / 294,286 (100 %)** | 10,315 |

**Finding:** the published AR labels are reproduced exactly by a
**look-back** window – "did this region start an M/X flare in the 24
hours *before* (and including) the image time?" – which is what
`step3_labeling.py` computes:

```python
(events_df['start_time'] >= timestamp - pd.Timedelta('1 days')) &
(events_df['start_time'] <= timestamp)
```

Example: HARP 362 (NOAA 11153). An M1.9 flare starts 2011-02-09 01:23.
The published labels mark patches from 2011-02-09 02:00 to 2011-02-10
01:00 as M1.9 / positive, and the patch at 2011-02-08 21:00 (before the
flare) as C1.0 / negative.

The full-disk legacy labels use a forward window (see
[full_disk.md](full_disk.md)), so the two published datasets use opposite
window directions. This needs Dr. Pandey's confirmation before any
active-region validation can be called complete.

## What this does *not* validate

- Patches removed by the legacy width (`Cropped_Width >= 70`) and longitude
  (`|LON_FWT| <= 30/60/90`) filters are not in the published files, so
  the full SHARP patch → HARP table (`meta_data_updated.csv`, referenced
  by `step2_data_filtering.py` but not public) is still missing.
- 286 patches are later than the last catalog flare (2018-12-29), so
  their labels reflect missing catalog coverage, not quiet regions
  ([D4](../decisions.md#d4-catalog-coverage)).
- SHARP filenames are in TAI; GOES times are UTC.

## Reproduce

```bash
python scripts/validate_active_region.py --summary docs/validation/active_region_summary.json
```

Downloads about 36 MB of CSV (plus the shared GOES catalog) into `scripts/.cache/`; runtime ≈ 100 s.
