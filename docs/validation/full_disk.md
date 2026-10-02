# Full-disk validation on the supplied real data

**Status:** reproduced on 2026-10-01 from this repository
(`scripts/validate_full_disk.py`), not copied from earlier reports.

## Inputs

| Input | Source | SHA-256 |
|---|---|---|
| HMI image list | `explainingFullDisk/labeling/data_labels/totalfiles_jpg_512.csv` | `767e5b17…26ddc` |
| GOES flare catalog | `explainingFullDisk/labeling/data_source/goes_flares_integrated.csv` | `2a1c76ff…cf48` |
| Legacy labels | `explainingFullDisk/labeling/data_labels/NEW_full_dataset_cleaned_1_hours_with_loc_and_time_new.csv` | `cc8cef71…c7ee` |

`explainingFullDisk` was at commit `e7d6481` (2023-03-07) when these files
were downloaded. Full hashes are in the JSON summaries next to this file.
No image files were downloaded – only these CSVs.

## Settings

Chosen to match the legacy script `labeling/labeling.py`:

| Setting | Value | Legacy code |
|---|---|---|
| `target` | `full_disk` | all flares |
| `event_time` | `start` | `df.start` |
| `interval_mode` | `right_closed` | `start > pws` and `start <= pwe` |
| `prediction_window` | `24h` **and** `23h59min59s` | `pwe = pws + 23:59:59` |
| threshold | `M` (≥ M1.0) | first letter `M` or `X` |
| image list | `strict=False`, `duplicate_policy="first"` | – |

## Results

| Quantity | Value |
|---|---|
| Paths in the image list | 63,729 |
| Excluded non-image paths (e.g. `labels_store/*.csv`) | 80 |
| Valid images (no duplicate timestamps) | 63,649 |
| Time range | 2010-12-06 07:00 → 2018-12-30 23:00 |
| Flares in the catalog | 14,401 |

| | window `24h` → (t, t+24h] | window `23h59min59s` → (t, t+23:59:59] |
|---|---|---|
| Positive labels | 9,006 | 9,000 |
| Negative labels | 54,643 | 54,649 |
| Agreement with legacy | 63,643 / 63,649 (99.9906 %) | **63,649 / 63,649 (100 %)** |
| Differences | 6 (all package = 1, legacy = 0) | 0 |

All six differences with the 24-hour window are images for which an
M-class flare starts **exactly** 24 hours later:

| Image time | Flare | Flare start |
|---|---|---|
| 2011-03-06 05:00 | M1.2 | 2011-03-07 05:00 |
| 2011-08-07 18:00 | M3.5 | 2011-08-08 18:00 |
| 2013-07-02 07:00 | M1.5 | 2013-07-03 07:00 |
| 2014-08-23 12:00 | M5.9 | 2014-08-24 12:00 |
| 2014-10-19 09:00 | M3.9 | 2014-10-20 09:00 |
| 2014-12-03 08:00 | M1.3 | 2014-12-04 08:00 |

The legacy window stops one second short of 24 hours, so these flares are
outside it. Setting `prediction_window="23h59min59s"` removes every
difference. Which boundary is intended is decision
[D1](../decisions.md#d1-24-hour-window-boundary).

Positive labels per year (24h window): 2010 0/562, 2011 1,237/6,528,
2012 1,358/6,515, 2013 1,437/7,657, 2014 2,707/8,408, 2015 1,664/8,511,
2016 250/8,426, 2017 353/8,488, 2018 0/8,554.

## Caveats

- This validates **agreement with the legacy implementation**, not the
  physical correctness of the catalog.
- The catalog does not state its observation coverage. Its latest
  recorded flare starts 2018-12-29 13:16; 58 images have windows ending
  after that time (34 images are later than it). They are labeled from
  the flares the catalog lists. That latest flare is not taken as the end
  of coverage – near solar minimum a long flare-free period is plausible –
  so whether these 0 labels reflect quiet Sun or missing coverage is
  unknown (decision [D4](../decisions.md#d4-catalog-coverage)).
- Both the package and the legacy script read the same catalog, so
  agreement says nothing about flares missing from it.
- Times are compared as given; no TAI/UTC conversion.

## Reproduce

```bash
pip install -e .
python scripts/validate_full_disk.py --legacy-labels default --prediction-window 24h \
    --summary docs/validation/full_disk_summary_24h.json
python scripts/validate_full_disk.py --legacy-labels default --prediction-window 23h59min59s \
    --summary docs/validation/full_disk_summary_23h59min59s.json
```

The first run downloads about 20 MB of CSV into `scripts/.cache/`
(ignored by git). The full label table (`scripts/output/full_disk_labels.csv`,
about 5.8 MB: `timestamp, image_path, label, max_flux`) is regenerated and not
committed. Runtime ≈ 30 s per run on a laptop.

Files here: `full_disk_summary_*.json` (counts, settings, hashes,
mismatch rows), `full_disk_log_*.txt` (console output).
