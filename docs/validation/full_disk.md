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

## Step by step, with real rows

What `scripts/validate_full_disk.py` does, shown on a few real rows of
each file. The rows below are copied from the actual input files and the
actual run (full files are not committed; see "Reproduce").

### Step 1 – image list → image index

**Input** (`totalfiles_jpg_512.csv`, no header, one path per line; 63,729
lines). First lines, and a line that is not an image:

```text
/data/hmi_jpgs_512/2010/12/21/HMI.m2010.12.21_21.00.00.jpg
/data/hmi_jpgs_512/2010/12/21/HMI.m2010.12.21_09.00.00.jpg
/data/hmi_jpgs_512/2010/12/21/HMI.m2010.12.21_20.00.00.jpg
...
/data/hmi_jpgs_512/labels_store/C_full_dataset_cleaned_9_hours.csv     (line 33,542)
```

**Code:**

```python
from flare_indexer import build_image_index_from_filenames

image_index, report = build_image_index_from_filenames(
    "totalfiles_jpg_512.csv", strict=False, duplicate_policy="first", return_report=True,
)
```

**What it does.** Reads the time from each file name
(`HMI.m2010.12.06_07.00.00.jpg` → `2010-12-06 07:00:00`), skips the 80
lines that are not image names (like the `labels_store/*.csv` line), and
sorts by time.

**Output** (first rows of 63,649):

```text
          timestamp                                                 image_path
2010-12-06 07:00:00 /data/hmi_jpgs_512/2010/12/06/HMI.m2010.12.06_07.00.00.jpg
2010-12-06 08:00:00 /data/hmi_jpgs_512/2010/12/06/HMI.m2010.12.06_08.00.00.jpg
2010-12-06 09:00:00 /data/hmi_jpgs_512/2010/12/06/HMI.m2010.12.06_09.00.00.jpg
```

### Step 2 – GOES catalog → flare catalog

**Input** (`goes_flares_integrated.csv`, 14,401 rows, 33 columns). The
columns that are used, first two rows:

```text
         start_time           peak_time            end_time goes_class  noaa_active_region     fl_lon  fl_lat
2010-05-01 01:34:00 2010-05-01 01:39:00 2010-05-01 01:43:00       C5.7             11067.0 -73.000000    23.0
2010-05-01 05:23:00 2010-05-01 05:27:00 2010-05-01 05:31:00       B1.6             11064.0 -24.824109    15.0
```

**Code:**

```python
from flare_indexer import adapt_goes_catalog

catalog = adapt_goes_catalog("goes_flares_integrated.csv")
catalog.to_csv("flare_catalog_adapted.csv", index=False)
```

**What it does.** Keeps the three times, renames `goes_class` → `class`
and `noaa_active_region` → `active_region`, checks every class, and drops
the other 26 columns.

**Output** (first two rows):

```text
         start_time           peak_time            end_time class  active_region     fl_lon  fl_lat
2010-05-01 01:34:00 2010-05-01 01:39:00 2010-05-01 01:43:00  C5.7        11067.0 -73.000000    23.0
2010-05-01 05:23:00 2010-05-01 05:27:00 2010-05-01 05:31:00  B1.6        11064.0 -24.824109    15.0
```

### Step 3 – labels

**Code** (the script runs this twice: once with
`BinaryThresholdStrategy("M")` for `label`, once with `MaxFlareStrategy()`
for `max_flux`):

```python
from flare_indexer import BinaryThresholdStrategy, DatasetBuilder

builder = DatasetBuilder(
    prediction_window=24,            # or "23h59min59s" for the legacy window
    strategy=BinaryThresholdStrategy(threshold="M"),
    target="full_disk",
    event_time="start",              # compare each flare's start_time
    interval_mode="right_closed",    # window (t, t + 24h]
)
labels = builder.build("image_index_normalized.csv", "flare_catalog_adapted.csv")
```

**What it does.** For each image time `t`, finds flares whose
`start_time` is in `(t, t + 24h]` and gives `1` if the strongest is M1.0
or stronger.

**Output** (`scripts/output/full_disk_labels.csv`, 63,649 rows; columns
`timestamp, image_path, label, max_flux`). First rows:

```text
          timestamp                                                 image_path  label     max_flux
2010-12-06 07:00:00 /data/hmi_jpgs_512/2010/12/06/HMI.m2010.12.06_07.00.00.jpg      0 0.000000e+00
2010-12-06 08:00:00 /data/hmi_jpgs_512/2010/12/06/HMI.m2010.12.06_08.00.00.jpg      0 1.900000e-07
2010-12-06 09:00:00 /data/hmi_jpgs_512/2010/12/06/HMI.m2010.12.06_09.00.00.jpg      0 1.900000e-07
```

`max_flux` is the strongest flare's X-ray flux in W/m² (a number, not
class text): `0` means no flare in the window, `1.9e-07` is a B1.9 flare.

**Why a label turns from 0 to 1** – three consecutive images:

```text
          timestamp  label  max_flux
2011-01-26 23:00:00      0  0.000002
2011-01-27 00:00:00      0  0.000002
2011-01-27 01:00:00      1  0.000013
```

The catalog has an M1.3 flare starting at `2011-01-28 00:44`. For the
00:00 image the window ends at `2011-01-28 00:00`, before the flare, so
only C-class flares count → `0`. For the 01:00 image the window ends at
`2011-01-28 01:00`, which includes 00:44 → max flux 1.3e-5 = M1.3 → `1`.

### Step 4 – comparison with the legacy labels

**Input** (`NEW_full_dataset_cleaned_1_hours_with_loc_and_time_new.csv`,
the legacy output). The `label` column holds the image path; `goes_class`
holds the strongest class, or `NF` for no flare:

```text
label,goes_class,fl_lon,fl_lat,rest_fl,rest_lon,rest_lat
2010/12/06/HMI.m2010.12.06_07.00.00.jpg,NF,unk,unk,,,
2010/12/06/HMI.m2010.12.06_08.00.00.jpg,B1.9,-38.5479308163,16.0,['B1.9'],[-38.5479308163],[16.0]
```

**What the script does.** Reads the time from the legacy file name,
turns `goes_class` into 0/1 (`M` or `X` → 1, otherwise 0), and compares
with our `label` image by image.

**One of the six differences**, both windows side by side:

```text
          timestamp  label_24h  label_23h59m59s  legacy goes_class
2014-08-23 11:00:00          0                0  C6.0
2014-08-23 12:00:00          1                0  C6.0
2014-08-23 13:00:00          1                1  M5.9
```

The catalog has an M5.9 flare starting at exactly `2014-08-24 12:00`. For
the 12:00 image that is exactly `t + 24h`: inside `(t, t + 24h]` → `1`,
but outside the legacy `(t, t + 23:59:59]` → `0`, which matches the
legacy `C6.0`. For the 13:00 image both windows include it.

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
