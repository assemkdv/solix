# User guide

> Compact reference. For a step-by-step guide with verified examples, start with the
> [researcher manual](researcher_manual.md).

Reference for `flare-indexer` as implemented in this repository. Each
statement here describes current code behavior; open scientific choices
are listed in [decisions.md](decisions.md).

Contents: [inputs](#1-inputs) · [parameters](#2-datasetbuilder-parameters) ·
[time windows](#3-time-windows) · [labels](#4-labeling-strategies) ·
[sequences](#5-sequences) · [active regions](#6-active-region-filtering) ·
[outputs](#7-output-schemas) · [missing data](#8-missing-and-malformed-data) ·
[errors](#9-errors-and-warnings) · [reproducibility](#10-reproducibility-checklist)

---

## 1. Inputs

### 1.1 Image index (CSV)

| Column | Required | Type | Notes |
|---|---|---|---|
| `timestamp` | always | anything `pandas.to_datetime` parses | Observation time of the image. |
| `active_region` | when `target="active_region"` | int, float or string | NOAA AR number (or any region key, see §6). `13600`, `13600.0` and `"13600"` are treated as equal. Blank = missing. |
| any other column | no | any | Ignored for matching. Copied to the output in sequence mode, and in single-image mode when `include_metadata=True`. |

Rules by mode:

- **Single image:** no ordering or uniqueness requirement; output rows
  are in input order, one per input row (duplicates give duplicate rows).
- **Row-based sequences:** timestamps must be sorted ascending and unique
  (within each region when `target="active_region"`).
- **Time-based sequences:** timestamps must be unique (within each region
  for `target="active_region"`; order does not matter).
- With `include_metadata=True` or in sequence modes, an index column must
  not be named like an output column (`label`, `sequence_start`,
  `sequence_end`, `timestamps`, `n_images`); this raises `ValueError`
  instead of silently overwriting the label.

### 1.2 Flare catalog (CSV)

Two schemas are accepted and detected from the column names.

**Full-timestamp schema (recommended)** – what `adapt_goes_catalog()` produces:

| Column | Required | Example |
|---|---|---|
| `start_time` | yes | `2024-03-01 23:52:00` |
| `peak_time` | yes | `2024-03-02 00:07:00` |
| `end_time` | yes | `2024-03-02 00:20:00` |
| `class` | yes | `M2.7` |
| `active_region` | yes (values may be blank) | `13601` |
| `fl_lon`, `fl_lat` | no | `35.0`, `-8.0` |

**Legacy HHMM schema** – the scraper output: `date, start, peak, end,
class, active_region`, where `start/peak/end` are 4-digit `HHMM` times
combined with `date`. All three times get the same calendar date, so a
flare that crosses midnight is represented incorrectly in this schema;
use the full-timestamp schema for such catalogs. If a file has the
columns of both schemas, the legacy schema is used.

`class` must be a GOES class: a letter `A, B, C, M, X` (any case) followed
by an optional positive number (`M`, `M2`, `M2.3`, `X10`); surrounding
spaces are ignored. It is converted to peak X-ray flux in W/m²
(`A=1e-8, B=1e-7, C=1e-6, M=1e-5, X=1e-4`, times the number), and all
comparisons are numeric, never alphabetical.

Every class is checked when the catalog is loaded (`EventMatcher`,
`DatasetBuilder.build`, `adapt_goes_catalog`). Missing values (blank,
NaN) and malformed values (`Q2.0`, `M2.3.4`, `M0`, `2.3`) raise one
`InvalidGOESClassError` that counts all bad rows and names the first five
by file line. They are not dropped, not treated as "no flare" and not
given a default class, because each of those choices would silently
decide whether a flare of unknown size crosses the threshold. Clean or
remove such rows explicitly before labeling. `InvalidGOESClassError` is
a `ValueError` (and, for backward compatibility, also a `KeyError` and
`IndexError`).

### 1.3 Converting real-world inputs

| Function | Input | Output |
|---|---|---|
| `build_image_index_from_filenames(paths, output_path=None, strict=True, duplicate_policy="error", return_report=False)` | List of paths, or a headerless one-path-per-line file (e.g. `totalfiles_jpg_512.csv`). Basenames must look like `HMI.m2010.12.21_21.00.00.jpg`. | DataFrame `timestamp, image_path`, sorted by time. With `return_report=True`, also an `ImageIndexReport`. |
| `adapt_goes_catalog(input_path, output_path=None)` | CSV with `start_time, peak_time, end_time, goes_class, noaa_active_region` (+ optional `fl_lon, fl_lat`), e.g. `goes_flares_integrated.csv`. | Full-timestamp catalog DataFrame. Other columns are dropped. |

`strict=False` excludes rows that are not valid HMI image filenames
instead of raising; the report lists how many and which (first 200).
`duplicate_policy` is `"error"`, `"first"` or `"last"`.

---

## 2. `DatasetBuilder` parameters

```python
DatasetBuilder(
    prediction_window, strategy,
    target="full_disk", event_time="peak", interval_mode="left_closed",
    missing_active_region="negative", include_metadata=False,
    # row-based sequences
    sequence_length=None, stride=None, cadence_minutes=None,
    # time-based sequences
    start=None, cadence=None, observation_window=None, sliding_window=None,
).build(image_index_path, event_catalog_path, return_report=False)
```

| Parameter | Values | Default | Meaning |
|---|---|---|---|
| `prediction_window` | number (hours) or duration (`"24h"`, `"23h59min59s"`, `pd.Timedelta`) | – | Window length after the reference time. Must be > 0. |
| `strategy` | `BinaryThresholdStrategy(threshold)`, `MaxFlareStrategy()`, a `DatasetBuildingPipeline`, or any object with `label(flares)` | – | Turns the flares found in the window into one label. |
| `target` | `"full_disk"`, `"active_region"` | `"full_disk"` | Whether any flare counts, or only flares of the image's region. |
| `event_time` | `"peak"`, `"start"` | `"peak"` | Which flare time is compared to the window. |
| `interval_mode` | `"left_closed"`, `"right_closed"` | `"left_closed"` | `[t, t+w)` or `(t, t+w]`. |
| `missing_active_region` | `"negative"`, `"drop"`, `"error"` | `"negative"` | AR mode only; see §6.3. |
| `include_metadata` | `bool` | `False` | Single-image mode: keep other image-index columns in the output. |
| `sequence_length` | int ≥ 1 | 1 | Row-based: images per sequence. |
| `stride` | int ≥ 1 | 1 | Row-based: rows between sequence starts. |
| `cadence_minutes` | int > 0 | none | Row-based: skip sequences whose images are not exactly this far apart. |
| `start` | timestamp | earliest image | Time-based: first sequence start; must be an existing image time. |
| `cadence` | duration | – | Time-based: spacing of images inside a sequence. |
| `observation_window` | duration | – | Time-based: span covered by one sequence. |
| `sliding_window` | duration | – | Time-based: shift between sequence starts. |
| `return_report` | `bool` | `False` | Time-based only: also return a `SequenceBuildReport`. |

Row-based and time-based parameters cannot be mixed.

---

## 3. Time windows

For a reference time `t` (the image time, or the **last** image of a
sequence) and window length `w`:

| `interval_mode` | Flare counted if its `event_time` is in | Includes `t`? | Includes `t + w`? |
|---|---|---|---|
| `left_closed` (default) | `[t, t + w)` | yes | no |
| `right_closed` | `(t, t + w]` | no | yes |

`event_time="peak"` (default) uses `peak_time`; `event_time="start"`
uses `start_time`.

**Reproducing the legacy full-disk labels** (`labeling.py` in
`explainingFullDisk`) requires:

```python
DatasetBuilder(prediction_window="23h59min59s", event_time="start",
               interval_mode="right_closed", strategy=BinaryThresholdStrategy("M"))
```

because the legacy script uses `start > t` and `start <= t + 23:59:59`.
With `prediction_window=24` instead, a flare starting exactly at `t + 24h`
is counted. Which one is scientifically intended is an open decision
([decisions.md](decisions.md#d1-24-hour-window-boundary)).

All timestamps are compared as given: no time-zone or time-scale
conversion is done. Make sure the image index and the catalog use the same
time scale (see §8 on TAI vs. UTC).

---

## 4. Labeling strategies

| Strategy | Label | No flare in window |
|---|---|---|
| `BinaryThresholdStrategy("M")` | `1` if the strongest flare's flux ≥ flux of the threshold (`"M"` = M1.0 = 1e-5 W/m²), else `0` | `0` |
| `BinaryThresholdStrategy("C4.2")` | same, threshold C4.2 | `0` |
| `MaxFlareStrategy()` | flux of the strongest flare, in W/m² | `0.0` |

Both are built from two smaller pieces, `MaxFluxReducer` (flares → max
flux) and a labeler (`BinaryThresholdLabeler`, `RegressionLabeler`).
For class labels `"NF"`, `"A"` … `"X"` (the class of the strongest flare
in the window, not every class present), pass
`DatasetBuildingPipeline(reducer=MaxFluxReducer(), labeler=MulticlassLabeler())`
as the strategy – see `examples/09_multiclass_labels.py`.
`DatasetBuildingPipeline(reducer, labeler)` lets you swap either piece or
inspect each stage – see `examples/06_modular_pipeline.py`.

---

## 5. Sequences

The reference time of a sequence is its **last** image (`sequence_end`).

### 5.1 Row-based (`sequence_length`, `stride`, `cadence_minutes`)

Takes `sequence_length` consecutive rows starting at row 0, `stride`,
`2·stride`, … A sequence is skipped (not an error) when `cadence_minutes`
is set and any two neighboring images differ by anything else. Use this
when the image index is already regular.

### 5.2 Time-based (`start`, `cadence`, `observation_window`, `sliding_window`)

1. The source cadence is inferred as the most common gap between sorted
   unique timestamps. `cadence` must be a whole multiple of it (so a 1-hour
   index can be sampled every 2 hours, but not every 30 minutes).
2. `observation_window` and `sliding_window` must be whole multiples of
   `cadence`. Images per sequence = `observation_window / cadence`.
3. Candidate starts: `start`, `start + sliding_window`, … while the last
   required image time `start + observation_window − cadence` is not after
   the last image in the index.
4. A candidate needs an image at **exactly** every `start + i·cadence`.
   If any is missing, the candidate is skipped and recorded in the report.
   Nothing is interpolated or replaced by a nearby image.

Example: `cadence="2h", observation_window="12h", sliding_window="6h"`
gives 6 images at `t, t+2h, …, t+10h`, a new sequence every 6 hours, and
the label is computed from `t+10h`.

`build(..., return_report=True)` returns a `SequenceBuildReport` with the
inferred cadence, candidate/emitted/skipped counts, and the first 50
skipped starts and missing timestamps. In active-region mode it also has
`region_count` and `skipped_sequence_regions` (the region of each listed
skipped start).

### 5.3 Active-region sequences

With `target="active_region"`, the index is grouped by region first and
each region's sequences are built from its own images only (§6.2).

---

## 6. Active-region filtering

### 6.1 Matching rule

With `target="active_region"`, a flare is counted only if its catalog
`active_region` equals the image's `active_region` after normalization
(blank → missing; `13600.0` → `"13600"`). Flares from other regions are
ignored even when stronger. The region key is just a string: it can be a
NOAA number or, if both files use it, a HARP number.

### 6.2 Sequences

Rows are grouped by normalized region before sequences are built, so
several regions may be observed at the same timestamps and a sequence
never mixes regions. Within each region:

- row-based mode needs sorted, unique timestamps and takes consecutive
  rows of that region (`cadence_minutes` skips sequences with gaps);
- time-based mode needs unique timestamps; the source cadence is inferred
  from all images; without `start` each region's grid begins at its first
  image, with `start` all regions share the grid `start + k·sliding_window`;
- a missing frame only affects its own region.

Output rows are sorted by `sequence_start`, then `active_region`
(missing region last). Images without a region never join a region's
sequence (§6.3).

### 6.3 Missing active regions – two different cases

| Case | What happens | Why it matters |
|---|---|---|
| Image has no `active_region` | Controlled by `missing_active_region`: `"negative"` (default) → label as if no flare, and emit a `UserWarning` with the count; `"drop"` → leave it out; `"error"` → `ValueError`. In sequence modes such images never join a region's sequence; with `"negative"` they form one separate group (`active_region` = missing) whose sequences are labeled as if no flare occurred. | The true label is unknown, so a 0 may be a false negative. Whether any particular row is affected cannot be told from the data. |
| Catalog flare has no `active_region` | It never matches any image. A `UserWarning` reports how many such flares the catalog has. | If the flare came from a region in the dataset, that region's images may get 0 where 1 would be correct (a possible false negative). In `goes_flares_integrated.csv`, 2,343 of 14,401 flares (33 of 792 M/X flares) have no NOAA region. |

Neither case ever falls back to full-disk matching.

---

## 7. Output schemas

| Mode | Columns |
|---|---|
| Single image, full disk | `timestamp, label` (+ other index columns if `include_metadata=True`) |
| Single image, active region | `timestamp, label`; with `include_metadata=True`: `timestamp, active_region, label, <other index columns>` |
| Sequence, full disk | `sequence_start, sequence_end, timestamps, n_images, label, <other index columns as lists>` |
| Sequence, active region | `sequence_start, sequence_end, timestamps, n_images, active_region, label, <other index columns as lists>` |

`label` is `int` (0/1) for `BinaryThresholdStrategy`, `float` flux for
`MaxFlareStrategy`. An empty image index (header only) returns an empty
DataFrame with these columns.

---

## 8. Missing and malformed data

| Situation | Behavior |
|---|---|
| Image missing from a time-based sequence grid | Sequence skipped, listed in `SequenceBuildReport`. |
| Gap inside a row-based sequence | Skipped only if `cadence_minutes` is set; otherwise the gap is silently included. |
| Non-image path in an HMI file list | `strict=True`: `ValueError`; `strict=False`: excluded and counted in `ImageIndexReport`. |
| Two images with the same timestamp | Loader: per `duplicate_policy`. Full-disk sequence modes: `ValueError`. Active-region sequence modes: allowed in different regions, `ValueError` within one region. Single-image mode: both kept. |
| Flare with a missing or invalid class (`""`, NaN, `Q2.0`, `M0`) | `InvalidGOESClassError` when the catalog is loaded, listing the bad lines (§1.2). |
| Time the catalog may not cover | Labeled from whatever the catalog contains, i.e. 0 if it lists no flare (§8.1). |
| Flare that crosses midnight | Correct in the full-timestamp schema; wrong in the HHMM schema. |
| Different time scales | Not converted. HMI SHARP filenames are in TAI, GOES times in UTC (TAI − UTC = 34–37 s in 2010–2018), which only matters for flares within seconds of a window edge. |

### 8.1 Catalog coverage

A flare catalog lists flares; it does not say during which periods flares
were observed and recorded. If an image's window falls (partly) outside
the catalog's real coverage, the package still labels it from the flares
the catalog lists, so a 0 there can be a false negative. The package does
**not** infer coverage from the first or last recorded flare: a long
period without flares (for example near solar minimum) looks the same as
a period that was not covered. To exclude uncovered periods, filter the
image index yourself using coverage dates documented for the catalog.
Built-in coverage checks would need such explicit dates (decision D4 in
[decisions.md](decisions.md)).

---

## 9. Errors and warnings

All input problems raise `ValueError` (or its subclass
`InvalidGOESClassError`) with the file path and the offending column or
value, except `DatasetBuildingPipeline` components without
`reduce`/`assign` (`TypeError`). Common messages:

| Message starts with | Cause | Fix |
|---|---|---|
| `Flare catalog at … is missing required column(s)` | Neither catalog schema matched. | Rename columns or use `adapt_goes_catalog`. |
| `Image index at … is missing required column: timestamp` / `active_region` | Missing column. | Add it; `active_region` is only needed for AR mode. |
| `… must have timestamps sorted in ascending order` | Row-based sequences on unsorted data. | Sort by `timestamp`. |
| `… contains duplicate timestamp(s)` | Full-disk sequence mode with repeated times. | Deduplicate, or use `target="active_region"` if the rows are different regions. |
| `… duplicate timestamp(s) within active_region …` | One region has two images at the same time. | Deduplicate that region. |
| `… within rows with a missing active_region` | Several unknown-region patches at one time in sequence mode. | `missing_active_region="drop"` or fill in regions. |
| `… missing or invalid GOES class value(s) …` | Blank or malformed `class` values in the catalog. | Fix or remove those rows. |
| `… would overwrite output column(s) …` | An index column is named like an output column (e.g. `label`). | Rename it. |
| `cadence … is smaller than / not an exact multiple of the inferred source cadence` | Requested sampling incompatible with the data. | Choose a multiple of the source cadence. |
| `start=… has no exact matching image timestamp` | `start` is not an image time. | Use an existing time. |
| `… have a missing active_region` (`ValueError` or `UserWarning`) | Images without a region (§6.3). | Fill the region, or choose `"drop"`. |
| `… flare(s) in the catalog … have no active_region` (`UserWarning`) | Catalog flares without a region (§6.3). | Informational; consider the scientific impact. |

---

## 10. Reproducibility checklist

When publishing labels, record:

- package version or git commit;
- SHA-256 of the image index and flare catalog (the validation script
  prints and stores them);
- `prediction_window`, `event_time`, `interval_mode`, `target`,
  threshold, and `missing_active_region`;
- sequence parameters and the `SequenceBuildReport` counts;
- how the image index was built (`strict`, `duplicate_policy`, the
  `ImageIndexReport` counts).

## Minimum flux, counts and presence

`MinFluxReducer` selects the weakest recorded event (0.0 when empty).
`ClassCountReducer` counts all matched events into A/B/C/M/X bins; pair it
with `ClassCountLabeler` for counts or `ClassPresenceLabeler` for 0/1
presence. Empty windows have five zero values. These dictionary-valued
labels need explicit JSON serialization or expansion into five columns.
See the API reference and `examples/12_minimum_and_counts.py`.
Cumulative indexing still requires a defined scientific formula.
