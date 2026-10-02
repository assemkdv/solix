# Researcher manual

This manual explains how to use flare-indexer to label solar images for
flare-prediction research. It assumes you know some Python and pandas,
but not the package. Every code block marked "Run" was executed against
the small synthetic files in `docs/fixtures/`, and the output shown is the
real output (checked automatically by `scripts/check_doc_examples.py`).

Related documents: [API reference](api_reference.md) (every public class
and function), [system design](architecture/system_design.md) (how the
code works inside), [decisions](decisions.md) (open scientific choices).

## Contents

1. [What the package does](#1-what-the-package-does)
2. [Installation](#2-installation)
3. [Key terms](#3-key-terms)
4. [First example, start to finish](#4-first-example-start-to-finish)
5. [Input formats](#5-input-formats)
6. [Output formats](#6-output-formats)
7. [Time windows](#7-time-windows)
8. [Strategies: binary labels and flux values](#8-strategies-binary-labels-and-flux-values)
9. [Single-image and sequence workflows](#9-single-image-and-sequence-workflows)
10. [Full-disk and active-region workflows](#10-full-disk-and-active-region-workflows)
11. [Missing regions and metadata](#11-missing-regions-and-metadata)
12. [Using each stage without DatasetBuilder](#12-using-each-stage-without-datasetbuilder)
13. [Custom reducers, labelers and strategies](#13-custom-reducers-labelers-and-strategies)
14. [Preparing real data](#14-preparing-real-data)
15. [Adapting the examples to your data](#15-adapting-the-examples-to-your-data)
16. [Errors and warnings](#16-errors-and-warnings)
17. [Limitations and unresolved scientific assumptions](#17-limitations-and-unresolved-scientific-assumptions)

---

## 1. What the package does

A flare-prediction model learns from pairs of *(image, label)*. The label
answers a question such as "did an M-class or stronger flare start in the
24 hours after this image?". flare-indexer computes those labels from:

- an **image index**: a table with one row per image and its time;
- a **flare catalog**: a table with one row per flare, its times, its
  GOES class and its active region.

It never opens image files. It can label single images or sequences of
images, use any flare on the Sun (full disk) or only flares from the
image's own active region, and return either a 0/1 label or the
strongest flare's X-ray flux.

Typical uses: building training/validation/test labels; reproducing labels
from an earlier study; checking how a window or threshold choice changes
class balance.

## 2. Installation

**Prerequisites:** Python 3.10 or newer (tested on 3.10–3.13), `pip`, and
`git`. The package depends only on `pandas` and `numpy`, which `pip`
installs automatically.

The package is **not on PyPI yet**. Install it from the GitHub repository:

```bash
git clone https://github.com/assemkdv/flare-indexer.git
cd flare-indexer
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # add ".[dev]" to also install pytest
```

**Verify the installation:**

```bash
python -c "import flare_indexer as f; print(len(f.__all__), 'public names'); print(f.DatasetBuilder)"
```

Expected:

```text
19 public names
<class 'flare_indexer.builder.DatasetBuilder'>
```

All examples below assume you run Python **from the repository root**,
because they read files from `docs/fixtures/`.

## 3. Key terms

| Term | Meaning |
|---|---|
| **GOES class** | Text naming a flare's peak X-ray brightness: a letter `A, B, C, M, X` and a number, e.g. `M2.3`. Each letter is 10 times stronger than the one before. |
| **X-ray flux** | The same strength as a number in W/m² (watts per square metre). `A1.0` = 1e-8, `B1.0` = 1e-7, `C1.0` = 1e-6, `M1.0` = 1e-5, `X1.0` = 1e-4; `M2.3` = 2.3e-5. The package always compares flux numbers, never class text. |
| **Active region (AR)** | A magnetically active area on the Sun, identified by a NOAA number (e.g. `13700`) or by an HMI HARP number. |
| **Full disk** | The whole visible Sun: any flare counts. |
| **Reference time `t`** | The time from which the window is measured: the image's time, or the **last** image's time for a sequence. |
| **Prediction window** | The period after `t` in which flares are looked for, e.g. 24 hours. |
| **Label** | The value produced per image or sequence: `0`/`1`, or a flux number. |
| **Reducer** | Turns a list of flares into one value (here: the maximum flux). |
| **Labeler** | Turns that value into the final label (here: compare with a threshold, or pass through). |
| **Strategy** | A ready-made reducer + labeler pair with a `label(flares)` method. |

## 4. First example, start to finish

The fixture files contain four invented flares and five invented images.

<!-- example: first_inputs -->
```python
from pathlib import Path

import pandas as pd

# Show the raw CSV text exactly as stored on disk.
print(Path("docs/fixtures/catalog.csv").read_text())
print(Path("docs/fixtures/images.csv").read_text())
```
<!-- output: first_inputs -->
```text
start_time,peak_time,end_time,class,active_region
2024-05-01 02:00:00,2024-05-01 02:15:00,2024-05-01 02:30:00,C4.0,13700
2024-05-01 05:30:00,2024-05-01 05:45:00,2024-05-01 06:00:00,M2.0,13700
2024-05-01 08:00:00,2024-05-01 08:10:00,2024-05-01 08:20:00,X1.2,13701
2024-05-02 00:00:00,2024-05-02 00:20:00,2024-05-02 00:40:00,M1.0,

timestamp,image_path
2024-05-01 00:00:00,hmi/HMI.m2024.05.01_00.00.00.jpg
2024-05-01 03:00:00,hmi/HMI.m2024.05.01_03.00.00.jpg
2024-05-01 06:00:00,hmi/HMI.m2024.05.01_06.00.00.jpg
2024-05-01 09:00:00,hmi/HMI.m2024.05.01_09.00.00.jpg
2024-05-01 18:00:00,hmi/HMI.m2024.05.01_18.00.00.jpg

```

Now label each image: "did an M-class or stronger flare **start** within
6 hours **after** the image?" (6 hours keeps this small example readable;
real studies typically use 24.)

<!-- example: first_build -->
```python
import flare_indexer as fidx

builder = fidx.DatasetBuilder(
    prediction_window=6,                         # hours
    strategy=fidx.BinaryThresholdStrategy("M"),  # 1 if the strongest flare is >= M1.0
    event_time="start",                          # compare the flare's start_time
    interval_mode="right_closed",                # window (t, t + 6h]
    include_metadata=True,                       # keep image_path in the output
)
labels = builder.build("docs/fixtures/images.csv", "docs/fixtures/catalog.csv")
print(labels.to_string(index=False))
```
<!-- output: first_build -->
```text
          timestamp  label                       image_path
2024-05-01 00:00:00      1 hmi/HMI.m2024.05.01_00.00.00.jpg
2024-05-01 03:00:00      1 hmi/HMI.m2024.05.01_03.00.00.jpg
2024-05-01 06:00:00      1 hmi/HMI.m2024.05.01_06.00.00.jpg
2024-05-01 09:00:00      0 hmi/HMI.m2024.05.01_09.00.00.jpg
2024-05-01 18:00:00      1 hmi/HMI.m2024.05.01_18.00.00.jpg
```

Why each label:

| Image `t` | Window `(t, t+6h]` | Flares starting in it | Strongest | Label |
|---|---|---|---|---|
| 00:00 | 00:00–06:00 | C4.0 (02:00), M2.0 (05:30) | M2.0 = 2.0e-5 ≥ 1e-5 | 1 |
| 03:00 | 03:00–09:00 | M2.0 (05:30), X1.2 (08:00) | X1.2 = 1.2e-4 | 1 |
| 06:00 | 06:00–12:00 | X1.2 (08:00) | X1.2 | 1 |
| 09:00 | 09:00–15:00 | none | – | 0 |
| 18:00 | 18:00–00:00 next day | M1.0 (starts 00:00 next day, exactly `t + 6h`; included because the right end is closed) | M1.0 = 1.0e-5 ≥ 1e-5 | 1 |

## 5. Input formats

### 5.1 Image index (CSV)

| Column | Required | Type | Meaning |
|---|---|---|---|
| `timestamp` | always | date-time text, e.g. `2024-05-01 03:00:00` | When the image was taken. |
| `active_region` | only with `target="active_region"` | integer or text (`13700`, `13700.0` and `"13700"` are the same) | Region shown in the image. Blank = unknown. |
| any other column | no | any | Carried through to the output in sequence mode, and in single-image mode with `include_metadata=True`. Must not be named `label`, `sequence_start`, `sequence_end`, `timestamps` or `n_images` when it is carried through. |

Timestamps carry no time zone and are compared exactly as written. Use
the same time scale (e.g. UTC) in the image index and the catalog; the
package does no conversion.

Single-image mode accepts any row order and repeated times. Sequence
modes need unique times (per region in active-region mode); row-based
sequences also need them sorted ascending.

### 5.2 Flare catalog (CSV)

**Full-timestamp schema** (recommended):

| Column | Required | Type | Meaning |
|---|---|---|---|
| `start_time`, `peak_time`, `end_time` | yes | date-time text | When the flare started, peaked and ended. |
| `class` | yes | GOES class text, e.g. `M2.0` | Flare strength. Checked when the file is loaded. |
| `active_region` | yes (cells may be blank) | integer or text | NOAA region of the flare. |
| `fl_lon`, `fl_lat` | no | number (degrees) | Flare location, kept on `FlareEvent` if present. |

**Legacy scraper schema:** `date, start, peak, end, class, active_region`,
where `start/peak/end` are 4-digit `HHMM` times (e.g. `0309`) combined
with `date`. All three get the same date, so a flare that crosses
midnight is wrong in this schema; use the full-timestamp schema for such
data. If a file has the columns of both schemas, the legacy schema wins.

**GOES class rules:** a letter `A`–`X` (any case), optionally followed by
a positive number; spaces around it are ignored. Blank or malformed
classes (`""`, `Q2.0`, `M2.3.4`, `M0`) stop loading with
`InvalidGOESClassError` naming the file lines. They are not skipped,
because a flare of unknown strength cannot be compared with a threshold.

## 6. Output formats

`build()` returns a pandas DataFrame. Columns by mode:

| Column | Appears in | Type | Meaning |
|---|---|---|---|
| `timestamp` | single image | Timestamp | The image's time (= reference time `t`). |
| `active_region` | single image AR mode with `include_metadata=True`; all AR sequence modes | text, or missing | Normalized region (`"13700"`). |
| `label` | always | `int` 0/1 (binary strategy) or `float` W/m² (max-flare strategy) | The answer for this image or sequence. |
| `sequence_start` | sequences | Timestamp | First image's time. |
| `sequence_end` | sequences | Timestamp | Last image's time (= reference time `t`). |
| `timestamps` | sequences | list of Timestamps | All image times in the sequence. |
| `n_images` | sequences | int | Number of images in the sequence. |
| other index columns | see 5.1 | as in the index (lists in sequence mode) | e.g. `image_path`. |

Column order: single image `timestamp, [active_region], label, [extra]`;
sequences `sequence_start, sequence_end, timestamps, n_images,
[active_region], label, [extra]`. An index with a header but no rows gives
an empty table with these columns.

## 7. Time windows

Three settings decide which flares count for reference time `t`:

| Setting | Values | Default | Effect |
|---|---|---|---|
| `prediction_window` | hours as a number (`24`, `0.5`) or a duration text (`"24h"`, `"23h59min59s"`) | – | Window length `w`. Must be > 0. |
| `event_time` | `"peak"`, `"start"` | `"peak"` | Which flare time must fall in the window. |
| `interval_mode` | `"left_closed"`, `"right_closed"` | `"left_closed"` | `[t, t+w)` includes `t`, excludes `t+w`; `(t, t+w]` excludes `t`, includes `t+w`. |

Windows always look **forward** from `t`. The package has no option to
look backward.

Run (requires `fidx` from §4):

<!-- example: windows -->
```python
for window, event_time, mode in [
    (6, "peak", "left_closed"),          # package defaults
    (6, "start", "right_closed"),        # lab full-disk convention
    ("5h59min59s", "start", "right_closed"),
]:
    result = fidx.DatasetBuilder(
        prediction_window=window, strategy=fidx.BinaryThresholdStrategy("M"),
        event_time=event_time, interval_mode=mode,
    ).build("docs/fixtures/images.csv", "docs/fixtures/catalog.csv")
    print(f"{str(window):>11} {event_time:5} {mode:12} labels={result['label'].tolist()}")
```
<!-- output: windows -->
```text
          6 peak  left_closed  labels=[1, 1, 1, 0, 0]
          6 start right_closed labels=[1, 1, 1, 0, 1]
 5h59min59s start right_closed labels=[1, 1, 1, 0, 0]
```

Only the last image (18:00) changes. The M1.0 flare *starts* at exactly
`t + 6h` and *peaks* 20 minutes later, so it counts only for the
start-based, right-closed 6-hour window. On real full-disk data the same
effect (with 24 hours) explains all six differences from the legacy
labels; see [decision D1](decisions.md#d1-24-hour-window-boundary).

## 8. Strategies: binary labels and flux values

| Strategy | Label when flares are found | Label when none are found |
|---|---|---|
| `BinaryThresholdStrategy("M")` | `1` if the strongest flare's flux ≥ the threshold's flux (`"M"` = M1.0 = 1e-5 W/m²; `"C4.2"` also works), else `0` | `0` |
| `MaxFlareStrategy()` | strongest flare's flux in W/m² (a number, not class text) | `0.0` |

Run (requires `fidx` from §4):

<!-- example: strategies -->
```python
flux = fidx.DatasetBuilder(
    prediction_window=6, strategy=fidx.MaxFlareStrategy(),
    event_time="start", interval_mode="right_closed",
).build("docs/fixtures/images.csv", "docs/fixtures/catalog.csv")
binary_c = fidx.DatasetBuilder(
    prediction_window=6, strategy=fidx.BinaryThresholdStrategy("C"),
    event_time="start", interval_mode="right_closed",
).build("docs/fixtures/images.csv", "docs/fixtures/catalog.csv")
print("max flux (W/m^2):", flux["label"].tolist())
print("binary, >= C1.0: ", binary_c["label"].tolist())
```
<!-- output: strategies -->
```text
max flux (W/m^2): [2e-05, 0.00012, 0.00012, 0.0, 1e-05]
binary, >= C1.0:  [1, 1, 1, 0, 1]
```

## 9. Single-image and sequence workflows

**Single image** (default): one row per image; `t` is the image's time.

**Sequences**: several consecutive images form one sample; `t` is the
**last** image's time. Two ways to define them (pick one per builder):

| Mode | Parameters | How sequences are formed |
|---|---|---|
| Row-based | `sequence_length`, `stride` (default 1), `cadence_minutes` (optional) | Take `sequence_length` consecutive rows, move `stride` rows, repeat. With `cadence_minutes`, skip any sequence whose images are not exactly that far apart. |
| Time-based | `cadence`, `observation_window`, `sliding_window`, optional `start` | Require images at exactly `start + k·cadence` for `observation_window / cadence` images; move the start by `sliding_window`. Incomplete sequences are skipped and reported, never filled in. |

Run (requires `fidx` from §4). The fixture images are at 00, 03, 06, 09
and 18 h; a 2-image, 3-hour grid is complete only for the first three
starts:

<!-- example: sequences -->
```python
sequences, report = fidx.DatasetBuilder(
    prediction_window=6, strategy=fidx.BinaryThresholdStrategy("M"),
    event_time="start", interval_mode="right_closed",
    cadence="3h", observation_window="6h", sliding_window="3h",
).build("docs/fixtures/images.csv", "docs/fixtures/catalog.csv", return_report=True)
print(sequences[["sequence_start", "sequence_end", "n_images", "label"]].to_string(index=False))
print("inferred cadence:", report.inferred_source_cadence)
print("candidates:", report.total_candidate_sequences,
      "emitted:", report.emitted_sequences, "skipped:", report.skipped_sequences)
print("first missing image of each skip:", [str(t) for t in report.missing_timestamps])
```
<!-- output: sequences -->
```text
     sequence_start        sequence_end  n_images  label
2024-05-01 00:00:00 2024-05-01 03:00:00         2      1
2024-05-01 03:00:00 2024-05-01 06:00:00         2      1
2024-05-01 06:00:00 2024-05-01 09:00:00         2      0
inferred cadence: 0 days 03:00:00
candidates: 6 emitted: 3 skipped: 3
first missing image of each skip: ['2024-05-01 12:00:00', '2024-05-01 12:00:00', '2024-05-01 15:00:00']
```

The 09:00 start needs a 12:00 image, the 12:00 start needs 12:00 and
the 15:00 start needs 15:00; none exist, so those three are skipped.

## 10. Full-disk and active-region workflows

`target="full_disk"` (default) counts every flare. `target="active_region"`
counts only flares whose catalog `active_region` equals the image's
`active_region`. The image index then needs an `active_region` column.

Run (requires `fidx` and `Path` from §4):

<!-- example: active_region -->
```python
import warnings

print(Path("docs/fixtures/ar_images.csv").read_text())   # raw input; the last region cell is blank
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    ar_labels = fidx.DatasetBuilder(
        prediction_window=6, strategy=fidx.BinaryThresholdStrategy("M"),
        target="active_region", event_time="start", interval_mode="right_closed",
        include_metadata=True,
    ).build("docs/fixtures/ar_images.csv", "docs/fixtures/catalog.csv")
print()
print(ar_labels.to_string(index=False, na_rep="(none)"))
print()
for warning in caught:
    print("WARNING:", warning.message)
```
<!-- output: active_region -->
```text
timestamp,active_region,image_path
2024-05-01 00:00:00,13700,patches/13700_0000.jpg
2024-05-01 00:00:00,13701,patches/13701_0000.jpg
2024-05-01 03:00:00,13700,patches/13700_0300.jpg
2024-05-01 03:00:00,13701,patches/13701_0300.jpg
2024-05-01 06:00:00,13700,patches/13700_0600.jpg
2024-05-01 06:00:00,,patches/unknown_0600.jpg


          timestamp active_region  label               image_path
2024-05-01 00:00:00         13700      1   patches/13700_0000.jpg
2024-05-01 00:00:00         13701      0   patches/13701_0000.jpg
2024-05-01 03:00:00         13700      1   patches/13700_0300.jpg
2024-05-01 03:00:00         13701      1   patches/13701_0300.jpg
2024-05-01 06:00:00         13700      0   patches/13700_0600.jpg
2024-05-01 06:00:00        (none)      0 patches/unknown_0600.jpg

WARNING: 1 of 4 flare(s) in the catalog at docs/fixtures/catalog.csv have no active_region. With target='active_region' they can never match any image, so a region that produced one of them may be labeled negative.
WARNING: 1 image(s) have a missing active_region (e.g. row 5 (2024-05-01 06:00:00)) and were labeled as if no flare occurred. Such labels are unknown, not confirmed negatives. Pass missing_active_region='drop' to leave them out, or 'error' to stop.
```

Why: at 00:00, region 13700 has C4.0 and M2.0 in `(00:00, 06:00]` → 1,
while region 13701's X1.2 starts at 08:00, outside the window → 0. At
03:00 the X1.2 is inside region 13701's window → 1. At 06:00 region 13700
has no flare → 0. The M1.0 flare has no region in the catalog, so it can
never count for any patch (first warning). The patch with a blank region
gets 0 by the default policy (see §11), and the second warning says so.

**AR sequences** are built **per region**: rows are grouped by region,
so several regions may share timestamps, and a sequence never mixes
regions.

<!-- example: ar_sequences -->
```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ar_seqs = fidx.DatasetBuilder(
        prediction_window=6, strategy=fidx.BinaryThresholdStrategy("M"),
        target="active_region", event_time="start", interval_mode="right_closed",
        missing_active_region="drop",
        cadence="3h", observation_window="6h", sliding_window="3h",
    ).build("docs/fixtures/ar_images.csv", "docs/fixtures/catalog.csv")
print(ar_seqs[["sequence_start", "sequence_end", "active_region", "label", "image_path"]].to_string(index=False))
```
<!-- output: ar_sequences -->
```text
     sequence_start        sequence_end active_region  label                                       image_path
2024-05-01 00:00:00 2024-05-01 03:00:00         13700      1 [patches/13700_0000.jpg, patches/13700_0300.jpg]
2024-05-01 00:00:00 2024-05-01 03:00:00         13701      1 [patches/13701_0000.jpg, patches/13701_0300.jpg]
2024-05-01 03:00:00 2024-05-01 06:00:00         13700      0 [patches/13700_0300.jpg, patches/13700_0600.jpg]
```

Region 13700 has images at 00, 03 and 06 h (two sequences); region 13701
only at 00 and 03 h (one sequence).

## 11. Missing regions and metadata

`missing_active_region` (active-region mode only) decides what happens to
images whose region is blank:

| Value | Single image | Sequences |
|---|---|---|
| `"negative"` (default) | label as if no flare, plus a `UserWarning` | such images form their own group (`active_region` missing), labeled as if no flare, plus a warning |
| `"drop"` | left out of the output | left out |
| `"error"` | `ValueError` | `ValueError` before any sequence is built |

A 0 produced by `"negative"` is a result of the policy, not an observed
absence of flares (see [decision D3](decisions.md#d3-images-without-an-active-region)).

`include_metadata=True` keeps the other index columns (and, in AR mode,
the normalized `active_region`) in single-image output. Without it the
output is only `timestamp, label`, and several patches taken at the same
time cannot be told apart.

Run (requires `fidx` and `warnings` from earlier sections):

<!-- example: policies -->
```python
for policy in ["drop", "error"]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = fidx.DatasetBuilder(
                prediction_window=6, strategy=fidx.BinaryThresholdStrategy("M"),
                target="active_region", missing_active_region=policy, include_metadata=True,
            ).build("docs/fixtures/ar_images.csv", "docs/fixtures/catalog.csv")
        print(policy, "->", len(out), "rows:", out["image_path"].tolist())
    except ValueError as error:
        print(policy, "-> ValueError:", error)
```
<!-- output: policies -->
```text
drop -> 5 rows: ['patches/13700_0000.jpg', 'patches/13701_0000.jpg', 'patches/13700_0300.jpg', 'patches/13701_0300.jpg', 'patches/13700_0600.jpg']
error -> ValueError: 1 image(s) have a missing active_region (e.g. row 5 (2024-05-01 06:00:00)). Their label is unknown, not negative. Fill in active_region, or pass missing_active_region='drop' or 'negative' to DatasetBuilder.
```

## 12. Using each stage without DatasetBuilder

`DatasetBuilder` repeats three stages for every image. You can call the
stages yourself, for one time point, to inspect or debug a label.

| Stage | Call | Returns |
|---|---|---|
| 1. Match events | `EventMatcher(catalog).query(t, window, ...)` or `pipeline.extract_events(...)` | `list[FlareEvent]` |
| 2. Reduce | `MaxFluxReducer().reduce(events)` or `pipeline.reduce_events(events)` | `float` (W/m²; `0.0` for no events) |
| 3. Assign | `BinaryThresholdLabeler("M").assign(value)` or `pipeline.assign_label(value)` | `int` 0/1 (binary) or the value itself (`RegressionLabeler`) |
| All three | `pipeline.run_one(matcher, t, window, ...)` | `PipelineResult(events, reduced_value, label)` |

Stop after **matching** (requires `fidx`, `pd` from §4):

<!-- example: stop_after_matching -->
```python
matcher = fidx.EventMatcher("docs/fixtures/catalog.csv")
t = pd.Timestamp("2024-05-01 03:00")
events = matcher.query(t, 6, event_time="start", interval_mode="right_closed")
print(type(events).__name__, len(events))
for event in events:
    print(" ", event.goes_class, event.start_time, "region", event.active_region)
```
<!-- output: stop_after_matching -->
```text
list 2
  M2.0 2024-05-01 05:30:00 region 13700
  X1.2 2024-05-01 08:00:00 region 13701
```

Stop after **reducing** (requires `events` from the previous block):

<!-- example: stop_after_reducing -->
```python
value = fidx.MaxFluxReducer().reduce(events)
print(type(value).__name__, value, "W/m^2")
```
<!-- output: stop_after_reducing -->
```text
float 0.00012 W/m^2
```

**Assign** a label to that value (requires `value`):

<!-- example: stop_after_assigning -->
```python
print("binary >= M:", fidx.BinaryThresholdLabeler("M").assign(value))
print("binary >= X2:", fidx.BinaryThresholdLabeler("X2").assign(value))
print("regression:", fidx.RegressionLabeler().assign(value))
```
<!-- output: stop_after_assigning -->
```text
binary >= M: 1
binary >= X2: 0
regression: 0.00012
```

All three stages at once, keeping every intermediate value (requires
`matcher`, `t`):

<!-- example: run_one -->
```python
pipeline = fidx.DatasetBuildingPipeline(fidx.MaxFluxReducer(), fidx.BinaryThresholdLabeler("M"))
result = pipeline.run_one(matcher, t, 6, event_time="start", interval_mode="right_closed")
print(type(result).__name__)
print("events:", [e.goes_class for e in result.events])
print("reduced_value:", result.reduced_value)
print("label:", result.label)
```
<!-- output: run_one -->
```text
PipelineResult
events: ['M2.0', 'X1.2']
reduced_value: 0.00012
label: 1
```

Region filtering works the same way: `matcher.query(t, 6,
active_region="13701", ...)` returns only that region's flares.

## 13. Custom reducers, labelers and strategies

No base class is needed; an object only has to have the right method:

| Role | Required method | Used by |
|---|---|---|
| Reducer (`EventReducer` contract) | `reduce(events) -> value` | `DatasetBuildingPipeline` |
| Labeler (`LabelAssigner` contract) | `assign(value) -> label` | `DatasetBuildingPipeline` |
| Strategy | `label(flares) -> label` | `DatasetBuilder(strategy=...)` |

A `DatasetBuildingPipeline` has a `label()` method, so it can be passed
as a strategy. Run (requires `fidx` from §4):

<!-- example: custom -->
```python
class FlareCount:
    """Reducer: number of flares in the window, any class."""
    def reduce(self, events):
        return len(events)

class AtLeastTwo:
    """Labeler: 1 if the window holds two or more flares."""
    def assign(self, value):
        return int(value >= 2)

counting = fidx.DatasetBuildingPipeline(FlareCount(), AtLeastTwo())
labels = fidx.DatasetBuilder(
    prediction_window=6, strategy=counting, event_time="start", interval_mode="right_closed",
).build("docs/fixtures/images.csv", "docs/fixtures/catalog.csv")
print(labels.to_string(index=False))

class StrongestClassText:
    """A strategy written from scratch: returns class text, or 'none'."""
    def label(self, flares):
        if not flares:
            return "none"
        return max(flares, key=lambda f: fidx.FluxConverter().to_flux(f.goes_class)).goes_class

text_labels = fidx.DatasetBuilder(
    prediction_window=6, strategy=StrongestClassText(), event_time="start", interval_mode="right_closed",
).build("docs/fixtures/images.csv", "docs/fixtures/catalog.csv")
print(text_labels["label"].tolist())
```
<!-- output: custom -->
```text
          timestamp  label
2024-05-01 00:00:00      1
2024-05-01 03:00:00      1
2024-05-01 06:00:00      0
2024-05-01 09:00:00      0
2024-05-01 18:00:00      0
['M2.0', 'X1.2', 'X1.2', 'none', 'M1.0']
```

An object without the required method is rejected when the pipeline is
created (`TypeError`).

## 14. Preparing real data

Two helpers convert common lab formats:

- `build_image_index_from_filenames(paths, ...)` reads HMI file paths
  (names like `HMI.m2024.05.01_00.00.00.jpg`) and returns
  `timestamp, image_path`. With `strict=False`, non-image paths are
  excluded and counted instead of raising an error.
- `adapt_goes_catalog(path, ...)` converts a catalog with `start_time,
  peak_time, end_time, goes_class, noaa_active_region` (like
  `goes_flares_integrated.csv`) to the full-timestamp schema.

Run (requires `fidx` from §4):

<!-- example: loaders -->
```python
index, report = fidx.build_image_index_from_filenames(
    "docs/fixtures/hmi_paths.txt", strict=False, return_report=True,
)
print(index.to_string(index=False))
print("rows:", report.total_input_rows, "excluded:", report.excluded_row_count, report.excluded_rows)
print()
catalog = fidx.adapt_goes_catalog("docs/fixtures/goes_catalog.csv")
print(catalog.to_string(index=False))
```
<!-- output: loaders -->
```text
          timestamp                                                 image_path
2024-05-01 00:00:00 /data/hmi_jpgs_512/2024/05/01/HMI.m2024.05.01_00.00.00.jpg
2024-05-01 03:00:00 /data/hmi_jpgs_512/2024/05/01/HMI.m2024.05.01_03.00.00.jpg
rows: 3 excluded: 1 ['/data/hmi_jpgs_512/labels_store/Fold1_train.csv']

         start_time           peak_time            end_time class active_region  fl_lon  fl_lat
2024-05-01 02:00:00 2024-05-01 02:15:00 2024-05-01 02:30:00  C4.0         13700   -12.0     8.0
2024-05-01 05:30:00 2024-05-01 05:45:00 2024-05-01 06:00:00  M2.0         13700   -11.5     8.0
```

Both helpers accept `output_path=` to write the converted CSV, which you
then pass to `DatasetBuilder.build()`. For the lab's real files, see
`scripts/validate_full_disk.py`.

## 15. Adapting the examples to your data

1. **Image index:** make a CSV with `timestamp` (and `active_region` for
   patches) plus any identifier columns you want back, e.g. `image_path`.
   Use `build_image_index_from_filenames` for HMI file lists.
2. **Catalog:** make a full-timestamp CSV, or convert a GOES catalog with
   `adapt_goes_catalog`. Check that `class` values are valid and that
   times use the same time scale as your images.
3. **Choose the question:** `prediction_window`, `event_time`,
   `interval_mode`, `target`, and a strategy with a threshold. Write
   these down; they define your labels.
4. **Choose single images or sequences** (§9).
5. **Run** `DatasetBuilder(...).build(index, catalog)`, read the
   warnings, and save the result with `to_csv`.
6. **Spot-check** a few labels with `EventMatcher.query` (§12).

For active-region patches identified by HARP numbers while the catalog
uses NOAA numbers, both files must use the same key; see
[active_region_workflow.md](active_region_workflow.md).

## 16. Errors and warnings

| Message (start) | Raised by | Cause | What to do |
|---|---|---|---|
| `Flare catalog at … is missing required column(s)` | loading the catalog | neither schema matched | rename columns or use `adapt_goes_catalog` |
| `Flare catalog at … has N missing or invalid GOES class value(s)` (`InvalidGOESClassError`) | loading the catalog | blank/malformed `class` | fix or remove those lines |
| `Image index at … is missing required column` | `build` | no `timestamp`, or no `active_region` in AR mode | add the column |
| `prediction_window must be …` | `DatasetBuilder(...)`, `query` | zero, negative, non-finite or unreadable window | use e.g. `24` or `"23h59min59s"` |
| `Cannot mix row-based sequence parameters …` | `DatasetBuilder(...)` | both sequence styles given | choose one |
| `… must have timestamps sorted …` / `… duplicate timestamp(s)` | `build` | sequence mode on unsorted or repeated times | sort, deduplicate, or use AR mode for several regions |
| `cadence … smaller than / not an exact multiple of the inferred source cadence` | `build` | sampling incompatible with the data | choose a multiple |
| `… would overwrite output column(s)` | `build` | index column named like an output column | rename it |
| `… have a missing active_region` (`ValueError` or `UserWarning`) | `build`, AR mode | blank regions in the index | fill them, or choose a policy (§11) |
| `… flare(s) in the catalog … have no active_region` (`UserWarning`) | `build`, AR mode | catalog flares without a region | informational; see §17 |

## 17. Limitations and unresolved scientific assumptions

**Software limitations**

- Windows only look forward. Reproducing the published active-region
  labels (which look back) is done only by
  `scripts/validate_active_region.py`, not by the package.
- No time-zone or TAI/UTC conversion.
- No catalog-coverage check: a catalog does not say which periods it
  covers, and the package does not guess.
- In AR sequence mode with the default policy, several region-less
  patches at the same time raise an error (use `"drop"`).

**Scientific assumptions that are not settled** (see
[decisions.md](decisions.md))

- Whether the window should end at `t + 24h` or `t + 23:59:59` (D1).
- Whether active-region labels should look forward or back (D2).
- How to treat images without a region (D3) and catalog flares without a
  NOAA region (D6). Both can cause false negatives (a 0 where 1 would be
  correct); which rows are affected is unknown.
- Whether a 0 in a period the catalog may not cover means "quiet Sun" or
  "no data" (D4).
- Whether image and catalog times share a time scale (SHARP file names
  are TAI; the GOES catalog is assumed to be UTC).

Matching published labels shows how they were computed, not that they are
scientifically correct.
