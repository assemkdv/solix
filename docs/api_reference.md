# API reference

Every public name exported by `flare_indexer` (the 24 names in
`flare_indexer.__all__`), every public method of those classes, and the
public module-level helpers that are not exported. A developer appendix
at the end describes private helpers; researchers should not depend on
them.

**How to read the examples.** Each "Run" block was executed from the
repository root, in order, in one Python session; the output shown is the
real output (checked by `scripts/check_doc_examples.py`). Blocks reuse
the names defined in [Setup](#setup) and, where stated, in earlier blocks.

**Units.** *GOES class* is text such as `"M2.0"`. *Flux* is a number in
W/m² (watts per square metre): `"M2.0"` → `2e-05`. Every comparison in the
package uses flux numbers, never class text.

## Inventory

| Name | Kind | Module | Section |
|---|---|---|---|
| `FluxConverter` | class | `classifier` | [link](#fluxconverter) |
| `FlareClassifier` | class | `classifier` | [link](#flareclassifier) |
| `InvalidGOESClassError` | exception | `classifier` | [link](#invalidgoesclasserror) |
| `FlareEvent` | data container | `events` | [link](#flareevent) |
| `EventMatcher` | class | `events` | [link](#eventmatcher) |
| `EventReducer` | Protocol (contract) | `reducers` | [link](#eventreducer) |
| `MaxFluxReducer` | class | `reducers` | [link](#maxfluxreducer) |
| `MinFluxReducer` | class | `reducers` | [link](#minfluxreducer) |
| `ClassCountReducer` | class | `reducers` | [link](#classcountreducer) |
| `ClassCountLabeler` | class | `labelers` | [link](#classcountlabeler) |
| `ClassPresenceLabeler` | class | `labelers` | [link](#classpresencelabeler) |
| `LabelAssigner` | Protocol (contract) | `labelers` | [link](#labelassigner) |
| `BinaryThresholdLabeler` | class | `labelers` | [link](#binarythresholdlabeler) |
| `RegressionLabeler` | class | `labelers` | [link](#regressionlabeler) |
| `MulticlassLabeler` | class | `labelers` | [link](#multiclasslabeler) |
| `BinaryThresholdStrategy` | class | `strategies` | [link](#binarythresholdstrategy) |
| `MaxFlareStrategy` | class | `strategies` | [link](#maxflarestrategy) |
| `PipelineResult` | data container | `pipeline` | [link](#pipelineresult) |
| `DatasetBuildingPipeline` | class | `pipeline` | [link](#datasetbuildingpipeline) |
| `DatasetBuilder` | class | `builder` | [link](#datasetbuilder) |
| `SequenceBuildReport` | report container | `builder` | [link](#sequencebuildreport) |
| `build_image_index_from_filenames` | function | `loaders` | [link](#build_image_index_from_filenames) |
| `adapt_goes_catalog` | function | `loaders` | [link](#adapt_goes_catalog) |
| `ImageIndexReport` | report container | `loaders` | [link](#imageindexreport) |

Public methods: `FluxConverter.to_flux`, `FluxConverter.to_class`, `FlareClassifier.is_strong`,
`EventMatcher.query`, `EventReducer.reduce` (contract),
`MaxFluxReducer.reduce`, `MinFluxReducer.reduce`, `ClassCountReducer.reduce`,
`ClassCountLabeler.assign`, `ClassPresenceLabeler.assign`, `LabelAssigner.assign` (contract),
`BinaryThresholdLabeler.assign`, `RegressionLabeler.assign`,
`MulticlassLabeler.assign`, `BinaryThresholdStrategy.label`, `MaxFlareStrategy.label`,
`DatasetBuildingPipeline.extract_events`, `.reduce_events`,
`.assign_label`, `.run_one`, `.label`, `DatasetBuilder.build`.

Public but not exported: `flare_indexer.classifier.parse_goes_class`,
`flare_indexer.classifier.validate_goes_class_column`, and module
constants (see [Module-level helpers and constants](#module-level-helpers-and-constants)).

## Setup

Run once before the other examples:

<!-- example: setup -->
```python
import warnings
from pathlib import Path

import pandas as pd
import flare_indexer as fidx

CATALOG = "docs/fixtures/catalog.csv"          # 4 synthetic flares (one without region)
IMAGES = "docs/fixtures/images.csv"            # 5 synthetic full-disk images
AR_IMAGES = "docs/fixtures/ar_images.csv"      # 6 synthetic patches (one without region)
print(Path(CATALOG).read_text())
```
<!-- output: setup -->
```text
start_time,peak_time,end_time,class,active_region
2024-05-01 02:00:00,2024-05-01 02:15:00,2024-05-01 02:30:00,C4.0,13700
2024-05-01 05:30:00,2024-05-01 05:45:00,2024-05-01 06:00:00,M2.0,13700
2024-05-01 08:00:00,2024-05-01 08:10:00,2024-05-01 08:20:00,X1.2,13701
2024-05-02 00:00:00,2024-05-02 00:20:00,2024-05-02 00:40:00,M1.0,

```

---

## classifier

### `FluxConverter`

```python
from flare_indexer import FluxConverter
FluxConverter()
```

**Purpose.** Converts GOES class text to X-ray flux in W/m².
**Inputs.** None.
**Public attributes.** `FLUX_MAP` (dict): letter → base flux,
`{"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}`;
`CLASS_LETTERS` (tuple): `("A", "B", "C", "M", "X")`, weakest to strongest.
This class is the only place that knows the GOES scale; `MulticlassLabeler`
and `ClassCountReducer` call it rather than repeating the boundaries.

### `FluxConverter.to_flux`

```python
FluxConverter().to_flux(goes_class: str) -> float
```

**Purpose.** Turn one class text into a flux number.

| Parameter | Type | Accepted values |
|---|---|---|
| `goes_class` | `str` | letter `A`, `B`, `C`, `M` or `X` (any case), optionally followed by a positive number; surrounding spaces ignored. `"M"` means `"M1.0"`. |

**What it does.** Parses the class, multiplies `FLUX_MAP[letter] × number`
in decimal arithmetic using the number exactly as written, and rounds to a
`float` once at the end. Equal values written differently give the
identical float: `"C10.0"` and `"M1.0"` are both exactly `1e-05`, `"M10.0"`
and `"X1.0"` both `0.0001`. (Multiplying floats, `1e-6 * 10.0`, would give
`9.999999999999999e-06`, just below an M1.0 threshold.)
**Returns.** `float`, W/m².

Run (requires Setup):

<!-- example: to_flux -->
```python
converter = fidx.FluxConverter()
for text in ["M2.0", "m2.0", "X1.2", "C4", "M", "C10.0", "M10.0"]:
    print(text, "->", converter.to_flux(text))
print(converter.to_flux("C10.0") == converter.to_flux("M1.0"))
```
<!-- output: to_flux -->
```text
M2.0 -> 2e-05
m2.0 -> 2e-05
X1.2 -> 0.00012
C4 -> 4e-06
M -> 1e-05
C10.0 -> 1e-05
M10.0 -> 0.0001
True
```

**Output explained.** `M` is 1e-5, so `M2.0` is 2e-5; case does not
matter; a bare letter means multiplier 1; `C10.0` is exactly M1.0 and
`M10.0` exactly X1.0.
**Errors.** `InvalidGOESClassError` for blank, NaN, unknown letters
(`"Z5.0"`), malformed numbers (`"M2.3.4"`) or zero (`"M0"`).
**Limitations.** Does not check physical plausibility (e.g. `"A900"` is
accepted).

### `FluxConverter.to_class`

```python
FluxConverter().to_class(flux) -> str
```

**Purpose.** The opposite direction of `to_flux`: the class letter of a
flux value.

| Parameter | Type | Accepted values |
|---|---|---|
| `flux` | number | positive, finite flux in W/m² |

**What it does.** Compares the flux with the lower boundary of each class
in `FLUX_MAP`, from `X` down: `A ≥ 1e-8`, `B ≥ 1e-7`, `C ≥ 1e-6`,
`M ≥ 1e-5`, `X ≥ 1e-4` (no upper bound). Each boundary belongs to the
higher class and the comparison is exact: one float step below `1e-5` is
`"C"`.
**Returns.** One of `"A"`, `"B"`, `"C"`, `"M"`, `"X"`.

Run (requires Setup):

<!-- example: to_class -->
```python
converter = fidx.FluxConverter()
for flux in [2e-05, 1e-05, 9.999999999999999e-06, 1e-08, 0.00012]:
    print(flux, "->", converter.to_class(flux))
print(converter.to_class(converter.to_flux("C10.0")))
```
<!-- output: to_class -->
```text
2e-05 -> M
1e-05 -> M
9.999999999999999e-06 -> C
1e-08 -> A
0.00012 -> X
M
```

**Output explained.** Exactly `1e-05` is `M`; the float just below it is
`C`; `C10.0` converts to exactly `1e-05`, so it is `M`.
**Errors.** `ValueError` for zero (no flare is not a class), a negative,
NaN or infinite value, or a positive flux below `1e-8` (the scale has no
letter for it); `TypeError` for a non-number, including class text and
booleans.

### `FlareClassifier`

```python
from flare_indexer import FlareClassifier
FlareClassifier()
```

**Purpose.** Answers "is this flare at least as strong as a threshold
letter?". **Inputs.** None.

### `FlareClassifier.is_strong`

```python
FlareClassifier().is_strong(goes_class: str, threshold: str = "M") -> bool
```

| Parameter | Type | Default | Accepted values |
|---|---|---|---|
| `goes_class` | `str` | – | a GOES class (see `to_flux`) |
| `threshold` | `str` | `"M"` | a single letter `A`, `B`, `C`, `M`, `X` (upper case) |

**What it does.** Returns `to_flux(goes_class) >= FLUX_MAP[threshold]`.
**Returns.** `bool`.

Run (requires Setup):

<!-- example: is_strong -->
```python
classifier = fidx.FlareClassifier()
print(classifier.is_strong("M2.0"), classifier.is_strong("C9.9"), classifier.is_strong("C9.9", threshold="C"))
```
<!-- output: is_strong -->
```text
True False True
```

**Output explained.** M2.0 ≥ M1.0 → `True`; C9.9 < M1.0 → `False`;
C9.9 ≥ C1.0 → `True`.
**Errors.** `InvalidGOESClassError` for a bad `goes_class`; `KeyError` for
a threshold that is not one of the five upper-case letters (unlike
`BinaryThresholdLabeler`, it does not accept `"C4.2"`).

### `InvalidGOESClassError`

```python
from flare_indexer import InvalidGOESClassError
```

**Purpose.** The exception for missing or malformed GOES classes. It is a
subclass of `ValueError`, and also of `KeyError` and `IndexError` so that
older code catching those still works. `str(error)` is a plain message.
**Raised by.** `FluxConverter.to_flux`, `parse_goes_class`,
`validate_goes_class_column`, `EventMatcher(...)` (via the catalog check),
`adapt_goes_catalog`, and `DatasetBuilder.build`.

Run (requires Setup):

<!-- example: invalid_class -->
```python
bad = Path("docs/fixtures") / "_bad_catalog.csv"
bad.write_text(Path(CATALOG).read_text().replace("C4.0", "Q4.0").replace("X1.2", ""))
try:
    fidx.EventMatcher(bad)
except fidx.InvalidGOESClassError as error:
    print(type(error).__name__, "| is ValueError:", isinstance(error, ValueError))
    print(str(error).replace(str(bad), "<file>"))
finally:
    bad.unlink()
```
<!-- output: invalid_class -->
```text
InvalidGOESClassError | is ValueError: True
Flare catalog at <file> has 2 missing or invalid GOES class value(s) in column 'class' (line 2: 'Q4.0', line 4: <missing>). Expected a letter A, B, C, M or X followed by an optional positive number, e.g. 'M2.3'. Fix or remove these rows explicitly: a flare of unknown class cannot be compared with a threshold.
```

**Output explained.** Both bad rows are reported at once, by file line
(the header is line 1); the blank class is shown as `<missing>`.

---

## events

### `FlareEvent`

```python
from flare_indexer import FlareEvent
FlareEvent(peak_time, goes_class, start_time, active_region, end_time=None, fl_lon=None, fl_lat=None)
```

**Purpose.** One flare, as returned by `EventMatcher.query`. A dataclass
(a simple container whose fields you read by name).

| Field | Type | Meaning |
|---|---|---|
| `peak_time` | `pandas.Timestamp` | time of peak X-ray flux |
| `goes_class` | `str` | class text exactly as in the catalog (not flux) |
| `start_time` | `pandas.Timestamp` | start time |
| `active_region` | `str` or `None` | normalized region (`"13700"`), `None` if missing |
| `end_time` | `pandas.Timestamp` or `None` | end time |
| `fl_lon`, `fl_lat` | `float` or `None` | location in degrees, if the catalog has those columns |

Run (requires Setup):

<!-- example: flare_event -->
```python
event = fidx.FlareEvent(
    peak_time=pd.Timestamp("2024-05-01 05:45"), goes_class="M2.0",
    start_time=pd.Timestamp("2024-05-01 05:30"), active_region="13700",
)
print(event)
print(event.goes_class, "->", fidx.FluxConverter().to_flux(event.goes_class), "W/m^2")
```
<!-- output: flare_event -->
```text
FlareEvent(peak_time=Timestamp('2024-05-01 05:45:00'), goes_class='M2.0', start_time=Timestamp('2024-05-01 05:30:00'), active_region='13700', end_time=None, fl_lon=None, fl_lat=None)
M2.0 -> 2e-05 W/m^2
```

**Output explained.** Fields not given default to `None`. The class stays
text; convert it with `FluxConverter` when you need a number.
**Limitations.** No validation on direct construction; validation happens
when a catalog is loaded.

### `EventMatcher`

```python
from flare_indexer import EventMatcher
EventMatcher(catalog_path: str | pathlib.Path)
```

**Purpose.** Loads a flare catalog once and answers time-window queries.

| Parameter | Type | Meaning |
|---|---|---|
| `catalog_path` | `str` or `Path` | CSV in the full-timestamp schema (`start_time, peak_time, end_time, class, active_region` [+ `fl_lon, fl_lat`]) or the legacy schema (`date, start, peak, end, class, active_region`, `HHMM` times) |

**What it does.** Reads the CSV; detects the schema (legacy wins if both
match); builds full timestamps (legacy: date + `HHMM`); checks that each
row's available times satisfy `start <= peak <= end` (equal times allowed;
a blank full-timestamp cell is skipped); validates every class; sorts by
`peak_time`.
**Public class attributes.** `REQUIRED_CATALOG_COLUMNS` (legacy schema
columns) and `FULL_TIMESTAMP_CATALOG_COLUMNS` (full-timestamp columns),
both `set` of `str`.
**Errors.** `ValueError` for an empty file, missing columns, invalid
`HHMM` values, unparsable timestamps, or times out of order (the message
names the data row, 1 = first row after the header; for the legacy schema
it explains that a flare crossing midnight needs the full-timestamp
schema, because dates are never shifted); `InvalidGOESClassError` for bad
classes. A header-only file is valid (every query returns `[]`).

### `EventMatcher.query`

```python
EventMatcher(...).query(image_time, prediction_window_hours, active_region=None,
                        event_time="peak", interval_mode="left_closed") -> list[FlareEvent]
```

| Parameter | Type | Default | Accepted values / units |
|---|---|---|---|
| `image_time` | `pandas.Timestamp` | – | reference time `t` |
| `prediction_window_hours` | number or duration | – | hours (`6`, `0.5`) or text/`Timedelta` (`"23h59min59s"`); > 0 and finite |
| `active_region` | `str` or `None` | `None` | `None` = any region (full disk); otherwise the normalized region text, e.g. `"13700"` |
| `event_time` | `str` | `"peak"` | `"peak"` or `"start"` |
| `interval_mode` | `str` | `"left_closed"` | `"left_closed"` = `[t, t+w)`, `"right_closed"` = `(t, t+w]` |

**What it does.** Finds flares whose chosen time is in the window after
`t` (binary search on a cached sorted column), then, if `active_region`
is given, keeps only flares with exactly that region.
**Returns.** `list[FlareEvent]`, empty if none.

Run (requires Setup):

<!-- example: query -->
```python
matcher = fidx.EventMatcher(CATALOG)
t = pd.Timestamp("2024-05-01 03:00")
print([e.goes_class for e in matcher.query(t, 6)])
print([e.goes_class for e in matcher.query(t, 6, event_time="start", interval_mode="right_closed")])
print([e.goes_class for e in matcher.query(t, 6, active_region="13700", event_time="start", interval_mode="right_closed")])
print([e.goes_class for e in matcher.query(pd.Timestamp("2024-05-01 18:00"), "6h", event_time="start", interval_mode="right_closed")])
```
<!-- output: query -->
```text
['M2.0', 'X1.2']
['M2.0', 'X1.2']
['M2.0']
['M1.0']
```

**Output explained.** Line 1 (defaults, peak time, `[03:00, 09:00)`):
M2.0 peaks 05:45, X1.2 peaks 08:10. Line 2 (start time): the same two.
Line 3: only region 13700's flare. Line 4: M1.0 starts exactly at
`18:00 + 6h`; the right-closed window includes it.
**Errors.** `ValueError` for an invalid `event_time`, `interval_mode` or
window. **Limitations.** The window always looks forward from `t`; there
is no backward option. The region string must already be normalized
(`"13700"`, not `13700`). No time-zone conversion.

---

## reducers

### `EventReducer`

```python
from flare_indexer import EventReducer   # typing.Protocol
```

**Purpose.** A *contract*, not a class to instantiate: any object with a
method `reduce(self, events: list[FlareEvent])` returning one value can be
used as a reducer in `DatasetBuildingPipeline`. No inheritance needed.

### `EventReducer.reduce`

Contract: `reduce(events: list[FlareEvent]) -> value`. Receives a list
(possibly empty) of flares for one reference time; returns anything the
labeler understands. The pipeline passes a copy of the list, so a reducer
cannot change the caller's list. A concrete implementation is
`MaxFluxReducer`; a custom one is shown under
[`DatasetBuildingPipeline`](#datasetbuildingpipeline).

### `MaxFluxReducer`

```python
from flare_indexer import MaxFluxReducer
MaxFluxReducer()
```

**Purpose.** The built-in reducer: strongest flare's flux.

### `MaxFluxReducer.reduce`

```python
MaxFluxReducer().reduce(events: list[FlareEvent]) -> float
```

**Inputs.** `events`: list of `FlareEvent`. **What it does.** Converts
each `goes_class` to flux and returns the maximum; `0.0` for an empty
list. **Returns.** `float`, W/m².

Run (requires `matcher` and `t` from `EventMatcher.query`):

<!-- example: reduce -->
```python
events = matcher.query(t, 6, event_time="start", interval_mode="right_closed")
print(fidx.MaxFluxReducer().reduce(events))
print(fidx.MaxFluxReducer().reduce([]))
```
<!-- output: reduce -->
```text
0.00012
0.0
```

**Output explained.** X1.2 (1.2e-4) beats M2.0 (2e-5); no events → `0.0`.
**Errors.** `InvalidGOESClassError` if an event has a bad class (only
possible for hand-built events; loaded catalogs are already checked).

---

## labelers

### `LabelAssigner`

```python
from flare_indexer import LabelAssigner   # typing.Protocol
```

**Purpose.** A *contract*: any object with `assign(self, reduced_value)`
returning a label can be the labeler of a `DatasetBuildingPipeline`.

### `LabelAssigner.assign`

Contract: `assign(reduced_value) -> label`. Receives whatever the reducer
returned; returns the final label. Concrete implementations:
`BinaryThresholdLabeler`, `RegressionLabeler`, `MulticlassLabeler`.

### `BinaryThresholdLabeler`

```python
from flare_indexer import BinaryThresholdLabeler
BinaryThresholdLabeler(threshold: str = "M")
```

| Parameter | Type | Default | Accepted values |
|---|---|---|---|
| `threshold` | `str` | `"M"` | any GOES class: a letter (`"M"` = M1.0) or a full class (`"C4.2"`) |

**Public attribute.** `threshold` (the text given).
**Errors.** `ValueError` ("Invalid GOES threshold …") for a bad threshold.

### `BinaryThresholdLabeler.assign`

```python
BinaryThresholdLabeler(...).assign(reduced_value) -> int
```

**Inputs.** `reduced_value`: a flux number in W/m². **Returns.** `1` if
`reduced_value >= flux(threshold)`, else `0`.

Run (requires Setup):

<!-- example: binary_assign -->
```python
labeler = fidx.BinaryThresholdLabeler("M")
print(labeler.assign(1.2e-4), labeler.assign(9.9e-6), labeler.assign(1e-5), labeler.assign(0.0))
print(fidx.BinaryThresholdLabeler("C4.2").assign(4.2e-6))
```
<!-- output: binary_assign -->
```text
1 0 1 0
1
```

**Output explained.** X1.2 → 1; C9.9 (9.9e-6) → 0; exactly M1.0 → 1
(the comparison is "≥"); nothing → 0; exactly C4.2 → 1.
**Limitations.** Expects a number; passing class text raises `TypeError`.

### `RegressionLabeler`

```python
from flare_indexer import RegressionLabeler
RegressionLabeler()
```

**Purpose.** Pass the reduced value through as the label.

### `RegressionLabeler.assign`

```python
RegressionLabeler().assign(reduced_value)
```

**What it does.** Returns integers as `int`, other real numbers as
`float` (e.g. `numpy.float64` → `float`), anything else unchanged.

Run (requires Setup):

<!-- example: regression_assign -->
```python
import numpy as np
for value in [1.2e-4, np.float64(2e-5), 3, "text"]:
    out = fidx.RegressionLabeler().assign(value)
    print(repr(out), type(out).__name__)
```
<!-- output: regression_assign -->
```text
0.00012 float
2e-05 float
3 int
'text' str
```

**Output explained.** Numbers become plain Python numbers; non-numbers
are returned as they are.

### `MulticlassLabeler`

```python
from flare_indexer import MulticlassLabeler
MulticlassLabeler()
```

**Purpose.** Turn a reduced flux into a GOES class label: `"NF"`, `"A"`,
`"B"`, `"C"`, `"M"` or `"X"`. Use it with `MaxFluxReducer`:

```python
fidx.DatasetBuildingPipeline(reducer=fidx.MaxFluxReducer(), labeler=fidx.MulticlassLabeler())
```

**What the label means.** The class of the *one* flux value the reducer
selected. With `MaxFluxReducer` that is the strongest flare in the
window; the label does not say which other classes occurred or how many
flares there were (one X1.0 and ten C flares → `"X"`). Use a custom
reducer for counts.

**Public attribute.** `CLASSES = ("NF", "A", "B", "C", "M", "X")`, in
increasing flux order (e.g. to map labels to class indices).

### `MulticlassLabeler.assign`

```python
MulticlassLabeler().assign(reduced_flux) -> str
```

**Inputs.** `reduced_flux`: a flux number in W/m² (`int`, `float`,
`numpy` number, `Fraction`).

**Returns.**

| Flux (W/m²) | Label |
|---|---|
| exactly `0.0` (empty window) | `"NF"` |
| `1e-8 ≤ flux < 1e-7` | `"A"` |
| `1e-7 ≤ flux < 1e-6` | `"B"` |
| `1e-6 ≤ flux < 1e-5` | `"C"` |
| `1e-5 ≤ flux < 1e-4` | `"M"` |
| `flux ≥ 1e-4` | `"X"` |

The comparison is numeric (boundaries from `FluxConverter.FLUX_MAP`);
each boundary belongs to the higher class, so `"M10.0"` and `"X1.0"` are
both `"X"`. The comparison is exact, with no tolerance: a float even one
step below a boundary (`math.nextafter(1e-5, 0)`) is in the lower class.
Class text reaches the boundary exactly because `FluxConverter.to_flux`
converts equivalent text to the identical float (`"C10.0"` → `1e-05`, so
`"M"`; `BinaryThresholdLabeler("M")` gives `1` for it too).

**Errors.**
- `ValueError` for a positive flux below `1e-8` (e.g. from a catalog class
  `"A0.5"`): the GOES scale has no letter for it, and it is a flare, so it
  is not labeled `"NF"`. Inside `DatasetBuilder` this stops the build.
- `ValueError` for a negative, NaN or infinite flux.
- `TypeError` for anything that is not a real number, including class
  text such as `"M2.0"` and `True`/`False`.

Run (requires Setup):

<!-- example: multiclass_assign -->
```python
to_flux = fidx.FluxConverter().to_flux
labeler = fidx.MulticlassLabeler()
for goes_class in ["A1.0", "B9.9", "C1.0", "C10.0", "M9.9", "M10.0", "X1.0"]:
    print(goes_class, to_flux(goes_class), labeler.assign(to_flux(goes_class)))
import math
print(labeler.assign(0.0), labeler.assign(1e-5), labeler.assign(math.nextafter(1e-5, 0)))
for bad in [5e-9, -1e-6, float("nan"), "M2.0"]:
    try:
        labeler.assign(bad)
    except (ValueError, TypeError) as error:
        print(type(error).__name__, str(error)[:60], "...")
```
<!-- output: multiclass_assign -->
```text
A1.0 1e-08 A
B9.9 9.9e-07 B
C1.0 1e-06 C
C10.0 1e-05 M
M9.9 9.9e-05 M
M10.0 0.0001 X
X1.0 0.0001 X
NF M C
ValueError MulticlassLabeler.assign got a positive flux 5e-09 W/m^2 bel ...
ValueError MulticlassLabeler.assign got a negative flux -1e-06; X-ray f ...
ValueError MulticlassLabeler.assign got a non-finite flux nan. ...
TypeError MulticlassLabeler.assign expects a flux number in W/m^2 (e.g ...
```

**Output explained.** C10.0 is exactly 1e-5 and so `"M"`; M10.0 and
X1.0 are both `"X"`; exactly 1e-5 is `"M"`, the next float below it is
`"C"`; nothing in the window is `"NF"`.

Run through `DatasetBuilder` (requires Setup): full-disk images, the same
images as time-based sequences (labeled from the last image), and
active-region patches. Window `(t, t + 6h]` on the flare start time.

<!-- example: multiclass_builder -->
```python
multiclass = fidx.DatasetBuildingPipeline(reducer=fidx.MaxFluxReducer(), labeler=fidx.MulticlassLabeler())
options = dict(prediction_window="6h", strategy=multiclass, event_time="start", interval_mode="right_closed")

print(fidx.DatasetBuilder(**options).build(IMAGES, CATALOG))
sequences = fidx.DatasetBuilder(**options, cadence="3h", observation_window="6h", sliding_window="3h") \
    .build(IMAGES, CATALOG)
print(sequences[["sequence_start", "sequence_end", "label"]])
ar = fidx.DatasetBuilder(**options, target="active_region", missing_active_region="drop", include_metadata=True)
print(ar.build(AR_IMAGES, CATALOG)[["timestamp", "active_region", "label"]])
```
<!-- output: multiclass_builder -->
```text
            timestamp label
0 2024-05-01 00:00:00     M
1 2024-05-01 03:00:00     X
2 2024-05-01 06:00:00     X
3 2024-05-01 09:00:00    NF
4 2024-05-01 18:00:00     M
       sequence_start        sequence_end label
0 2024-05-01 00:00:00 2024-05-01 03:00:00     X
1 2024-05-01 03:00:00 2024-05-01 06:00:00     X
2 2024-05-01 06:00:00 2024-05-01 09:00:00    NF
            timestamp active_region label
0 2024-05-01 00:00:00         13700     M
1 2024-05-01 00:00:00         13701    NF
2 2024-05-01 03:00:00         13700     M
3 2024-05-01 03:00:00         13701     X
4 2024-05-01 06:00:00         13700    NF
```

**Output explained.** Image 00:00 sees C4.0 and M2.0 → `"M"`; 03:00 and
06:00 see X1.2 → `"X"`; 09:00 sees nothing → `"NF"`; 18:00 sees the M1.0
starting at midnight → `"M"`. Each sequence takes the label of its last
image's window (the 09:00–12:00 sequence is skipped: no image at 12:00).
In active-region mode region 13701 at 00:00 is `"NF"` while full-disk
says `"M"`: the M2.0 belongs to 13700. The patch without a region is
dropped.

---

## strategies

### `BinaryThresholdStrategy`

```python
from flare_indexer import BinaryThresholdStrategy
BinaryThresholdStrategy(threshold: str = "M")
```

**Purpose.** Ready-made strategy for `DatasetBuilder`: `MaxFluxReducer`
followed by `BinaryThresholdLabeler(threshold)`. **Inputs/attribute.**
`threshold` as for `BinaryThresholdLabeler`. **Errors.** `ValueError` for
a bad threshold.

### `BinaryThresholdStrategy.label`

```python
BinaryThresholdStrategy(...).label(flares: list[FlareEvent]) -> int
```

**Returns.** `1` if the strongest flare's flux ≥ the threshold's flux,
else `0` (also `0` for no flares).

Run (requires `events` from `MaxFluxReducer.reduce`):

<!-- example: binary_strategy -->
```python
print(fidx.BinaryThresholdStrategy("M").label(events), fidx.BinaryThresholdStrategy("X2").label(events),
      fidx.BinaryThresholdStrategy().label([]))
```
<!-- output: binary_strategy -->
```text
1 0 0
```

**Output explained.** `events` holds M2.0 and X1.2: ≥ M → 1; X1.2 < X2.0
→ 0; no flares → 0.

### `MaxFlareStrategy`

```python
from flare_indexer import MaxFlareStrategy
MaxFlareStrategy()
```

**Purpose.** Ready-made strategy returning the strongest flux:
`MaxFluxReducer` followed by `RegressionLabeler`.

### `MaxFlareStrategy.label`

```python
MaxFlareStrategy().label(flares: list[FlareEvent]) -> float
```

**Returns.** Strongest flux in W/m² (a number, not class text); `0.0`
for no flares.

Run (requires `events`):

<!-- example: max_strategy -->
```python
print(fidx.MaxFlareStrategy().label(events), fidx.MaxFlareStrategy().label([]))
```
<!-- output: max_strategy -->
```text
0.00012 0.0
```

---

## pipeline

### `PipelineResult`

```python
from flare_indexer import PipelineResult
PipelineResult(events, reduced_value, label)
```

**Purpose.** Dataclass returned by `DatasetBuildingPipeline.run_one`,
holding every intermediate value of one run.

| Field | Type | Meaning |
|---|---|---|
| `events` | `list[FlareEvent]` | matched flares |
| `reduced_value` | any (built-ins: `float` W/m²) | reducer output |
| `label` | any (built-ins: `int` or `float`) | labeler output |

An inspection example is under `DatasetBuildingPipeline.run_one`.

### `DatasetBuildingPipeline`

```python
from flare_indexer import DatasetBuildingPipeline
DatasetBuildingPipeline(reducer, labeler)
```

| Parameter | Required structure |
|---|---|
| `reducer` | any object with a callable `reduce(events)` |
| `labeler` | any object with a callable `assign(value)` |

**Purpose.** Run match → reduce → assign as separate, inspectable steps.
Stores no per-call state. **Public attributes.** `reducer`, `labeler`.
**Errors.** `TypeError` if a method is missing.

Run (requires Setup) – a custom reducer satisfying the `EventReducer`
contract, used with a built-in labeler:

<!-- example: pipeline_init -->
```python
class CountReducer:
    def reduce(self, events):
        return len(events)

counting = fidx.DatasetBuildingPipeline(CountReducer(), fidx.RegressionLabeler())
pipeline = fidx.DatasetBuildingPipeline(fidx.MaxFluxReducer(), fidx.BinaryThresholdLabeler("M"))
try:
    fidx.DatasetBuildingPipeline(object(), fidx.RegressionLabeler())
except TypeError as error:
    # The message starts with the rejected object's repr; print the rule it states.
    print("TypeError: reducer=<...> " + str(error).split("> ", 1)[1].split(";")[0])
```
<!-- output: pipeline_init -->
```text
TypeError: reducer=<...> must provide a callable 'reduce' method (e.g. `def reduce(self, ...): ...`)
```

### `DatasetBuildingPipeline.extract_events`

```python
pipeline.extract_events(matcher, reference_time, prediction_window, active_region=None,
                        event_time="peak", interval_mode="left_closed") -> list[FlareEvent]
```

**Inputs.** An `EventMatcher` and the same window arguments as
`EventMatcher.query`. **What it does.** Calls `matcher.query` and returns
a new list. **Returns.** `list[FlareEvent]` (stage 1 result).

### `DatasetBuildingPipeline.reduce_events`

```python
pipeline.reduce_events(events) -> value
```

Calls `self.reducer.reduce` on a copy of `events`. Returns the reducer's
value (stage 2 result; `float` for `MaxFluxReducer`).

### `DatasetBuildingPipeline.assign_label`

```python
pipeline.assign_label(reduced_value) -> label
```

Calls `self.labeler.assign(reduced_value)`. Returns the label (stage 3).

Run (requires `pipeline`, `counting`, `matcher`, `t`):

<!-- example: pipeline_stages -->
```python
stage1 = pipeline.extract_events(matcher, t, 6, event_time="start", interval_mode="right_closed")
stage2 = pipeline.reduce_events(stage1)
stage3 = pipeline.assign_label(stage2)
print(type(stage1).__name__, [e.goes_class for e in stage1])
print(type(stage2).__name__, stage2)
print(type(stage3).__name__, stage3)
print("custom count reducer:", counting.reduce_events(stage1))
```
<!-- output: pipeline_stages -->
```text
list ['M2.0', 'X1.2']
float 0.00012
int 1
custom count reducer: 2
```

**Output explained.** Stage 1 is a list of two flares; stage 2 the
strongest flux; stage 3 the 0/1 label. The counting pipeline reduces the
same list to `2`.

### `DatasetBuildingPipeline.run_one`

```python
pipeline.run_one(matcher, reference_time, prediction_window, active_region=None,
                 event_time="peak", interval_mode="left_closed") -> PipelineResult
```

Runs all three stages and returns a `PipelineResult`.

Run (requires `pipeline`, `matcher`, `t`):

<!-- example: run_one -->
```python
result = pipeline.run_one(matcher, t, 6, event_time="start", interval_mode="right_closed")
print(type(result).__name__, [e.goes_class for e in result.events], result.reduced_value, result.label)
```
<!-- output: run_one -->
```text
PipelineResult ['M2.0', 'X1.2'] 0.00012 1
```

### `DatasetBuildingPipeline.label`

```python
pipeline.label(events) -> label
```

**Purpose.** Lets a pipeline be used as `DatasetBuilder(strategy=pipeline)`:
reduces then assigns. **Returns.** The label.

Run (requires `pipeline`, `stage1`, `counting`):

<!-- example: pipeline_label -->
```python
print(pipeline.label(stage1), counting.label(stage1))
print(fidx.DatasetBuilder(prediction_window=6, strategy=counting, event_time="start",
                          interval_mode="right_closed").build(IMAGES, CATALOG)["label"].tolist())
```
<!-- output: pipeline_label -->
```text
1 2
[2, 2, 1, 0, 1]
```

**Output explained.** The counting pipeline used as a strategy gives the
number of flares starting in each image's window.

---

## builder

### `DatasetBuilder`

```python
from flare_indexer import DatasetBuilder
DatasetBuilder(prediction_window, strategy, sequence_length=None, stride=None, cadence_minutes=None,
               target="full_disk", event_time="peak", interval_mode="left_closed",
               start=None, cadence=None, observation_window=None, sliding_window=None,
               missing_active_region="negative", include_metadata=False)
```

**Purpose.** Labels a whole image index.

| Parameter | Type | Default | Accepted values / meaning |
|---|---|---|---|
| `prediction_window` | number or duration | – | hours, or text/`Timedelta`; > 0, finite |
| `strategy` | object with `label(flares)` | – | built-in strategy, pipeline, or custom |
| `sequence_length` | `int` | `None` (= 1) | row-based: images per sequence, ≥ 1 |
| `stride` | `int` | `None` (= 1) | row-based: rows between sequence starts, ≥ 1 |
| `cadence_minutes` | `int` | `None` | row-based: required spacing; sequences with other gaps are skipped |
| `target` | `str` | `"full_disk"` | `"full_disk"` or `"active_region"` |
| `event_time` | `str` | `"peak"` | `"peak"` or `"start"` |
| `interval_mode` | `str` | `"left_closed"` | `"left_closed"` or `"right_closed"` |
| `start` | timestamp | `None` | time-based: first sequence start; must be an image time |
| `cadence` | duration | `None` | time-based: spacing inside a sequence; multiple of the inferred source cadence |
| `observation_window` | duration | `None` | time-based: span of one sequence; multiple of `cadence` |
| `sliding_window` | duration | `None` | time-based: shift between starts; multiple of `cadence` |
| `missing_active_region` | `str` | `"negative"` | `"negative"`, `"drop"`, `"error"` (AR mode only) |
| `include_metadata` | `bool` | `False` | single-image: keep other index columns |

Row-based and time-based parameters cannot be combined; giving any
time-based parameter requires `cadence`, `observation_window` and
`sliding_window`. **Public attributes.** The arguments as given
(`prediction_window`, `strategy`, `target`, `event_time`,
`interval_mode`, `missing_active_region`, `include_metadata`), plus
`time_based` (`bool`); row-based: `sequence_length`, `stride`,
`cadence_minutes`; time-based: `cadence`, `observation_window`,
`sliding_window` (parsed to `Timedelta`), `start`, `number_of_images`,
`slide_steps`. Class attribute `VALID_TARGETS`.
**Errors.** `ValueError` for any invalid value or combination.

### `DatasetBuilder.build`

```python
DatasetBuilder(...).build(image_index_path, event_catalog_path, return_report=False)
```

| Parameter | Type | Meaning |
|---|---|---|
| `image_index_path` | `str` or `Path` | image index CSV (`timestamp` [+ `active_region`] [+ other columns]) |
| `event_catalog_path` | `str` or `Path` | flare catalog CSV (see `EventMatcher`) |
| `return_report` | `bool` | time-based mode only: also return a `SequenceBuildReport` |

**What it does.** Reads the index, loads the catalog, and for every
reference time calls `EventMatcher.query` then `strategy.label`
(details: [system design: processing](architecture/system_design.md#processing)).
**Returns.** `pandas.DataFrame`, or `(DataFrame, SequenceBuildReport)`.
Columns are described in the [researcher manual §6](researcher_manual.md#6-output-formats).

Run (requires Setup):

<!-- example: build -->
```python
single = fidx.DatasetBuilder(prediction_window=6, strategy=fidx.MaxFlareStrategy(),
                             event_time="start", interval_mode="right_closed").build(IMAGES, CATALOG)
print(single.to_string(index=False, float_format=lambda v: f"{v:.1e}"))
print(list(single.columns), single["label"].dtype)
```
<!-- output: build -->
```text
          timestamp   label
2024-05-01 00:00:00 2.0e-05
2024-05-01 03:00:00 1.2e-04
2024-05-01 06:00:00 1.2e-04
2024-05-01 09:00:00 0.0e+00
2024-05-01 18:00:00 1.0e-05
['timestamp', 'label'] float64
```

**Output explained.** One row per image; with `MaxFlareStrategy` the
label is the strongest flux (float W/m²), `0.0` for 09:00.
**Warnings.** In AR mode: `UserWarning` when catalog flares have no
region, and (policy `"negative"`) when images have no region.
**Errors.** `ValueError` for a missing/empty index file or column,
ordering/duplicate problems in sequence modes, incompatible cadence, an
index column that would overwrite an output column, `return_report=True`
outside time-based mode, and region policy `"error"`; catalog errors as
for `EventMatcher`.
**Limitations.** Forward windows only; no TAI/UTC or coverage handling.

### `SequenceBuildReport`

```python
from flare_indexer import SequenceBuildReport   # returned by build(..., return_report=True)
```

| Field | Type | Meaning |
|---|---|---|
| `inferred_source_cadence` | `Timedelta` | most common gap between sorted unique image times |
| `requested_cadence` | `Timedelta` | the `cadence` used |
| `observation_window`, `sliding_window` | `Timedelta` | as used |
| `effective_start` | `Timestamp` or `None` | `start`, or the earliest image time |
| `total_candidate_sequences` | `int` | candidate starts considered |
| `emitted_sequences` | `int` | sequences produced |
| `skipped_sequences` | `int` | candidates with a missing image |
| `skipped_sequence_starts` | `list[Timestamp]` | first 50 skipped starts |
| `missing_timestamps` | `list[Timestamp]` | first missing image time of each listed skip (first 50) |
| `skipped_sequence_regions` | `list` | AR mode: region of each listed skip; empty for full disk |
| `region_count` | `int` or `None` | AR mode: number of region groups; `None` for full disk |

Run (requires Setup):

<!-- example: sequence_report -->
```python
_, report = fidx.DatasetBuilder(
    prediction_window=6, strategy=fidx.BinaryThresholdStrategy("M"),
    cadence="3h", observation_window="6h", sliding_window="3h",
).build(IMAGES, CATALOG, return_report=True)
for name, value in vars(report).items():
    print(f"{name}: {value}")
```
<!-- output: sequence_report -->
```text
inferred_source_cadence: 0 days 03:00:00
requested_cadence: 0 days 03:00:00
observation_window: 0 days 06:00:00
sliding_window: 0 days 03:00:00
effective_start: 2024-05-01 00:00:00
total_candidate_sequences: 6
emitted_sequences: 3
skipped_sequences: 3
skipped_sequence_starts: [Timestamp('2024-05-01 09:00:00'), Timestamp('2024-05-01 12:00:00'), Timestamp('2024-05-01 15:00:00')]
missing_timestamps: [Timestamp('2024-05-01 12:00:00'), Timestamp('2024-05-01 12:00:00'), Timestamp('2024-05-01 15:00:00')]
skipped_sequence_regions: []
region_count: None
```

**Output explained.** The images are at 00, 03, 06, 09, 18 h, so the most
common gap is 3 h. Six 2-image grids fit before 18:00; the ones starting
at 09, 12 and 15 h lack an image and are skipped.

---

## loaders

### `build_image_index_from_filenames`

```python
from flare_indexer import build_image_index_from_filenames
build_image_index_from_filenames(paths, output_path=None, strict=True, duplicate_policy="error",
                                 return_report=False)
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `paths` | `str`/`Path` to a headerless one-path-per-line file, or an iterable of path strings | – | image paths; only the file name is parsed, e.g. `HMI.m2024.05.01_00.00.00.jpg` |
| `output_path` | `str`/`Path` | `None` | also write the result as CSV |
| `strict` | `bool` | `True` | `True`: error on a non-matching path; `False`: exclude it and report it |
| `duplicate_policy` | `str` | `"error"` | `"error"`, `"first"`, `"last"` for repeated timestamps |
| `return_report` | `bool` | `False` | also return an `ImageIndexReport` |

**Returns.** `DataFrame` with `timestamp` (sorted) and `image_path`; or
`(DataFrame, ImageIndexReport)`.

Run (requires Setup):

<!-- example: build_index -->
```python
index, image_report = fidx.build_image_index_from_filenames(
    "docs/fixtures/hmi_paths.txt", strict=False, return_report=True,
)
print(index.to_string(index=False))
try:
    fidx.build_image_index_from_filenames("docs/fixtures/hmi_paths.txt")
except ValueError as error:
    print("strict=True ->", str(error)[:60], "...")
```
<!-- output: build_index -->
```text
          timestamp                                                 image_path
2024-05-01 00:00:00 /data/hmi_jpgs_512/2024/05/01/HMI.m2024.05.01_00.00.00.jpg
2024-05-01 03:00:00 /data/hmi_jpgs_512/2024/05/01/HMI.m2024.05.01_03.00.00.jpg
strict=True -> Filename 'Fold1_train.csv' (from path '/data/hmi_jpgs_512/la ...
```

**Errors.** `ValueError` for a bad `duplicate_policy`, a non-matching
file name (`strict=True`), an impossible date in a well-formed name
(`strict=True`), or duplicates with `"error"`.

### `ImageIndexReport`

```python
from flare_indexer import ImageIndexReport   # returned by build_image_index_from_filenames(..., return_report=True)
```

| Field | Type | Meaning |
|---|---|---|
| `total_input_rows` | `int` | non-blank input paths |
| `valid_image_rows_before_dedup` | `int` | paths parsed as images |
| `excluded_row_count` | `int` | paths excluded (exact count) |
| `excluded_rows` | `list[str]` | first 200 excluded paths |
| `duplicate_timestamp_count` | `int` | distinct timestamps that occurred more than once |
| `duplicate_rows` | `list[str]` | first 200 paths involved in duplicates |
| `final_row_count` | `int` | rows returned |
| `duplicate_policy` | `str` | policy applied |

Run (requires `image_report` from the previous block):

<!-- example: image_report -->
```python
for name, value in vars(image_report).items():
    print(f"{name}: {value}")
```
<!-- output: image_report -->
```text
total_input_rows: 3
valid_image_rows_before_dedup: 2
excluded_row_count: 1
excluded_rows: ['/data/hmi_jpgs_512/labels_store/Fold1_train.csv']
duplicate_timestamp_count: 0
duplicate_rows: []
final_row_count: 2
duplicate_policy: error
```

### `adapt_goes_catalog`

```python
from flare_indexer import adapt_goes_catalog
adapt_goes_catalog(input_path, output_path=None) -> pandas.DataFrame
```

| Parameter | Type | Meaning |
|---|---|---|
| `input_path` | `str`/`Path` | CSV with `start_time, peak_time, end_time, goes_class, noaa_active_region` (+ optional `fl_lon, fl_lat`); other columns ignored |
| `output_path` | `str`/`Path` | optional CSV to write |

**What it does.** Validates classes; keeps full timestamps; renames
`goes_class` → `class` and `noaa_active_region` → `active_region`
(normalized: `13700.0` → `"13700"`, blank → missing).
**Returns.** `DataFrame` in the full-timestamp schema accepted by
`EventMatcher`.

Run (requires Setup):

<!-- example: adapt -->
```python
adapted = fidx.adapt_goes_catalog("docs/fixtures/goes_catalog.csv")
print(adapted.to_string(index=False))
print(list(adapted.columns))
```
<!-- output: adapt -->
```text
         start_time           peak_time            end_time class active_region  fl_lon  fl_lat
2024-05-01 02:00:00 2024-05-01 02:15:00 2024-05-01 02:30:00  C4.0         13700   -12.0     8.0
2024-05-01 05:30:00 2024-05-01 05:45:00 2024-05-01 06:00:00  M2.0         13700   -11.5     8.0
['start_time', 'peak_time', 'end_time', 'class', 'active_region', 'fl_lon', 'fl_lat']
```

**Errors.** `ValueError` for an empty file or missing columns;
`InvalidGOESClassError` for bad classes.

---

## Module-level helpers and constants

Public (no leading underscore) but not exported from `flare_indexer`;
import them from their module.

### `parse_goes_class`

```python
from flare_indexer.classifier import parse_goes_class
parse_goes_class(goes_class) -> tuple[str, float]
```

Splits a class into (upper-case letter, multiplier); raises
`InvalidGOESClassError` on bad input. Used by `FluxConverter.to_flux`.

<!-- example: parse_goes_class -->
```python
from flare_indexer.classifier import parse_goes_class
print(parse_goes_class("m2.3"), parse_goes_class("X"))
```
<!-- output: parse_goes_class -->
```text
('M', 2.3) ('X', 1.0)
```

### `validate_goes_class_column`

```python
from flare_indexer.classifier import validate_goes_class_column
validate_goes_class_column(values: pandas.Series, source, column="class") -> None
```

Checks every value; returns `None` if all are valid, otherwise raises one
`InvalidGOESClassError` listing the count and first five bad lines
(`source` and `column` are used in the message). Used by `EventMatcher`
and `adapt_goes_catalog`.

<!-- example: validate_column -->
```python
from flare_indexer.classifier import validate_goes_class_column
print(validate_goes_class_column(pd.Series(["M2.0", "c1"]), "example.csv"))
try:
    validate_goes_class_column(pd.Series(["M2.0", None]), "example.csv")
except fidx.InvalidGOESClassError as error:
    print(str(error)[:80], "...")
```
<!-- output: validate_column -->
```text
None
Flare catalog at example.csv has 1 missing or invalid GOES class value(s) in col ...
```

**Constants.** `flare_indexer.events.VALID_EVENT_TIMES` (`{"peak",
"start"}`), `flare_indexer.events.VALID_INTERVAL_MODES`
(`{"left_closed", "right_closed"}`),
`flare_indexer.builder.VALID_MISSING_ACTIVE_REGION_POLICIES`
(`{"negative", "drop", "error"}`), `flare_indexer.loaders.VALID_DUPLICATE_POLICIES`
(`{"error", "first", "last"}`), `flare_indexer.loaders.GOES_CATALOG_REQUIRED_COLUMNS`
and `GOES_CATALOG_OPTIONAL_COLUMNS` (columns read by `adapt_goes_catalog`).

---

## Minimum flux and per-class outputs

### `MinFluxReducer`
Selects the weakest recorded event's numeric peak flux, in W/m². This is
not background-flux sampling. Pair with `RegressionLabeler` for numeric
output or `MulticlassLabeler` for the selected event's class.

### `MinFluxReducer.reduce`
`reduce(events: list[FlareEvent]) -> float`: minimum flux, or `0.0` if empty.
With binary labeling this asks whether the weakest event meets the threshold,
not whether any event does. Below-A numeric flux is accepted by the reducer,
but rejected when paired with the existing multiclass labeler.

### `ClassCountReducer`
Counts all matched events by numeric class, in stable `A/B/C/M/X` order.
`C10.0` counts as M and `M10.0` as X. Each supplied event counts once;
deduplication is an input-preparation responsibility. Below-A events raise
an error consistently with `MulticlassLabeler`.

### `ClassCountReducer.reduce`
`reduce(events: list[FlareEvent]) -> dict[str, int]`: all five keys, with
zero values for an empty window. It does not consume a maximum-flux scalar.

### `ClassCountLabeler`
Validates and copies a mapping of per-class counts. Pair with
`ClassCountReducer`, not a flux reducer.

### `ClassCountLabeler.assign`
`assign(counts) -> dict[str, int]`: requires exactly A/B/C/M/X and nonnegative
integer counts; rejects booleans. Missing/extra keys raise `ValueError`,
noninteger inputs raise `TypeError`. Returns a fresh mapping.

### `ClassPresenceLabeler`
Produces a multi-label presence mapping from the same full-event counts.

### `ClassPresenceLabeler.assign`
`assign(counts) -> dict[str, int]`: the same validation, then 1 for each
positive count and 0 otherwise. No NF bin: an empty window is all zeros.

```python
from flare_indexer import (DatasetBuildingPipeline, MinFluxReducer,
    RegressionLabeler, ClassCountReducer, ClassCountLabeler, ClassPresenceLabeler)
minimum = DatasetBuildingPipeline(MinFluxReducer(), RegressionLabeler())
counts = DatasetBuildingPipeline(ClassCountReducer(), ClassCountLabeler())
presence = DatasetBuildingPipeline(ClassCountReducer(), ClassPresenceLabeler())
```

Pass any of these as `DatasetBuilder(..., strategy=...)`. Counts/presence
are dictionary values in the `label` column; export them explicitly as
JSON or expand to five columns before CSV/framework use. Plain CSV does
not preserve Python mappings. Cumulative indexing is not implemented:
its formula remains to be specified.

## Developer appendix: private helpers

Names starting with `_` are internal. They may change without notice; do
not call them from research code. Listed here so the code is fully
accounted for.

### `classifier.py`

| Helper | Input → output | Called by |
|---|---|---|
| `_GOES_CLASS_PATTERN` | regular expression for a class | `parse_goes_class` |
| `_MAX_INVALID_EXAMPLES` | int (5) | `validate_goes_class_column` |
| `_parse_goes_class_decimal` | class text → (letter, multiplier as exact `Decimal`); `InvalidGOESClassError` on bad input | `parse_goes_class`, `FluxConverter.to_flux` |

### `events.py`

| Helper | Purpose | Input → output | Called by |
|---|---|---|---|
| `_parse_prediction_window` | read a window | number (hours) or duration → positive `Timedelta`; `ValueError` otherwise | `EventMatcher.query`, `DatasetBuilder.__init__`, validation script |
| `_normalize_active_region` | one spelling per region | value → `str` or `None` (`13700.0` → `"13700"`) | `EventMatcher`, `DatasetBuilder`, `adapt_goes_catalog` |
| `_normalize_hhmm` | check one legacy time cell | value → 4-digit `"HHMM"` text or `ValueError` | `_combine_date_and_hhmm` |
| `_combine_date_and_hhmm` | legacy date + time | DataFrame, column names → `Series` of timestamps | `EventMatcher._load_legacy_hhmm_catalog` |
| `_parse_full_timestamp_column` | parse a time column with a clear error | DataFrame, column → `Series` of timestamps | `EventMatcher._load_full_timestamp_catalog` |
| `_validate_time_order` | reject rows with start > peak, peak > end or start > end (missing times skipped) | DataFrame, path, `hhmm` flag → `None` or `ValueError` naming the data row | `EventMatcher._load_legacy_hhmm_catalog`, `EventMatcher._load_full_timestamp_catalog` |
| `EventMatcher._load_catalog` | read CSV and pick a schema | path → sorted DataFrame | `EventMatcher.__init__` |
| `EventMatcher._load_legacy_hhmm_catalog` | load the legacy schema | DataFrame → DataFrame | `_load_catalog` |
| `EventMatcher._load_full_timestamp_catalog` | load the full-timestamp schema | DataFrame → DataFrame | `_load_catalog` |
| `EventMatcher._sorted_view` | cached sorted times for binary search | `event_time` → (sorted values, original positions) | `EventMatcher.query` |

### `reducers.py`, `labelers.py`, `strategies.py`, `pipeline.py`, `loaders.py`

| Helper | Purpose | Called by |
|---|---|---|
| `_flux_converter` (reducers, labelers) | shared `FluxConverter` instance | `MaxFluxReducer.reduce`, `BinaryThresholdLabeler.__init__`, `MulticlassLabeler.assign` (class boundaries) |
| `_max_flux_reducer` (strategies) | shared `MaxFluxReducer` instance | both strategies |
| `_require_method` (pipeline) | check `reduce`/`assign` exist; `TypeError` otherwise | `DatasetBuildingPipeline.__init__` |
| `_read_paths_file` (loaders) | read a one-path-per-line file (BOM-safe) → `list[str]` | `build_image_index_from_filenames` |
| `_FILENAME_PATTERN`, `_MAX_REPORT_SAMPLES` (loaders) | HMI file-name regex; report sample cap (200) | `build_image_index_from_filenames` |

### `builder.py`

| Helper | Purpose | Input → output | Called by |
|---|---|---|---|
| `_parse_duration` | positive duration | value, name → `Timedelta` | `DatasetBuilder.__init__` |
| `_parse_timestamp_param` | timestamp parameter | value, name → `Timestamp` | `DatasetBuilder.__init__` |
| `_infer_source_cadence` | most common gap | timestamps → `Timedelta` | time-based builders |
| `_validate_requested_cadence` | cadence is a multiple of the source cadence | two `Timedelta`s → `None` or `ValueError` | time-based builders |
| `_extra_columns` | columns to carry; refuse name clashes | DataFrame, excluded, reserved, path → `list[str]` | all builders that copy columns |
| `_sequence_frame` | AR sequence output, sorted | records, columns → DataFrame | AR sequence builders |
| `_SEQUENCE_OUTPUT_COLUMNS`, `_ROW_BASED_PARAM_NAMES`, `_TIME_BASED_PARAM_NAMES`, `_MAX_SEQUENCE_REPORT_SAMPLES` | constants | – | builder |
| `DatasetBuilder._build_single_image` | full-disk, one row per image | index, matcher, path → DataFrame | `build` |
| `DatasetBuilder._build_single_image_active_region` | AR, one row per image, region policy | index, matcher, path → DataFrame | `build` |
| `DatasetBuilder._report_missing_active_regions` | warn or raise for region-less images | list, unit → `None` | `_build_single_image_active_region` |
| `DatasetBuilder._warn_catalog_missing_active_region` | warn about region-less flares | matcher, path → `None` | `build` |
| `DatasetBuilder._validate_sequence_timestamps` | sorted + unique | timestamps, path, group → `None` or `ValueError` | row-based builders |
| `DatasetBuilder._validate_unique_timestamps` | unique | timestamps, path, group → `None` or `ValueError` | sequence builders |
| `DatasetBuilder._sequence_violates_cadence` | `cadence_minutes` check | window times → `bool` | row-based builders |
| `DatasetBuilder._build_sequences` | full-disk row-based | index, matcher, path → DataFrame | `build` |
| `DatasetBuilder._region_groups` | split index by region, apply policy | index, path → list of (region, rows) | AR sequence builders |
| `DatasetBuilder._warn_missing_region_sequences` | warning for region-less rows in sequences | index, count → `None` | AR sequence builders |
| `DatasetBuilder._group_name` | readable group name for messages | region → `str` | AR sequence builders |
| `DatasetBuilder._query_region` | query one region (never full disk for `None`) | matcher, time, region → list | AR sequence builders |
| `DatasetBuilder._build_sequences_active_region` | AR row-based, per region | index, matcher, path → DataFrame | `build` |
| `DatasetBuilder._grid_candidates` | walk a time grid | position map, first start, last time → emitted, counts, samples | time-based builders |
| `DatasetBuilder._validated_start` | effective start | timestamps, path → `Timestamp` | time-based builders |
| `DatasetBuilder._time_based_candidates` | full-disk grid + report | index, path → (candidates, report) | `_build_time_based_sequences` |
| `DatasetBuilder._build_time_based_sequences` | full-disk time-based | index, matcher, path → (DataFrame, report) | `build` |
| `DatasetBuilder._build_time_based_sequences_active_region` | AR time-based, one grid per region | index, matcher, path → (DataFrame, report) | `build` |
