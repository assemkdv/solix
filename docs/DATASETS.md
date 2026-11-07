# Datasets

## Inputs

`build(index, flare_catalog)` reads two CSV files.

**Index**: one row per instance. An instance is one point in time and
whatever file belongs to it: an image, a `.npy` array, anything. The
package never opens the files.

| Column | Needed |
|---|---|
| `timestamp` | always: the instance's observation time |
| `active_region` | for the two active-region classes: the NOAA region of the instance |
| anything else, e.g. `image_path` | optional; kept in the output (see below) |

**Flare catalog**: one row per flare.

| Column | Needed |
|---|---|
| `start_time`, `peak_time`, `end_time` | full date and time, e.g. `2024-03-01 04:00:00` (a blank cell is allowed; a flare with a blank `start_time` can never match, and the build warns) |
| `class` | GOES class, e.g. `M1.4` |
| `active_region` | the flare's NOAA region (blank if unknown) |
| `rxfi` | only for the RXF labels ([LABELS.md](LABELS.md)) |

[Preparing real data](#preparing-real-data) shows how to make both files from
HMI file names and a GOES catalog.

### Your own column names

If your files use other names, tell `build` which column is which, as
`{your name: package name}`:

```python
labels = builder.build(
    "my_index.csv", "my_flares.csv",
    index_columns={"obs_time": "timestamp", "noaa_ar": "active_region"},
    catalog_columns={"goes_class": "class", "begin": "start_time"},
)
```

The result uses the package's names (`timestamp`, `active_region`, ...);
other columns keep their own. Nothing is guessed: a column that is not in
the file, a name the package does not use, two columns given the same
name, or a name that already exists in the file stops the build with an
error.

## The prediction window

Each instance at time `t` is labeled from the flares in `[t, t + w)`: a flare
counts if it starts at `t` or later and before `t + w`. `w` is
`prediction_window`, e.g. `"24h"`, `"12h"` or a number of hours (`24`).

In `examples/data/`, the instance at 04:00 counts the C3.2 flare that starts at
exactly 04:00 (here with `MaxFluxReducer()` and `RegressionLabeler()`, whose
label is the strongest flare's peak flux, and `include_flares=True`):

```text
          timestamp    label             flares
2024-03-01 00:00:00 0.000027 [C3.2, M1.4, M2.7]
2024-03-01 04:00:00 0.000027 [C3.2, M1.4, M2.7]
```

A flare counts by its start time: one that starts inside the window counts
even if it peaks after the window ends.

## The four classes

| Class | One row per | Flares counted |
|---|---|---|
| `FullDiskSingleInstance` | full-disk instance | all |
| `FullDiskSequence` | sequence of full-disk instances | all |
| `ActiveRegionSingleInstance` | active-region instance | its own region only |
| `ActiveRegionSequence` | sequence of instances of one region | that region only |

Every class takes `prediction_window` first and `reducer` and `labeler` by
name (see [LABELS.md](LABELS.md)), then its own settings by name:

| Setting | Single instance | Sequence | Default |
|---|---|---|---|
| `cadence` | optional | required | none |
| `start` | optional, with `cadence` | optional | the first instance |
| `observation_window`, `sliding_window` | | required | |
| `include_metadata` | ✓ | (always on) | `False` |
| `include_flares` | ✓ | ✓ | `False` |
| `verbose` | ✓ | ✓ | `False` |
| `missing_active_region` | active region only | active region only | `"negative"` |

## Single instances

```python
builder = fidx.FullDiskSingleInstance(
    "24h", reducer=fidx.MaxFluxReducer(), labeler=fidx.BinaryThresholdLabeler("M"),
)
labels = builder.build("index.csv", "flare_catalog.csv")
```

The result has the columns `timestamp, label`.

`cadence` keeps only some of the instances: `cadence="6h"` keeps the
instances at the first instance's time, +6 h, +12 h, and so on (or from
`start=`, which must be an instance time). The cadence must be a multiple
of the time between the instances.

## Sequences

```python
builder = fidx.FullDiskSequence(
    "24h", reducer=fidx.MaxFluxReducer(), labeler=fidx.BinaryThresholdLabeler("M"),
    cadence="1h",              # time between instances in a sequence
    observation_window="3h",   # time one sequence covers: 3 instances
    sliding_window="1h",       # a new sequence starts every hour
)
sequences = builder.build("index.csv", "flare_catalog.csv")
```

```text
     sequence_start        sequence_end                                                      timestamps  n_instances  label
2024-03-01 00:00:00 2024-03-01 02:00:00 [2024-03-01 00:00:00, 2024-03-01 01:00:00, 2024-03-01 02:00:00]            3      1
2024-03-01 01:00:00 2024-03-01 03:00:00 [2024-03-01 01:00:00, 2024-03-01 02:00:00, 2024-03-01 03:00:00]            3      1
```

- Each sequence is labeled from its last instance: the window is
  `[sequence_end, sequence_end + w)`.
- Other index columns, such as `image_path`, come as lists in the
  same order as `timestamps`.
- A sequence with a missing instance is skipped; no nearby instance is used
  in its place.
- `cadence` may be coarser than the index (hourly sequences from a
  12-minute index), not finer, and must be a multiple of the index's own
  spacing.
  `observation_window` must be a multiple of `cadence`.
- `sliding_window` may be smaller than `cadence`: from a 12-minute index,
  `cadence="1h"` and `sliding_window="12min"` start a new hourly sequence
  every 12 minutes. It must be a multiple of the index's own spacing.
- `build(..., return_report=True)` also returns a `SequenceBuildReport`: how
  many sequences were made and skipped, and which instance times were missing.

## Active regions

```python
builder = fidx.ActiveRegionSingleInstance(
    "24h", reducer=fidx.MaxFluxReducer(), labeler=fidx.BinaryThresholdLabeler("M"),
)
labels = builder.build("active_region_index.csv", "flare_catalog.csv")
```

```text
          timestamp active_region  label
2024-03-01 00:00:00         13600      1
2024-03-01 00:00:00         13601      1
```

- Only the flares of an instance's own region count.
- Several regions can share a timestamp, so the result always has an
  `active_region` column.
- `ActiveRegionSequence` takes the same settings as `FullDiskSequence` and
  builds each region's sequences from that region's instances only.
- A catalog flare with no region can never match an instance, and a warning
  says how many there are.

`missing_active_region` decides what happens to an instance with no region
in the index:

| Value | Instance with no region |
|---|---|
| `"negative"` | labeled as if no flare occurred, with a warning |
| `"drop"` | left out |
| `"error"` | stops the build with an error |

## Extra columns

- `include_metadata=True` (single instances) keeps every other index
  column, e.g. `image_path`.
- `include_flares=True` adds a `flares` column next to the label: the
  classes of the flares in each window, e.g. `["C3.2", "M1.4"]`, so any
  label can be checked by eye.
- `verbose=True` prints a short summary of each build:

```text
Index: 118 instances (examples/data/full_disk_index.csv)
Flare catalog: 8 flares (examples/data/flare_catalog.csv)
Settings: window [t, t + 24h) on start_time, full disk, single instances
Result: 118 rows; label 0: 41, label 1: 77
```

## Preparing real data

```python
index = fidx.build_index_from_hmi_filenames(
    "hmi_image_list.txt", output_path="index.csv",
    strict=False)  # skip lines that are not HMI images
catalog = fidx.adapt_goes_catalog(
    "goes_flares.csv", output_path="flare_catalog.csv")
```

- `build_index_from_hmi_filenames` reads HMI full-disk file names such as
  `HMI.m2010.12.21_21.00.00.jpg` (a list or a text file with one per line)
  and writes the index. The timestamp comes from the file name. Other
  names, including SHARP patch names, stop it with an error (or are left
  out with `strict=False`); for patches, write the index yourself with
  `timestamp` and `active_region`.
- `adapt_goes_catalog` converts an integrated GOES flare catalog into the
  catalog format above. NOAA region numbers passed 9999 on 2002-06-14, and
  many flare lists kept only four digits, so for flares from that day on
  it adds the leading 1 back (3324 → 13324). A flare that is then listed
  twice (same start, peak, end, class and region) is kept once. A warning
  gives both counts.
- `add_relative_flux` adds the `rxfi` column ([LABELS.md](LABELS.md#relative-x-ray-flux-rxf)).
