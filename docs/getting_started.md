# Getting started

> A ten-minute introduction. The full guide is the [researcher manual](researcher_manual.md).

This page is for someone who has never used `flare-indexer`. In about ten
minutes you will install it, label a small synthetic dataset, and know
where to look next.

## What the package does, in one paragraph

Machine-learning models for solar flare prediction learn from pairs of
*(solar image, label)*. The label answers a question such as *"Did an
M-class or stronger flare happen in the 24 hours after this image was
taken?"*. `flare-indexer` computes those labels. You give it two tables:

1. an **image index** – one row per image, with its timestamp (and, for
   active-region patches, the region it shows);
2. a **flare catalog** – one row per flare, with its start/peak/end time,
   GOES class (e.g. `M2.3`) and NOAA active region number.

It returns one label per image (or per sequence of images). It never
opens the image files; it only reads their timestamps.

## A concrete example

Image index (`timestamp`, `image_path`):

```
timestamp,image_path
2024-03-04 00:00:00,hmi/2024/03/04/HMI.m2024.03.04_00.00.00.jpg
```

Flare catalog (only one flare shown):

```
start_time,peak_time,end_time,class,active_region
2024-03-05 00:00:00,2024-03-05 00:10:00,2024-03-05 00:25:00,M1.0,13600
```

The flare starts exactly 24 hours after the image. Whether that image is
labeled `1` or `0` depends on how the window is defined:

| Window rule | Label |
|---|---|
| `peak_time` in `[t, t+24h)` (package default) | 0 |
| `start_time` in `(t, t+24h]` | 1 |
| `start_time` in `(t, t+23:59:59]` (what the legacy full-disk script does) | 0 |

This is not a contrived corner case: on the real 2010–2018 full-disk data,
this boundary is the *only* source of disagreement with the legacy labels
(see [validation/full_disk.md](validation/full_disk.md)).

## 1. Install

Python 3.10 or newer. The package is not on PyPI yet, so install it from
a clone of the repository:

```bash
git clone https://github.com/assemkdv/flare-indexer.git
cd flare-indexer
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                             # optional: run the test suite
```

## 2. Run the examples

The `examples/` folder has six short scripts and a small, fully synthetic
dataset in `examples/data/`. Run them from the repository root:

```bash
python examples/01_full_disk_single_image.py   # one 0/1 label per image
python examples/02_window_boundaries.py        # why the 24h boundary matters
python examples/03_sequences.py                # labels for image sequences
python examples/04_active_region.py            # per-region labels, missing regions
python examples/05_real_data_loaders.py        # converting real input formats
python examples/06_modular_pipeline.py         # inspect / customize each stage
```

Outputs are written to `examples/output/` (ignored by git).

## 3. Label your own images

```python
import flare_indexer as fidx

builder = fidx.DatasetBuilder(
    prediction_window=24,                         # hours, or a duration like "23h59min59s"
    strategy=fidx.BinaryThresholdStrategy("M"),   # 1 if any flare >= M1.0 in the window
    event_time="start",                           # which flare time is checked
    interval_mode="right_closed",                 # (t, t + window]
    include_metadata=True,                        # keep image_path etc. in the output
)
labels = builder.build("my_image_index.csv", "my_flare_catalog.csv")
labels.to_csv("labels.csv", index=False)
```

If your inputs are an HMI file listing and a GOES catalog like the lab's
`goes_flares_integrated.csv`, convert them first – see
`examples/05_real_data_loaders.py`.

## Where to go next

- [User guide](user_guide.md) – every input column, parameter, output
  column, window rule and error message.
- [Active-region workflow](active_region_workflow.md) – patches, HARP/NOAA
  numbers, and what is still missing for real active-region validation.
- [Architecture](architecture/architecture.md) – how the code is organized.
- [Open decisions](decisions.md) – choices that need a scientist's sign-off.
