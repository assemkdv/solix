# flare-indexer

flare-indexer builds labeled datasets for solar flare prediction. You give it
the times of your solar images and a GOES flare catalog. For each image, or
each sequence of images, it finds the flares in the prediction window after
the image and turns them into a label: binary, flux, class, counts or a
cumulative index. Full-disk and active-region data are both supported.

![Architecture](docs/architecture/overview.svg)

## Installation

Python 3.10 or newer. The only dependencies are pandas and numpy.

```bash
git clone https://github.com/assemkdv/flare-indexer.git
cd flare-indexer
pip install .
```

## Inputs

Two CSV files:

| File | Columns |
|---|---|
| Image index | `timestamp`, plus `active_region` for active-region labels, plus any columns you want to keep (e.g. `image_path`) |
| Flare catalog | `start_time, peak_time, end_time, class, active_region` (full date and time in every time cell), plus `rxfi` for the relative X-ray flux labels |

## Quick start

```python
import flare_indexer as fidx

builder = fidx.DatasetBuilder(
    prediction_window="24h",                     # window (t, t + 24h)
    strategy=fidx.BinaryThresholdStrategy("M"),  # 1 if a flare >= M1.0 starts in it
)
labels = builder.build("examples/data/full_disk_image_index.csv",
                       "examples/data/flare_catalog.csv")
```

The result is a pandas DataFrame with one row per image and a `label`
column.

The window is `(t, t + w)`: a flare counts if its start time is after the
image time `t` and before `t + w`. Use `event_time="peak"` to match the peak
time instead.

## Labels

| Label | Strategy |
|---|---|
| 0/1, strongest flare ≥ a GOES class | `BinaryThresholdStrategy("M")` |
| Peak flux of the strongest flare (W/m²) | `MaxFlareStrategy()` |
| Class of the strongest flare (NF, A, B, C, M, X) | `DatasetBuildingPipeline(MaxFluxReducer(), MulticlassLabeler())` |
| Number of flares per class | `DatasetBuildingPipeline(ClassCountReducer(), ClassCountLabeler())` |
| Cumulative index ΣC + 10ΣM + 100ΣX | `DatasetBuildingPipeline(CumulativeClassIndexReducer(), RegressionLabeler())` |
| Cumulative index ≥ a number, as 0/1 | `DatasetBuildingPipeline(CumulativeClassIndexReducer(), NumericThresholdLabeler(10))` |
| Largest relative X-ray flux increase (RXF) | `DatasetBuildingPipeline(MaxRelativeFluxReducer(), RegressionLabeler())` |
| Sum of RXF values | `DatasetBuildingPipeline(SumRelativeFluxReducer(), RegressionLabeler())` |

RXF is a flare's peak X-ray flux divided by the background X-ray flux of the
24 hours before it starts, read from the catalog's `rxfi` column. Add
`NumericThresholdLabeler(10)` for a 0/1 label. The RXF labels are for single
images only.

A pipeline is a reducer (flares → one value) and a labeler (value → label).
Any object with a `label(events)` method can be passed as `strategy`.

## Sequences

```python
builder = fidx.DatasetBuilder(
    prediction_window="24h",
    strategy=fidx.BinaryThresholdStrategy("M"),
    cadence="1h", observation_window="3h", sliding_window="1h",  # 3 images, 1 h apart
)
sequences = builder.build("examples/data/full_disk_image_index.csv",
                          "examples/data/flare_catalog.csv")
```

Each sequence is labeled from the time of its last image. Sequences with a
missing image are skipped. For fixed-length sequences of consecutive rows,
use `sequence_length=` and `stride=` instead.

## Active regions

```python
builder = fidx.DatasetBuilder(
    prediction_window="24h",
    strategy=fidx.BinaryThresholdStrategy("M"),
    target="active_region",
)
labels = builder.build("examples/data/active_region_image_index.csv",
                       "examples/data/flare_catalog.csv")
```

Only flares from the image's own region count. Sequences are built
separately for each region.

## Preparing real data

```python
index = fidx.build_image_index_from_filenames(
    "examples/data/hmi_image_list.txt", output_path="image_index.csv",
    strict=False)  # skip lines that are not HMI images
catalog = fidx.adapt_goes_catalog(
    "examples/data/goes_flares_sample.csv", output_path="flare_catalog.csv")
```

The first function reads HMI or SHARP file names and writes the image
index. The second converts an integrated GOES catalog into the catalog
format above.

## Examples

| Script | Shows |
|---|---|
| `examples/01_full_disk.py` | single-image labels |
| `examples/02_sequences.py` | image sequences |
| `examples/03_active_region.py` | active-region labels |
| `examples/04_loaders.py` | preparing real inputs |
| `examples/05_cumulative_index.py` | cumulative flare index |
| `examples/06_pytorch_images.py`, `07_pytorch_sequences.py` | real HMI images in PyTorch |
| `examples/08_tensorflow_images.py`, `09_tensorflow_sequences.py` | real HMI images in TensorFlow |

Examples 01–05 use the synthetic data in `examples/data/`. Examples 06–09
need `pip install ".[pytorch-examples]"` or `".[tensorflow-examples]"` and
download about 10 MB of images; see
[examples/real_data/README.md](examples/real_data/README.md).

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
