# flare-indexer

flare-indexer builds **labeled datasets for solar flare prediction**: given
solar image timestamps and a NOAA flare catalog, it looks forward in time
from each image (or each sequence of images), checks whether a flare occurs
within a configurable prediction window, and returns a table of
image/sequence → label pairs ready for training.

```
image timestamp(s) + flare catalog
        │
        ▼
  EventMatcher      flares in the window after each image
        │           (optionally only from the image's active region)
        ▼
  reducer + labeler one value per window: binary (≥ M?), max flux, class, counts
        │
        ▼
  DatasetBuilder    the labeled table (single images or sequences)
```

## Installation

Not yet published to PyPI. Install from source (Python ≥ 3.10; needs only
pandas and numpy):

```bash
git clone https://github.com/assemkdv/flare-indexer.git
cd flare-indexer
pip install .
```

Optional extras for the real-image examples: `pip install ".[pytorch-examples]"`
(torch, pillow) or `pip install ".[tensorflow-examples]"` (tensorflow, pillow;
Python 3.10–3.13).

## Quick example

```python
import flare_indexer as fidx

builder = fidx.DatasetBuilder(
    prediction_window="24h",                      # look 24 h ahead of each image
    strategy=fidx.BinaryThresholdStrategy("M"),   # 1 if any flare >= M1.0, else 0
    target="full_disk",                           # or "active_region"
    event_time="peak",                            # which flare time is checked
    interval_mode="left_closed",                  # [t, t + 24h)
)
labels = builder.build("examples/data/full_disk_image_index.csv",
                       "examples/data/flare_catalog.csv")
print(labels)
```

Inputs are two CSV files: an **image index** (`timestamp`, plus
`active_region` and any identifier columns you want back) and a **flare
catalog** (`start_time, peak_time, end_time, class, active_region`, every
time with its full date). Every labeling choice (window length, which
flare time, boundary rule, threshold) is an explicit parameter; the
defaults are listed in the researcher manual.

## Documentation

- **[Researcher manual](docs/researcher_manual.md)** — install, inputs and
  outputs, windows, strategies, sequences, active regions, real data,
  errors, limitations. Start here.
- **[API reference](docs/api_reference.md)** — every public class, function
  and method.
- **[System design](docs/architecture/system_design.md)** — how the code
  works inside, with one inspectable example.
- [Active-region workflow](docs/active_region_workflow.md) — labeling
  SHARP/HARP patches step by step.
- [Open decisions](docs/decisions.md) — labeling choices that still need a
  scientific sign-off (window edge, look-back, missing regions, coverage).
- Real-data validation: [full disk](docs/validation/full_disk.md),
  [active region](docs/validation/active_region.md) — reproduction of the
  lab's published labels, with the documented catalog corrections;
  [cumulative index](docs/validation/cumulative_index.md) — reproduction
  of the public SuryaBench cumulative flare index.
- [CHANGELOG](CHANGELOG.md).

The examples in the manual, the API reference and the system design are
run by `python scripts/check_doc_examples.py` and in the test suite, so
their recorded output is always current.

## Examples

`examples/01`–`06`, `09`, `12` and `13` run on the synthetic data in
`examples/data/` (single images, window boundaries, sequences, active
regions, real-data loaders, the modular pipeline, multiclass labels, counts,
the cumulative class index).
`examples/07`–`08` (PyTorch) and `10`–`11` (TensorFlow) label real SDO/HMI
magnetograms and load them as `[B,C,H,W]` / `[B,T,C,H,W]` and
`[B,H,W,C]` / `[B,T,H,W,C]` batches; see
[examples/real_data/README.md](examples/real_data/README.md).

```bash
python examples/01_full_disk_single_image.py
```

## Development

```bash
pip install -e ".[dev]"
pytest                                   # framework tests skip without torch/tensorflow
python scripts/check_doc_examples.py     # documentation examples
```

`scripts/validate_full_disk.py` and `scripts/validate_active_region.py`
rerun the real-data validation against the lab's published files (they
download about 90 MB into `scripts/.cache/`);
`scripts/validate_cumulative_index.py` compares the cumulative class index
with the public SuryaBench labels (about 13 MB).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
