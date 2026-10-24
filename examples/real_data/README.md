# Real-data PyTorch and TensorFlow examples (06-11)

Six examples label real SDO/HMI data with `flare_indexer` and load it with
a deep-learning framework. Examples 06-09 use full-disk images; 10-11 use
active-region patches. Framework code lives only in these scripts, not in
the package. PyTorch, TensorFlow, Pillow and astropy are optional; the base
package and its tests need none of them.

| Example | Dataset class | Sample | Batch shape |
|---|---|---|---|
| [`06_pytorch_images.py`](../06_pytorch_images.py) | `FlareDataset` (PyTorch `Dataset`) | one image, labeled from its observation time `t` | `[B, 1, 256, 256]` |
| [`07_pytorch_sequences.py`](../07_pytorch_sequences.py) | `FlareSequenceDataset` (PyTorch `Dataset`) | 3 images 1 h apart, labeled from the last time `t_end` | `[B, 3, 1, 256, 256]` |
| [`08_tensorflow_images.py`](../08_tensorflow_images.py) | `FlareDataset` (`keras.utils.PyDataset`) | as 06 | `[B, 256, 256, 1]` |
| [`09_tensorflow_sequences.py`](../09_tensorflow_sequences.py) | `FlareSequenceDataset` (`keras.utils.PyDataset`) | as 07, oldest image first | `[B, 3, 256, 256, 1]` |
| [`10_pytorch_active_region.py`](../10_pytorch_active_region.py) | `FlareDataset` (PyTorch `Dataset`) | one patch, labeled from its own region's flares | `[B, 1, 128, 128]` |
| [`11_tensorflow_active_region.py`](../11_tensorflow_active_region.py) | `FlareDataset` (`keras.utils.PyDataset`) | as 10 | `[B, 128, 128, 1]` |

All examples label with a 24 h prediction window and
`BinaryThresholdStrategy("M")`: a sample is 1 if an M- or X-class flare
starts in `[t, t + 24h)`. The settings are written out in each script.
The TensorFlow examples use the same data, decoder and label checks as the
PyTorch ones and move the channel axis last before batching.

Results:

- 06 and 08: 9 images, 4 positive, 5 negative.
- 07 and 09: 2 sequences, both 0.
- 10 and 11: 16 patches, 2 positive, 14 negative.

Each example recomputes every label from the catalog without
`flare_indexer` and stops if any label differs.

## Data

Full disk (06-09): SDO/HMI line-of-sight magnetograms (4096 x 4096, 8-bit
JPEG 2000) from the [Helioviewer API](https://api.helioviewer.org/docs/v2/),
requested hourly on 2017-09-03 from 01:00 to 10:00 UTC. The 07:00 request
has no image within 60 s and is recorded as missing, so 9 images are used.
[`fetch_real_data.py`](fetch_real_data.py) downloads them to `images/` (not
in git) and checks each file against the SHA-256 in
[`image_manifest.csv`](image_manifest.csv).

Active regions (10-11): 16 SHARP magnetograms (`hmi.sharp_cea_720s`) of
two regions, exported from [JSOC](http://jsoc.stanford.edu) with
[`fetch_sharp_patches.py`](fetch_sharp_patches.py) and kept in
`sharp_patches/` (listed in [`sharp_manifest.csv`](sharp_manifest.csv)).
JSOC has no records at 07:00 and 08:00 TAI.

| HARP | NOAA region | Flares |
|---|---|---|
| 7115 | 12673 | M1.2 starting 2017-09-04 05:36, then more M-class flares (up to M5.5) |
| 7120 | 12675 | none in 2017-09-01 .. 06 |

Timestamps are the observation times from each file's header: `DATE_OBS`
for the JPEG 2000 images, `DATE-OBS` for the SHARP files. Both are UTC and
used at full precision, never rounded.

Flare catalog: the NOAA SWPC GOES flare list from the
[Heliophysics Event Knowledgebase](https://www.lmsal.com/hek/) for
2017-09-01 to 2017-09-06 (55 flares). Raw result:
[`hek_swpc_flares.csv`](hek_swpc_flares.csv); package schema:
[`flare_catalog.csv`](flare_catalog.csv); query in
[`catalog_provenance.json`](catalog_provenance.json).

The docstrings of `fetch_real_data.py`, `fetch_sharp_patches.py`,
[`hmi_jp2.py`](hmi_jp2.py) and [`sharp_fits.py`](sharp_fits.py) give the
details: acquisition and replay, manifest columns, and hash checks.

## Setup and run

From the repository root:

```bash
pip install -e ".[pytorch-examples]"           # torch, pillow, astropy
python examples/real_data/fetch_real_data.py   # downloads the 9 images once (about 10 MB)
python examples/06_pytorch_images.py
python examples/07_pytorch_sequences.py
python examples/10_pytorch_active_region.py
pytest tests/test_pytorch_examples.py tests/test_fetch_real_data.py
```

TensorFlow needs Python 3.13 or older:

```bash
pip install -e ".[tensorflow-examples]"        # tensorflow, pillow, astropy
python examples/08_tensorflow_images.py
python examples/09_tensorflow_sequences.py
python examples/11_tensorflow_active_region.py
pytest tests/test_tensorflow_examples.py
```

Without the package installed, prefix the commands with
`PYTHONPATH=$PWD/src`. Outputs go to `examples/output/` (or `--output DIR`).
The pip source package does not include the SHARP patch files, so
examples 10-11 need a checkout of the repository.

## Preprocessing

Full disk ([`hmi_jp2.py`](hmi_jp2.py)): each JPEG 2000 image is decoded at
resolution level 4 (256 x 256) and divided by 255 to give float32 values in
[0, 1]. There is no rotation, cropping, limb masking or standardization.

Patches ([`sharp_fits.py`](sharp_fits.py)): missing pixels are set to 0 G,
the field is clipped to [-256, 255] G (the clip used in arpatch-sfp) and
scaled to [0, 1], and each patch is resized to 128 x 128 so that patches of
different sizes can be batched.

## Limitations

- The data cover one morning. They show how the package works; they are
  not a training set. The Helioviewer images are 8-bit display products,
  not calibrated data.
- Sequences use exact timestamps. Some HMI times are 0.1 s or 45 s off the
  hourly grid, and the 07:00 image is missing, so 6 of 8 candidate
  sequences are skipped. The package has
  no tolerance for this, and the examples do not add one.
