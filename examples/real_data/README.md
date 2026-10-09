# Real-image PyTorch and TensorFlow examples (07, 08, 10, 11)

Four examples label the same real SDO/HMI full-disk images with
`flare_indexer` and load them with a deep-learning framework:

| Example | Framework | Sample | Batch shape |
|---|---|---|---|
| [`07_pytorch_full_disk_images.py`](../07_pytorch_full_disk_images.py) | PyTorch `Dataset`/`DataLoader` | one image, labeled from its observation time `t` | `[B, C, H, W]` = `[B, 1, 256, 256]` |
| [`08_pytorch_full_disk_sequences.py`](../08_pytorch_full_disk_sequences.py) | PyTorch `Dataset`/`DataLoader` | 3 images 1 h apart, labeled from the final observation time `t_end` | `[B, T, C, H, W]` = `[B, 3, 1, 256, 256]` |
| [`10_tensorflow_full_disk_images.py`](../10_tensorflow_full_disk_images.py) | `tf.data.Dataset` | as 07 | `[B, H, W, C]` = `[B, 256, 256, 1]` |
| [`11_tensorflow_full_disk_sequences.py`](../11_tensorflow_full_disk_sequences.py) | `tf.data.Dataset` | as 08, oldest image first | `[B, T, H, W, C]` = `[B, 3, 256, 256, 1]` |

Framework code lives only in these example scripts, not in the package.
PyTorch, TensorFlow and Pillow are optional dependencies; the base
package and its tests need none of them.

The TensorFlow examples use the same manifest, cached images, catalog,
decoder, provenance checks and independent label checks as the PyTorch
ones. The shared decoder returns channel-first `[C, H, W]`; examples 10
and 11 convert each image explicitly to TensorFlow's channel-last
`[H, W, C]` (`to_channels_last` in example 10) before batching.

## Labeling settings

All four examples set every labeling choice explicitly; the package defaults
are not used and not changed.

```python
fidx.DatasetBuilder(
    prediction_window="24h",
    strategy=fidx.BinaryThresholdStrategy("M"),  # 1 if any flare >= M1.0
    target="full_disk",
    event_time="start",
    interval_mode="right_closed",                # (t, t + 24h]
    # examples 08 and 11 only:
    cadence="1h", observation_window="3h", sliding_window="1h",
)
```

The window edge (`24h` vs `23h59min59s`) and catalog coverage are open
questions D1 and D4 in [`docs/decisions.md`](../../docs/decisions.md);
these examples do not settle them.

## Setup and run

From the repository root:

```bash
pip install -e ".[pytorch-examples]"              # flare_indexer + torch + pillow
python examples/real_data/fetch_real_data.py      # replays the committed manifest (~9 MB once)
python examples/07_pytorch_full_disk_images.py
python examples/08_pytorch_full_disk_sequences.py
pytest tests/test_pytorch_examples.py tests/test_fetch_real_data.py
```

TensorFlow (2.21 publishes wheels for Python 3.10–3.13, not 3.14):

```bash
python3.13 -m venv .venv-tf && . .venv-tf/bin/activate
pip install -e ".[tensorflow-examples]"           # flare_indexer + tensorflow + pillow
python examples/10_tensorflow_full_disk_images.py
python examples/11_tensorflow_full_disk_sequences.py
pytest tests/test_tensorflow_examples.py          # add torch to also compare with 07/08
```

On macOS with a python.org interpreter, downloads can fail with
`CERTIFICATE_VERIFY_FAILED` until "Install Certificates.command" (in the
Python application folder) has been run, or with
`SSL_CERT_FILE=$(python -m certifi)` set. The script then stops without
changing the manifest or writing partial files.

Without the package installed, prefix the commands with
`PYTHONPATH=$PWD/src`. Outputs go to `examples/output/` (or `--output DIR`).
Without PyTorch, Pillow or the downloaded images the examples stop with a
message and a non-zero exit code, and the tests that need them are skipped.

## Data and provenance

Acquisition ([`fetch_real_data.py`](fetch_real_data.py)) is separate from
indexing and labeling. It needs network access only for files that are not
cached yet; the examples never use the network.

### Acquisition modes

| Command | When | What it does |
|---|---|---|
| `fetch_real_data.py` with no `image_manifest.csv` | first acquisition | asks Helioviewer `getClosestImage` for each requested time, downloads the images, reads their headers, writes the manifest |
| `fetch_real_data.py` with the manifest present (the normal case) | replay | uses only the recorded image ids, URLs and SHA-256: no `getClosestImage`, nothing downloaded for rows recorded as `missing` or `duplicate`; already-cached files are not downloaded again |
| `fetch_real_data.py --refresh` | explicit refresh | runs discovery again and rewrites the manifest; the archive may now return other images |

Every cached or downloaded file must match the SHA-256 in the manifest. On
a mismatch the script stops with `HashMismatchError`: a downloaded file is
not written, a cached file is not touched, and the recorded hash is not
replaced. Replay rewrites only the columns read from the verified files'
headers (`timestamp`, `date_obs`, `offset_seconds`, `T_OBS`, ...). During
discovery, a request answered by an image id or observation time already
used for an earlier request is recorded as `duplicate` (with
`duplicate_of`) and gets no `image_path`; with hourly requests this cannot
occur, but it is handled. `--refresh-catalog` queries HEK again.

**Images.** SDO/HMI line-of-sight magnetograms (`INSTRUME=HMI_FRONT2`,
`CONTENT=MAGNETOGRAM`, 6173 Å, 4096 × 4096, 8-bit JPEG 2000) from the
[Helioviewer API](https://api.helioviewer.org/docs/v2/):
`getClosestImage?sourceId=19` for each requested time, then
`getJP2Image?id=...`. Ten times were requested, hourly from 2017-09-03
01:00 to 10:00 UTC. Files are cached in `images/` (not in git).

**Timestamps.** Each image's timestamp is `DATE_OBS` (UTC) read from the
FITS header embedded in the downloaded JP2, **at full precision**
(e.g. `2017-09-03 00:59:49.900000`). It is used unchanged for the image
index, for building sequences and as the label reference time; nothing is
rounded and there is no cadence tolerance. Helioviewer's own `date`
(whole seconds) is kept as `helioviewer_date` for provenance only; the
script checks that it equals a rounded copy of `DATE_OBS`. The requested
time is kept only as a record.

[`image_manifest.csv`](image_manifest.csv) records, per request:
`requested_time`, `status` (`ok`, `missing`, `duplicate`), `duplicate_of`,
`timestamp`, `date_obs`, `offset_seconds`, `image_path`, `helioviewer_id`,
`helioviewer_date`, `T_OBS`, `T_REC` (TAI), `QUALITY`, `CROTA2`, size,
`sha256` and `source_url`. For the `missing` row, `sha256` and `bytes`
describe the rejected candidate, which is not kept.

| Requested (UTC) | Observed `DATE_OBS` | Status |
|---|---|---|
| 01:00, 02:00, 03:00, 04:00 | 00:59:49.9, 01:59:49.9, 02:59:49.9, 03:59:49.9 | ok |
| 05:00, 06:00 | 04:59:50.0, 05:59:50.0 | ok (0.1 s later than the hourly grid from 00:59:49.9) |
| 07:00 | 07:36:35.0 (+36 min) | **missing** – no image within 60 s; not replaced |
| 08:00 | 07:59:50.0 | ok |
| 09:00 | 09:00:34.9 (+35 s) | ok, but off the hourly grid |
| 10:00 | 09:59:49.9 | ok |

`QUALITY` is non-zero for all images (1024, 1056 or 66560). The examples
record it but do not interpret or filter on it.

**Flare catalog.** NOAA SWPC GOES X-ray flare list, queried from the
[Heliophysics Event Knowledgebase](https://www.lmsal.com/hek/) (`FRM_NAME=SWPC`)
for 2017-09-01 00:00 to 2017-09-06 00:00 (exact URL in
[`catalog_provenance.json`](catalog_provenance.json)): 55 flares, all
with a GOES class. Raw result: [`hek_swpc_flares.csv`](hek_swpc_flares.csv);
in the package schema: [`flare_catalog.csv`](flare_catalog.csv). For this
period it lists the same 55 (start time, class) pairs as the lab catalog
`goes_flares_integrated.csv` (checked by
`test_catalog_agrees_with_lab_catalog_when_cached` when that file is in
`scripts/.cache/`).

Before labeling, the examples stop unless every window `(t, t + 24h]`
lies inside the queried interval (last window ends 2017-09-04 09:59:50).
A flare list cannot show that GOES observed without gaps, so a 0 label
means "no M/X flare in this list", as discussed in D4.

## Preprocessing

`hmi_jp2.decode_magnetogram`, identical in both examples:

1. Decode the 4096 × 4096 8-bit JP2 at JPEG 2000 resolution level 4
   (OpenJPEG wavelet reduction) → 256 × 256.
2. Divide by 255 → float32 in [0, 1]. These are Helioviewer's 8-bit
   display values, not calibrated Gauss.
3. Add a channel axis → `[1, 256, 256]`.

No rotation (`CROTA2 = 0`, already north-up), cropping, limb masking or
standardization.

## Results (executed 2026-10-09)

Example 07 — 9 images, 4 positive, 5 negative:

```
              timestamp  label                           image_path
2017-09-03 00:59:49.900      0 images/hmi_magnetogram_106225337.jp2
2017-09-03 01:59:49.900      0 images/hmi_magnetogram_106225417.jp2
2017-09-03 02:59:49.900      0 images/hmi_magnetogram_106225497.jp2
2017-09-03 03:59:49.900      0 images/hmi_magnetogram_106225575.jp2
2017-09-03 04:59:50.000      0 images/hmi_magnetogram_106225659.jp2
2017-09-03 05:59:50.000      1 images/hmi_magnetogram_106225744.jp2
2017-09-03 07:59:50.000      1 images/hmi_magnetogram_106225795.jp2
2017-09-03 09:00:34.900      1 images/hmi_magnetogram_106225863.jp2
2017-09-03 09:59:49.900      1 images/hmi_magnetogram_106225950.jp2

batch 0: images [4, 1, 256, 256] torch.float32 range [0.000, 1.000]  labels [0, 0, 0, 0]
batch 1: images [4, 1, 256, 256] torch.float32 range [0.000, 1.000]  labels [0, 1, 1, 1]
batch 2: images [1, 1, 256, 256] torch.float32 range [0.000, 1.000]  labels [1]

  t=2017-09-03 05:59:50  package label=1  catalog M/X flares: M1.2 at 2017-09-04 05:36:00
  t=2017-09-03 04:59:50  package label=0  catalog M/X flares: none
```

The label changes between 04:59:50.0 and 05:59:50.0 because the first
M-class flare in the catalog after the images, M1.2 from NOAA 12673,
starts 2017-09-04 05:36, which is inside `(t, t + 24h]` only from 05:36 on
09-03.

Example 08 — 8 candidate sequences, 2 emitted, 6 skipped, both labeled 0:

```
Time grid: inferred source cadence 0 days 01:00:00, start 2017-09-03 00:59:49.900000
candidates 8, emitted 2, skipped 6
  skipped sequence starting 2017-09-03 02:59:49.900000: no image at 2017-09-03 04:59:49.900000
  skipped sequence starting 2017-09-03 03:59:49.900000: no image at 2017-09-03 04:59:49.900000
  skipped sequence starting 2017-09-03 04:59:49.900000: no image at 2017-09-03 04:59:49.900000
  skipped sequence starting 2017-09-03 05:59:49.900000: no image at 2017-09-03 05:59:49.900000
  skipped sequence starting 2017-09-03 06:59:49.900000: no image at 2017-09-03 06:59:49.900000
  skipped sequence starting 2017-09-03 07:59:49.900000: no image at 2017-09-03 07:59:49.900000

         sequence_start            sequence_end  n_images  label
2017-09-03 00:59:49.900 2017-09-03 02:59:49.900         3      0
2017-09-03 01:59:49.900 2017-09-03 03:59:49.900         3      0

batch 0: images [2, 3, 1, 256, 256] torch.float32  labels [0, 0]
```

The grid starts at the first image, 00:59:49.9. The images at 04:59:50.0
and 05:59:50.0 are 0.1 s after their grid points, 07:00 was never
acquired, and the image near 09:00 was observed at 09:00:34.9, so 6 of 8
candidates lack a grid image and are skipped. No image is used in place
of a grid time.

**Change from the first version of this example.** The first version
rounded `DATE_OBS` to whole seconds. That turned 00:59:49.9 … 03:59:49.9
into 00:59:50 … 03:59:50, the same whole-second grid as 04:59:50.0 and
05:59:50.0, and gave 4 sequences, one of them positive (ending 05:59:50).
That positive sequence existed only because of the rounding; with the
actual observation times it is incomplete and is skipped. The data were not changed to recover it.
Single-image labels are the same in both versions.

Examples 10 and 11 (TensorFlow 2.21.0, Python 3.13, executed 2026-10-09)
produce the same label tables as 07 and 08 (byte-identical CSVs) and the
same samples in the same order:

```
10: tf.data.Dataset.from_generator(...).batch(4)  element_spec images: TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32, name=None)
batch 0: images [4, 256, 256, 1] float32 range [0.000, 1.000]  labels [0, 0, 0, 0]
batch 1: images [4, 256, 256, 1] float32 range [0.000, 1.000]  labels [0, 1, 1, 1]
batch 2: images [1, 256, 256, 1] float32 range [0.000, 1.000]  labels [1]
9 images: 4 positive, 5 negative; all checks passed

11: tf.data.Dataset.from_generator(...).batch(2)  element_spec images: TensorSpec(shape=(None, 3, 256, 256, 1), dtype=tf.float32, name=None)
candidates 8, emitted 2, skipped 6
batch 0: images [2, 3, 256, 256, 1] float32  labels [0, 0]
2 sequences: 0 positive, 2 negative; all checks passed
```

`test_tensorflow_and_pytorch_loaders_yield_the_same_samples` checks, for
every sample, that the TensorFlow array equals the PyTorch tensor after
moving the channel axis last, and that timestamps, image paths and labels
are identical.

## Saved sequence index

`08_sequence_labels.csv` has two list-valued columns, `timestamps` and
`image_path`. Examples 08 and 11 write them as JSON lists (timestamps as ISO
strings) and read them back with `load_sequences` (both in `real_examples.py`), which parses them into
lists of `pd.Timestamp` and `str`; the examples check the reloaded table
equals the original. A plain `pd.read_csv` returns these columns as
strings.

## Checks performed

- Every loaded image's header time equals its index time.
- Within each sequence: oldest image first, images exactly 1 h apart,
  last image time = `sequence_end`.
- DataLoader / tf.data order, labels and image paths equal the label table.
- Batch shapes and dtype (`float32`, values in [0, 1]) in every batch.
- Every label is recomputed without `flare_indexer` from the catalog rows
  (`t < start_time ≤ t + 24h`, class M or X), with `t` parsed directly from
  the full-precision `DATE_OBS` string in the manifest (which must equal
  the index timestamp; for sequences, the last image's).
- `image_path` survives into the single-image labels and the sequence table.

## Limitations

- 9 images from one morning; a demonstration, not a training set. Labels
  are 4/5 (images) and 0/2 (sequences): with the actual observation times
  no complete sequence ends late enough to be positive.
- Helioviewer's 8-bit JP2 are display products. Science use should start
  from calibrated FITS (e.g. JSOC `hmi.M_720s`), which needs a registered
  JSOC export e-mail and was not used here.
- `QUALITY` flags are recorded, not interpreted.
- Catalog coverage is the query interval only (D4). The window edge `24h`
  is the example's explicit choice, not a resolution of D1.
- The grid uses exact timestamps. HMI observation times here drift by
  0.1 s, which is enough to break the hourly grid. Building longer
  sequences from such data would need an explicit, documented tolerance
  policy; the package has none, and these examples do not add one.
- A replay depends on Helioviewer still serving the recorded ids with the
  same bytes. If it does not, the SHA-256 check stops the script rather
  than using different files.
