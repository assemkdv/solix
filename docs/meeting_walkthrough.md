# Meeting walkthrough

A short guide for presenting flare-indexer to Dr. Pandey. It separates
**finished software work** from **open scientific decisions** and
**missing data**.

## Files to open, in order

All paths are relative to the repository root
(`/Users/assemkadirova/Developer/flare-indexer`).

| # | File | What to show |
|---|---|---|
| 1 | [`docs/validation/full_disk.md`](validation/full_disk.md) | 63,649 / 63,649 labels reproduced with a 23h59min59s window; the six 24-hour boundary cases. |
| 2 | [`docs/validation/active_region.md`](validation/active_region.md) | Section A: all 511,421 published AR rows are reproduced by a look-back window. Section B: what could not be validated. |
| 3 | [`docs/decisions.md`](decisions.md) | The questions (D1–D6), starting with D2. |
| 4 | [`docs/architecture/system_design.md`](architecture/system_design.md) | How the code works, with a traced example. |
| 5 | [`docs/architecture/flare_indexer_architecture.pdf`](architecture/flare_indexer_architecture.pdf) | Pages 1 (overview) and 3 (time windows) if asked. |
| 6 | [`docs/researcher_manual.md`](researcher_manual.md) | How a researcher would use the package. |
| 7 | [`docs/api_reference.md`](api_reference.md) | Reference only. |

On a Mac:

```bash
cd /Users/assemkadirova/Developer/flare-indexer
open -e docs/validation/full_disk.md docs/validation/active_region.md docs/decisions.md
open docs/architecture/flare_indexer_architecture.pdf
```

## One working demo

From the repository root, with the package installed (`pip install -e .`):

```bash
python examples/02_window_boundaries.py
```

Expected output (synthetic data):

```text
Image time t = 2024-03-04 00:00:00

package default:  peak_time in [t, t+24h)          label=0  flares: none
legacy-like:      start_time in (t, t+24h]         label=1  flares: M1.0 (start 03-05 00:00)
legacy exact:     start_time in (t, t+23:59:59]    label=0  flares: none

The M1.0 flare starts at exactly t + 24h. It is inside (t, t+24h] but outside
[t, t+24h) and outside (t, t+23:59:59]. On the real full-disk data this single
boundary choice explains all 6 differences from the legacy labels.
```

What to say: "The same image and the same flare give label 0, 1, 0
depending only on where the window ends. This is exactly what happens in
the six real full-disk differences, and it is decision D1."

## What to say (about one minute)

> "flare-indexer turns image timestamps and a flare catalog into training
> labels, for example: did an M-class or stronger flare start in the 24
> hours after this image?
>
> **Software work that is finished and tested.** The package reproduces
> your full-disk labels exactly, 63,649 of 63,649, when the window ends at
> 23:59:59 as in your script. It builds labels for single images and for
> sequences, for the full disk and for active regions, including several
> regions observed at the same time. It checks inputs, warns about
> missing regions, and is documented with runnable examples. The test
> suite passes on GitHub for Python 3.10 to 3.13.
>
> **What we found.** The published active-region labels look 24 hours
> *back* from each image, not forward. That rule reproduces all 511,421
> published rows. The package itself only looks forward.
>
> **What needs your decision.** Should the active-region window look back
> or forward? Should the full-disk window end at 24 hours or one second
> earlier? What should happen to images with no region?
>
> **What data is missing.** The full SHARP patch table and the catalog's
> coverage dates. Without them we can reproduce the published labels, but
> not validate the full original dataset."

## Keep these three things separate

| Category | Items |
|---|---|
| **Verified** | Full-disk reproduction (100 % with 23h59min59s). Published AR labels reproduced by look-back (100 %, label and class). Tests and builds pass in GitHub CI. |
| **Open decisions** | D1 window end, D2 AR window direction, D3 missing-region policy, D5 defaults, D6 flares without a NOAA region, TAI/UTC handling. |
| **Missing data** | `meta_data_updated.csv` (full SHARP patch table); coverage dates of `goes_flares_integrated.csv`. |

Do not claim: that the labels are scientifically correct (we showed
reproduction), that the AR validation is complete, or that the package
supports a look-back window (it exists only in
`scripts/validate_active_region.py` for comparison).
