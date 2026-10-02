# Open decisions

Choices that change labels and therefore need a scientist's sign-off
(Dr. Pandey). For each, the package already supports the options listed;
nothing below has been decided in code. Defaults are unchanged from
v0.2.1 until a decision is recorded here.

| ID | Question | Status |
|---|---|---|
| D1 | Full-disk window end: `t + 24h` or `t + 23:59:59`? | open |
| D2 | Active-region window: before or after the image? | open – blocks AR validation |
| D3 | Images with no active region: negative, dropped, or error? | open |
| D4 | What to do where the catalog has no coverage? | open |
| D5 | Package defaults (`peak` / `left_closed`) vs. lab convention | open |
| D6 | Catalog flares with no NOAA region in AR mode | open |

---

## D1. 24-hour window boundary

**Fact.** The legacy full-disk script counts flares with
`t < start ≤ t + 23:59:59`. With `prediction_window=24` the package uses
`t < start ≤ t + 24:00:00`. On 2010–2018 data this changes 6 of 63,649
labels (all six flares start exactly on the hour, 24 h after an image).
`prediction_window="23h59min59s"` reproduces all 63,649 legacy labels.

**Implication.** Images are hourly and catalog start times have minute
resolution, so "exactly 24 h later" happens whenever a flare starts on the
hour. Take an M flare starting at 05:00 on day 2:

- `(t, t+24h]` makes it count for the hourly images from 05:00 on day 1
  to 04:00 on day 2 – 24 images;
- `(t, t+23:59:59]` makes it count from 06:00 on day 1 to 04:00 on day 2 –
  23 images.

So the legacy rule gives an on-the-hour flare one positive image fewer
than every other flare. Both are defensible; the question is whether the
window means "24 hours" or "just under 24 hours".

**Options.** (a) keep `24h` – a clean 24-hour definition, 6 labels differ
from published work; (b) use `23h59min59s` – exact reproduction of the
published dataset; (c) `[t, t+24h)` – half-open in the other direction.

**Question for Dr. Pandey.** Was `23:59:59` intended as "24 hours,
exclusive end", and should new datasets match the published one exactly?

## D2. Active-region window direction

**Fact.** All 294,286 published AR labels (arpatch-sfp train/val/test) are
reproduced exactly by `start_time in [t − 24h, t]` (flares *before* the
image). A forward window `(t, t+24h]` agrees on 96.567 % and gives 7,831
positives instead of 10,315. Details: [validation/active_region.md](validation/active_region.md).

**Implication.** A look-back label means "this region flared in the last
24 h", which is a different task from forecasting. If the published labels
were meant to be forecasts, they include information from after the
prediction time; if look-back was intended (e.g. as a nowcast or for a
model that uses a later shift), the package needs a look-back option to
reproduce it. The package currently only looks forward.

**Question.** Is the look-back window in `step3_labeling.py` intentional?
Which direction should flare-indexer use for AR datasets?

## D3. Images without an active region

**Fact.** Before this branch, such images were silently labeled 0. Now the
default still labels them 0 but warns; `missing_active_region="drop"`
removes them and `"error"` stops.

**Implication.** An image whose region is unknown has an *unknown* label.
Putting it in the negative class can add real flaring patches to the
negatives, which lowers recall and biases skill scores (TSS/HSS) in ways
that are hard to see later. Dropping them removes data but keeps the
negative class clean. In the published AR files every patch has a HARP
number, so this mainly affects other datasets.

**Recommendation to discuss.** Make `"drop"` or `"error"` the default in
the next minor release; keep `"negative"` available for reproducing old
results.

## D4. Catalog coverage

**Fact.** The catalog's last flare starts 2018-12-29 13:16. The package
cannot tell "no flare happened" from "the catalog does not cover this
time". 58 full-disk images and 333 AR patches have windows extending past
that point (286 AR patches are entirely after it).

**Options.** (a) document only; (b) add a `catalog_coverage=(start, end)`
parameter that drops or flags images whose window is not fully covered.

## D5. Package defaults

**Fact.** Defaults are `event_time="peak"`, `interval_mode="left_closed"`
(`[t, t+w)` on peak time). The lab's full-disk convention is
`event_time="start"`, `interval_mode="right_closed"`.

**Question.** Should the defaults follow the lab convention before the
first public release? Changing them later breaks users' results.

## D6. Flares without a NOAA region in AR mode

**Fact.** 2,343 of 14,401 flares (33 of 792 M/X flares) have no NOAA
region; 108 more have a NOAA region not in `allharp2noaa.csv`. In AR mode
they can never be assigned to a patch, so the patch that produced them is
labeled negative for that flare. The package now warns about this.

**Options.** (a) accept and document; (b) assign by flare location
(`fl_lon`, `fl_lat`) when the patch's bounding box is known – needs the
SHARP metadata table.
