# Open decisions

Choices that change labels and therefore need a scientist's sign-off
(Dr. Pandey). For each, the package already supports the options listed;
nothing below has been decided in code. Defaults are unchanged from
v0.2.1 until a decision is recorded here.

| ID | Question | Status |
|---|---|---|
| D1 | Full-disk window end: `t + 24h` or `t + 23:59:59`? | open |
| D2 | Active-region window: before or after the image? | open – blocks matching the published AR labels with the package |
| D3 | Images with no active region: negative, dropped, or error? | open |
| D4 | How to record the catalog's observation coverage? | open – needs coverage information |
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

**Fact.** All 511,421 rows of the published arpatch-sfp `-90_to_90`
fold files are reproduced exactly – both the `label` and the full
`goes_class` string – by the rule in `step3_labeling.py`:
same HARP, `start_time` in `[t − 1 day, t]` (flares *before* the image).
A forward window `(t, t+24h]` agrees on 499,009 rows (97.573 %).
Sources, commands and limitations: [validation/active_region.md](validation/active_region.md).

**Implication.** A look-back label answers "did this region start an
M/X flare in the 24 hours up to the image?", which is a different task
from forecasting the next 24 hours. If the dataset was meant for
forecasting, the published labels do not encode that target. If look-back
was intended (for example, as a nowcast or with a shift applied later),
the package needs a look-back option to reproduce it. The package
currently only looks forward. This finding concerns the published label
files; it is not a validation of the complete original patch set.

**Question.** Is the look-back window in `step3_labeling.py` intentional?
Which direction should flare-indexer use for AR datasets?

## D3. Images without an active region

**Fact.** Before this branch, such images were silently labeled 0. Now the
default still labels them 0 but warns; `missing_active_region="drop"`
removes them and `"error"` stops. In sequence modes such images never
join a region's sequence; with the default they form one separate group.

**Implication.** For an image whose region is unknown, a 0 label is an
assumption, not an observation. If any such image actually belongs to a
region that flared in its window, its 0 is a false negative. Nothing in
the data tells us whether that happened, so the labels may be wrong, not
that they are. The effect on a trained model or on skill scores (TSS,
HSS, recall) is not known in advance; it depends on how many such images
there are and what their true labels would be. Dropping them avoids
assigning a label that is not supported by data, at the cost of losing
those images. In the published AR files every patch has a HARP number,
so this mainly matters for other datasets.

**Question.** Should the default become `"drop"` or `"error"` in the next
minor release (keeping `"negative"` available to reproduce old results)?

## D4. Catalog coverage

**Fact.** A flare catalog lists flares; it does not state the periods
during which flares were observed and recorded. The package therefore
cannot distinguish "no flare happened in this window" from "this window
was not covered by the catalog", and labels both 0. The latest flare
recorded in `goes_flares_integrated.csv` starts 2018-12-29 13:16, but that
is not evidence that coverage ends there: the end of 2018 was near solar
minimum, so a long quiet period is plausible. For reference only: 58
full-disk images have a window ending after that time, and 1,244
published AR patches are later than it.

**Implication.** Where the catalog is incomplete, 0 labels can be false
negatives; where it is complete, they are correct. Without explicit
coverage information we cannot say which applies.

**Options.** (a) document only (current state); (b) the user supplies
explicit coverage intervals (e.g. from the catalog's provenance or GOES
data availability), and the package drops, flags or rejects images whose
window is not fully inside them. The package must not infer coverage
from the first or last recorded flare.

**Question.** Is the coverage period of `goes_flares_integrated.csv`
documented anywhere (start and end, and any known gaps)?

## D5. Package defaults

**Fact.** Defaults are `event_time="peak"`, `interval_mode="left_closed"`
(`[t, t+w)` on peak time). The lab's full-disk convention is
`event_time="start"`, `interval_mode="right_closed"`.

**Question.** Should the defaults follow the lab convention before the
first public release? Changing them later breaks users' results.

## D6. Flares without a NOAA region in AR mode

**Fact.** 2,343 of 14,401 flares (33 of 792 M/X flares) have no NOAA
region; 108 more have a NOAA region not in `allharp2noaa.csv`. In AR mode
they cannot be assigned to any patch. The package warns about this.

**Implication.** If one of these flares came from a region that has
patches in the dataset, that patch may be labeled 0 when the flare
would have made it 1 – a possible false negative. Which patches, if any,
are affected cannot be determined without a location-based assignment.

**Options.** (a) accept and document; (b) assign by flare location
(`fl_lon`, `fl_lat`) when the patch's bounding box is known – needs the
SHARP metadata table.
