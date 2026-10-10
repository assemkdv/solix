# Open decisions

Choices that change labels and therefore need a scientist's sign-off
(Dr. Pandey). The seven questions below are the ones sent to him on
2026-10-09, in the same order. For each, the package already supports the
options listed; nothing has been decided in code. Defaults are unchanged
from v0.2.1 until an answer is recorded here.

| ID | Question | Status |
|---|---|---|
| Q1 | What should cumulative indexing calculate? | open – not implemented |
| Q2 | Active-region labels: flares before or after the image? | open – blocks reproducing the published AR labels with the package API |
| Q3 | Default flare time (start/peak) and window boundaries | open |
| Q4 | Missing region numbers: on images, and on catalog flares | open |
| Q5 | Does an empty window mean "no flare" or "no observations"? | open – needs coverage information |
| Q6 | Image timestamps: cadence tolerance, TAI vs UTC | open |
| Q7 | Flux below A1.0 | open |

(Earlier numbering: D1 and D5 are now Q3, D2 is Q2, D3 and D6 are Q4,
D4 is Q5. Q1, Q6 and Q7 are new.)

---

## Q1. What should cumulative indexing calculate?

**Fact.** Implemented: maximum flux (`MaxFluxReducer`), minimum flux
(`MinFluxReducer`), per-class counts (`ClassCountReducer` +
`ClassCountLabeler`) and class presence (`ClassPresenceLabeler`).
Cumulative indexing is not implemented. The AIKE slides show only the
maximum-flare example.

**Options.** (a) sum of the peak fluxes of every flare in the window –
for C2.0, C3.0 and M1.0 that is 2e-6 + 3e-6 + 1e-5 = 1.5e-5 W/m²;
(b) another formula (e.g. time-integrated flux, which needs start/end
times and a flare profile the catalog does not have).

**Question.** Add the peak fluxes, or use another formula?

## Q2. Active-region labels: flares before or after the image?

**Fact.** All 511,421 rows of the published arpatch-sfp `-90_to_90`
fold files are reproduced exactly – both the `label` and the full
`goes_class` string – by the rule in `step3_labeling.py` (lines 22–23):
same HARP, `start_time` in `[t − 1 day, t]` (flares *before* the image).
A forward window `(t, t+24h]` agrees on 499,009 rows (97.573 %).
Sources, commands and limitations: [validation/active_region.md](validation/active_region.md).

**Implication.** A look-back label answers "did this region start an
M/X flare in the 24 hours up to the image?", which is a different task
from forecasting the next 24 hours. If the dataset was meant for
forecasting, the published labels do not encode that target. If look-back
was intended (for example, as a nowcast or with a shift applied later),
the package needs a look-back option to reproduce it. The package
currently only looks forward; the look-back rule exists only in
`scripts/validate_active_region.py`. This finding concerns the published
label files; it is not a validation of the complete original patch set.

**Question.** Was the look-back intended? Which direction should the
package support as the default (the other stays available as an option)?

## Q3. Default flare time and window boundaries

**Fact (flare time).** The package can match by `start_time` or
`peak_time` (`event_time`). The lab's full-disk `labeling.py` uses start
time; the package default is `event_time="peak"`.

**Fact (boundaries).** The package default is
`interval_mode="left_closed"`, `[t, t+w)`. The legacy full-disk script
counts flares with `t < start ≤ t + 23:59:59`, i.e. `right_closed`. With
`prediction_window=24` and `right_closed` the package uses
`t < start ≤ t + 24:00:00`; on 2010–2018 data this changes 6 of 63,649
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
window means "24 hours" or "just under 24 hours". Changing defaults after
the first public release would silently change users' results.

**Options.** Flare time: `start` (lab convention) or `peak`. Window:
(a) `(t, t+24h]` – a clean 24-hour definition, 6 labels differ from the
published work; (b) `(t, t+23:59:59]` – exact reproduction; (c)
`[t, t+24h)` – half-open in the other direction (current default).

**Question.** Change the default to start time? Should a flare starting
exactly at the image time count, and one starting exactly 24 h later?

## Q4. Missing region numbers

**Fact (images).** An image with no region number is labeled 0 with a
warning (`missing_active_region="negative"`, the default);
`"drop"` removes it and `"error"` stops. In sequence modes such images
never join a region's sequence; with the default they form one separate
group. In the published AR files every patch has a HARP number, so this
mainly matters for other datasets.

**Fact (catalog flares).** 2,343 of 14,401 flares (33 of 792 M/X flares)
in `goes_flares_integrated.csv` have no NOAA region; 108 more have a NOAA
region not in `allharp2noaa.csv`. In AR mode they cannot be assigned to
any patch. The package warns about this.

**Implication.** For an image whose region is unknown, a 0 label is an
assumption, not an observation; if its region flared, the 0 is a false
negative. Likewise a region-less flare that came from a region with
patches in the dataset may leave those patches at 0. Which patches, if
any, are affected cannot be determined without a location-based
assignment (`fl_lon`, `fl_lat` against the patch's bounding box, which
needs the SHARP metadata table). The effect on skill scores is not known
in advance.

**Question.** Should the default for region-less images become `"drop"`
or `"error"`? For region-less flares: exclude with a warning and document
the limitation, or is matching by location required?

## Q5. Does an empty window mean "no flare"?

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

**Question.** Is coverage information available for this catalog (start,
end, known gaps)? Should the package flag or exclude windows outside
confirmed coverage?

## Q6. Image timestamps: cadence tolerance, TAI vs UTC

**Fact (tolerance).** The real-image examples (07, 08, 10, 11) use the
full-precision `DATE_OBS` from each JPEG 2000 header. Some images differ
from the exact hourly grid by 0.1 s, so the time-based sequence rule
(`start + i × cadence`, no tolerance) keeps 2 of 8 candidate sequences
and skips 6. Missing grid points are never substituted. The lab's own
image lists use filename times on exact hours, where this does not arise.

**Fact (TAI).** SHARP filenames carry TAI times; GOES catalog times are
UTC. TAI − UTC was 34–37 s in 2010–2018. Neither the package nor the
legacy scripts convert; this matters only for flares within seconds of a
window edge.

**Options.** Tolerance: (a) none (current); (b) a configurable
`cadence_tolerance`, with a documented default. TAI: (a) document only
(current); (b) a conversion option in `build_image_index_from_filenames`.

**Question.** Allow a configurable tolerance, and what value? Convert TAI
to UTC before matching?

## Q7. Flux below A1.0

**Fact.** The slides describe flux below A1.0 as flare-quiet. In the
package, `"NF"` means an empty event window (reduced flux exactly 0.0).
A catalog event such as `A0.5` converts to 5e-9 W/m², and
`MulticlassLabeler` (through `FluxConverter.to_class`) raises
`ValueError` for it: the GOES scale has no letter for that flux, and it
is a recorded flare, so it is not "no flare". Binary and regression
labels handle it without error. No such event occurs in the lab catalog.

**Options.** (a) keep the error (current); (b) label it `"NF"`; (c) a
separate quiet class; (d) `"A"`.

**Question.** Which behavior should other users get?
