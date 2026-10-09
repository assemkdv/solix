# Architecture

Editable diagrams: [`flare_indexer_architecture.drawio`](flare_indexer_architecture.drawio)
(open in [diagrams.net](https://app.diagrams.net) or the draw.io desktop app;
one tab per page). Vector PDF: [`flare_indexer_architecture.pdf`](flare_indexer_architecture.pdf).
Both are generated from [`build_diagrams.py`](build_diagrams.py); edit that
script rather than the outputs, then run
`python docs/architecture/build_diagrams.py` (needs `matplotlib`).

| Page | Shows |
|---|---|
| 1 Overview | Inputs, loaders, `DatasetBuilder`, `EventMatcher`, strategy chain, outputs |
| 2 build() flow | Validation steps, mode dispatch, per-region grouping, per-reference-time loop |
| 3 Time windows | `[t, t+w)`, `(t, t+w]`, `(t, t+23:59:59]`, and the AR look-back window |
| 4 Sequences | Row-based vs. time-based construction, skipped candidates, per-region grids |
| 5 Active regions | Region normalization and grouping, missing-region policy, HARP/NOAA mapping, real-data facts |
| 6 Modules and tests | Module dependencies, scripts, examples, tests |

## Modules

| Module | Responsibility | Public names |
|---|---|---|
| `classifier.py` | GOES class parsing/validation, class → flux (W/m²) | `FluxConverter`, `FlareClassifier`, `InvalidGOESClassError` |
| `events.py` | Load/validate catalog (two schemas, GOES classes), window query | `EventMatcher`, `FlareEvent` |
| `reducers.py` | Flares → one value | `EventReducer` (protocol), `MaxFluxReducer` |
| `labelers.py` | Value → label | `LabelAssigner` (protocol), `BinaryThresholdLabeler`, `RegressionLabeler`, `MulticlassLabeler` |
| `strategies.py` | Ready-made reducer + labeler pairs | `BinaryThresholdStrategy`, `MaxFlareStrategy` |
| `pipeline.py` | Stage-by-stage, inspectable labeling | `DatasetBuildingPipeline`, `PipelineResult` |
| `builder.py` | Image index → labeled table; sequences (grouped by region in AR mode); missing-region policy | `DatasetBuilder`, `SequenceBuildReport` |
| `loaders.py` | Real-world input conversion | `build_image_index_from_filenames`, `adapt_goes_catalog`, `ImageIndexReport` |

## Design notes

- **One query per reference time.** `EventMatcher` keeps a sorted copy of
  the catalog per `event_time` and uses binary search (`numpy.searchsorted`)
  for the window, so the full-disk validation (63,649 images, two
  strategies) runs in about 30 s on a laptop.
- **Strategies are duck-typed.** Anything with `label(flares)` works as a
  `strategy`; anything with `reduce(events)` / `assign(value)` works in a
  `DatasetBuildingPipeline`.
- **No silent fallbacks.** A missing active region never falls back to
  full-disk matching; incomplete time-based sequences are skipped and
  reported, never interpolated.
- **Pure functions of the inputs.** No network access, no global state;
  the same CSVs and parameters always give the same labels.

## Known limitations

- Windows only look forward from the reference time.
- No time-scale conversion (TAI vs UTC).
- No catalog-coverage check: a catalog does not state its coverage, and
  the package does not infer it from recorded flares.
- In AR sequence modes with the default `missing_active_region="negative"`,
  several unknown-region patches at the same time raise `ValueError`
  (use `"drop"`).
