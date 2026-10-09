# Architecture

Overview of the package components, labeling choices, and limitations.
For the data flow and a worked example, see [system_design.md](system_design.md).

## Modules

| Module | Responsibility | Public names |
|---|---|---|
| `classifier.py` | Validate GOES classes and convert strength to flux, e.g. M2.0 → `2e-5 W/m²` | `FluxConverter`, `FlareClassifier`, `InvalidGOESClassError` |
| `events.py` | Read the catalog, check classes and timestamp order, find flares in a time window | `EventMatcher`, `FlareEvent` |
| `reducers.py` | Select maximum/minimum event flux or count events by class | `EventReducer` (protocol), `MaxFluxReducer`, `MinFluxReducer`, `ClassCountReducer` |
| `labelers.py` | Turn the summary into binary, numerical, multiclass, count, or presence labels | `LabelAssigner` (protocol), `BinaryThresholdLabeler`, `RegressionLabeler`, `MulticlassLabeler`, `ClassCountLabeler`, `ClassPresenceLabeler` |
| `strategies.py` | Ready-made maximum-flux combinations for binary or numerical labels | `BinaryThresholdStrategy`, `MaxFlareStrategy` |
| `pipeline.py` | Combine a reducer and labeler; inspect events, summary, and final label | `DatasetBuildingPipeline`, `PipelineResult` |
| `builder.py` | Build a labeled table for images or sequences; group AR sequences by region; handle missing regions | `DatasetBuilder`, `SequenceBuildReport` |
| `loaders.py` | Extract image timestamps from supported filenames and adapt lab GOES catalogs | `build_image_index_from_filenames`, `adapt_goes_catalog`, `ImageIndexReport` |
| `__init__.py` | Export the public tools for imports from `flare_indexer` | Names listed above |

## Labeling choices

- **Binary:** `BinaryThresholdStrategy("M")` — 1 if an M-or-stronger event occurs.
- **Maximum flux:** `MaxFlareStrategy()` — strongest event's flux.
- **Minimum flux:** `MinFluxReducer` + `RegressionLabeler` — weakest event's flux.
- **Multiclass:** `MaxFluxReducer` + `MulticlassLabeler` — strongest event's class.
- **Counts:** `ClassCountReducer` + `ClassCountLabeler` — event count per class.
- **Presence:** `ClassCountReducer` + `ClassPresenceLabeler` — 0/1 per class.

The last four use `DatasetBuildingPipeline(reducer=..., labeler=...)` as
`DatasetBuilder`'s strategy. Counts and presence use all matched events.

## Design notes

- **Efficient event matching.** `EventMatcher` caches sorted start/peak times
  and uses `numpy.searchsorted` to find window boundaries.
- **Replaceable components.** Any object with `label(flares)` can be a
  strategy; `reduce(events)` and `assign(value)` define pipeline components.
- **Explicit missing-data handling.** Missing regions never become full-disk
  queries. Incomplete time-based sequences are skipped and reported,
  never interpolated.
- **Timestamp checks.** Available times must satisfy `start <= peak <= end`.
  Midnight-crossing events require full timestamps with explicit dates.
- **Repeatable labeling.** Core labeling uses local inputs; with deterministic
  components, the same files and settings produce the same labels.
- **Separate research corrections.** Both validation scripts use
  `scripts/catalog_corrections.py` for seven documented lab-catalog peak-time
  corrections. The original catalog and start-time results are unchanged.

## Known limitations

- Cumulative indexing is not implemented; its formula is still undefined.
- Core windows only look forward. AR look-back is a validation-script comparison.
- No time-scale conversion (TAI vs UTC).
- No catalog-coverage check: an empty window does not prove observations were available.
- Time-based sequences require exact cadence; no timestamp tolerance.
- Multiclass and counts reject positive flux below A1.0; the policy is under review.
- In AR sequence modes with `missing_active_region="negative"`, multiple
  unknown-region patches at the same time raise `ValueError` (use `"drop"`).
