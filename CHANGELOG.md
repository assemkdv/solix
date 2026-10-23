# Changelog

## Unreleased

Not yet released; version in `setup.cfg` is still 0.2.1.

### Changed (prediction window, 2026-10-23)
- The prediction window is always `(t, t + w)`: a flare at exactly the
  image time `t` or exactly `t + w` does not count. With minute-resolution
  catalogs a 24-hour window is `t < start <= t + 23:59:59`, the SuryaBench
  rule. The `interval_mode` parameter is removed from
  `DatasetBuilder`, `EventMatcher.query` and `DatasetBuildingPipeline`,
  and `--interval-mode` from the command line; passing it is now a
  `TypeError`. Code that used either former mode gets the new window; a
  flare starting exactly at an image time no longer counts for that image.
- The default `event_time` is now `"start"` (was `"peak"`): a flare counts
  when its start time is in the window. Pass `event_time="peak"` for the
  old behavior.
- The command line's `--window` default stays `24h`.

### Fixed (input validation, review of 2026-10-17)
- A duration given as bare numeric text (`prediction_window="24"`,
  `cadence="3600"`) was read by pandas as nanoseconds and silently
  labeled everything 0. Such text is now rejected with a request for a
  unit (`"24h"`). A bare number keeps its documented meaning only where
  one exists: `prediction_window` reads a number as hours and
  `cadence_minutes` as minutes; the time-based `cadence`,
  `observation_window` and `sliding_window` (which pandas also read as
  nanoseconds) now reject a bare number and take duration text or a
  `Timedelta`.
- A catalog flare whose `start_time` (or `peak_time`) is blank can never
  match under `event_time="start"` (or `"peak"`). This was silent;
  `DatasetBuilder.build` now warns with the count for the selected
  column. The catalog still loads, as before.
- A numeric time column (`20240101100000`, epoch seconds) in the catalog
  or the image index was read as nanoseconds since 1970 and matched
  nothing. It is now rejected, naming file, column and data row.
- Region identifiers were normalized only when pandas inferred a numeric
  column, so text `"13700.0"` or `" 13700"` in a column that also held
  one text value never matched 13700. Equivalent spellings now match
  regardless of column dtype, the integer is taken from the text (no
  float rounding), whitespace is stripped, and non-numeric identifiers
  are kept. `EventMatcher.query(active_region=...)` applies the same
  normalization; a blank region is an error rather than full-disk matching.
- A blank `timestamp` cell in the image index crashed single-image mode
  with a `TypeError`, was reported as "not sorted" in row-based mode and
  was silently dropped in time-based mode with a report that showed no
  skip. It is now rejected before any sample is built, listing the rows.
- Time-zone policy, now one rule everywhere: a timestamp with an explicit
  UTC offset is converted to UTC and stored naive (offsets may differ
  within a column, e.g. across a daylight-saving change; `Z`, `+00:00`
  and `+05:00` spellings of one instant give one value); a naive
  timestamp is assumed to be on the catalog's scale; a column that mixes
  naive and offset cells is rejected, naming the first row of each kind.
  Catalogs with `Z`/`+00:00` times previously loaded and then failed on
  every `query`; offset-aware image times previously worked only through
  numpy with a warning per query. No TAI/UTC conversion is implied.
- A catalog with both a `class` and a `goes_class` column reported bogus
  invalid classes; it is now rejected with a message naming the columns.
- `BinaryThresholdLabeler.assign` compared whatever it was given: NaN
  gave 0, `True` gave 1, `Decimal` gave 0. It now accepts real numbers
  only (converted with `float()`), raising `TypeError` for booleans,
  `Decimal`, text or `None` and `ValueError` for NaN/infinite values, like
  `NumericThresholdLabeler`. A numpy `float32` value is compared after
  conversion to `float`, as `MulticlassLabeler` already did.
- `sequence_length`/`stride` given as floats (`2.0`, `2.5`) or booleans
  passed the constructor and failed inside `build`; `cadence_minutes`
  given as text failed with a bare `TypeError`. All three are validated
  in the constructor (`TypeError` for the wrong type, `ValueError` for a
  bad value).

### Added
- `flare-indexer` command (`flare_indexer.cli`, a console script registered
  in `setup.cfg`): `flare-indexer label IMAGES.csv CATALOG.csv --output
  OUT.csv` runs `DatasetBuilder` with the built-in labelings (binary,
  max-flux, multiclass, class-counts, class-presence, cumulative with a
  raw or thresholded output), both targets, the window settings and both
  sequence modes. Index columns are carried through; list and dict cells
  are JSON text; durations need a unit; an existing output needs
  `--overwrite`; warnings and errors go to stderr; a failed run leaves no
  partial file. Researcher manual §18. No new dependency.
- `CumulativeClassIndexReducer`: the cumulative normalized GOES subclass
  index Σ C + 10 Σ M + 100 Σ X of Hong, Ji, Pandey, Aydin (DaWaK 2023)
  and SuryaBench, summed in decimal arithmetic; A/B add 0 (selected
  convention, pending confirmation); empty window 0.0.
  `NumericThresholdLabeler(threshold)`: binary label from a plain numeric
  threshold, separate from GOES-class thresholds. Example:
  `examples/13_cumulative_index.py`; validation against the public
  SuryaBench labels: `scripts/validate_cumulative_index.py`,
  `docs/validation/cumulative_index.md`.
- Minimum event-flux reduction and all-event per-class counts/presence,
  composable through the existing pipeline.
- `prediction_window` accepts a pandas duration as well as hours
  (e.g. `"23h59min59s"`), in `DatasetBuilder` and `EventMatcher.query`.
- `DatasetBuilder(missing_active_region="negative" | "drop" | "error")`.
- `DatasetBuilder(include_metadata=True)` keeps image-index columns in
  single-image output (and `active_region` in AR mode).
- `UserWarning` when active-region labels are produced for images with no
  region, and when catalog flares have no region.
- `scripts/validate_full_disk.py --prediction-window --summary`.
- `scripts/validate_active_region.py`.
- `examples/` (six scripts at that point, synthetic data; later entries
  below bring the count to 13) and `docs/` (at that point: getting
  started, user guide, AR workflow, architecture diagrams, validation
  reports, open decisions). Later in this cycle the getting-started and
  user-guide pages were merged into `researcher_manual.md` and
  `api_reference.md`, and the diagram images and their generator were
  replaced by the Mermaid diagram in `docs/architecture/system_design.md`.
- CI builds the sdist and wheel and checks them with `twine check`.
- `MulticlassLabeler`: maps a reduced flux to `"NF"`, `"A"`, `"B"`, `"C"`,
  `"M"` or `"X"` by numeric class boundaries; use it as
  `DatasetBuildingPipeline(reducer=MaxFluxReducer(), labeler=MulticlassLabeler())`.
  The label is the class of the reducer's representative flux (the
  strongest flare), not the set or count of classes in the window.
  `0.0` is `"NF"`; a positive flux below A1.0 and negative, non-finite or
  non-numeric values raise errors. Boundaries are exact (no tolerance).
  Example: `examples/09_multiclass_labels.py`.
- `examples/07_pytorch_full_disk_images.py` and
  `examples/08_pytorch_full_disk_sequences.py`: real SDO/HMI images
  (Helioviewer) labeled with the HEK/SWPC GOES flare list and loaded with a
  PyTorch `Dataset`/`DataLoader`; download script and provenance in
  `examples/real_data/`. Image timestamps are the full-precision header
  `DATE_OBS`; the fetch script replays the recorded manifest (ids, URLs,
  SHA-256) by default and rediscovers images only with `--refresh`.
  Optional extra `pytorch-examples` (torch, pillow).
- `examples/10_tensorflow_full_disk_images.py` and
  `examples/11_tensorflow_full_disk_sequences.py`: the same real images,
  catalog, labels and checks loaded as `tf.data.Dataset` batches
  `[B, H, W, C]` and `[B, T, H, W, C]` (explicit channel-first →
  channel-last conversion). Optional extra `tensorflow-examples`
  (tensorflow, pillow; TensorFlow has no Python 3.14 wheels yet).

- Active-region sequences are built per region: the index is grouped by
  region first, so several regions may share timestamps.
  `SequenceBuildReport` gains `region_count` and `skipped_sequence_regions`.
- GOES classes are validated when a catalog is loaded; missing or
  malformed values raise `InvalidGOESClassError` (a `ValueError`, also a
  `KeyError`/`IndexError` for backward compatibility) naming the lines.
- `scripts/validate_active_region.py` reproduces the published arpatch-sfp
  labels from the four `-90_to_90` fold files at a pinned commit and
  compares both `label` and `goes_class`.

- CI job `framework-examples` (PyTorch and TensorFlow): installs the built
  wheel with the extra, replays the pinned example images with SHA-256
  checks, runs examples 07/08 or 10/11 and their tests, and fails on any
  unexpected skip. The `build` job also checks the sdist contents and
  imports the installed wheel. (This entry was written before the push;
  the job first ran on GitHub on 2026-10-09, run 37912188090, and has
  passed on every `main` run since.)
- Release-readiness checks of the built wheel and sdist, done at that
  point: clean-venv installs on 3.12 and 3.13, both extras, examples
  from the sdist. CI repeats only part of this: the `build` job installs
  the wheel in a clean venv on 3.12 and imports it, and the
  `framework-examples` job installs the wheel with each extra and runs
  the examples; the 3.13 clean-venv install and the run from the sdist
  are not in CI.

### Changed
- `FluxConverter.to_class(flux)` is the one place that turns a flux into a
  GOES class letter; `MulticlassLabeler` and `ClassCountReducer` call it.
  `reducers` no longer imports `labelers`, so the modules form strict
  layers (classifier → events/labelers → reducers → strategies/pipeline/
  builder), documented in `docs/architecture/system_design.md` and checked
  by `tests/test_module_dependencies.py`. Labels and error messages are
  unchanged; `FluxConverter.CLASS_LETTERS` is new.
- The sdist no longer includes generated `examples/output/` files or the
  downloaded image cache (`MANIFEST.in` prunes them).
- Images with no active region are still labeled 0 by default, but no
  longer silently: a warning reports how many.
- `FluxConverter.to_flux` multiplies in decimal arithmetic from the class
  text and rounds to `float` once, so equivalent classes give identical
  floats (`"C10.0"` and `"M1.0"` → `1e-05`; `"M10.0"` and `"X1.0"` →
  `0.0001`). Before, `"C10.0"` gave `9.999999999999999e-06` and was below
  an M1.0 threshold (`BinaryThresholdLabeler("M")` → 0, now 1). Other
  values can change in the last 1–2 bits (`"M9.9"`:
  `9.900000000000001e-05` → `9.9e-05`). On the cached 2010–2018 data no
  label changed (full-disk 24h and 23h59min59s, published AR labels); 99 of
  the 298 distinct catalog classes and 14,552 of 63,649 `max_flux` values
  (24h) differ by at most 2 ulp, and the order of all classes is unchanged.

### Removed
- The `date, start, peak, end, class, active_region` catalog schema with
  `HHMM` times. `EventMatcher` reads one schema: `start_time, peak_time,
  end_time, class, active_region`, every time with its full date. The old
  schema could not represent a flare that crosses midnight; convert such
  files to full timestamps before use. The package no longer references
  the `solar-event-scraper` repository.
- `data/solar_events.csv`, a sample catalog with undocumented provenance
  that was used only by an end-to-end smoke test. The test now uses
  `tests/fixtures/synthetic_same_day_catalog.csv` and
  `tests/fixtures/synthetic_full_timestamp_catalog.csv` (invented events,
  see `tests/fixtures/README.md`) and checks exact timestamps and labels.

### Compatibility notes (behavior that differs from v0.2.1 / main)
- Active-region sequences: an index whose region changes along the series
  no longer raises "mixed active_region values"; each region gets its own
  sequences. A row without a region no longer turns the surrounding
  sequence into a missing-region sequence; it is excluded from every
  region's sequences (with the default policy it forms its own group).
  `missing_active_region="error"` now raises for any such row in sequence
  modes, before any sequence is built.
- A catalog with a missing or malformed GOES class now fails when loaded,
  even if that flare never falls inside a window. A class with a zero
  multiplier (`M0`) is rejected; lower-case classes and surrounding
  spaces are accepted.
- An image-index column named like an output column (e.g. `label`) now
  raises `ValueError` in sequence modes and with `include_metadata=True`;
  it used to overwrite the computed label silently.
- `prediction_window=True`, `inf` or `nan` now raise `ValueError`
  (`True` was previously taken as 1 hour).
- Loading a catalog now checks that each row's available times satisfy
  `start <= peak <= end` (equal times allowed; blank cells are skipped)
  and raises `ValueError` naming the data row otherwise. Before, such
  rows loaded silently. The cached lab GOES catalog used by `scripts/validate_full_disk.py` and
  `scripts/validate_active_region.py` has 7 such rows (all C-class,
  2011–2017, peak recorded as 00:00 on the start date); the validation
  scripts therefore apply documented corrections to a copy of that
  catalog (`scripts/catalog_corrections.py`): six peaks move to the
  next day's 00:00, as NOAA's GOES X-ray reports record the maximum at
  `0000`; the 2011-04-11 C2.5 peak becomes missing, as NOAA records none.
  The downloaded file is unchanged, every record is matched on all its
  original values, every correction is printed and stored in the summary
  JSON, and `--skip-catalog-corrections` turns them off. Start-time
  results in `docs/validation/` are unchanged; the flare without a peak
  cannot be matched by peak time.

### Since v0.2.1 (already on main before this branch)
- Package renamed to `flare_indexer`; `FluxConverter` split from
  `FlareClassifier`; real-data loaders; configurable `event_time` and
  `interval_mode`; time-based sequences; modular pipeline.
