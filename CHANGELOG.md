# Changelog

## Unreleased

Not yet released; version in `setup.cfg` is still 0.2.1.

### Added
- `prediction_window` accepts a pandas duration as well as hours
  (e.g. `"23h59min59s"`), in `DatasetBuilder` and `EventMatcher.query`.
- `DatasetBuilder(missing_active_region="negative" | "drop" | "error")`.
- `DatasetBuilder(include_metadata=True)` keeps image-index columns in
  single-image output (and `active_region` in AR mode).
- `UserWarning` when active-region labels are produced for images with no
  region, and when catalog flares have no region.
- `scripts/validate_full_disk.py --prediction-window --summary`.
- `scripts/validate_active_region.py`.
- `examples/` (six scripts, synthetic data), `docs/` (getting started,
  user guide, AR workflow, architecture diagrams, validation reports,
  open decisions).
- CI builds the sdist and wheel and checks them with `twine check`.

- Active-region sequences are built per region: the index is grouped by
  region first, so several regions may share timestamps.
  `SequenceBuildReport` gains `region_count` and `skipped_sequence_regions`.
- GOES classes are validated when a catalog is loaded; missing or
  malformed values raise `InvalidGOESClassError` (a `ValueError`, also a
  `KeyError`/`IndexError` for backward compatibility) naming the lines.
- `scripts/validate_active_region.py` reproduces the published arpatch-sfp
  labels from the four `-90_to_90` fold files at a pinned commit and
  compares both `label` and `goes_class`.

### Changed
- Images with no active region are still labeled 0 by default, but no
  longer silently: a warning reports how many.

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

### Since v0.2.1 (already on main before this branch)
- Package renamed to `flare_indexer`; `FluxConverter` split from
  `FlareClassifier`; real-data loaders; configurable `event_time` and
  `interval_mode`; time-based sequences; modular pipeline.
