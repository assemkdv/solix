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

### Changed
- Images with no active region are still labeled 0 by default, but no
  longer silently: a warning reports how many.

### Since v0.2.1 (already on main before this branch)
- Package renamed to `flare_indexer`; `FluxConverter` split from
  `FlareClassifier`; real-data loaders; configurable `event_time` and
  `interval_mode`; time-based sequences; modular pipeline.
