# Changelog

## Unreleased

Not yet released; version in `setup.cfg` is still 0.2.1.

### Added
- Minimum event-flux reduction and all-event per-class counts/presence,
  composable through the existing pipeline; cumulative formula remains open.
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
  imports the installed wheel. Not yet run on GitHub.
- `docs/release_verification.md`: what was verified for the built
  artifacts and what remains open.

### Changed
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
