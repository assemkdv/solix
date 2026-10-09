# Release-readiness verification (0.2.1, unreleased)

What was checked before any release, how, and what is still open. This is
**not** a declaration that the package is ready to publish: the scientific
decisions D1–D6 in [decisions.md](decisions.md), the below-A-class policy
of `MulticlassLabeler`, and the scope of a 1.0 release still need review.
Nothing was uploaded to TestPyPI or PyPI.

## Verified locally (macOS arm64, 2026-10-09)

Interpreters: CPython 3.12.0 and 3.13.12 (python.org builds), 3.14.3 for
the repository test suite. Build tools: `build` and `twine` from a fresh
virtual environment.

| Check | Command (summary) | Result |
|---|---|---|
| Build into a new directory | `python -m build --outdir <tmp>/dist .` | `flare_indexer-0.2.1-py3-none-any.whl`, `flare_indexer-0.2.1.tar.gz` |
| Metadata | `twine check <tmp>/dist/*` | both PASSED |
| Wheel contents | `python -m zipfile -l` | the 9 modules of `flare_indexer`, `LICENSE`, metadata with extras `dev`, `pytorch-examples`, `tensorflow-examples`; nothing else |
| sdist contents | `tar tzf` | source, tests and fixtures, docs, scripts, all 11 examples, `examples/data/`, `examples/real_data/` (manifest, catalog, raw HEK result, provenance JSON, README, scripts); **no** `examples/output/`, `.jp2`, caches, virtual environments or secrets |
| Wheel install | clean venv, `pip install <wheel>`, run from outside the repository with `PYTHONPATH` unset | imports from `site-packages` on 3.12 and 3.13 |
| Smoke test of the installed wheel | small hand-made catalog and index | binary, regression, multiclass (incl. `C10.0` = M1.0), active-region isolation, time-based sequences and the below-A `ValueError` as expected on 3.12 and 3.13 |
| sdist install | clean venv, `pip install <sdist>` (builds a wheel from the sdist) | installs; smoke test passes |
| Tests from the sdist | extracted sdist with `src/` moved away, against the installed package | 408 passed, 8 skipped (all skips need downloaded images or the private lab catalog) |
| `pytorch-examples` extra | clean 3.13 venv, `pip install "<wheel>[pytorch-examples]"` (torch 2.14.1, pillow 12.3.0, pandas 3.0.6) | examples 07 and 08, copied from the sdist and run outside the repository, pass all checks |
| `tensorflow-examples` extra | clean 3.13 venv, `pip install "<wheel>[tensorflow-examples]"` (tensorflow 2.21.0) | examples 10 and 11 pass all checks |
| Image replay without a cache | empty `images/`, `fetch_real_data.py` | all 9 images downloaded from the recorded URLs; every SHA-256 matched; the `missing` row was not downloaded; manifest byte-identical |
| CI test step, simulated | the `framework-examples` pytest command with each wheel-installed extra | pytorch 21 passed, tensorflow 7 passed, no skips; with images removed pytest still exits 0 (4 skipped), which the job's skip check turns into a failure |

Example results with the installed packages (identical to the repository
runs): images `[4,1,256,256]`, `[4,1,256,256]`, `[1,1,256,256]` (PyTorch)
and `[4,256,256,1]`, `[4,256,256,1]`, `[1,256,256,1]` (TensorFlow), 9 images
with 4 positive and 5 negative labels; sequences: 8 candidates, 2 emitted,
6 skipped, batch `[2,3,1,256,256]` / `[2,3,256,256,1]`, labels `[0, 0]`.

## Fixed during this verification

- The sdist shipped 13 generated files from `examples/output/`, because
  `MANIFEST.in` included every `*.csv` under `examples/`. `MANIFEST.in`
  now prunes `examples/output/` and `examples/real_data/images/` and
  excludes byte code, `*.part` files and `.DS_Store`.

## CI configuration (written, not yet run on GitHub)

`.github/workflows/test.yml`:

- `test`: unchanged base matrix, Python 3.10–3.13, editable install, full
  suite. Framework and real-image tests skip there.
- `build`: builds both artifacts, `twine check`, fails if the sdist
  contains example outputs, images, partial downloads or byte code or lacks
  the provenance files, and imports the installed wheel outside the
  checkout.
- `framework-examples` (new; matrix `pytorch`, `tensorflow`; Python 3.12):
  installs the **built wheel** with the extra (plus CPU-only PyTorch, so
  the TensorFlow job also compares against PyTorch), restores the image
  cache keyed on the manifest hash, replays the pinned images with
  SHA-256 verification (three attempts, never `--refresh`), checks that the
  manifest and catalog are unchanged, runs the two examples, and runs the
  real-image tests. Any skip other than the private lab-catalog test fails
  the job, so a missing dependency or failed download cannot pass as
  skipped.

This workflow has only been parsed and its shell steps reproduced
locally; it has not executed on GitHub Actions. The first GitHub run is
still needed to confirm the CPU PyTorch index, TensorFlow on the Ubuntu
runner, and network access to Helioviewer from the runner.

## Known limitations and open items

- **Release blockers (scientific):** D1–D6, the below-A policy, and the
  strategies in scope for 1.0. The version stays 0.2.1.
- **Network dependency:** the framework job depends on Helioviewer serving
  the recorded image ids with identical bytes. If the archive changes, the
  job fails on the hash check (by design).
- **macOS python.org interpreters** may fail HTTPS downloads with
  `CERTIFICATE_VERIFY_FAILED` until "Install Certificates.command" is run
  (or `SSL_CERT_FILE` points to a CA bundle such as certifi's). This
  happened here with 3.13; the script stopped without changing anything.
- **TensorFlow** has no wheels for Python 3.14; the extra needs 3.10–3.13.
- **Python 3.10 and 3.11** were not available locally; they are covered
  only by the CI matrix.
- **Build warning:** setuptools reports that the license classifier is
  deprecated in favor of an SPDX expression. `twine check` passes; the
  metadata was left unchanged.
- The repository has no `pyproject.toml`; builds use setuptools' legacy
  backend with an unpinned setuptools.
