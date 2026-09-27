# Built vs planned

What is built, what is scaffolded, and what has not been started. This file is the
reference for what may be claimed about the project: nothing is listed as built until it
runs and is reproducible from the repository.

**This is a personal portfolio project.** No patients, clinicians or business decisions
depend on it.

Last updated: 2026-09-27.

## Built

Nothing yet.

## Scaffolded

- HAM10000 download with checksum verification, metadata validation, and a lesion-level
  split (`src/cdm/data.py`). The split is tested for leakage on synthetic data in CI and
  on the real metadata when it is present locally: 5,229 / 1,120 / 1,121 lesions in
  train / val / test. Not yet used by a training run.
- Package layout, lint, type check and CI (`pyproject.toml`, `.github/workflows/ci.yml`).

## Not started

- Phase 0 to Phase 6 as listed in `PLAN.md`. The course notebook in `reference/` is
  history, not a starting point: its numbers are superseded once Phase 0 reruns them.
