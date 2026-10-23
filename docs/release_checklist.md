# Release checklist (Zenodo)

A release archives the **code** on Zenodo with a DOI. No data goes with it: every source keeps
its own license and is cited in [DATA_CARD.md](DATA_CARD.md).

Owner steps (they need the owner's GitHub and Zenodo logins):
1. Check `LICENSE` (MIT) is on `main`.
2. Sign in to zenodo.org with GitHub, open **GitHub** in the account menu, and switch on
   `erjonb19/clinical-drift-monitor`.
3. Check CI is green on `main` and `results/mlflow_runs.json` matches the committed results
   (`pytest tests/test_evidence.py`).
4. Update `version` and `date-released` in `CITATION.cff`, and the README status line.
5. Create a GitHub release with a tag such as `v0.1.0`. Zenodo archives it and mints a DOI,
   reading `.zenodo.json` for the title, description, creators and license.
6. Add the DOI to `CITATION.cff` (`doi:`) and a badge to the README, in a normal PR.

Done for v1.0.0 on 2026-10-23: the owner switched on the Zenodo integration and approved the
release; `v1.0.0` was created on `main` at cf2ebcd, and Zenodo minted DOI
10.5281/zenodo.23600426 (concept DOI 10.5281/zenodo.23600425). Step 4 was done after the
release, in the same PR as step 6, so the archived `CITATION.cff` has no version or DOI.
