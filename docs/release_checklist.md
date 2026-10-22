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

Not done yet: steps 2 to 6. No release or DOI exists.
