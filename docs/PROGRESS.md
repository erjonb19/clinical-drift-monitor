# Progress

One line per finished item from [BACKLOG.md](BACKLOG.md): what changed, and how to check it.
Nothing goes here until the item's "Done when" is met.

| Date | Item | What changed | How to check |
| --- | --- | --- | --- |
| 2026-11-11 | 37 | zizmor (pinned by hash) audits every workflow change and must first catch a planted template injection | The "Workflow audit" job on any PR that touches `.github/workflows/`; `pytest tests/test_supply_chain.py` |
| 2026-11-11 | 46 | The G5 CSV error sheets write formula-like text as text | `pytest tests/test_extract.py -k "spreadsheet or formulas or whitespace"` |
| 2026-11-11 | 45 | The $5 Haiku cap fails closed: a missing or shrunken spend log refuses calls once spending has begun | `pytest tests/test_extract.py -k spend_log` |
| 2026-11-11 | 44 | The API sends security headers, serves no OpenAPI schema, returns only its response models' fields, and logs refusals without an address or body | `pytest tests/test_api.py` |
| 2026-11-11 | 43 | Dependabot alerts, grouped security updates and malware alerts on (owner) | Settings, Code security (owner) |
| 2026-11-11 | 42 | `SECURITY.md` points to GitHub's private vulnerability reporting (on), with no email address | Read `SECURITY.md`; `pytest tests/test_security_md.py` |
| 2026-11-11 | 41 | `.env.example` lists every environment variable the code reads, with no values | `pytest tests/test_env_example.py` |
| 2026-11-11 | 39 | The image scan fails on fixable HIGH findings as well as CRITICAL; the image drops setuptools and applies Debian's security updates to pass | The `image` job's Trivy step on any PR; `pytest tests/test_supply_chain.py` |
| 2026-11-11 | 38 | Dependabot waits 7 days after a release (version updates only) | `pytest tests/test_supply_chain.py`; read `.github/dependabot.yml` |
| 2026-11-11 | 36 | ruff's security rules (`S`) on; asserts outside tests and notebooks became explicit errors | `ruff check .`; read `[tool.ruff.lint]` in `pyproject.toml` |
| 2026-11-11 | 35 | The Docker base image is pinned by digest; Dependabot updates it | Read the `FROM` line in `Dockerfile`; `pytest tests/test_supply_chain.py` |
| 2026-11-11 | 34 | Every `uses:` in the workflows is pinned to a commit SHA with its release in a comment | `grep -rn "uses:" .github/workflows/`; `pytest tests/test_supply_chain.py` |
| 2026-11-11 | 33 | CIFAR-10 batch files are unpickled only after their published MD5 matches | `pytest tests/test_build_benchmarks.py` |
| 2026-11-11 | 32 | The serving bundle refuses a manifest that leaves a checkpoint or detector unhashed, a path-like name or no seeds; the deploy script applies the same check | `pytest tests/test_serving.py` |
| 2026-11-11 | 31 | Every `torch.load` passes `weights_only=True`; torch floor 2.6 | `pytest tests/test_weights_only.py` |
| 2026-11-11 | 30 | Uploads are opened with the JPEG and PNG decoders only; other formats are named and refused before any decoder runs | `pytest tests/test_contract.py` |
| 2026-11-11 | 29 | Images over 50 megapixels are refused from their header, before decoding (silent failure #16) | `pytest tests/test_contract.py tests/test_api.py tests/test_serving.py` |
| 2026-11-11 | 28 | Every workflow's token is read-only by declaration, and no checkout keeps its credentials | `pytest tests/test_supply_chain.py`; read the top of each `.github/workflows/*.yml` |
| 2026-11-11 | 25 | gitleaks scans the whole history on every push and PR (pinned, checksummed, with a planted-key self-test); a pre-commit hook for the laptop | The `Secrets` workflow's run on any PR; `pytest tests/test_supply_chain.py` |
| 2026-11-11 | 24 | Branch protection on `main` (owner): `quality`, `image` and both CodeQL `analyze` checks required | Settings, Branches, `main` (owner) |
| 2026-11-11 | 11 | Supply chain and merge rules: Dependabot, CodeQL, pip-audit with a known-bad self-test, Trivy, and branch protection on `main` (owner) | Read `.github/dependabot.yml` and `.github/workflows/ci.yml`; `pytest tests/test_supply_chain.py` |
| 2026-11-11 | 9 | The change-control plan fits one page and covers serving, monitoring and retraining; the model card adds the input contract, targets and rollback; CI fails if a generated doc is stale or hand-edited | `python scripts/build_docs.py && git diff docs/` (no change); `pytest tests/test_build_docs.py` |
| 2026-11-11 | 40 | `.gitignore` covers the standard's secret and state files | `pytest tests/test_gitignore.py` |
| 2026-11-10 | 22 | Security audit against `docs/SECURITY_STANDARDS.md` (plan only): 0 Critical, 0 High, 5 Medium, 27 Low; items 23 to 47; a proposed `.claude/settings.json` | Read `docs/security-audit-2026-11-10.md` |
| 2026-11-09 | 10 | README "Reviewer feedback" section (date, reviewer role, comment, what changed, link), empty until a real review | Read the README section; `pytest tests/test_readme_feedback.py` |

## Open checks (not results)

| Date | Check | Status |
| --- | --- | --- |
| 2026-11-06 | Did the monthly drift job's first scheduled run (due 1 November 2026, 06:00 UTC) run, and was it green? | **No run.** The owner received no email (inbox, spam and promotions checked). Owner checked the job page on 2026-11-06: **no run on 1 November**; the only run is the manual backfill of 22 October. Cause (inferred from the owner's four workspace queries, 2026-11-06, not documented by Databricks): the schedule was active and correct, and the scheduler did not fire. Evidence under item 1 in BACKLOG.md. |
| 2026-11-07 | Does the Free Edition scheduler start runs at all? Scheduler probe `cdm-scheduler-probe` (job 325029901767432), due at 06:00 UTC on 8, 9 and 10 November 2026 | **Pending; fires not reported.** The 8 and 9 November fires' outcomes were not reported (the owner's answers, 2026-11-09, were left as placeholders both times, and again in the owner's message of 2026-11-10). Deployed by the owner on 2026-11-07. A manual test run (621176150087695) succeeded in 27 seconds with `trigger_type` `one_time`, and its email arrived; that tests the notebook and the email, not the scheduler. |
| 2026-11-07 | Can a Free Edition serverless job download ISIC image files? | **Yes, once.** The probe's manual run read `ISIC_6589782`'s metadata from `api.isic-archive.com` and downloaded the file (2,658,276 bytes) from `isic-archive.s3.amazonaws.com`. |

## LLM spend (the one exception to $0: Claude Haiku, capped at $5 in total)

| Date | Use | Spend | Cumulative |
| --- | --- | --- | --- |
| 2026-10-23 | G5 clinical text extraction: all 85 Haiku calls, dev runs and scoring (`results/research/g5_extraction/score.json`, `llm_calls`, `llm_spend_usd`) | $0.0265 | $0.0265 |
