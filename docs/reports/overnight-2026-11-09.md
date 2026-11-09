# Overnight run, 2026-11-09

Repo-side work only, as asked. No Databricks deploy or workspace change, no GPU run, no Haiku
spend, no account or service principal, no release or image pushed, no branch-protection change,
no other repo touched, and the scheduler probe was left alone. Every PR below got a fresh
review, and its findings were fixed before merge; CI (`ci` and `docker`) was green on the
merged commit.

## Merged

| PR | Backlog item | What |
|---|---|---|
| #56 | order | Your "Order of work" and items 11 to 21 (read and merged as asked). |
| #57 | 1 | Deploy guard: `deploy_databricks.py --job` reads the job back and refuses a missing, changed or paused schedule. |
| #58 | 1 | Watchdog workflow (07:17 UTC on the 2nd) for job 524433148303929; fails "not configured" until you add the three secrets. |
| #59 | 1 | Label-free drift signals for months after the replay (`cdm.post_replay`) and their spec, committed before any run. |
| #60 | 6 | CI and Docker on every push to every branch; the champion smoke test, skipping until the bundle asset and image manifest exist. |
| #61 | 11 | Dependabot, CodeQL, pip-audit and Trivy. **The first Trivy run found five fixable CRITICAL Pillow 12.3.0 CVEs in the API image** (one in JPEG decoding); Pillow is now 13.0.0. |
| #62 | 4, 5 | Model targets (`config/slo.json`), the pooled melanoma window with a Wilson interval, and the alert path (a failed run with an "ALERT" message); checked on the committed replay; README thresholds table. |
| #63 | 18 | Input contract in the API (422 with the reason; blur and exposure report-only). |
| #64 | 13 | Serving benchmark on the untrained test bundle (serving speed only). |
| #65 | 15, 17 | Specs only: detector quality on staged shifts; the retraining pipeline. |

No backlog item was ticked: each still waits on a deploy, an asset or a decision below.
CI on `main`: green at 415506a (after #64: `quality`, `image`, CodeQL); the run for the final
merge, 5648df6 (#65, docs only), was still running when this was written.

## Open, and why

- **#50** (item 8 eval scenarios): waits for your pass thresholds, as you said. Not touched.
- **#66, #67, #68**: Dependabot's first grouped PRs, opened after #61 merged (item 11's "first
  Dependabot PR appears" is met). Left for you, not merged: they are not mine, and **#66 moves
  the API image from Python 3.11 to 3.15**, against the repo's Python 3.11 standard. A
  Dependabot `ignore` rule for Python minor and major versions in the Docker group would stop
  that; not added without your OK. #67 updates three Actions; #68 updates 33 Python packages.

## Waits on you

1. **Deploys** (laptop): the monthly job now needs a redeploy with the new notebook, wheel and
   `config/slo.json` staged (the job pins the notebook's hash), so the targets, alerts and the
   referral count run. The deploy guard runs on that deploy.
2. **Service principal and three repo secrets** for the watchdog (`DATABRICKS_HOST`,
   `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET`), after the view-only checks you planned.
3. **Champion bundle as a release asset** (URL and SHA-256 into `config/champion_smoke.json`), and
   running `scripts/make_smoke_manifest.py` where the ISIC API is reachable: the smoke test
   (item 6) and the champion's benchmark (item 13) wait on these.
4. **Branch protection** on `main`: require `quality`, `image` and the CodeQL `analyze` checks.
5. **Dependabot PRs #66 to #68:** review and merge or close; #66 (Python 3.15) is the risky one.
6. **Decisions:**
   - the targets' values (melanoma referral sensitivity 0.90 on the point estimate, delayed
     balanced accuracy 0.55, abstain share 0.15) and the post-replay signal thresholds;
   - the input contract's thresholds (blur and exposure stay report-only until calibrated);
   - the accepted `cryptography` and `uv` findings, both held back by `flwr` 1.39.0, or a
     `flwr` upgrade (it changes the stack behind the federated results);
   - item 15's run (about 2.5 CPU hours, an estimate) and item 17's design (a separate,
     hand-run promotion job is a proposed change to the item's wording);
   - the melanoma target cannot be judged until replay months 1 to 12 are backfilled from the
     audit table (a Databricks run; the code is not written).
7. **The scheduler probe:** your 8 November answer was a placeholder, so that fire is recorded
   as not reported. A scheduled task reminds you at 08:00 UTC on 10 November to send the run
   timeline and probe files and to delete the probe job.

## Found along the way

- The new supply-chain checks found real issues on their first run: Pillow's CVEs (fixed), and
  two dependencies pinned old by `flwr` (accepted with reasons and a review date of 2026-12-09).
- The reviews caught, before merge: a chi-square crash on a class the baseline never predicted;
  a melanoma target that could never be judged as first written; an alert that would have
  stopped a multi-month backfill; torch silently skipped by pip-audit; a decompression bomb
  turning into a 500; and a benchmark reading that wrongly said three networks.
- The benchmark's 18 ms median per request is about 5 times faster than Phase 3's PyTorch
  timing on Databricks; the gap is disclosed, not explained.

## How to check each item

**Item 1 (deploy guard, watchdog, post-replay code)**
1. Open `scripts/deploy_databricks.py` and find `check_deployed_schedule`.
2. Run `pytest tests/test_deploy.py tests/test_watchdog.py tests/test_post_replay.py`.
3. In GitHub, open Actions → "Drift watchdog" → Run workflow with month `2026-11`: without
   secrets it fails as inconclusive ("watchdog not configured"); once the secrets exist it
   should fail "MISSED", since no run started on 1 November (BACKLOG item 1).

**Item 6 (CI on every push, smoke test)**
1. Push any branch: the CI and Docker workflows start.
2. Run `pytest -rs tests/test_champion_smoke.py`: one test passes, one skips and names the two
   missing inputs.

**Item 11 (supply chain)**
1. Open the Docker run of #61's last commit: the Trivy step lists no CRITICAL finding.
2. Open the CI run: "The audit catches a known-vulnerable pin" passes, naming `requests 2.19.0`.
3. Read `config/pip_audit_ignore.json` for the accepted findings and their reasons.

**Items 4 and 5 (targets and alerts)**
1. Read the README section "Alerts and model targets".
2. Run `python scripts/slo_report.py`: months 1 to 6 list no alert; 7 and 8 list skin type II;
   9 to 12 list broken targets. It rewrites `results/phase3/monthly/slo_replay.json`; `git diff`
   should then show no change.
3. Run `pytest tests/test_slo.py`.

**Item 18 (input contract)**
1. Run `pytest tests/test_contract.py tests/test_api.py`.
2. Read `CONTRACT` in `src/cdm/contract.py` for every threshold and its reason.

**Item 13 (benchmark)**
1. Read `results/phase3/serving/benchmark.json`, including `annotations_by_hand`.
2. Optionally run Actions → "Serving benchmark" for a GitHub-runner measurement.

**Items 15 and 17 (specs)**
1. Read `results/phase3/detector_quality/spec.json` and `docs/specs/item17-retraining-pipeline.md`.
