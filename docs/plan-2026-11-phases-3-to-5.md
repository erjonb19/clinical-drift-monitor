# Plan: finishing Phases 3 to 5 from a cloud session

> This file is kept as the proposal the owner answered. The answers to section 7 (2026-11-03)
> are recorded in PLAN.md's decisions log by the PR that follows this one; that entry governs.

Nov 3, 2026 · drafted by Claude Code for Erjon Brucaj's review

This is a proposal. Nothing in it is built, and nothing in it changes PLAN.md until the owner
approves it. Phase 6 stays deferred (decisions log, 2026-10-23) and is not covered here.

Earlier work ran on the owner's laptop, which had a Databricks CLI login. This session runs in a
cloud container with the repository and GitHub access only: **no Databricks credentials, no
LLM keys, and no data or model weights.** Each item below says which of these it needs.

Key: **Cloud** = can be done here (code, tests, docs, CI) with no Databricks and no paid
calls. **Databricks** = needs workspace access (what access is stated). **Decision** = needs
the owner to choose before any work.

## 1. What was checked before writing this

- `main` is at c86f8d9 (PR #46, G5 score, merged 2026-10-23).
- **`phase3/retraining` is already on main in substance.** Its one unmerged commit (70961ba,
  "Re-review fixes ...") was applied to main as 684d7ea with the same message;
  `git cherry origin/main origin/phase3/retraining` marks it as an equivalent patch. It no
  longer merges cleanly only because main later added silent-failures #14. No other remote
  branch has commits that are not on main.
- The monthly job's first scheduled run was due on 1 November at 06:00 UTC. **Whether it ran,
  and whether it was green, is unknown from here** (needs Databricks read access).
- Billing in the cost table goes up to 2026-10-22 23:00 UTC: 34.3 GPU and 26.7 CPU DBUs
  ($31.62 at list price; Free Edition charges nothing). The retraining, serving-bundle,
  monthly-backfill and later runs ran after that cutoff, so **their DBUs are not yet counted**
  and the real total is higher than 34.3 GPU DBUs.

## 2. Every open item in Phases 3, 4 and 5

### Phase 3 (serving and MLOps)

| # | Item (PLAN.md) | State today | Needs |
|---|---|---|---|
| 3.1 | Close out `phase3/retraining` | Its fixes are on main (684d7ea) | **Decision:** OK to delete the branch (nothing to merge) |
| 3.2 | Silent-failures #12 (open): champion validation score 68.45% vs committed 68.26% | Not caught by code | **Cloud:** make the retraining notebook record the champion weight hashes and models-file SHA-256 in `decision.json`, and refuse when the champion's validation score is off the committed one by more than a set tolerance; tests. No rerun proposed |
| 3.3 | Retraining: bundle rebuild and registry move on promotion | Written path saves checkpoints only; rebuild and alias move not built | **Cloud** for code and tests; **Databricks** only if a promotion ever happens (none pending) |
| 3.4 | Docker image deployed free on Render | Blocked 2026-10-23 (HF 402; Render 512 MiB vs 442 MiB peak; owner declined to pay) | **Decision:** keep blocked, or amend Phase 3's "Done when" (see 3.13) |
| 3.5 | Monthly job: one green scheduled run | Due 1 Nov; outcome unknown | **Databricks** (read-only): run status and the run's results file |
| 3.6 | Gram backup detector in the monthly job | Not started; a CPU refit missed the GPU normalizer by up to 11.8% in Phase 2 | **Decision** (spend is likely to end in "not usable on CPU"), then **Databricks** CPU job |
| 3.7 | Test the actual serving setup (GPU-fit detector on CPU features) | Not started | **Decision:** the shipped bundle fits each seed's Mahalanobis detector on CPU features (`databricks/notebooks/serving_bundle.py`), so the setup this item tests is not the one served. Propose closing it as superseded, with a decisions-log entry; or run it (**Databricks** CPU) |
| 3.8 | CPU batch-level drift (spec `results/phase2/batch_drift/spec.json`) | Not started | **Databricks** CPU job. Cheap if the serving-bundle run saved per-image CPU scores in the volume; unknown from here |
| 3.9 | Per-run cost tracking (ticked when a refresh covers the 18 untraced results) | Partly built | **Databricks:** SELECT on `system.billing.usage`, `system.billing.list_prices`, `system.lakeflow.*`, plus a SQL warehouse or a CPU job |
| 3.10 | Unity Catalog lineage end to end | Gap: no model-to-audit edge; no screenshot | **Cloud** to make the monthly job load the model via `models:/...@champion`; **Databricks** CPU run that scores; **owner** takes the screenshot in the UI |
| 3.11 | `diagnosis_confirm_type` in bronze and silver | Not started | **Cloud** for schema, gates and tests; **Databricks** pipeline update with a bronze metadata refresh (writes production tables, so write-audit-publish gates apply) |
| 3.12 | Staged shifts scored for time to detect and false-alarm rate | Not started | **Cloud** for the spec (committed before any run, labelled *staged*); **Databricks** CPU job to score blurred and brightened images. The age-mix shift resamples existing scores and may need no new features |
| 3.13 | Phase 3 "Done when": live API URL, one green scheduled run, CI blocks a regression | CI gate done; URL blocked; scheduled run unconfirmed | **Decision:** amend to "API built and smoke-tested in Docker CI; public deploy blocked (decisions log 2026-10-23)", or leave Phase 3 open |

### Phase 4 (LLM layer)

| # | Item | Needs |
|---|---|---|
| 4.0 | Which LLM writes the report | **Decision**, see section 4 |
| 4.1 | 8D report generator from a drift-table row (problem, containment, root cause, permanent fix, verification, prevention) | **Cloud**, with recorded responses |
| 4.2 | Groundedness check: every number in the report matches the drift table | **Cloud**. Read governed-clinical-agent's check for the pattern first (needs that repo added to this session, read-only), then write fresh code |
| 4.3 | Eval set: 20 to 30 synthetic drift scenarios with known findings | **Cloud**. Every scenario labelled *synthetic (staged)*; pass thresholds committed before any live call (EVAL_PLAN amendment) |
| 4.4 | Eval harness: correctness, groundedness, cost and latency per call; provider outages counted apart from accuracy drops | **Cloud**, replayed in CI |
| 4.5 | Human review queue: a report counts as sent only after review | **Cloud**: a status file (draft, approved, rejected, sent) and a CLI; a test that an unreviewed report cannot be marked sent. **Decision:** reviewer is the owner |
| 4.6 | The live eval run that produces the recordings | **Owner** runs the manual workflow (section 4), **last** |
| 4.7 | "When the drift job flags": calling the report from the Databricks job | **Decision.** Proposed: the report runs from the committed drift rows in a GitHub workflow, not inside the Databricks job, so no LLM key goes into Databricks. Wiring it into the job later would need a Databricks secret scope |

Phase 4 "Done when": the eval passes a threshold set in advance and every report traces back
to drift-table rows. Both are reachable from the cloud, apart from the live run in 4.6.

### Phase 5 (packaging)

| # | Item | Needs |
|---|---|---|
| 5.1 | One-command demo (`docker compose up`) with a sample batch that trips the alarm | **Decision** on weights and the sample batch (below); then **Cloud** for compose, scripts and a CI smoke test with the untrained test bundle |
| 5.2 | Runbook in 8D steps | **Cloud** (after Phase 4, so it can point at the real report and queue) |
| 5.3 | Model card (CHAI v0.2), data card | Drafted (`docs/model-card.md`, `docs/DATA_CARD.md`). **Cloud**: refresh through `scripts/build_docs.py` for Phases 3 to 4, then tick |
| 5.4 | Change-control plan (FDA PCCP structure) | Drafted. **Cloud**: refresh against the built retraining rule, then tick |
| 5.5 | Standards alignment (GMLP, TRIPOD+AI, IMDRF) | Drafted, with stale gaps (for example "Security: gap until serving exists"). **Cloud**: refresh, then tick |
| 5.6 | "Structured after, not compliant with" and "not for clinical use" in every standards doc | Present in all three. **Cloud**: a test that keeps it there, then tick |
| 5.7 | README in the reckoner style, results first | **Cloud**, near the end, after the numbers it reports exist |
| 5.8 | Optional 3-minute demo recording | **Owner**, last |

**Decision for 5.1.** The demo needs the real serving bundle, which is not in git and must
not be. The owner approved public weights on 2026-10-23, but no place to download them from was
chosen. Options:
(a) attach the bundle to a GitHub release (free; the bundle comes out of the Databricks volume,
    via a read of the volume or the owner's own download), and the demo verifies its SHA-256;
(b) the demo runs on the untrained CI test bundle, labelled as such everywhere. Honest, but it
    cannot show a real alarm;
(c) the demo replays committed drift rows only, with no model in the loop.
**Recommended: (a).** The sample batch is not committed either: a committed manifest lists
ISIC image IDs, license, attribution and SHA-256 (reckoner pattern), and the demo downloads
them at start-up. Proposed batch: about 50 PAD-UFES-20 images (licenses recorded per image),
which alerted in replay months 10 to 12.

## 3. Proposed order (one PR per item)

Each PR: a short plan first, a fresh review before merge, CI green, docs and
`BUILT_VS_PLANNED.md` updated in the same PR, and nothing called built until it runs.

**A. No Databricks and no spend (start here)**

1. This plan (docs only).
2. Stale docs on main (section 6), with the PLAN amendments the owner approves (3.7, 3.13).
3. Close out retraining: delete `phase3/retraining` (3.1, owner's OK) and the #12 guard (3.2).
4. Phase 4, one PR each: 4.3 scenario set and thresholds (first, so the answers are fixed
   before the generator exists) → 4.1 generator on recorded responses → 4.2 groundedness check
   → 4.4 eval harness and CI replay → 4.5 review queue → the manual live-eval workflow,
   merged without being run.
5. Phase 5: 5.3 to 5.6 docs refresh → 5.2 runbook → 5.1 demo (after the weights decision).
6. Recordings last: 4.6 live eval (owner runs it), then 5.7 README refresh with those numbers,
   then 5.8 demo recording.

**B. Databricks (after access is given, each run approved one at a time)**

1. 3.5 read the 1 November run (read-only, no compute).
2. 3.9 cost refresh.
3. 3.8 CPU batch drift, 3.12 staged shifts (spec committed first), 3.11 confirm-type ingest,
   3.10 lineage run.
4. 3.6 Gram backup and 3.7 serving-setup check only if the owner keeps them.

## 4. The LLM layer: CI and live calls

- **CI never calls an LLM and needs no key.** The generator takes a transport, as
  `cdm.extract` already does: CI uses a recorded transport that replays committed responses,
  keyed by prompt hash, model and scenario. A missing recording fails the test; it never falls
  through to a live call.
- **Live calls only through a manual workflow** (`workflow_dispatch`, never on push or pull
  request, so no secret reaches a fork). It runs the eval with a hard cap (calls per run, and
  dollars where the provider charges), writes the spend log and the new recordings as a
  workflow artifact, and the owner commits them in a normal PR. It should run in a GitHub
  Environment with the owner as required reviewer, so every run needs a click from the owner.
- **Decision 4.0, which provider:**
  - **Gemini free tier, as PLAN.md's stack says.** Secret: `GEMINI_API_KEY`. No dollar spend,
    but the key must come from a Google project **with no billing account attached** (CLAUDE.md:
    no GCP billing). Free-tier prompts may be used by Google to improve its products; only
    synthetic scenarios and aggregate drift numbers would be sent, no images and no patient
    data. Free-tier rate limits and terms should be checked on the day; not verified here.
  - **Claude Haiku**, reusing `cdm.extract`'s transport, spend log and $5 cap. Secret:
    `ANTHROPIC_API_KEY`. CLAUDE.md allows Haiku "for G5 clinical text extraction and evals", which can be read
    as G5 evals only; writing the reports is outside it either way, so CLAUDE.md and PLAN.md would need amending. Rough size, using the price
    recorded in `src/cdm/extract.py` ($0.10 in, $0.50 out per million tokens, not re-checked):
    30 scenarios × about 3k input and 2k output tokens is about $0.04 per eval pass. G5 has
    spent $0.0265 of the $5.
  - **Recommended: Gemini free tier**, because it is what PLAN.md says and adds no spend. The
    transport interface keeps a switch to Haiku a one-file change if the owner prefers it.

## 5. Databricks: credentials and DBU estimates

**What a workflow or this session would need.** Workspace URL (`DATABRICKS_HOST`) and a token.
The Databricks CLI and SDK read both from environment variables.

**Safest way to give it (recommended):**
1. Prefer a **service principal** with OAuth machine-to-machine credentials
   (`DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET`), granted only: CAN_MANAGE_RUN on the
   named `cdm-*` jobs, READ VOLUME on `workspace.cdm.raw`, SELECT on `workspace.cdm` and the
   `system.billing` and `system.lakeflow` tables. **Not verified:** whether Free Edition lets
   you create service principals and grant system-table access to them.
2. If not, a **personal access token with a short lifetime** (for example 7 days), created only
   for this work and revoked after. A PAT carries all of the owner's rights, so this is the
   weaker option.
3. Store it as an **environment secret** (a GitHub Environment named, say, `databricks`, with
   the owner as required reviewer), used only by `workflow_dispatch` workflows that run one named
   job and download its results file. Not a plain repo secret, and never on `pull_request`.
4. **Not in this session.** A token in this container's environment would be readable by
   everything that runs here. Results come back as workflow artifacts and land through a PR.
   (Whether this container can even reach the workspace URL depends on the environment's
   network policy; not checked.)

**DBU estimates.** These are estimates, not results. The basis is the committed cost table
(billing to 2026-10-22): GPU jobs ran at about 0.065 DBU per wall-clock minute (cdm-phase2-v2,
5.24 DBUs in 80.8 minutes; cdm-final-seeds, 2.08 in 30.5), CPU jobs at about 0.02 to 0.06
(batch-drift reference 3.31 in 169 minutes; ONNX 0.68 in 12.1), and a pipeline update at about
0.46 (5.55 DBUs over 12 updates). Runs after the cutoff, such as the serving bundle's CPU
feature extraction, are not costed yet, which makes the CPU feature-extraction figures the
least certain.

| Item | GPU DBUs | CPU DBUs | Why |
|---|---|---|---|
| 3.5 read the 1 Nov run | 0 | 0 | Reads run status and a file; the run itself is already billed |
| 3.9 cost refresh | 0 | 0.1 to 0.5 | A few queries on system tables |
| 3.8 CPU batch drift | 0 | 0.2 to 0.5 if CPU scores are saved; 1 to 3 if features must be recomputed | NumPy sampling vs CPU feature extraction |
| 3.12 staged shifts | 0 | 2 to 6 | CPU features for shifted copies of the test images, 3 seeds; age mix about 0 |
| 3.11 confirm-type ingest | 0 | 0.5 to 2 | One or two pipeline updates with a metadata refresh |
| 3.10 lineage run | 0 | 0.3 to 1 | One monthly-job run that loads the model through the registry |
| 3.6 Gram backup (if kept) | 0 | 2 to 6 | CPU refit needs intermediate-layer features for the training split |
| 3.7 serving-setup check (if kept) | 0 | 0.5 to 2 | CPU features for validation and test |
| Retraining rerun | not proposed | | About 3 GPU DBUs at the 45-minute widget deadline, which is checked only between seeds; up to about 3.9 at the job's 60-minute `timeout_seconds` |
| **Total if all run** | **0** | **about 6 to 21** | No GPU work is proposed |

Free Edition limits daily compute; if a job hits the limit it stops until the next day.
No run starts without the owner's approval for that run.

## 6. Stale on main

- **STATUS.md** (2026-10-20) calls itself superseded, lists only Phase 1 and 2 jobs, and
  points at laptop paths. Proposed: cut it to a short pointer to the README status and
  `docs/BUILT_VS_PLANNED.md`, or delete it (owner's choice).
- **README** says the first scheduled run "is due on the 1st of next month" and that the
  monthly job's new write path "has not run yet"; both are now out of date in tense, and the
  outcome is unknown until 3.5. The README status line is dated 2026-10-23.
- **`docs/report/report.tex`** says "twelve bugs" in the silent-failures list; it has 14.
- **PLAN.md** "What each lane gets" says the model was "served through ONNX Runtime"; ONNX was
  closed as attempted and serving is PyTorch on CPU.
- **PLAN.md** open questions from week 1 (ISIC S3 read, Asset Bundles, upload dates, start
  date) are still unticked; some look answered by Phases 1 and 2. To be checked against the
  phase notes, not assumed.
- **`docs/standards-alignment.md`** says "Security: gap until serving exists"; the API is
  built and its privacy and rate-limit checks run in CI (still not deployed).
- **Phase 5 checkboxes** for the model card, change-control plan and standards alignment are
  unticked although drafts exist (5.3 to 5.6).
- **Costs** in the README and `BUILT_VS_PLANNED.md` stop at 2026-10-22 billing (3.9).
- **27 feature branches** remain on origin, including `phase3/retraining`; all but that one are fully merged, and its one commit is on main as an equivalent patch. Deleting
  them is the owner's call.

## 7. Decisions needed from the owner

1. Delete `phase3/retraining` (and the other merged branches)?
2. Phase 3 "Done when" with the API deploy blocked (3.13)?
3. Close 3.7 as superseded, and keep or drop 3.6 Gram backup?
4. LLM provider for the reports (4.0), and the secret to add.
5. Reports from committed drift rows in a GitHub workflow, not inside the Databricks job (4.7)?
6. Where public weights are downloaded from for the demo (5.1)?
7. Databricks access: service principal or short PAT, as an Environment secret (section 5)?
8. STATUS.md: shorten or delete?
