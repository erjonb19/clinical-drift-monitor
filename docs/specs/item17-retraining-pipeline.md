# Spec: an automated retraining pipeline (backlog item 17)

Written by Claude on 2026-11-09; the separate `promote_challenger` job was approved by the
owner the same day. Nothing is built or run. Any GPU run
needs the owner's approval of its estimate first.

## Today

The retraining loop ran once by hand (2026-10-23, `databricks/notebooks/retrain.py`, run
624636213399909): the challenger was not promoted, and its weights were not kept. The champion
check (silent-failures #12) and the decision rule (`results/phase3/retrain/spec.json`) exist.
Item 3 decided that every challenger is registered, whether or not it is promoted, with a
`model_gen` tag.

## The pipeline: one Lakeflow job, five tasks

1. **trigger_check** (CPU): reads `gold_drift_monthly`. It continues only when a month is in
   alert **and** that month's labels have arrived (its delayed accuracy is recorded), so the
   challenger can train on labelled new-site images. Otherwise it ends the run green with the
   status `no_trigger`, written to `gold_model_decisions`.
2. **train_challenger** (GPU, one A10 task, the deadline from `retrain.py`): fine-tunes a
   challenger per champion seed, as `retrain.py` does today, after the champion check passes.
   **A change from today:** it always saves the challenger's checkpoints to the volume (today
   they are saved only on promotion), so every challenger can be registered.
3. **evaluate** (GPU or CPU): scores champion and challenger on the same held-out images
   (patient-disjoint new-site images and client validation) and applies the pre-committed rule
   (`results/phase3/retrain/spec.json`: gain of at least 2.0 points, paired lower bound above
   0, client validation change of at least -1.0 point), plus the model targets of item 4
   (`config/slo.json`, built 2026-11-09) on the held-out set.
4. **decide** (CPU): writes `decision.json` and a row in `gold_model_decisions` with
   `promoted_by_rule` true or false.
5. **register** (CPU): always registers the challenger in Unity Catalog with `model_gen`, the
   SHA-256 of the decision file just written and the run ID as tags, then ends the run. It
   never moves the `champion` alias.

**Change to item 17's wording (approved by the owner, 2026-11-09; BACKLOG item 17 is updated):**
the item asked for one job that "stops for a human approval". This spec uses a separate job, `promote_challenger`, run by
hand by the owner, because this repo knows of no approval step inside a Lakeflow job on Free
Edition (not checked against the docs). The human approval is the act of running it with two
inputs:
the registry version to promote and the decision file's SHA-256. It refuses unless that
decision says `promoted_by_rule: true` and its hash matches; then it moves `champion`, rebuilds
the serving bundle, and writes the old and new versions to `gold_model_decisions`, so the
rollback in item 2 can move the alias back.

## Done when (from BACKLOG item 17)

The job definition is committed and tested; one approved run goes from trigger to a registered
version and a decision file; the alias moves only after the owner runs `promote_challenger`.

## Changes made while building the promotion half (2026-11-09, for the owner to re-approve)

Found when `promote_challenger` was built (`cdm.promotion`); nothing has run:

1. **The decision key is `promoted`**, the key `retrain.py` already writes
   (`results/phase3/retrain/decision.json`), not `promoted_by_rule` as written above. The
   future `decide` task should keep writing `promoted`; if it wrote `promoted_by_rule`, every
   promotion would be refused (safe, but it would need this check changed).
2. **The serving bundle is built and checked before registration, not by `promote_challenger`.**
   The registration step tags the version with `decision_sha256` and
   `serving_bundle_sha256`; `promote_challenger` refuses a version without both, so the alias
   can never point at a version whose bundle was not checked. After the move it records the
   `check_registered_model` run as pending, as the rollback steps do (`docs/rollback.md`).
3. **The trigger on the committed replay is months 7 to 11**, not "9 to 12" as written below:
   months 7 and 8 alerted on skin type II, 9 to 12 on the overall test, and month 12's labels
   arrive only after the replay.

## Tests before any run

- `trigger_check` logic on committed month records (the replay's months 9 to 12 would trigger
  only once their labels are recorded).
- The promote job refuses a wrong hash, a rule-rejected decision and a missing version.
- The job file: one GPU task with a deadline, no retries, `max_concurrent_runs` 1, unpaused
  only if the owner wants it on a schedule (otherwise triggered by the monthly job's alert).

## Cost (estimate, not a result)

The 2026-10-23 run used one A10 task under a 45-minute deadline (60-minute hard timeout);
at about 0.065 GPU DBU a minute that is about 3 to 4 GPU DBUs a run, before start-up time and
any GPU time for the evaluate task. **(ask)** before any.

## Open for the owner

- Should a scheduled monthly job start the pipeline on its own, or only the owner?
- Which challenger data: the new site's labelled images only, or those plus the clients'
  training images (today's choice)?
