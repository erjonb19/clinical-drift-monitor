# Spec: an automated retraining pipeline (backlog item 17)

Written by Claude on 2026-11-09 for the owner to approve. Nothing is built or run. Any GPU run
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
3. **evaluate** (GPU or CPU): scores champion and challenger on the same held-out images
   (patient-disjoint new-site images and client validation) and applies the pre-committed rule
   (`results/phase3/retrain/spec.json`: gain of at least 2.0 points, paired lower bound above
   0, client validation change of at least -1.0 point), plus the model targets of item 4
   (`config/slo.json`) on the held-out set.
4. **register** (CPU): always registers the challenger in Unity Catalog with `model_gen`, the
   decision file's SHA-256 and the run ID as tags. It never moves the `champion` alias.
5. **decide** (CPU): writes `decision.json` and a row in `gold_model_decisions` with
   `promoted_by_rule` true or false, and ends the run.

**Promotion is a separate job**, `promote_challenger`, run by hand by the owner. Lakeflow jobs
have no approval step, so the human approval is the act of running this job with two inputs:
the registry version to promote and the decision file's SHA-256. It refuses unless that
decision says `promoted_by_rule: true` and its hash matches; then it moves `champion`, rebuilds
the serving bundle, and writes the old and new versions to `gold_model_decisions`, so the
rollback in item 2 can move the alias back.

## Done when (from BACKLOG item 17)

The job definition is committed and tested; one approved run goes from trigger to a registered
version and a decision file; the alias moves only after the owner runs `promote_challenger`.

## Tests before any run

- `trigger_check` logic on committed month records (the replay's months 9 to 12 would trigger
  only once their labels are recorded).
- The promote job refuses a wrong hash, a rule-rejected decision and a missing version.
- The job file: one GPU task with a deadline, no retries, `max_concurrent_runs` 1, unpaused
  only if the owner wants it on a schedule (otherwise triggered by the monthly job's alert).

## Cost (estimate, not a result)

The 2026-10-23 run used one A10 task under a 45-minute deadline (60-minute hard timeout);
at about 0.065 GPU DBU a minute that is about 3 to 4 GPU DBUs a run. **(ask)** before any.

## Open for the owner

- Should a scheduled monthly job start the pipeline on its own, or only the owner?
- Which challenger data: the new site's labelled images only, or those plus the clients'
  training images (today's choice)?
