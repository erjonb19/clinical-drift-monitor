# Backlog

The source of truth for what to work on and in what order (from 2026-11-06). Work the first
unchecked item. The owner edits this file as plans change. When an item is done it is ticked
here, gets one line in [PROGRESS.md](PROGRESS.md), and goes on the README's "Production
readiness" checklist.

Each item has a **Done when** line (the check), and a **Today** line: what already exists in
the repo on 2026-11-06, so no item is ticked for work that only partly exists. A blocked item
gets a **Blocked:** line with the reason.

## Items, in order

1. [ ] **Scheduled drift job.** A Databricks job scores drift on a schedule without anyone
   pressing anything.
   - **Done when (first version, replaced below):** one run started by the schedule (not by
     hand) finishes green, and its results file and job run ID are committed, with the run's
     trigger shown as scheduled.
   - **Today:** `databricks/monthly_drift_job.json` is scheduled at 06:00 UTC on the 1st, and
     emails the owner on success and failure. The 12-month replay ran once as a manual backfill.
     The first scheduled run was due on 1 November 2026; its outcome is not recorded in the
     repo. After the replay, scheduled runs only log the ISIC live count and score nothing
     new, so a green run alone would not show drift being scored.
   - **Found in the workspace (owner, 2026-11-06):** no run on 1 November. The job's only run
     ever is the manual backfill on 22 October 2026 at 10:02 PM (succeeded, 15m 14s). The
     schedule reads "At 06:00 AM, on day 1 of the month (UTC)" and shows a pause control, so it
     looks active. The job runs as the owner on serverless compute. No email arrived, which fits
     a run that never started.
   - **Cause: not found yet.** No run record at all means the scheduler did not start a run, not
     that a run failed. Candidates, most likely first:
     1. The schedule was paused, or the job's settings were changed, between the backfill and
        1 November. The job file changed twice after the backfill (81a1590 on 22 October,
        27 minutes after it, and a846b19 on 23 October); whether either was deployed is not
        recorded. Every committed version says `"pause_status": "UNPAUSED"`, and
        `scripts/deploy_databricks.py` applies the file with `jobs reset`, so a redeploy from
        the repo would unpause the job, not pause it. A pause would have come from a UI edit or
        another definition. Any of these is a job update in the audit log. A reply on the
        Databricks Community forum
        (community.databricks.com/t5/databricks-free-edition-help/jobs-suddenly-stopped-running-with-no-errors/td-p/169936)
        says the docs describe no case where Databricks pauses a schedule on its own.
     2. A run was skipped (for example the Free Edition quota, or a run already active). A
        skipped run is a real status and would be listed; the owner saw none, so this is less
        likely, but the run timeline lists any skipped run.
     3. The Free Edition scheduler did not fire, for no recorded reason. A Free Edition user
        reported the same in September 2026 (the thread above); no cause was confirmed.
   - **Needed from the workspace** (SQL editor; replace `<job_id>` with the job's ID from its
     page). Whether Free Edition exposes `system.access.audit` is not known; the repo has read
     `system.access.table_lineage` and `system.billing` before.
     1. Job changes: `SELECT event_time, action_name, user_identity.email, request_params
        FROM system.access.audit WHERE service_name = 'jobs' AND event_date BETWEEN
        '2026-10-22' AND '2026-11-06' AND request_params['job_id'] = '<job_id>' ORDER BY
        event_time`
     2. Schedule history: `SELECT change_time, paused, trigger FROM system.lakeflow.jobs
        WHERE job_id = '<job_id>' ORDER BY change_time`
     3. Every run the job has, including skipped ones (a run that was never created has no row
        at all): `SELECT run_id, period_start_time, trigger_type, result_state,
        termination_code FROM system.lakeflow.job_run_timeline WHERE job_id = '<job_id>' ORDER
        BY period_start_time DESC`

        Query 2 only covers changes made after the system table was enabled; column names in 2
        and 3 are from the docs and may need adjusting.
     4. The exact word on the schedule's button ("Pause" means active, "Resume" means paused),
        and the next run time if one is shown.
   - **Fix (planned, written before code):**
     1. Whatever the cause: a missed run must not be silent. A watchdog checks on the 2nd of
        each month that the job has a run started by the schedule since the 1st, and fails
        loudly if not. Proposed: a scheduled GitHub Actions workflow calling the Jobs API with
        a read-only credential (Can View on this job only). Behind the approval gate, a
        scheduled workflow would sit waiting (up to 30 days) until approved, which defeats a
        watchdog, so this credential would sit outside the gate. **Owner decision.** Databricks job
        notifications cover start, success, failure and
        duration, not a run that never starts, so the check has to live outside the job.
     2. A guard, not the likely cause (a redeploy from the repo unpauses):
        `scripts/deploy_databricks.py --job` reads the job back after `jobs reset`, refuses if
        the schedule is not `UNPAUSED`, and prints the next run time. A test checks that every
        committed job with a schedule says `UNPAUSED`.
     3. Then a scheduled run is shown green once (this item's "Done when").
   - **After replay month 12, written before code.** The owner's rules (2026-11-06): score
     whatever new ISIC images arrive; report drift signals that need no labels (input or
     embedding drift, prediction mix, confidence); if no new images arrive, record "no new data"
     as an explicit status, never a silent green run; add labelled metrics later, when labels
     exist. **Proposed by Claude, for the owner to confirm:**
     1. New images are public ISIC images whose IDs the job has not seen before, kept in a
        table of seen IDs. Whether ISIC records an upload date is still an open question in
        PLAN.md, so the job would not rely on one.
     2. They are scored with the model under the `champion` alias. Each image's license and
        attribution is recorded. Each signal is compared with the validation baseline: the OOD
        score (the existing KS and PSI test), the embedding drift, the prediction mix across the
        7 classes, and the confidence (top softmax probability). Each signal's test and
        threshold go in a spec committed before the first such run.
     3. The status is one of `scored` (with counts), `no_new_data` or `failed`.
     4. **Limit to check first:** Free Edition restricts outbound internet to a set of trusted
        domains. The job reached `api.isic-archive.com` for the count; whether it can download
        image files, which may be served from another host, is not known.
   - **Done when (updated):** one run started by the schedule (its `trigger_type` from the
     run timeline committed with it) finishes green and records `scored` or `no_new_data`,
     with its results file and run ID committed; and, if the owner approves the watchdog, it
     exists and has been seen to fail on a missing run.

2. [ ] **Model versioning and rollbacks** through the MLflow model registry, with the steps to
   roll back to the previous version.
   - **Done when:** the registry holds two versions, a committed script moves the `champion`
     alias to the newer one and back, a check after the rollback shows the served answers
     match the older version, and the rollback steps are in the runbook.
   - **Today:** `workspace.cdm.skin_lesion_classifier` has one version (v1, alias `champion`).
     With one version there is nothing to roll back to yet.

3. [ ] **Champion vs challenger.** The current model (v1) is the champion and v2 is the
   challenger. Score both on the same data, compare them, and promote the challenger only if it
   wins on the agreed targets (the ML version of a canary release).
   - **Done when:** a spec committed before the run names the data and the targets; one run
     scores both models on the same images; the decision file is committed; and the registry
     alias moves only if the challenger wins.
   - **Today:** a retraining loop with this rule exists and ran once (2026-10-23): the
     challenger gained 6.2 points at Buenos Aires but lost 1.6 on client validation (limit 1.0),
     so it was not promoted, and its weights were not kept. Registry "v1" is the model the
     earlier phases call "v2" (the 3-seed ensemble).
   - **Decided (owner, 2026-11-06):** train a new challenger on Databricks serverless GPU (a run
     the owner approves, with its GPU estimate, before it starts). Every challenger is
     registered in MLflow whether or not it is promoted; only the `champion` alias decides what
     serves. Every registry version carries a `model_gen` tag naming its model generation, so
     registry v1 is tagged `model_gen=v2`.

4. [ ] **SLOs as model targets**, such as a minimum melanoma sensitivity, written in the README,
   with an alert when drift breaks a target.
   - **Done when:** the README has an SLO table (each target, its value, why, and where it is
     measured); the monthly job checks each target it can measure; and a broken target raises an
     alert.
   - **Today:** CI floors exist for committed metrics (`config/eval_gate.json`), but no runtime
     SLOs. **Limit:** the job reports melanoma sensitivity only for a month with at least 100
     melanomas; the replay months had 5 to 64, so a monthly melanoma-sensitivity SLO can't be
     measured at 300 images a month without pooling months or changing the rule.
   - **Rule (owner, 2026-11-06), written before any code:** melanoma sensitivity is measured on
     a trailing window of labelled months. The window starts at the latest month whose labels
     have arrived and adds earlier months one at a time until it holds at least 100 melanoma
     images. Sensitivity is reported with a 95% Wilson score interval, with the months in the
     window and the melanoma count. If all available months together have fewer than 100
     melanomas, the month is marked "insufficient data" and the target is neither met nor
     broken. The target's value, and whether it is judged on the point estimate or the
     interval's lower bound, are set in the item's PR before any month is judged.

5. [ ] **Alerts** when drift, or equity drift by Fitzpatrick skin type, crosses a threshold,
   with the thresholds and why.
   - **Done when:** a threshold crossing reaches the owner (not only a flag in a table), and the
     README states each threshold and the reason for it.
   - **Today:** the job flags an alert (KS p < 0.01 and PSI >= 0.2) or a warning (PSI >= 0.1),
     and runs the same test per skin type with at least 100 images. The rules were fixed in
     `results/phase3/monthly/spec.json` before any month ran. The flags are written to
     `gold_drift_monthly`. The job's email goes out on every success or failure, not on an
     alert. **Limit:** skin type is recorded only at the two new hospitals, so a skin-type alert
     so far detects the site, not anything about skin type.

6. [ ] **Tests and model evals in CI** on every push, with pass and fail thresholds.
   - **Done when:** CI runs the tests and the eval gate on every push and pull request, the
     thresholds are listed in one place, and a self-test shows a regression fails the build.
   - **Today:** built for pushes to `main` and every pull request, not for pushes to other
     branches (`.github/workflows/ci.yml`, `docker.yml`): ruff, mypy, pytest, and the eval gate
     with a regression self-test, plus the Docker smoke test. **The eval gate does not run
     inference:** `tests/test_eval_gate.py`
     reads committed results files (`config/eval_gate.json` says so). The Docker smoke test
     runs inference, but on an untrained test bundle.
   - **Decided (owner, 2026-11-06):** add a CPU smoke test in CI that loads the champion and
     scores a fixed set of about 50 images, then mark the item done. This needs the champion
     bundle downloadable in CI (the release asset with a checked SHA-256, decided 2026-11-03)
     and a committed manifest of the images (ISIC IDs, license, attribution, SHA-256), fetched
     at test time and never committed.

7. [ ] **Observability on Databricks:** a prediction audit table, Unity Catalog lineage, and
   per-run cost tracking.
   - **Done when:** the audit table is written by every scoring run; lineage shows bronze to
     silver to gold to the registered model to the audit table, with a screenshot; and the cost
     table covers every run behind a committed result.
   - **Today:** the audit table is built. Lineage has no edge from the registered model to the
     audit table, and there is no screenshot. Costs are traced for 13 of 31 results files, with
     billing up to 2026-10-22.

8. [ ] **8D incident reports as postmortems:** run one real drift incident end to end and write
   it up.
   - **Done when:** one alert month goes through the whole path: report written, every number
     checked against the drift table, the report reviewed in the queue, and the runbook's
     containment steps done. The write-up is committed and says whether the drift was real,
     replayed or staged.
   - **Today:** 30 synthetic (staged) eval scenarios and a proposed eval spec are in open PR #50,
     with thresholds waiting for the owner. There is no report generator, groundedness check,
     review queue or runbook yet. The replay's alert months use real images in an order we
     chose, so they count as replayed, not as live drift.

9. [ ] **Model card and change-control plan:** a CHAI-format applied model card, and a one-page
   change-control plan mapped to FDA's Predetermined Change Control Plan guidance.
   - **Done when:** both are regenerated from committed evidence and cover the built serving,
     monitoring and retraining parts, and the change-control plan fits on one page.
   - **Today:** both are drafted and generated by `scripts/build_docs.py`. The change-control
     plan is 67 lines, longer than one page.

10. [ ] **Reviewer feedback section** in the README, to record what outside reviewers say and
    what changed.
    - **Done when:** the README has the section with a fixed format (date, reviewer role,
      comment, what changed, link), empty or with real entries only.
    - **Today:** not started.

## Later (kept, not started)

- Abstain and risk-coverage analysis (referral workload vs accuracy and melanoma sensitivity).
  *Note:* a validation risk-coverage curve and the abstain threshold were already built in
  Phase 3 (`results/phase3/serving/results.json`).
- A more prominent CMS readmission drift model. *Note:* deferred on 2026-10-23, because no
  monthly history of the CMS data exists yet.
- The research informatics extensions: OMOP and SNOMED CT mapping, a vision-language baseline,
  subgroup evaluation, interpretability and human review, multimodal fusion. *Note:* Phases A
  to E are built. F is a draft waiting for the owner's review. In G, G5 and G6 are ticked in
  PLAN.md but filed under Scaffolded in BUILT_VS_PLANNED, and G1 to G4 are open. See
  `docs/BUILT_VS_PLANNED.md` and PLAN.md.

## Open from PLAN.md, not placed (owner to decide)

These were open in PLAN.md and are not in the list above. They stay here until the owner adds
them to the list or drops them.

- Recompute the gold embeddings and baseline score distributions with the registered model (a
  Phase 2 item moved to Phase 3).
- Phase 3: CPU batch-level drift, staged shifts scored for time to detect and false-alarm rate,
  `diagnosis_confirm_type` in bronze and silver, and the free public API deploy (blocked by
  free hosting limits, 2026-10-23).
- Phase 4: the report generator, groundedness check, eval harness, human review queue and
  manual live-eval workflow (item 8 needs most of these).
- Phase 5: the one-command demo (`docker compose up`), the 8D runbook, the README refresh
  (results first), the standards-alignment document, the "structured after, not compliant
  with" check across the standards documents, and the optional demo recording.
- Research: Phase F (the cohort contract, a draft waiting for review) and G1 to G4.
