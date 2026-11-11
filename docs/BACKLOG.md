# Backlog

The source of truth for what to work on and in what order (from 2026-11-06). Work the first
unchecked item **in the "Order of work" below** (from 2026-11-08; item numbers never change,
the order is set there). The owner edits this file as plans change. When an item is done it is ticked
here, gets one line in [PROGRESS.md](PROGRESS.md), and goes on the README's "Production
readiness" checklist.

Each item has a **Done when** line (the check), and a **Today** line: what already exists in
the repo on 2026-11-06, so no item is ticked for work that only partly exists. A blocked item
gets a **Blocked:** line with the reason.

## Order of work (owner, 2026-11-08)

**First (owner, 2026-11-09): item 22, the security audit** (plan only), after any PR already in progress. The items it adds are placed by the owner when he merges it.

1. **Item 1**, the parts that need no workspace access first: the deploy guard (fix 2), the
   watchdog workflow (written and tested, with no credential stored until the owner creates
   the service principal), and the post-month-12 scoring code and its spec. The probe's
   scheduled fires (8 to 10 November) and every Databricks deploy are the owner's.
2. **Item 6**: CI on every push to every branch. The CPU smoke test waits on the champion
   bundle as a release asset (the owner uploads it); build the test so it skips with a clear
   message until the asset exists.
3. **Item 11** (supply chain and merge rules).
4. **Items 4 and 5** (SLOs and alerts): the window rule, the targets and the alert path in
   code, tested on committed results; the live alert is checked on the next scheduled run.
5. **Items 18 and 13** (an input contract for images; a load test and serving benchmark).
6. **Items 2 and 3** (registry versions, rollback, champion vs challenger). The challenger's GPU
   run waits for the owner's approval of its estimate.
7. **Item 7**, then **items 15, 17, 16, 14** (detector quality, an automated retraining
   pipeline, shadow scoring, batch scoring at scale).
8. **Item 8** once the owner sets PR #50's thresholds, then **items 9 and 10**.
9. **Item 12** (a live endpoint) and **item 19** (least-privilege access) when the owner has
   made their accounts and decisions; **item 20** after the governed clinical agent's item 22.
10. **Item 21** (showcase) last.

**Rules for items 11 to 21 (owner, 2026-11-08):**
- **(ask)** before: a GPU run, a new account or hosted service, anything that could bill,
  Haiku spend, publishing a release or image, a change to another repo, and any deploy to the
  Databricks workspace (the owner deploys from the laptop; cloud sessions have no workspace
  access by design, decided 2026-11-03).
- **Numbers are measured and labelled:** what ran, on what hardware, with which model (a
  test bundle's latency is serving speed, never accuracy). Staged drift stays labelled
  staged, as the honesty rules say.
- **Unattended runs:** when an item is blocked or needs an (ask), write the reason under it and
  move to the next item that does not depend on it, instead of stopping.
- **Merging (owner, 2026-11-11, replacing 2026-11-10's rule):** Claude merges PRs itself,
  including ones marked "owner merges", once CI is green on the final commit and a fresh review
  by a separate agent that did not write the change finds nothing blocking. The owner is asked
  first, in chat, for spending, GPU runs, release tags, Databricks deploys, deleting data and any
  permission-settings change (those PRs are opened and left). PR #50 stays with the owner (his
  threshold decisions).

## Items, in order

1. [ ] **Scheduled drift job.** A Databricks job scores drift on a schedule without anyone
   pressing anything.
   - **Done when (first version, replaced below):** one run started by the schedule (not by
     hand) finishes green, and its results file and job run ID are committed, with the run's
     trigger shown as scheduled.
   - **Today:** `databricks/monthly_drift_job.json` is scheduled at 06:00 UTC on the 1st, and
     emails the owner on success and failure. The 12-month replay ran once as a manual backfill.
     The first scheduled run was due on 1 November 2026; it did not start (below). After the replay, scheduled runs only log the ISIC live count and score nothing
     new, so a green run alone would not show drift being scored.
   - **Found in the workspace (owner, 2026-11-06):** no run on 1 November. The job's only run
     ever is the manual backfill on 22 October 2026 at 10:02 PM (succeeded, 15m 14s). The
     schedule reads "At 06:00 AM, on day 1 of the month (UTC)" and shows a pause control, so it
     looks active. The job runs as the owner on serverless compute. No email arrived, which fits
     a run that never started.
   - **Cause (inferred, not documented by Databricks), 2026-11-06:** the schedule was active
     and correct on 1 November, and the Free Edition scheduler did not start a run. No pause
     and no bad edit was found. Evidence, from four queries the owner ran in the workspace:
     1. `system.access.audit`, jobs events for this job, 22 October to 6 November: three
        events, all by the owner on 23 October: `runNow` at 02:02:13 UTC (the backfill), then
        `reset` at 02:29:20 and 16:21:02 UTC. No pause event.
     2. `system.lakeflow.jobs`: two rows, both at 2026-10-23 02:02:05 UTC, `paused = false`,
        quartz `0 0 6 1 * ?`. **Anomaly:** no rows for the two later resets, which query 1
        shows; why this table missed them is not known.
     3. `system.lakeflow.job_run_timeline`: one run only (ID starting 8308376346509, the
        backfill, 2026-10-23 02:02 UTC, trigger `ONETIME`, `SUCCEEDED`, termination code
        `SUCCESS`). Nothing on
        1 November, not even a skipped run.
     4. The schedule each reset set (`get_json_object(request_params['new_settings'],
        '$.schedule')` on the query 1 rows): both set `0 0 6 1 * ?`, `UTC`, `UNPAUSED`.

     Earlier candidates ruled out: a pause or edit (queries 1, 2 and 4) and a skipped run
     (query 3). A Free Edition user reported schedules that stopped firing with no record in
     September 2026 on the Databricks Community forum
     (community.databricks.com/t5/databricks-free-edition-help/jobs-suddenly-stopped-running-with-no-errors/td-p/169936);
     no cause was confirmed there either. Notification address (owner, 2026-11-07): the monthly
     job (524433148303929) and the probe both email the owner's login address, an inbox the
     owner watches; it does not change the cause, since no run means no email was due.
   - **Scheduler probe (owner OK, 2026-11-06; temporary):** `databricks/scheduler_probe_job.json`
     and `databricks/notebooks/scheduler_probe.py`, a separate job with the same kind of
     schedule (`0 0 6 8,9,10 * ?`, UTC), firing at 06:00 UTC on 8, 9 and 10 November 2026. Each
     run writes one JSON file to `/Volumes/workspace/cdm/raw/results/scheduler_probe/` (run ID,
     trigger type and time) and tries the one-image ISIC download, recorded without failing the
     run. It touches no table. Written and tested in CI; deployed by the owner (notebook import,
     then `scripts/deploy_databricks.py --job`); to be deleted after 10 November. The production
     job is not changed.
     - **Deployed by the owner, 2026-11-07:** job ID 325029901767432; notebook at
       `/Users/<owner>/cdm/scheduler_probe`. The deploy needed the repo's virtual environment
       (`.venv/Scripts/python.exe` on Windows); plain `python` failed with "No module named
       'cdm'". `databricks jobs get` showed the schedule `0 0 6 8,9,10 * ?`, UTC, `UNPAUSED`,
       the notebook path above, and success and failure emails to the owner's login address.
     - **Manual test run, 2026-11-07:** run 621176150087695, `TERMINATED` / `SUCCESS`, 27 seconds;
       its file recorded `run_utc` 2026-11-07T13:57:24+00:00 and `trigger_type` `one_time`. The
       success email arrived (started 2026-11-07 13:57:17 UTC, "Launched: Manually"). This tests
       the notebook and the email, not the scheduler: the scheduled fires are on 8 to 10 November.
     - **Scheduled fires, 8 and 9 November: not reported.** The owner's answers for both days
       (2026-11-09) arrived as unfilled placeholders ("[scheduled email arrived / no email]"), so
       neither fire is recorded either way. To settle it: the run timeline for job
       325029901767432 and the files in `/Volumes/workspace/cdm/raw/results/scheduler_probe/`.
       The owner's message of 2026-11-10 again held the unfilled placeholders for both days, so
       they stay not reported.
   - **Fix (planned, written before code):**
     1. Whatever the cause: a missed run must not be silent. A watchdog checks on the 2nd of
        each month that the job has a run started by the schedule since the 1st, and fails
        loudly if not. Proposed: a scheduled GitHub Actions workflow calling the Jobs API with
        a read-only credential (Can View on this job only). Behind the approval gate, a
        scheduled workflow would sit waiting (up to 30 days) until approved, which defeats a
        watchdog, so this credential would sit outside the gate. **Owner decision (2026-11-06):**
        OK, only with a credential that can view this one job and nothing else; if Free Edition
        allows only a personal access token, stop and tell the owner before storing anything.
        **Checked (owner, 2026-11-07):** "Add new" exists under Settings, Identity and access,
        Service principals, so a service principal can be created; nothing is created or stored
        yet.
        Databricks job notifications cover start, success, failure and duration, not a run
        that never starts, so the check has to live outside the job. **Built 2026-11-09, not
        configured:** `scripts/watchdog.py` and `.github/workflows/drift-watchdog.yml` (07:17 UTC
        on the 2nd, and by hand for any month; never on push or pull request). It asks the Jobs
        API for a run of job 524433148303929 with trigger `PERIODIC` since 06:00 UTC on the 1st,
        and exits 0 (ran; state printed), 1 (missed) or 2 (inconclusive: no credential or API
        error), never passing without an answer. It needs three repo secrets from the owner's
        view-only service principal: `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`,
        `DATABRICKS_CLIENT_SECRET`. Until they exist every run fails as "not configured".
        It reads every page of runs, counts a `SKIPPED` run as missed, and never counts a later
        month's run. The secret must allow the `all-apis` scope. GitHub disables scheduled
        workflows after 60 days without repository activity, which would silence it too.
        Tested with fake API replies; not yet run against the workspace.
     2. A guard, not the likely cause (a redeploy from the repo unpauses):
        `scripts/deploy_databricks.py --job` reads the job back after `jobs reset` or `jobs
        create`, refuses if the schedule is missing, different from the committed one or not
        `UNPAUSED`, and prints the schedule. A test checks that every committed job with a
        schedule says `UNPAUSED`. **Built 2026-11-09** (`check_deployed_schedule`, tested in
        `tests/test_deploy.py` with fake CLI replies); not yet used in a real deploy. The
        next-run time is not printed: `jobs get` does not return one.
     3. Then a scheduled run is shown green once (this item's "Done when").
   - **After replay month 12, written before code.** The owner's rules (2026-11-06): score
     whatever new ISIC images arrive; report drift signals that need no labels (input or
     embedding drift, prediction mix, confidence); if no new images arrive, record "no new data"
     as an explicit status, never a silent green run; add labelled metrics later, when labels
     exist. **Proposed by Claude, confirmed by the owner on 2026-11-06:**
     1. New images are public ISIC images whose IDs the job has not seen before, kept in a
        table of seen IDs. Whether ISIC records an upload date is still an open question in
        PLAN.md, so the job would not rely on one.
     2. They are scored with the model under the `champion` alias. Each image's license and
        attribution is recorded. Each signal is compared with the validation baseline: the OOD
        score (the existing KS and PSI test), the embedding drift, the prediction mix across the
        7 classes, and the confidence (top softmax probability). Each signal's test and
        threshold go in a spec committed before the first such run.
     3. The status is one of `scored` (with counts), `no_new_data` or `failed`.
     - **Built 2026-11-09 (code and spec, not run):** `cdm.post_replay` (new-ID diff; OOD,
       embedding, prediction-mix and confidence signals against the validation baseline; a
       100-image minimum; the explicit status) and `results/phase3/monthly/post_replay_spec.json`
       (each test and threshold, chosen by Claude before any such run, accepted by the       owner on 2026-11-09).
       Tested on synthetic data in `tests/test_post_replay.py`. Not yet wired into
       `databricks/notebooks/monthly_drift.py`, and no month has been scored with it.
       **Owner decision, 2026-11-09:** the signal thresholds are accepted as specced.
     4. **Limit checked (2026-11-07, the probe's manual run):** Free Edition restricts outbound
        internet to a set of trusted domains, but a serverless job reached both hosts it needs.
        For `ISIC_6589782` it read the metadata from `api.isic-archive.com` (license CC-BY,
        attribution Memorial Sloan Kettering Cancer Center) and downloaded the file, 2,658,276
        bytes, from `isic-archive.s3.amazonaws.com`. One image on one day; the SHA-256 was not
        captured (cut off in the owner's terminal).
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
   - **Built 2026-11-09 (not used):** `cdm.registry` and `scripts/move_champion.py` move the
     `champion` alias (dry run unless `--apply`), refusing if the alias is not where the owner
     expects, or if the target is missing, has no `model_gen` tag or was `rejected` by the
     promotion rule; a promotion must name its decision file's SHA-256. Each applied move is read
     back and logged to `results/phase3/registry/alias_moves.jsonl`. The steps, and the check
     after a rollback (the existing `check_registered_model` job against the older version's
     bundle), are in `docs/rollback.md`. Tested with a fake registry. **Waits on:** a second
     registry version (item 3's challenger, a GPU run the owner approves), and tagging v1
     `model_gen=v2` (the owner, step 2 of `docs/rollback.md`).

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
   - **(ask), 2026-11-09:** the challenger needs one Databricks serverless GPU run, which waits
     for the owner's approval. Estimate (not a result): the 2026-10-23 retraining run used one
     A10 task under a 45-minute deadline, about 3 to 4 GPU DBUs, before start-up time. The
     decision rule is already committed (`results/phase3/retrain/spec.json`). **Proposed:** make
     this run the first run of item 17's pipeline once it is built, so one GPU run registers the
     challenger (item 3), gives item 2 its second version to roll back from, and tests item 17;
     item 16's shadow scoring then uses the same challenger. Nothing started.
   - **Owner decision, 2026-11-10:** approved as the first run of item 17's pipeline; the owner
     starts it from the laptop. **It cannot start yet:** the pipeline's training tasks (train with
     checkpoints kept, evaluate, decide, build and check the serving bundle, register with tags)
     are not built (item 17). **GPU estimate (from committed records, not a result; replaces the 2026-11-09
     figure of 3 to 4 DBUs, which assumed the whole 45-minute deadline was used):** the
     2026-10-23 retraining notebook ran 366 seconds on one A10 (`results/phase3/retrain/
     decision.json`); its wall time with start-up is not in the cost table (billing was traced
     to 2026-10-22). 12 of the 16 traced GPU jobs used 0.065 to 0.076 DBU per wall minute
     (`results/phase3/costs/job_costs.json`; all 16 range from 0.028 to 0.090), so a 10- to
     25-minute GPU task is about 0.7 to 1.9 GPU DBUs, and the 60-minute hard timeout caps it
     near 4.6 (5.4 at the highest rate seen). Free Edition charges nothing; at the recorded
     list price (about $0.65 per DBU) that is $0.45 to $1.25, capped near $3.00 ($3.50 at the
     highest rate). Two start-up failures on 2026-10-23 (900 seconds each) may also count;
     unknown. The 10 to 25 minutes assumes the 6-minute notebook plus start-up and the
     pipeline's extra tasks; that is a guess, not a measurement.

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
   - **Built 2026-11-09 (code, not run):** `cdm.slo` (the window rule above, the Wilson
     interval, three targets in `config/slo.json`: melanoma referral sensitivity 0.90, judged on
     the point estimate; delayed balanced accuracy 0.55, from 100 labelled images; abstain share
     0.15), `delayed_accuracy` now counts melanomas referred, and the monthly notebook checks the
     targets after saving each month. Checked on the committed replay
     (`results/phase3/monthly/slo_replay.json`); the README has the table. **Waits on:** a
     redeploy (the values were accepted on 2026-11-09, below) (the job pins the notebook's hash
     and needs `config/slo.json` staged). **Melanoma sensitivity cannot be judged yet:** the
     replay records (months 1 to 12) have no referral count, and after the replay no labelled
     months arrive, so it stays "insufficient data" until months 1 to 12 are backfilled from
     `gold_prediction_audit.refer_melanoma` and their labels (a Databricks run for the owner;
     the backfill code is not written).
   - **Owner decision, 2026-11-09:** the three values are accepted (melanoma referral
     sensitivity 0.90 on the point estimate, delayed balanced accuracy 0.55, abstain share
     0.15) and are **provisional until the months 1 to 12 backfill exists**; Claude writes the
     backfill code, and the owner runs it on Databricks with the redeploy.
   - **Backfill code written 2026-11-09 (not run):** `cdm.backfill` and
     `databricks/notebooks/melanoma_backfill.py` rebuild each replay month's delayed-accuracy
     record, with `melanoma_referred`, from the rows the replay wrote to `gold_prediction_audit`
     and the labels in silver; nothing is re-scored. The run fails if a stored `refer_melanoma`
     flag disagrees with the bundle's threshold, or if any rebuilt month's labelled count,
     melanoma count or balanced accuracy differs from the committed record (months 1 to 11;
     month 12 has none and is marked unchecked). It writes `gold_melanoma_backfill` once and
     `melanoma_backfill.json`; the monthly job merges the table into the target's history, and
     `scripts/slo_report.py` uses the committed file (month m sees only months before m).
     `databricks/melanoma_backfill_job.json` holds `STAGE_FIRST` values that the deploy refuses
     until `scripts/stage_research_job.py melanoma_backfill results/phase3/serving/results.json
     results/phase3/monthly/month_01.json ... month_12.json` has filled them. Tested on
     synthetic data in `tests/test_backfill.py`.

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
   - **Built 2026-11-09 (code, not run):** the alert path in `cdm.slo`: after saving every
     month of a run the job fails the run with "ALERT (not a job error)" and each reason (a broken
     target, an overall drift alert, a skin-type drift alert); the failure email reaches the
     owner. Thresholds and reasons are in the README's "Alerts and model targets" table. Not
     ticked: no run has sent an alert yet.

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
   - **Built 2026-11-09, waiting on the owner:** `ci.yml` and `docker.yml` now run on every
     push to every branch and on every pull request. `tests/test_champion_smoke.py` loads the
     bundle with `cdm.serving.Service` (every file's SHA-256 and each seed's weights hash
     checked), scores the manifest's images, and fails if more than 20% abstain
     (`config/champion_smoke.json`). It **skips, naming what is missing,** until (1) the owner
     attaches the champion bundle to a release and fills `bundle_asset` (URL and SHA-256), and
     (2) `scripts/make_smoke_manifest.py` writes `config/champion_smoke_images.csv` (50 images
     of the clients' test split; needs the ISIC API, unreachable from cloud sessions; its
     search query syntax is unverified). Not ticked: the smoke test has not run.

7. [ ] **Observability on Databricks:** a prediction audit table, Unity Catalog lineage, and
   per-run cost tracking.
   - **Done when:** the audit table is written by every scoring run; lineage shows bronze to
     silver to gold to the registered model to the audit table, with a screenshot; and the cost
     table covers every run behind a committed result.
   - **Today:** the audit table is built. Lineage has no edge from the registered model to the
     audit table, and there is no screenshot. Costs are traced for 13 of 31 results files, with
     billing up to 2026-10-22.
   - **Blocked, 2026-11-09 (needs the workspace):** every remaining part runs in Databricks,
     which cloud sessions cannot reach: the lineage screenshot, and the billing query that
     traces the other 18 results files. **Proposed (owner to decide):** the missing edge exists
     because the monthly job loads the serving bundle from a volume, not the registered model;
     loading the champion through `models:/workspace.cdm.skin_lesion_classifier@champion` in the
     scoring job would let Unity Catalog record registered model to audit table. It changes the
     scoring job (and adds MLflow to its environment), so it waits for the owner's OK.
   - **Owner decision, 2026-11-10:** load the champion through the registry
     (`models:/workspace.cdm.skin_lesion_classifier@champion`) in the scoring job, so lineage
     records the registered model to audit table edge. A change to the monthly job; the owner
     deploys it.

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

10. [x] **Reviewer feedback section** in the README, to record what outside reviewers say and
    what changed.
    - **Done when:** the README has the section with a fixed format (date, reviewer role,
      comment, what changed, link), empty or with real entries only.
    - **Today (2026-11-06):** not started.
    - **Done 2026-11-09:** the README's "Reviewer feedback" section has the five-column table,
      empty ("No entries yet."); `tests/test_readme_feedback.py` fails if the header changes or
      a row lacks a field. Done out of order. Every item before it in the Order of work is
      unticked: the remaining parts of items 1, 6, 11, 4, 5, 18, 13, 2, 3, 7, 15, 17, 16, 14 and
      8 wait on the owner (deploys, the bundle asset, approvals, GPU or Databricks runs, PR #50's
      thresholds), but **item 9 is not blocked** and was skipped in this session; it is the next
      item to work. Item 10 depends on none of them.

## Production architecture (owner, 2026-11-08)

What real ML, data and software engineering teams would add, checked against the repo on
2026-11-08. Today the repo already has the `src/cdm` package, ruff, mypy, pytest, a Docker
image smoke-tested in CI, a `v1.0.0` tag (with no assets) and the Unity Catalog registry.

11. [x] **Supply chain and merge rules.** Dependabot for pip, Docker and GitHub Actions
    (grouped, weekly); CodeQL for Python and the workflows; `pip-audit` and a Trivy scan of
    the API image in CI, failing on fixable critical findings (accepted ones in an ignore file
    with the reason); branch protection on `main`, which the **owner turns on** in GitHub
    settings, requiring the `ci` and `docker` workflows' jobs.
    - **Done when:** the first Dependabot PR appears; a known-vulnerable pin fails the build;
      `gh api repos/erjonb19/clinical-drift-monitor/branches/main --jq .protected` is `true`.
    - **Today:** none of these; `main` is unprotected; CI runs on pushes to `main` and on pull
      requests only.
    - **Built 2026-11-09 (repo side), waiting on the owner:** `.github/dependabot.yml` (pip,
      Docker, Actions; weekly, grouped); `.github/workflows/codeql.yml` (Python and Actions);
      in `ci.yml`, `pip-audit` 2.10.1 audits a freeze of the installed packages from its own
      environment (torch's `+cpu` labels stripped so it is audited, and any package it cannot audit
      fails the step) and fails on any finding not accepted in `config/pip_audit_ignore.json`; a
      second step shows it reports a known-vulnerable pin by name (`tests/fixtures/`); in `docker.yml`,
      Trivy (pinned to a commit) fails on fixable CRITICAL findings in the API image, with
      accepted ones in `.trivyignore` (none). **Its first run found five fixable CRITICAL
      Pillow 12.3.0 findings in the API image** (CVE-2026-112393, -112398, -112403, -112900,
      -96639; fixed in 13.0.0), including a heap overflow in JPEG decoding, which the API does
      on every upload; `docker/constraints.txt` now pins Pillow 13.0.0 (the full test suite
      passes with it). **Accepted (confirmed by the owner, 2026-11-09):** four
      `cryptography` 46.0.7 findings and one `uv` 0.11.33 finding, both held back by `flwr`
      1.39.0 (`federated` extra, training only); `flwr` 1.43.0 would allow the fixed version but changes the stack behind the
      committed federated results. **Owner:** turn on branch protection for `main` (Settings,
      Branches) requiring the `quality`, `image` and CodeQL `analyze` checks; the first
      Dependabot PR appears on its weekly run. **Gap:** Dependabot's Docker updates cover the
      `FROM` image only, not the torch pin in the Dockerfile's `RUN` line or
      `docker/constraints.txt`; those still need a manual bump (the Trivy scan reports their
      vulnerabilities).
    - **Owner decisions, 2026-11-09:** the `cryptography` and `uv` findings stay accepted until
      the 2026-12-09 review, and `flwr` is not upgraded. Dependabot's first PRs: #67 (three
      Actions) merged green; #66 (Python 3.11 to 3.15 in the image) and #68 (33 packages,
      including `flwr` 1.43.0 and torch without its `+cpu` label) closed. `dependabot.yml` now
      ignores Python minor and major updates of the image, and gives torch, torchvision, numpy,
      scikit-learn, `flwr` and MLflow patch updates only (a test checks both rules); the rest
      of the Python group is merged when its next PR is green. The owner turns on branch
      protection (`quality`, `image`, `analyze`).
    - **Dependabot's next Python group (#74, 2026-11-09): not merged, CI red.** `quality`
      failed at install: pip could not resolve the dependencies (`ResolutionImpossible`),
      because the PR pins `uvicorn==0.54.0` and `flwr` 1.39.0 (held on purpose) needs
      `uvicorn<0.50.0`. It also drops the `+cpu` label from torch 2.14.1 in
      `docker/constraints.txt` (the new test would fail on it; CI stopped before the tests),
      raises lower bounds by a major version (`mlflow-skinny` 2.16 to 3.18, `duckdb` 1 to 2) and
      moves the `transformers` pin from 4 to 5, so the patch-only rule did not stop floor raises
      in `pyproject.toml`. Left open for the owner. Possible fixes: set the pip entry's
      `versioning-strategy` to `increase-if-necessary` (keeps `>=` floors; exact pins are still
      bumped), ignore uvicorn minor updates while flwr is held at 1.39.x, and bump torch by hand
      in both the Dockerfile and the constraints.
    - **Owner decision, 2026-11-10:** #74 closed; the suggested fixes applied in
      `dependabot.yml`: `versioning-strategy: increase-if-necessary` for pip, uvicorn patch
      updates only while flwr is held, and torch and torchvision ignored outright (replacing the
      2026-11-09 patch-only rule for those two, since even a patch bump drops `+cpu`). A torch
      bump is a hand change to the Dockerfile and `docker/constraints.txt` together, checked by
      the `+cpu` test. Tests check all three.
    - **Done 2026-11-11:** all three checks met. The first Dependabot PR appeared (#66,
      2026-11-09); CI's self-test step shows pip-audit exits 1 and names the known-vulnerable pin
      (`requests 2.19.0`, a test fixture), and the main audit step fails the build on any finding
      not accepted in `config/pip_audit_ignore.json`; and
      `GET /repos/erjonb19/clinical-drift-monitor/branches/main` returns `"protected": true`
      with required checks `quality`, `image`, `analyze (actions)` and `analyze (python)`
      (checked 2026-11-11). **Note:** enforcement is `non_admins`, so the owner's admin account
      can bypass the checks; including administrators is the owner's setting.
    - **Dependabot's next Python group (#83, 2026-11-11): held, closed.** `quality` failed at
      install: it moved fastapi to 0.162.0, and `flwr` 1.39.0 needs fastapi `>=0.138,<0.139`. It
      also moved the research Phase B stack a major version (`open_clip_torch` 3 to 4,
      `transformers` 4 to 5). `dependabot.yml` now gives fastapi, `open_clip_torch` and
      `transformers` patch updates only (a test checks it); the rest of the group comes back on
      Dependabot's next run.

12. [ ] **A live endpoint** **(ask)**. The free public deploy has been blocked since 2026-10-23
    (Hugging Face now charges for Docker Spaces; `scripts/deploy_space.py`). Compare, from each
    vendor's own pages: Databricks Model Serving on Free Edition (Databricks stays central;
    CPU only, limits to check), and the ONNX export on a small free host such as Render
    (512 MB; the ONNX model's memory measured first). Write the choice under this item; the
    owner creates any account.
    - **Done when:** a public URL answers `/health` and a scoring request, the README says
      where it runs and its limits (cold starts, quotas), and every request is logged.
    - **Today:** the API runs only in the CI smoke test and locally.

13. [ ] **A load test and serving benchmark.** Locust or k6 against the API container at 1, 5
    and 20 concurrent users, reporting p50, p95 and p99 latency, throughput and memory; the
    same for the ONNX export, and for dynamic int8 quantization if its accuracy on the
    validation set stays within the gate. A dispatch-only CI job; numbers in
    `results/phase3/serving/benchmark.json` with the hardware named.
    - **Done when:** the committed benchmark shows the numbers and names the bottleneck; the
      README's serving section links it.
    - **Today:** an ONNX export exists (`results/phase3/onnx/`), not shipped (PyTorch serves);
      no load test.
    - **Built 2026-11-09, partly measured:** `scripts/load_test.py` (1, 5 and 20 concurrent
      users; p50, p95 and p99 latency, throughput, server memory), `scripts/bench_app.py` (the
      API's own app with the rate limiter opened wide) and the dispatch-only
      `.github/workflows/benchmark.yml`. **Measured** in this cloud session on the untrained
      test bundle (one network; the champion runs three), so serving speed only:
      `results/phase3/serving/benchmark.json`, linked from the README. The machine saturates
      (throughput flat at about 44 to 53 images a second, latency growing with users); the mechanism and
      a 5x gap with Phase 3's PyTorch timing are not explained.
      **Not done:** the champion's numbers (needs its bundle), the ONNX export (failed parity in
      Phase 3) and int8 quantization (needs the champion's accuracy on validation). Not ticked.

14. [ ] **Batch scoring at scale.** Score a large slice of the public ISIC archive (for example
    50,000 images) on Databricks serverless as a distributed job (Spark with a pandas UDF, or
    batch inference over Delta), with throughput, cost from the billing table, and the
    bottleneck found. **(ask)** before the run, with its estimate; respect each image's
    license.
    - **Done when:** the run's results file and cost are committed and the README states
      images per minute and cost per 1,000 images, measured.
    - **Today:** the monthly job scores about 300 images a month.
    - **(ask), 2026-11-09, not started:** no estimate is good enough to ask with yet. At Phase
      3's batch-1 timing (about 93 ms per network per image on serverless CPU, three networks)
      50,000 images are roughly 4 CPU hours of scoring before any parallelism, and the images
      themselves (about 2.6 MB each, from one probe download) are on the order of 130 GB through
      Free Edition's restricted outbound access, whose limits for that volume are unknown. A
      smaller first slice (for example 5,000 images) would measure both before the owner is
      asked for the full run.
    - **Owner decision, 2026-11-10:** measure a 5,000-image slice first.

15. [ ] **Detector quality: time to detect and false alarms.** Staged shifts (labelled staged)
    of known size and start month, run through the monthly drift test; report how many months
    each shift takes to detect, and how often the test alarms on months with no shift. Ties
    the thresholds in item 5 to measured error rates.
    - **Done when:** a spec committed before the run, the results file, and a README table
      (shift size, months to detect, false-alarm rate).
    - **Today:** listed as open in PLAN.md Phase 3; PR #50's 30 scenarios are for the incident
      report evals (item 8), not for this.
    - **Spec written 2026-11-09 (not run, (ask)):** `results/phase3/detector_quality/spec.json`:
      staged blur, brightness, JPEG and age-mix shifts at three sizes and four shares, starting
      in month 4 of 12-month synthetic sequences (200 repeats each), scored by the shipped
      detector on CPU like the monthly job; time to detect and false-alarm rate (with a Wilson
      interval; the months reuse one pool, so it is not a population interval), with a no-shift
      control, and the post-replay rule's false-alarm rate. Needs a Databricks run: about 2.5
      CPU hours by Phase 3's batch-1 timing (an estimate), or less batched or on GPU.
    - **Owner decision, 2026-11-09:** approved, to run on a GitHub Actions runner (dispatch
      only), not on Databricks.
    - **Built 2026-11-09 (not run):** `cdm.detector_quality` (the shifts, applied to the exact
      224 px input the model sees; month draws; each sequence paired with a control that draws
      the same months with no image altered; time to detect; false-alarm rates with Wilson
      intervals; the post-replay rule's false-alarm rate), `scripts/detector_quality.py` (fetch
      by ID from the ISIC Archive, score with the champion bundle on CPU, simulate) and
      `.github/workflows/detector-quality.yml`. The spec is amended before any run: the runner
      fetches exactly silver's labelled 3,262 and 3,210 images by ID, and fails on any other
      count. Tested on synthetic scores (`tests/test_detector_quality.py`). **Waits on:** (1)
      the champion bundle as a release asset (as item 6 does), and (2) the owner exporting
      `config/detector_quality_ids.csv` once from Databricks (`isic_id, split`; public IDs only;
      the query is in the script):
      `SELECT isic_id, split FROM workspace.cdm.silver_images WHERE role = 'client' AND split
      IN ('val', 'test') AND label IS NOT NULL ORDER BY split, isic_id`. The workflow fails,
      saying which is missing, until both exist.

16. [ ] **Shadow scoring.** The challenger scores the same monthly images as the champion
    without serving them; both scores land in the audit table with the model version, and a
    monthly comparison is written. The ML version of a canary, before item 3's promotion.
    - **Done when:** one monthly run writes both models' scores and the comparison file.
    - **Today:** champion vs challenger is an offline comparison only (item 3).
    - **Blocked, 2026-11-09:** needs a registered challenger (item 3's GPU run, an (ask)).

17. [ ] **An automated retraining pipeline.** One multi-task Databricks job: a drift alert
    with labels available starts it; train the challenger, evaluate it against the
    pre-committed targets, register it (always, with its `model_gen` tag), and end. The
    `champion` alias moves only through a separate `promote_challenger` job that the owner runs
    by hand with the registry version and the decision file's SHA-256 (the human approval;
    wording changed by the owner on 2026-11-09, from "stop for a human approval" inside the one
    job). **(ask)** before any GPU run.
    - **Done when:** both job definitions are committed and tested; one run (approved) goes from
      trigger to a registered version and a decision file, and the alias moves only when the
      owner runs `promote_challenger`.
    - **Today:** the retraining loop ran once by hand (2026-10-23).
    - **Spec written 2026-11-09 (not built):** `docs/specs/item17-retraining-pipeline.md`: one
      job (trigger check, challenger training, evaluation, decision, then registration of every
      challenger) and a separate `promote_challenger` job the owner runs by hand with the
      decision file's hash (**approved by the owner, 2026-11-09**; the item's wording above is
      updated). About 3 to 4 GPU DBUs a run (an estimate); (ask) before any.
    - **Built 2026-11-09 (the approval half; not run):** `cdm.promotion.trigger_check` (a month
      triggers when it alerted and its labels have arrived; on the committed replay, months 7 to
      11 would) and `cdm.promotion.verify_promotion`, with
      `databricks/notebooks/promote_challenger.py` and `databricks/promote_challenger_job.json`
      (no schedule; the owner passes the registry version, the expected current version, the
      decision file and its SHA-256). It refuses unless that decision's rule promoted and the
      version was registered from it with a checked serving bundle (tags `decision_sha256` and
      `serving_bundle_sha256`), then moves the alias through `cdm.registry`, reads it back and
      logs the move to `gold_model_decisions`. Tested in `tests/test_promotion.py`. **Not built:**
      the training job's tasks (trigger check, train with checkpoints always kept, evaluate,
      decide, build and check the serving bundle, register with those tags): they change the
      GPU notebook `retrain.py` and can only be tried with a GPU run, an (ask). Two departures from the
      approved spec, re-approved by the owner on 2026-11-10, are listed in the spec's "Changes made while
      building" section: the decision key stays `promoted`, and the serving bundle is checked
      before registration rather than rebuilt by `promote_challenger`.

18. [ ] **An input contract for images.** Before scoring, each image is checked: format,
    resolution, aspect ratio, blur, exposure, and duplicates (perceptual hash); failures are
    quarantined with a reason, counted per month, and alerted on a spike, in the API and in the
    monthly job alike.
    - **Done when:** tests with bad images show each one quarantined with its reason; the
      monthly results carry the counts.
    - **Today:** no image quality checks before scoring.
    - **Built 2026-11-09 (API and library; monthly job not yet):** `cdm.contract` checks format
      (JPEG or PNG), resolution (shorter side at least 224 px, the model's input), aspect ratio
      (at most 3:1) and near-duplicates (64-bit difference hash within 4 bits; monthly job only,
      since the API keeps no hashes); these quarantine with a reason. Blur (Laplacian variance)
      and exposure (mean brightness, clipped share) are **report-only** until calibrated on real
      images; their values are guesses. `month_counts` gives counts per reason and a spike flag
      (over 10% quarantined). The API returns 422 with the reason, counts `quarantined` in
      `/health`, and returns `input_flags`; tested in `tests/test_contract.py` and
      `tests/test_api.py` on synthetic images. **Not done:** the monthly job does not run the
      contract yet (it belongs with the post-month-12 scoring wiring), so no monthly counts
      exist; blur and exposure need calibrating (16-bit
      PNGs, for one, read as heavily clipped). Decompression bombs (over the 50M-pixel limit)
      are unreadable (400), as before; the check runs off the event loop and computes no hash
      in the API; flat images hash to zero and are never matched as duplicates.
    - **Owner decision, 2026-11-09:** the contract's thresholds are accepted; blur and exposure
      stay report-only until calibrated on real images.

19. [ ] **Least-privilege access.** The watchdog's service principal (item 1) can view one job
    and nothing else, with the `users` group's default grants checked and removed; the
    scoring job runs as its own principal, not as the owner; the API takes per-client keys
    with a request log. **(ask)** before creating any principal (the owner creates them).
    - **Done when:** a committed grants report shows each principal's rights, and a test
      shows the API refuses a missing or revoked key.
    - **Today:** jobs run as the owner; the API has no authentication.

20. [ ] **Drift results through the governed clinical agent** **(ask: touches the other
    repo)**. After that repo's item 22, the monthly drift tables become a read-only data
    source there, so a person can ask in plain English which month and which site drifted,
    through its guard.
    - **Done when:** one question about a drift month answers correctly from the other repo.

21. [ ] **Showcase.** A one-page `docs/ARCHITECTURE.md` (one diagram; how it scales and where
    it breaks, using items 13 to 15's numbers), the README refreshed results-first (an open
    Phase 5 item), `docker compose up` as the one-command demo (also open in Phase 5), and a
    three-minute walkthrough script for the owner to record.
    - **Done when:** all four exist and the README links them.

22. [x] **Security audit against `docs/SECURITY_STANDARDS.md`** (owner, 2026-11-09). Plan only, change no code. Audit the repo against Section 19.3 of `docs/SECURITY_STANDARDS.md` and every section that applies to this stack, and report each gap in the file's SEVERITY format. Add one backlog item per gap, numbered after this one, with anything Critical (CVSS 9+) first in the Order of work. Propose a `.claude/settings.json` from Section 17.2, tailored so unattended runs don't stall: keep the deny list, and explain each change to the ask list. Check every vendor-specific setting (npm, uv, GitHub, Claude Code permission syntax) against the vendor's current docs before proposing it. One docs-only PR, which the owner reviews and merges himself (security decisions are his). Include Section 4.5: model checkpoints loaded with pickle (torch.load) should use `weights_only=True` or safetensors.
    - **Done 2026-11-10:** `docs/security-audit-2026-11-10.md`: no Critical or High finding, no
      secret in the tree or in any commit's history (414 counted); 5 Medium and 27 Low findings, each in
      the SEVERITY format, with items 23 to 47 below; a proposed `.claude/settings.json`
      (not added); the Section 4.5 `torch.load` call sites (item 31). The owner places the new
      items when he merges the audit's PR.

**Items from the security audit (2026-11-10).** None is Critical, so none goes first; the owner
places them. Each names its finding in `docs/security-audit-2026-11-10.md`. "(owner)" marks a
change only the owner can make (repository settings, accounts, consoles).

23. [ ] **Add `.claude/settings.json`** from the audit's proposal, after the owner's review.
    - **Done when:** the file is committed and a cloud session's `/permissions` lists its rules;
      `ruff check .` runs without a prompt, `curl -X POST` and a push to `main` are denied.
24. [x] **Branch protection on `main`** (owner, Medium): a ruleset requiring a PR and the
    `quality`, `image` and CodeQL `analyze` checks; no force-push or deletion.
    - **Done when:** `GET /repos/erjonb19/clinical-drift-monitor/branches/main` says
      `"protected": true` (as item 11 already asks).
    - **Done 2026-11-11 (owner):** turned on, requiring `quality`, `image` and both CodeQL
      `analyze` checks.
25. [ ] **Secret scanning before commit and in CI** (Medium): `.pre-commit-config.yaml` with
    gitleaks pinned; a CI job scanning full history (action pinned by SHA, `contents: read`).
    - **Done when:** a planted fake key fails the CI job, and the job is green on the real repo.
26. [ ] **Locked, hashed, cooled-down Python installs** (Medium; a new tool, uv, needs the
    owner's OK first): `uv.lock`, `exclude-newer = "7 days"`, `uv sync --locked` in every
    workflow, hashed exports for Docker and the Streamlit app.
    - **Done when:** CI and the app install only from the lock or hashes, and a test fails if a
      workflow installs without them.
27. [ ] **Hashed installs in the API image** (Medium): hashed requirements (with the `+cpu`
    torch wheels), `--require-hashes --no-deps`, pinned build backend.
    - **Done when:** the image builds with `--require-hashes` and the Trivy step passes.
28. [x] **Read-only token in `ci.yml`** (Medium): `permissions: contents: read`,
    `persist-credentials: false`; the repository default set to read-only (owner).
    - **Done when:** every workflow declares its permissions, checked by a test.
    - **Done 2026-11-11:** `ci.yml` now sets `permissions: contents: read` like the other five
      workflows, every checkout sets `persist-credentials: false`, and
      `tests/test_supply_chain.py` fails if a workflow lacks a top-level `contents: read` block or
      a checkout keeps its credentials (checked by removing the block from `ci.yml`). **Owner:**
      the repository's default workflow permission (Settings, Actions, General) should be
      read-only; it cannot be read without admin access.
29. [x] **Refuse 50 to 100 megapixel images** (silent failure #16): treat Pillow's
    `DecompressionBombWarning` as unreadable, or cap `img.size` before decoding.
    - **Done when:** a 60-megapixel PNG gets a 400 from the API and `unreadable` from the
      contract, in tests.
    - **Done 2026-11-11:** `cdm.contract.open_image` refuses any image over 50 megapixels from
      its header, before decoding; the contract check and `Service.predict` both use it (a size
      check rather than a warnings filter, which would be process-wide and not thread-safe in
      the API's thread pool). Tests in `tests/test_contract.py`, `tests/test_api.py` and
      `tests/test_serving.py`; silent failure #16 now names this check.
30. [ ] **Open uploads with JPEG and PNG decoders only** (`Image.open(..., formats=...)`).
    - **Done when:** a TIFF or ICO upload is refused before any decoder other than JPEG or PNG
      runs, in tests.
31. [x] **`torch.load(..., weights_only=True)` everywhere; floor `torch>=2.6`** (Section 4.5;
    the call sites are listed in the audit).
    - **Done when:** a test fails on any `torch.load(` without `weights_only=True`.
    - **Done 2026-11-11:** all 18 calls pass `weights_only=True` (each loads a plain
      `state_dict`; 15 are in notebooks or a script that only run on Databricks or the owner's
      laptop, converted by their call shape and not run here, and the other 3 run in the test
      suite); the floor is
      `torch>=2.6` in `pyproject.toml` and the seven job files. `tests/test_weights_only.py`
      parses `src`, `scripts`, `tests`, `app` and the notebooks and fails on any `torch.load`
      without a literal `weights_only=True` (aliases included), with a planted-call self-test.
      Safetensors for the bundle stays a later option.
32. [ ] **Bundle manifest must cover every checkpoint and detector**; reject path-like names.
    - **Done when:** a bundle whose manifest omits a checkpoint fails `Service.load`, in tests.
33. [x] **Checksum the CIFAR-10 files before unpickling** (`scripts/build_benchmarks.py`).
    - **Done when:** a changed CIFAR file is refused before `pickle.load`.
    - Done 2026-11-11: each batch file's MD5 is checked against torchvision's published list before `pickle.loads`; `tests/test_build_benchmarks.py` shows a planted pickle is refused without running and an unknown file name is refused. The benchmark build itself was not rerun here.
34. [ ] **Pin GitHub's own actions by SHA**; drop the test's owner exemption.
    - **Done when:** `tests/test_supply_chain.py` requires a SHA for every `uses:`.
35. [x] **Pin the Docker base image by digest.**
    - **Done when:** the `FROM` line carries `@sha256:` and Dependabot updates it.
    - **Done 2026-11-11:** `FROM python:3.11.17-slim@sha256:b7a9e5f6…` (the multi-platform
      index digest, read from Docker Hub's registry on 2026-11-11); `.github/dependabot.yml`'s
      docker entry updates digests along with tags (its Python minor and major ignore stays),
      and `tests/test_supply_chain.py` fails if any `FROM` lacks a digest.
36. [x] **Turn on ruff's `S` rules**, fixing or justifying each hit.
    - **Done when:** `ruff check .` passes with `S` selected.
    - **Done 2026-11-11:** `S` is selected in `pyproject.toml`; `ruff check .` passes. Fixed: 16
      `assert`s outside tests and notebooks became explicit `raise`s (several were data
      integrity checks that `python -O` would have skipped), and the three MD5 checksums of
      published files say `usedforsecurity=False`. Accepted, each with its reason in
      `pyproject.toml`: argument-list subprocess calls (S603, S607; `shell=True` stays
      flagged); asserts in tests and in Databricks notebooks (never run under `-O`); SQL built
      from repository constants or the owner's own CLI arguments in eight named files; `urlopen` on fixed hosts in six named
      files, and on URLs returned by the ISIC API in four (restricting those to the ISIC host is
      item 47).
37. [ ] **zizmor on workflow changes.**
    - **Done when:** a zizmor job runs on `.github/workflows/**` and passes.
38. [x] **Dependabot cooldown** (`cooldown: default-days: 7` per ecosystem).
    - **Done when:** each ecosystem has a cooldown, checked by a test.
    - **Done 2026-11-11:** pip, docker and github-actions each have `cooldown: default-days: 7`
      in `.github/dependabot.yml`; `tests/test_supply_chain.py` fails if an ecosystem loses it or
      goes under 7 days. GitHub applies a cooldown to version updates only, so security updates
      still open at once. GitHub's default is a 3-day cooldown, so this raises it to 7. The file
      parses with `yaml.safe_load`; Dependabot validates it again once it is on main.
39. [ ] **Trivy fails on fixable HIGH as well as CRITICAL.**
    - **Done when:** the scan uses `CRITICAL,HIGH` and passes (accepted findings in
      `.trivyignore` with reasons).
40. [x] **`.gitignore` covers Section 3.1's secret and state files.**
    - **Done when:** `git check-ignore` matches `.env.local`, `x.pem`, `x.key`, `x.p12`,
      `secrets/a` and `.databrickscfg`, checked by a test.
    - **Done 2026-11-11:** `.gitignore` adds `.env.*` (except `.env.example`), `*.pem`, `*.key`,
      `*.p12`, `*.pfx`, Terraform state, `secrets/`, credential and service-account JSON and
      `.databrickscfg`; no tracked file matches them. `tests/test_gitignore.py` checks each with
      `git check-ignore`, and that `.env.example` stays committable.
41. [x] **`.env.example`** with variable names and placeholders only.
    - **Done when:** it exists and lists every environment variable the code reads.
    - **Done 2026-11-11:** `.env.example` names the nine variables the code reads: the Haiku
      key and the watchdog's three Databricks values as empty names, and the data and notes
      folders, the tier-2 switch, and the API's bundle folder and proxy hops commented out with
      their defaults (an empty value would override a default; an empty `TRUSTED_PROXY_HOPS`
      would stop the API). `tests/test_env_example.py` scans `src`, `scripts` and `app` for
      `os.environ` reads and fails on any unlisted name or any filled-in value.
42. [ ] **Private vulnerability reporting** (owner turns it on) and a `SECURITY.md` pointing to
    it, with no personal email.
    - **Done when:** both exist.
    - **2026-11-11:** private vulnerability reporting turned on by the owner; `SECURITY.md` to
      come.
43. [x] **Dependabot alerts and security updates on** (owner).
    - **Done when:** the repository's security settings show both enabled.
    - **Done 2026-11-11 (owner):** Dependabot security updates (grouped) and malware alerts on
      (`security_and_analysis.dependabot_security_updates: enabled`, checked through the API).
      Dependabot (vulnerability) alerts are on because GitHub enables security updates only
      when alerts are on; the owner named security updates and malware alerts, not alerts.
44. [ ] **API hardening before any deploy:** security headers, `openapi_url=None`, response
    models, and structured security-event logging with no address or body.
    - **Done when:** tests check the headers, the absent schema, the response models and one
      logged rejection.
45. [x] **The Haiku cap fails closed:** a Console spend limit on the key (owner); refuse when
    the spend log is missing after spending began; date the price constant.
    - **Done when:** a test shows a missing log refuses a call, and the Console limit is noted.
    - **Done 2026-11-11:** `SpendLog` keeps a high-water mark beside the log
      (`~/.cdm/llm_spend.jsonl.high_water.json`), raised on every recorded call; once spending
      has begun, a missing log or one recording less than the mark refuses every call
      (`tests/test_extract.py`). The price constant carries the date it was last matched
      (`PRICE_CHECKED`, 2026-11-11: $0.10 and $0.50 per million tokens for `claude-haiku-5-5`,
      from the Claude API reference's model table). **Owner:** a spend limit on this key in the
      Anthropic Console is the real backstop; whether one is set cannot be checked from here.
      Limits: deleting the log and its mark together still resets the client-side count, and a
      log that predates this change has no mark until its next call.
46. [ ] **Neutralise formulas in the G5 CSV error sheets.**
    - **Done when:** a cell starting with `=` is written with a leading `'`, in tests.
47. [ ] **Public demo hardening:** DuckDB external access off and locked, error details hidden,
    temporary folders cleaned, a per-session cap on definitions, and the ISIC client restricted
    to its own host with a byte cap.
    - **Done when:** tests cover each, and the deployed demo is redeployed (owner).
    - **Built 2026-11-11, waiting on the owner's redeploy:** the demo's DuckDB connection has
      external access off and its configuration locked (`read_csv`, `ATTACH`, `COPY` and
      re-enabling all fail, in tests); `.streamlit/config.toml` hides error details and the
      page shows a generic message for fetch and load errors; a session's scratch folder is a
      temporary directory removed with the session, and a session may hold at most 50
      definitions; the ISIC client follows `next` only on https to its own host and caps
      response bodies (20 MB for JSON, 100 MB for images). Tests in `tests/test_cohort.py`,
      `tests/test_demo.py` and `tests/test_sources.py`. **Owner:** redeploy the Streamlit app
      (Community Cloud picks up `main` on reboot), then tick this item.

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
