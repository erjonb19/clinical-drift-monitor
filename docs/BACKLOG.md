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
   - **Done when:** one run started by the schedule (not by hand) finishes green, and its
     results file and job run ID are committed, with the run's trigger shown as scheduled.
   - **Today:** `databricks/monthly_drift_job.json` is scheduled at 06:00 UTC on the 1st, and
     emails the owner on success and failure. The 12-month replay ran once as a manual backfill.
     The first scheduled run was due on 1 November 2026; its outcome is not recorded in the
     repo. After the replay, scheduled runs only log the ISIC live count and score nothing
     new, so a green run alone would not show drift being scored.

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
     earlier phases call "v2" (the 3-seed ensemble). Which model is "v2" here needs the owner's
     answer before any run.

4. [ ] **SLOs as model targets**, such as a minimum melanoma sensitivity, written in the README,
   with an alert when drift breaks a target.
   - **Done when:** the README has an SLO table (each target, its value, why, and where it is
     measured); the monthly job checks each target it can measure; and a broken target raises an
     alert.
   - **Today:** CI floors exist for committed metrics (`config/eval_gate.json`), but no runtime
     SLOs. **Limit:** the job reports melanoma sensitivity only for a month with at least 100
     melanomas; the replay months had 5 to 64, so a monthly melanoma-sensitivity SLO can't be
     measured at 300 images a month without pooling months or changing the rule.

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
   - **Today:** built (ruff, mypy, pytest, and the eval gate with a regression self-test, plus
     the Docker smoke test). The gate reads committed results; it does not run a model in CI.
     Owner to decide whether that counts as done.

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
  to E and G are already built; see `docs/BUILT_VS_PLANNED.md`.

## Open from PLAN.md, not placed (owner to decide)

These were open in PLAN.md and are not in the list above. They stay here until the owner adds
them to the list or drops them.

- Phase 3: CPU batch-level drift, staged shifts scored for time to detect and false-alarm rate,
  and `diagnosis_confirm_type` in bronze and silver.
- Phase 4: the report generator, groundedness check, eval harness, human review queue and
  manual live-eval workflow (item 8 needs most of these).
- Phase 5: the one-command demo (`docker compose up`), the 8D runbook, the README refresh
  (results first), and the standards-alignment document.
