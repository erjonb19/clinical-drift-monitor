# Drift incident runbook (8D)

What to do when the monthly drift job puts a month in alert (backlog item 8). Each incident's
report and review live in `results/phase4/review_queue/`; this runbook and those reports are the
project's postmortems. Research prototype: no clinical use, so "hold" and "abstain" below act on
this project's model and demo, not on care.

**Severity and containment** come from the drift table row and the fixed rules
(`cdm.scenarios.RULES`): `alert` → `abstain_and_hold`, `warning` → `monitor`, `none` → `none`.
Only an alert gets an 8D report; a warning or a quiet month gets findings only.

## D1. Team

The owner is the reviewer and decides on every alert. Claude may draft reports and fixes in
PRs; nothing it drafts counts until the owner approves it in the review queue.

## D2. Problem

1. Open the month's row: `results/phase3/monthly/month_<n>.json` (committed replay) or the
   `workspace.cdm.gold_drift_monthly` row for a live month. Note `drift` (KS p, PSI, alert),
   `by_skin_type`, `by_source`, `abstain_share` and `delayed_accuracy_of_month`.
2. Write the report into the queue:
   `python scripts/incident_report.py write --month <n> --kind <real|replayed|staged>`
   (`--writer gemini` for the LLM report, which needs `GEMINI_API_KEY`). The script runs the
   groundedness check: a report with an unmatched number is kept for the record but cannot be
   approved.
3. Say plainly whether the drift is **real**, **replayed** (real images in an order we chose)
   or **staged** (synthetic). Never mix them in one number.

## D3. Containment (`abstain_and_hold`)

1. **Abstain:** check the month's `abstain_share`. The served model flags every image above the
   abstain threshold (`abstain: true` in the API's answer) for a person to read instead of the
   model. Nothing to switch on: it is always on. If the share is above the target in
   `config/slo.json` (0.15), say so in the report.
2. **Hold:** do not promote a challenger or move the `champion` alias while the alert is open
   (`scripts/move_champion.py` and `promote_challenger` are run by hand; this is a procedure, not
   a lock in code). If the champion was moved recently, consider a rollback (`docs/rollback.md`).
3. Record in the review note that both were checked.

## D4. Root cause

1. Compare `by_source` with the months before: a new source in the batch is the usual cause.
2. A skin-type alert detects the site, not anything about skin type (skin type is recorded only
   at the new sites).
3. The drift test shows that the scores moved, not why. Write the cause as a hypothesis until
   the reviewer confirms it.

## D5. Permanent fix

Relabel a sample of the new source's images, then train a challenger
(`databricks/challenger_job.json`, a GPU run the owner approves) and promote it only if it wins
under the committed rule (`results/phase3/retrain/spec.json`), through `promote_challenger`.

## D6. Verification

Next month's run reports the alerted month's delayed balanced accuracy. Compare it with the
floor (the lowest in-distribution month, 0.591) and with the target in `config/slo.json`. Add
the result to the incident's review note.

## D7. Prevention

Keep the monthly drift test, the per-skin-type test and the abstain-share target running; keep
the input contract (`cdm.contract`) in front of scoring.

## D8. Close

Approve or reject the report:
`python scripts/incident_report.py review results/phase4/review_queue/<file>.json --decision approved --reviewer <name> --note "<what was checked>"`.
Only an approved report counts as sent. Commit the queue file through a PR.
