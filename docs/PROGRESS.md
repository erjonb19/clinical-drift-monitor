# Progress

One line per finished item from [BACKLOG.md](BACKLOG.md): what changed, and how to check it.
Nothing goes here until the item's "Done when" is met.

| Date | Item | What changed | How to check |
| --- | --- | --- | --- |
| 2026-11-10 | 22 | Security audit against `docs/SECURITY_STANDARDS.md` (plan only): 0 Critical, 0 High, 5 Medium, 26 Low; items 23 to 47; a proposed `.claude/settings.json` | Read `docs/security-audit-2026-11-10.md` |
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
