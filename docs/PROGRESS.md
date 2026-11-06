# Progress

One line per finished item from [BACKLOG.md](BACKLOG.md): what changed, and how to check it.
Nothing goes here until the item's "Done when" is met.

| Date | Item | What changed | How to check |
| --- | --- | --- | --- |

## Open checks (not results)

| Date | Check | Status |
| --- | --- | --- |
| 2026-11-06 | Did the monthly drift job's first scheduled run (due 1 November 2026, 06:00 UTC) run, and was it green? | **Unknown.** The owner received no email (inbox, spam and promotions checked). The job emails its deployer's Databricks login on success and on failure, so either it did not run, or the email went elsewhere or was not sent. To be read in the Databricks run history (item 1). |

## LLM spend (the one exception to $0: Claude Haiku, capped at $5 in total)

| Date | Use | Spend | Cumulative |
| --- | --- | --- | --- |
| 2026-10-23 | G5 clinical text extraction: all 85 Haiku calls, dev runs and scoring (`results/research/g5_extraction/score.json`, `llm_calls`, `llm_spend_usd`) | $0.0265 | $0.0265 |
