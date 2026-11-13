# Owner steps

This file lists every step only the owner can do: laptop commands, Databricks, GitHub or
Console settings. Each section has a title, why it matters, and the exact commands for
Windows PowerShell. Claude writes the steps and ticks one when the owner says it is done.

There are no secrets in this file. Where a step needs one, the command prompts for it, and
you type or paste it into that prompt only, never into a chat.

Commands assume PowerShell in the repository folder on the laptop (`cd` there first). They
also assume the GitHub CLI (`gh`) and the Databricks CLI (`databricks`) are installed and
logged in, as for earlier deploys.

## Open

### S1. Upload the champion bundle as a release asset (items 6, 13, 15, 21; before Nov 26)

- [ ] Done

**Why:** the champion smoke test (item 6), the champion benchmark (13), detector quality (15)
and the `docker compose up` demo (21) all need the evaluated model in CI. CI downloads it from a
GitHub release and checks its SHA-256 before use. Today every one of them skips or waits.

**Which bundle:** registry version 1, alias `champion`. Its serving bundle is the Databricks
folder `/Volumes/workspace/cdm/raw/deploy/serving/d7191f4`, the one `databricks/monthly_drift_job.json`
and `docs/rollback.md` name.

**Which release:** attach it to the existing release `v1.0.0` (published 2026-10-23), so no
new tag is needed.

**Licence:** you approved public weights on 2026-10-23, for non-commercial research use with
attribution, because the training images include CC-BY-NC ones. The release asset is public.

1. Download the bundle folder from the volume:
   ```powershell
   databricks fs cp -r dbfs:/Volumes/workspace/cdm/raw/deploy/serving/d7191f4 .\dist\champion\d7191f4
   ```
2. Check it is the evaluated bundle: its manifest's SHA-256 must be
   `5734add1790eadd6f2e36d767bd78d6fcf5a791484756ed616edf721e8eedee5`
   (`bundle.manifest_sha256` in `results/phase3/serving/results.json`):
   ```powershell
   (Get-FileHash .\dist\champion\d7191f4\manifest.json -Algorithm SHA256).Hash.ToLower()
   ```
   Then check it with the repository's own loader, which needs the laptop's `.venv` with torch
   and `cdm` installed. It checks every file's SHA-256 and each seed's weights hash, and must
   print `ok`:
   ```powershell
   .venv\Scripts\python.exe -c "from pathlib import Path; from cdm.serving import Service; Service.load(Path('dist/champion/d7191f4')); print('ok')"
   ```
3. Pack it into one archive (`tar` is built into Windows 10 and 11):
   ```powershell
   tar -czf .\dist\cdm-champion-bundle-d7191f4.tar.gz -C .\dist\champion d7191f4
   ```
4. Check its size, which must be under 2 GB (GitHub's limit per asset):
   ```powershell
   "{0:N0} MB" -f ((Get-Item .\dist\cdm-champion-bundle-d7191f4.tar.gz).Length / 1MB)
   ```
5. Upload it to the `v1.0.0` release:
   ```powershell
   gh release upload v1.0.0 .\dist\cdm-champion-bundle-d7191f4.tar.gz --repo erjonb19/clinical-drift-monitor
   ```
   Then add the terms to the release notes. Open the release page
   (`gh release view v1.0.0 --web --repo erjonb19/clinical-drift-monitor`), click **Edit**, and
   add a line such as: "cdm-champion-bundle-d7191f4.tar.gz: model weights trained on ISIC
   images including CC-BY-NC ones. Non-commercial research use only, with attribution to the
   ISIC Archive and the image contributors. Not for clinical use."
6. Get its SHA-256 in lower case:
   ```powershell
   (Get-FileHash .\dist\cdm-champion-bundle-d7191f4.tar.gz -Algorithm SHA256).Hash.ToLower()
   ```
7. Send Claude two lines: the SHA-256 from step 6, and this URL:
   `https://github.com/erjonb19/clinical-drift-monitor/releases/download/v1.0.0/cdm-champion-bundle-d7191f4.tar.gz`.
   Claude fills `bundle_asset` in `config/champion_smoke.json`, and CI then runs the smoke test.

`dist/` is ignored by git, so nothing here gets committed.

### S2. Create the watchdog's view-only service principal (item 1)

- [ ] Done

**Why:** Databricks does not email about a scheduled run that never starts, which is what
happened on 1 November. The watchdog (`.github/workflows/drift-watchdog.yml`, 07:17 UTC on the
2nd) asks the Jobs API whether the monthly job ran. It needs its own identity, with **Can View**
on that one job and nothing granted directly beyond it. Like every new identity it also joins
the workspace `users` group and gets whatever that group holds (step 6 checks what that is).
Until it exists, every watchdog run fails as "not configured".

1. In the Databricks workspace, click your user icon (top right), then **Settings**.
2. Go to **Identity and access**. Next to **Service principals**, click **Manage**, then
   **Add service principal**, then **Add new**. Name it `cdm-watchdog` and click **Add**.
3. Open `cdm-watchdog`. On its **Configurations** tab, leave only **Workspace access**
   ticked. The API needs it. Untick anything else, such as Databricks SQL access or cluster
   creation.
4. On its **Secrets** tab, click **Generate secret**. Leave the window open: S3 copies the
   **Client ID** and the **Secret** from it, and the secret is shown only once. Note the
   secret's expiry date: when it expires, the watchdog reports "inconclusive" until a new
   secret is generated and S3 step 3 is repeated.
5. Give it view access to the monthly job only. Go to **Jobs & Pipelines** and open
   `cdm-phase3-monthly-drift` (job 524433148303929). Click **Permissions** (or the **⋮** menu,
   then **Edit permissions**), add `cdm-watchdog` with **Can View**, and click **Save**.
   Don't grant it anything else.

6. Check what the `users` group gives it: open **Catalog**, select the `workspace` catalog,
   then its **Permissions** tab, and look for `users` (or "All workspace users"). Tell Claude
   what it lists; nothing needs changing unless it is more than read access.

If a label differs from these, stop and tell Claude what you see. Don't give the principal any
wider access to work around it.

### S3. Add the three repository secrets for the watchdog (item 1)

- [ ] Done

**Why:** the watchdog reads its credential from GitHub secrets. Only the scheduled or
hand-started watchdog uses them; pushes and pull requests never see them.

Each command prompts for the value. Paste it into the prompt, not into a chat.

1. The workspace URL, as in your browser's address bar up to `.com`, for example
   `https://dbc-1234abcd-5678.cloud.databricks.com`:
   ```powershell
   gh secret set DATABRICKS_HOST --repo erjonb19/clinical-drift-monitor
   ```
2. The **Client ID** from S2, step 4:
   ```powershell
   gh secret set DATABRICKS_CLIENT_ID --repo erjonb19/clinical-drift-monitor
   ```
3. The **Secret** from S2, step 4:
   ```powershell
   gh secret set DATABRICKS_CLIENT_SECRET --repo erjonb19/clinical-drift-monitor
   ```
4. Check that the three names are listed (`gh` never shows the values):
   ```powershell
   gh secret list --repo erjonb19/clinical-drift-monitor
   ```

### S4. Run the watchdog once for November (item 1)

- [ ] Done

**Why:** this proves the credential works before the real check on 2 December. November's
monthly run never started, so the right answer is **missed**. The run should end in failure,
with a line starting `MISSED: job 524433148303929 has no run started by its schedule between
2026-11-01T06:00:00+00:00 and`. A "not configured" or "inconclusive" message means the
credential or its permission is wrong.

1. Start it:
   ```powershell
   gh workflow run drift-watchdog.yml --repo erjonb19/clinical-drift-monitor -f month=2026-11
   ```
2. Wait about ten seconds so the new run exists, then follow it to the end. Pick the newest
   run (the top one) if asked:
   ```powershell
   gh run watch --repo erjonb19/clinical-drift-monitor
   ```
3. Print the watchdog's answer (pick the same run if asked) and tell Claude the line it shows:
   ```powershell
   gh run view --log --repo erjonb19/clinical-drift-monitor | Select-String -Pattern "ok:|MISSED|inconclusive|not configured"
   ```

### S5. Delete the scheduler probe job (item 1)

- [ ] Done

**Why:** the probe (job 325029901767432) did its work, and its results will be committed in
`results/phase3/scheduler_probe/runs.json` by #127. Its schedule (`0 0 6 8,9,10 * ?`) fires on
the 8th, 9th and 10th of every month, so it runs again on 8 December unless it is deleted. Do
this once #127 is merged, and before 8 December.

```powershell
databricks jobs delete 325029901767432
```

The four probe files in `/Volumes/workspace/cdm/raw/results/scheduler_probe/` can stay; they
are small. Tell Claude when it is deleted, so the probe's files in `databricks/` can go too.

### S6. Find why the monthly job did not run on 1 November (item 1)

- [ ] Done

**Why:** the probe showed the Free Edition scheduler does fire this kind of schedule, so the
cause lies with the monthly job itself (job 524433148303929).

1. Its schedule and pause state now. Check `quartz_cron_expression` is `0 0 6 1 * ?`,
   `timezone_id` is `UTC` and `pause_status` is `UNPAUSED`:
   ```powershell
   New-Item -ItemType Directory -Force .\dist | Out-Null
   databricks jobs get 524433148303929 -o json | Out-File -Encoding utf8 .\dist\monthly_job.json
   Get-Content .\dist\monthly_job.json | Select-String -Pattern "quartz|timezone|pause_status|creator|run_as|created_time"
   ```
2. Every run the job has had, including skipped or failed attempts:
   ```powershell
   databricks jobs list-runs --job-id 524433148303929 -o json | Out-File -Encoding utf8 .\dist\monthly_runs.json
   Get-Content .\dist\monthly_runs.json | Select-String -Pattern "run_id|trigger|life_cycle_state|result_state|start_time"
   ```
3. Its change history since 20 October, in a SQL editor in the workspace. This includes any
   deploy (`reset`, `update`, `create`) and any event by another identity:
   ```sql
   SELECT event_time, action_name, user_identity.email AS who, response.status_code,
          request_params['job_id'] AS job_id
   FROM system.access.audit
   WHERE service_name = 'jobs'
     AND event_date >= '2026-10-20'
     AND (request_params['job_id'] = '524433148303929'
          OR action_name IN ('create', 'delete'))
   ORDER BY event_time;
   ```
   If the query says the table does not exist or you lack access, skip it and say so;
   your earlier queries of 2026-11-06 used it, so it should work.
4. Send Claude the matching lines from 1 and 2 and the rows from 3. Remove anything you consider
   private, such as email addresses, first.

### S7. Add `gitleaks` to the required checks on `main` (item 25)

- [ ] Done

**Why:** the secret scan runs on every push, but merging does not wait for it unless it is a
required check.

1. On GitHub, open the repository, then **Settings**, **Branches** (or **Rules**,
   **Rulesets**), and edit the rule for `main`.
2. Under **Require status checks to pass**, add `gitleaks`. Keep `quality`, `image`,
   `analyze (actions)` and `analyze (python)`.
3. Click **Save changes**.

### S8. Make the default workflow token read-only (item 28)

- [ ] Done

**Why:** every workflow already asks for `contents: read`, but the repository default decides
what a new workflow gets if it forgets.

1. On GitHub: **Settings**, **Actions**, **General**, **Workflow permissions**.
2. Choose **Read repository contents and packages permissions**, and click **Save**.

### S9. Check `.claude/settings.json` is loaded (item 23)

- [ ] Done

**Why:** the deny rules from #97 only count once a new Claude Code session reads them.

1. Start a new Claude Code session on this repository and run `/permissions`.
2. Check the deny list includes `Bash(rm -rf *)`, the `.env` read rules and the `find -delete`
   and `find -exec` rules. Tell Claude what it shows.

### S10. Confirm the security alerts are closed (#123, #124)

- [ ] Done

**Why:** Claude cannot read the alert lists: the API returns 403 without admin access.

1. On GitHub: **Security**, **Dependabot**. Alerts #1 to #5 (transformers) should be closed
   by #123.
2. **Security**, **Code scanning**. Alert #3 (`app/api/main.py`) should be closed by #124's
   next CodeQL run.
3. Tell Claude which, if any, are still open.

### S11. Deploy the scheduled rehearsal of the monthly job (item 1; by 19 November)

- [ ] Done

**Why:** the deadline (26 November) comes before the real monthly run on 1 December. The
rehearsal is a temporary copy of the monthly job (`databricks/monthly_drift_rehearsal_job.json`),
scheduled for **06:00 UTC on 20 November** (the cron repeats each 20 November, so S12 deletes
the job afterwards). It runs the monthly notebook as of the merged commit, with the same bundle. The real job still
runs an older copy (staged at 44451c6); the current one also has the targets and alert path
(items 4 and 5), the rehearsal switch, and a fix that keeps later months' labels out of a
month's targets, so this is also the first Databricks run of that code. It replays month 1
into its own tables
(`workspace.cdm.rehearsal_drift_monthly`, `rehearsal_prediction_audit` and
`rehearsal_drift_baseline`) and its own folder
(`/Volumes/workspace/cdm/raw/results/rehearsal/monthly`). It never writes the real `gold_*`
tables, the real results folder, or the real job's notebook: it is imported under another name.
Its results are labelled a rehearsal everywhere. It shows that a fresh scheduled copy of the
job fires and runs end to end; it cannot show why the real job missed 1 November (S6 looks
for that). The real 1 December run is still checked by the watchdog on 2 December.

If it can't be deployed by 19 November, tell Claude, who will move the date.

1. Get the merged code:
   ```powershell
   git checkout main
   git pull
   ```
2. Stage the notebook, the wheel and the three input files under this commit. It prints a short
   JSON block with no secrets: send it to Claude. The script builds the wheel with pip; if your
   `.venv` was made with `uv sync` it has no pip, so run
   `.venv\Scripts\python.exe -m ensurepip` first.
   ```powershell
   .venv\Scripts\python.exe scripts\stage_research_job.py --workspace-name rehearsal_monthly_drift monthly_drift results/phase3/monthly/spec.json results/phase3/serving/results.json config/slo.json
   ```
3. Claude puts those values into the rehearsal job file in place of `STAGE_FIRST` and merges it.
   Then get it:
   ```powershell
   git pull
   ```
4. Create the job. The script reads the job back and stops if the schedule is paused or
   different. Send Claude the job ID it prints.
   ```powershell
   .venv\Scripts\python.exe scripts\deploy_databricks.py --job databricks\monthly_drift_rehearsal_job.json
   ```
5. If S2 is done: give `cdm-watchdog` **Can View** on `cdm-rehearsal-monthly-drift` too, the
   same way as S2 step 5.

### S12. Collect the rehearsal's evidence (item 1; 20 November, after 06:30 UTC)

- [ ] Done

**Why:** Claude commits the evidence as a rehearsal: the run's trigger (it must say
`PERIODIC`), its result and its month file.

Replace `<JOB_ID>` with the ID from S11 step 4.

1. The run, with its trigger and result:
   ```powershell
   New-Item -ItemType Directory -Force .\dist | Out-Null
   databricks jobs list-runs --job-id <JOB_ID> -o json | Out-File -Encoding utf8 .\dist\rehearsal_runs.json
   Get-Content .\dist\rehearsal_runs.json | Select-String -Pattern "run_id|trigger|life_cycle_state|result_state|start_time"
   ```
2. Its files:
   ```powershell
   databricks fs cp -r dbfs:/Volumes/workspace/cdm/raw/results/rehearsal/monthly .\dist\rehearsal
   Get-ChildItem .\dist\rehearsal
   ```
   Send Claude the lines from step 1 and the contents of `month_01.json` and `slo_01.json`.
3. If S2 to S4 are done, run the watchdog against the rehearsal job. It should end **ok**:
   ```powershell
   gh workflow run drift-watchdog.yml --repo erjonb19/clinical-drift-monitor -f month=2026-11 -f job_id=<JOB_ID>
   gh run watch --repo erjonb19/clinical-drift-monitor
   gh run view --log --repo erjonb19/clinical-drift-monitor | Select-String -Pattern "ok:|MISSED|inconclusive|not configured"
   ```
   Pick the same run (the top one) if asked. Send Claude the line it prints.
4. Afterwards, delete the rehearsal job and its notebook copy. Dropping its three tables is
   your call; they are separate from the real ones.
   ```powershell
   databricks jobs delete <JOB_ID>
   $me = (databricks current-user me -o json | ConvertFrom-Json).userName
   databricks workspace delete "/Users/$me/cdm/rehearsal_monthly_drift"
   ```
   ```sql
   DROP TABLE IF EXISTS workspace.cdm.rehearsal_drift_monthly;
   DROP TABLE IF EXISTS workspace.cdm.rehearsal_prediction_audit;
   DROP TABLE IF EXISTS workspace.cdm.rehearsal_drift_baseline;
   ```

## Done

Ticked steps move here with the date. None yet.
