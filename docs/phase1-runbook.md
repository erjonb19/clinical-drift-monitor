# Phase 1 runbook: deploy and run the lakehouse

How to put the Phase 1 pipeline into your Databricks Free Edition workspace and run it.
Everything runs in Databricks' cloud except step 3, which uploads files from this laptop.

What gets created, all pinned to one commit hash:

- Schemas `workspace.cdm` and `workspace.cdm_broken`, and the volume `workspace.cdm.raw`.
- In the volume: the `cdm` wheel, `sites.json`, `HAM10000_metadata.csv` (MD5-checked), and
  the ImageNet EfficientNet-B0 weights (hash-checked).
- Pipelines `cdm-phase1` (bronze and silver in `workspace.cdm`) and `cdm-phase1-broken`
  (the same code on a faulted copy, writing to `workspace.cdm_broken`).
- Jobs `cdm-phase1` (ingest, then pipeline, then gold) and `cdm-phase1-broken-demo`.

## 1. Install the Databricks CLI (once)

In PowerShell:

```powershell
winget install Databricks.DatabricksCLI
```

Close and reopen PowerShell, then check `databricks --version` prints a version.

## 2. Log in (once)

```powershell
databricks auth login --host https://<your-workspace>.cloud.databricks.com
```

Use your workspace URL from the browser address bar. A browser window asks you to approve.
Check with `databricks current-user me`, which prints your email.

## 3. Deploy

From the repository folder, on a clean commit:

```powershell
cd C:\Users\Erjon\clinical-drift-monitor
$env:CDM_DATA = "C:\Users\Erjon\data\cdm"
.venv\Scripts\python scripts\deploy_databricks.py --dry-run
.venv\Scripts\python scripts\deploy_databricks.py
```

The dry run prints every command and writes the rendered definitions to `dist\` without
touching the workspace. The real run ends by printing the two job IDs and the commands to
run them.

## 4. Run the pipeline

```powershell
databricks jobs run-now <cdm-phase1 job id>
```

The first run downloads about 26,600 images (HAM10000 10,015; Barcelona 5,000; Buenos
Aires 1,616; MSK 3,728; PAD-UFES-20 2,298). Follow it under **Jobs & Pipelines** in the
workspace. Later runs download nothing that is already in the volume.

**Success looks like:** all three tasks green, and in a SQL editor:

```sql
SELECT * FROM workspace.cdm.gate_results;          -- every row: violations = 0
SELECT source, count(*) FROM workspace.cdm.silver_images GROUP BY source;
SELECT count(*) FROM workspace.cdm.silver_quarantine;
SELECT * FROM workspace.cdm.gold_baseline;
```

## 5. Run the broken-input demo

After step 4 has succeeded once:

```powershell
databricks jobs run-now <cdm-phase1-broken-demo job id>
```

**This run must fail**, at the pipeline task, on the `gate_results` expectation. The
failed gates must be exactly these (written by the ingest task's `--make-broken` log):

- `split_fingerprint_matches_phase0` (one HAM10000 image moved to another lesion)
- `no_missing_labels`, `class_counts_as_expected`, `diagnosis_mapping_matches_ham10000`
  (one HAM10000 label removed)
- `no_ham10000_image_in_a_site`, `no_duplicate_files` (one HAM10000 image slipped into
  Barcelona)

and `workspace.cdm_broken.silver_quarantine` must hold the one corrupt JPEG
(`ISIC_BROKEN_0000001`) without it failing anything. Because the gate table fails its own
expectation, read the gate details in the pipeline's event log (the failed flow's
expectation message), or from `silver_metadata` in `workspace.cdm_broken`.

## If something fails

- **ingest fails with "HAM10000_metadata.csv is missing" or an MD5 error:** rerun step 3.
- **ingest fails with "selection fingerprint ... does not match":** ISIC changed the
  Barcelona collection. Stop and tell Claude; do not edit the fingerprint by hand.
- **"N images failed to land":** a transient download problem; rerun the job. Landed files
  are kept and not downloaded again.
- **pipeline fails on `gate_results` in the real run:** a gate found a problem in real
  data. Nothing downstream ran. Report the gate name and detail; do not rerun hoping it
  passes.
- **gold fails installing torch:** check the task's environment log; torch comes from PyPI.

## What this has and has not been tested on

The ingest logic, gates, image decoding, gold calculations and deploy templates are tested
in CI without Databricks, including an end-to-end run of the ingest task and the gates on a
fake ISIC, and the broken-input demo's expected gate failures. The pipeline file itself
(`pipelines/lakehouse.py`), the CLI commands in `scripts/deploy_databricks.py` and the job
definitions have **not** run on Databricks yet; the first real run is their test.
