# Architecture

One page: what runs where, how it scales, and where it breaks. Every number is from a committed
file named beside it. A research prototype, **not for clinical use**. Dashed lines are planned
or hand-run steps, not automated ones.

```mermaid
flowchart LR
  ISIC[ISIC Archive<br/>public images] -->|Lakeflow pipeline,<br/>quality gates| LAKE[(Delta: bronze, silver, gold<br/>Unity Catalog)]
  LAKE --> TRAIN[Training on Databricks<br/>3 seeds, by-lesion splits]
  TRAIN --> BUNDLE[Serving bundle<br/>hashed manifest]
  BUNDLE --> REG[Unity Catalog registry<br/>v1, alias champion]
  BUNDLE --> API[FastAPI on CPU, Docker<br/>input contract, abstain flag]
  BUNDLE --> MONTH[Monthly Lakeflow job<br/>drift test, targets]
  MONTH --> AUDIT[(gold_prediction_audit)]
  MONTH -. ALERT on a failed run<br/>wired, not deployed .-> OWNER[Owner]
  MONTH -. trigger planned, item 17 .-> RETRAIN[Retraining job<br/>challenger vs champion]
  RETRAIN -. owner runs promote_challenger<br/>by hand, never run .-> REG
  CI[GitHub Actions<br/>tests, eval gate, image scan] -. checks .-> API
```

## What runs where

- **Databricks Free Edition (serverless, $0):** the data pipeline (Delta tables, data-quality
  gates that fail the update), training, the serving bundle, the registry and the monthly drift
  job. The retraining loop ran once, by hand, and did not promote; its alert trigger, the bundle
  rebuild and the registry move are not built (`docs/BACKLOG.md` item 17). Job results are
  logged as MLflow runs by hand (`results/mlflow_runs.json`; a test checks logged files are not
  stale, not that every file is logged), and DBUs are in `results/phase3/costs/result_costs.json`
  where billing reached the table.
- **The API (`app/api`):** FastAPI and PyTorch on CPU in a Docker image, loading the bundle file
  (not through the registry). CI builds and scans it on every push. Not deployed.
- **GitHub Actions:** ruff, mypy, pytest, the eval gate (committed metrics against
  `config/eval_gate.json`; it reads files and runs no model), secret scanning, the image scan.

## How it scales

- **Data:** 27,656 images, five datasets from six sites
  (`results/phase2/v2/data_checks.json`). The monthly job replays 300 images a month
  (`results/phase3/monthly/spec.json`).
- **Serving (item 13, `results/phase3/serving/benchmark.json`):** with an untrained one-network
  test bundle, no Docker and a CUDA torch build on a 4-CPU shared VM, the API serves about 44 to
  53 requests a second from 1 to 20 users while median latency grows from 18.4 ms to 458.6 ms.
  Throughput is flat from one user, so the machine is likely saturated already (CPU use was not
  measured). That is serving speed only, not the shipped model.
- **The shipped model runs three networks per image:** the ensemble's PyTorch median is
  288.6 ms per image at batch size 1 on Databricks serverless CPU
  (`results/phase3/onnx/results.json`). The test-bundle benchmark's 18 ms per whole request on
  other hardware is about 5 times faster per network; the gap is not explained. The champion
  itself, on a 4-CPU GitHub runner (`results/phase3/serving/benchmark_champion.json`): median 158.8 ms per request for one user,
  rising to 1,365.2 ms at 20 users, while throughput rises only from 5.97 to 14.21 images a
  second. The bottleneck is CPU compute, measured: in a second run the server used 3.95 of 4
  cores at every level and the machine was over 99% busy
  (`results/phase3/serving/benchmark_champion_cpu.json`, a different runner CPU, so its times
  differ).
- **Not measured yet:** batch scoring at scale (item 14; the owner chose a 5,000-image slice
  first) and detector quality on staged shifts (item 15; approved, code and spec committed, not yet
  run; the champion bundle it needs is a release asset since 2026-11-18).

## Where it breaks

1. **CPU saturation.** With the one-network test bundle, latency grew in step with concurrent
   users from the first one. With the champion's three networks on 4 CPUs, throughput still
   rose from 1 to 20 users, but only 2.4 times while median latency rose 8.6 times (item 13). Likely levers, not measured: more processes with fewer threads each, or a
   lighter model. ONNX failed its parity check (`results/phase3/onnx/results.json`).
2. **Drift scores depend on hardware.** Scores recomputed on CPU differed from the GPU fit, so
   the baseline and every monitored month are scored on the same hardware
   (`docs/model-card.md`, section 5).
3. **Labels arrive late.** The melanoma target needs each labelled month's referral count; it
   reads "insufficient data" in all 12 replay months until they are backfilled, and no labels
   arrive after the replay (`results/phase3/monthly/slo_replay.json`, `config/slo.json`).
4. **Skin type is recorded only at the new sites,** so a per-skin-type alert also measures the
   site shift (`results/phase3/monthly/`).
5. **Free Edition limits:** serverless only, with restricted outbound access. Archive-scale
   batch scoring (an estimate, not a measurement: about 130 GB of images for 50,000, item 14)
   may meet limits whose size is unknown.
6. **The detector is one score.** An alert says the month looks unlike validation, not why;
   the response is a human look and, with labels, retraining.
