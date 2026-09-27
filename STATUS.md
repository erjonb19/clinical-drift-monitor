# Status

Updated 2026-09-27 18:55. Phase 0 (fix the science): the full 3-seed run is **training
locally on CPU**, not on Colab (PLAN.md updated). No results yet.

## Done (committed, CI green)

- Code for every Phase 0 item: checksummed data, lesion-level split, EfficientNet-B0
  training, Mahalanobis / max softmax / energy, AUROC and FPR@95TPR, mean and range over
  seeds, one command (`python -m cdm.reproduce`).
- A full run saves each seed as it finishes and resumes from saved seeds on restart. It
  refuses to reuse a seed made by different code, data or settings, and refuses to start
  from uncommitted code.
- 20 tests: no lesion in two splits, scores point the right way, no refit when scoring,
  resume rules, end to end on tiny synthetic images.
- Docs: data card, silent-failures (3 entries), `docs/phase0-note.md` on the old 4.1% AUROC.
- All data downloaded and checksums verified: HAM10000, PathMNIST 224 px (12.63 GB),
  CIFAR-10. It lives in `C:\Users\Erjon\data\cdm`, outside the repo.
- Real-data smoke run (256 images per split, 1 epoch) passed end to end: exit 0, no
  AUROC below 50%. Its numbers are plumbing checks, not results, and are not committed.

## In progress: the full run

- Started 2026-09-27 18:26 as a detached process (not tied to any terminal), code version
  `2ccda00`, 12 epochs x seeds 0, 1, 2, all 10,015 images, 2,000 images per OOD set.
- Log: `logs\phase0_run.log`. Results: `results\phase0\` (`seedN.json` per finished seed,
  `summary.json` and `results.md` rewritten after each seed).
- Measured speed: epoch 1 took 25 min (val balanced accuracy 70.5% after one epoch).
  That is about 5.2 h per seed, so the whole run should finish **around 10:00 on
  2026-09-28**, not overnight. Estimated finishes: seed 0 about 23:40, seed 1 about 04:55,
  seed 2 about 10:10.
- The process asks Windows not to sleep, but that does not cover closing the lid. Keep the
  laptop plugged in with the lid open.

## Check progress in the morning

Open PowerShell and run:

```powershell
cd C:\Users\Erjon\clinical-drift-monitor

# 1. Still running? Prints process IDs if yes, nothing if it has stopped.
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object CommandLine -like '*cdm.reproduce*' | Select-Object ProcessId, CreationDate

# 2. Latest progress lines (the log also holds library warnings; this filters to ours).
Select-String -Path logs\phase0_run.log -Pattern '^\[' | Select-Object -Last 8 | ForEach-Object Line

# 3. Finished seeds, and the results table so far.
Get-ChildItem results\phase0 -Filter seed*.json
Get-Content results\phase0\results.md
```

How to read it:
- The last log line is `done`: all three seeds finished and every AUROC is at least 50%.
- `IMPLAUSIBLE RESULTS`: finished, but some AUROC is below 50%. Do not report it; that
  gets investigated first.
- Nothing is running and the log ends in a `Traceback` or mid-epoch: it crashed or the
  laptop slept. Restart it (below). Finished seeds are kept.

## Restart after a crash (resumes from the last finished seed)

In a Command Prompt (`cmd`), not PowerShell, so the log stays in one encoding:

```bat
cd /d C:\Users\Erjon\clinical-drift-monitor
set CDM_DATA=C:\Users\Erjon\data\cdm&& set PYTHONIOENCODING=utf-8&& .venv\Scripts\python.exe -u -m cdm.reproduce --workers 4 >> logs\phase0_run.log 2>&1
```

Leave that window open until it finishes. The seed that was in progress restarts from
epoch 1; finished seeds are loaded, not retrained.

## Next step after the run finishes

Claude checks `results/phase0/` for sanity, commits `results/phase0/` (JSON and
`results.md`), fills the README results table from it, links the corrected numbers from
`docs/phase0-note.md`, updates `docs/BUILT_VS_PLANNED.md`, and ticks the Phase 0 boxes in
PLAN.md. Then you confirm Phase 0 is done before Phase 1 starts.
