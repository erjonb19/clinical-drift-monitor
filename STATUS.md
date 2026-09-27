# Status

Paused 2026-09-27 during Phase 0 (fix the science). No results exist yet: nothing has been
trained on real data.

## Done (committed, CI green)

- PLAN.md updated (8D report and runbook, CM study Oct 2; Phase 0 reproduces with one
  command on a GPU machine).
- Scaffold: package, ruff, mypy, pytest, GitHub Actions CI.
- `src/cdm/data.py`: HAM10000 from Harvard Dataverse with MD5 checks, metadata validation,
  lesion-level 70/15/15 split (5,229 / 1,120 / 1,121 lesions). PathMNIST at native 224 px
  and CIFAR-10 as fixed OOD subsets.
- `src/cdm/ood.py`, `src/cdm/eval.py`: Mahalanobis (nearest class, fit on train), max
  softmax, energy; AUROC and FPR@95TPR; mean and range over seeds.
- `src/cdm/train.py`, `src/cdm/reproduce.py`: class-weighted EfficientNet-B0 training and
  the one-command run (`python -m cdm.reproduce`). Exits non-zero on any AUROC below 50%.
- 19 tests: no lesion in two splits (synthetic and real metadata), scores point the right
  way, no refit when scoring, end to end on tiny synthetic images.
- Docs: data card (HAM10000 is CC BY-NC 4.0), silent-failures (3 entries),
  `docs/phase0-note.md` on why the notebook's 4.1% AUROC was wrong.

## In progress (stopped, local only, outside the repo in `C:\Users\Erjon\data\cdm`)

- HAM10000: complete. Checksums verified and 10,015 images extracted.
- PathMNIST `medmnist/pathmnist_224.npz`: **11.67 of 12.63 GB (92.4%)**, partial.
- CIFAR-10 `cifar10/cifar-10-python.tar.gz`: 119 of ~170 MB, partial. torchvision
  re-downloads it on the next run because the checksum will not match.
- A real-data CPU smoke run was started and stopped before it finished, so it is
  **not verified**: `prepare_ham` and extraction worked, but training and scoring on real
  images have not completed once.

## Exact next step

1. Resume the PathMNIST download in Git Bash:
   `curl -C - -L -o /c/Users/Erjon/data/cdm/medmnist/pathmnist_224.npz "https://zenodo.org/records/10519652/files/pathmnist_224.npz?download=1"`
2. From the repo, run the CPU smoke test on real data:
   `CDM_DATA=/c/Users/Erjon/data/cdm .venv/Scripts/python -m cdm.reproduce --limit 64 --epochs 1 --seeds 0 --n-ood 64 --workers 0`
   It writes to `results/smoke/` (gitignored). Fix anything it surfaces.
3. Then Claude writes the numbered Colab Pro steps for the 3-seed GPU run and you run them.
   The data folder should live on Google Drive so the 12.6 GB PathMNIST file is not
   downloaded again each session.
