# Phase 1 note: the lakehouse on Databricks

Phase 1 turns the ISIC Archive and HAM10000 into a medallion lakehouse on Databricks Free
Edition: an ingest job task, a Lakeflow Declarative Pipeline (bronze and silver, with gates)
and a gold job task. Every number here is from `results/phase1/run_evidence.json`, written by
`scripts/phase1_export_evidence.py`, unless it is quoted from a Databricks error message.

## Outcome

**One command rebuilt bronze to gold, and all 11 gates passed on real data.** Job run
`606896740511548` (code `4f320a7`), 2026-09-29, 21:19 to 21:54 UTC: ingest 13 min, pipeline
7 min, gold 15 min, all SUCCESS.

| Source | Bronze files | Silver rows | Labelled | Unique manifest SHA-256 | Silver images | Quarantined | Gold embeddings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HAM10000 | 10,015 | 10,015 | 10,015 | 10,015 | 10,015 | 0 | 10,015 |
| Barcelona | 5,000 | 5,000 | 4,440 | 5,000 | 5,000 | 0 | 5,000 |
| Buenos Aires | 1,616 | 1,616 | 1,616 | 1,616 | 1,616 | 0 | 1,616 |
| MSK | 3,728 | 3,728 | 3,365 | 3,728 | 3,728 | 0 | 3,728 |
| PAD-UFES-20 | 2,298 | 2,298 | 2,298 | 2,298 | 2,298 | 0 | 2,298 |

Barcelona is exactly 5,000 at every layer, the pinned selection. Every gate row has 0
violations, and `split_fingerprint_matches_phase0` reads "got cc2b196cd5bf58ad, expected
cc2b196cd5bf58ad": the Phase 0 lesion split is reproduced exactly from ISIC's copies of
HAM10000 (train / val / test: 7,044 / 1,443 / 1,528 images, 5,229 / 1,120 / 1,121 lesions).

**A broken input failed a gate loudly.** Job run `161990905473833` (code `c74e24a`) built a
small faulted copy (335 HAM10000 images from 30 whole lesions per class, 20 images per site,
plus the faults) and its pipeline update failed on the `gate_results` expectation.

## The broken-input demo: which gates failed, and how that is known

Databricks surfaced **1** failing gate, because an expect-or-fail stops at the first
violating row:

```
[EXPECTATION_VIOLATION.VERBOSITY_ALL] Flow 'workspace.cdm_broken.gate_results' failed to meet
the expectation. Violated expectations: 'gate_passes'. ... Output record:
'{"name":"split_fingerprint_matches_phase0","violations":1,"detail":"got f68c84e0ec16beea,
expected db50c5237ef36a16"}'
```

The **other 5** were confirmed by running `cdm.gates.run_all` on
`workspace.cdm_broken.silver_metadata` (built before the gate stopped the update) with the
demo's own `broken/sites.json`. Exactly the six gates the faults were built to trip fail, and
the other five pass:

| Gate | Violations | Detail | Fault |
| --- | ---: | --- | --- |
| split_fingerprint_matches_phase0 | 1 | got f68c84e0…, expected db50c523… | a HAM10000 image moved to another lesion |
| class_counts_as_expected | 1 | bcc: got 48, expected 49 | a HAM10000 label removed |
| no_missing_labels | 1 | ISIC_0024331 | a HAM10000 label removed |
| diagnosis_mapping_matches_ham10000 | 1 | ISIC_0024331 | a HAM10000 label removed |
| no_ham10000_image_in_a_site | 1 | ISIC_0024332 | a HAM10000 image slipped into Barcelona |
| no_duplicate_files | 2 | ISIC_0024332 twice | a HAM10000 image slipped into Barcelona |
| split_computed, no_lesion_in_two_splits, license_and_attribution, every_image_landed, files_match_manifest | 0 | | |

The corrupt JPEG (`ISIC_BROKEN_0000001`) is in `workspace.cdm_broken.silver_quarantine`
("cannot decode image: Truncated File Read") and failed no gate, as designed.

## What went wrong on the way

**1. The first pipeline run failed analysis (run `330838862474565`, code `62b1b45`).**
Ingest landed all 22,657 images; the pipeline then failed before writing any table, in 6
updates (the original plus 5 retries):

```
[INTERNAL_ERROR] Found the unresolved operator: 'Project [unresolvedordinal(1) AS
unresolvedordinal(1)#12503, source#12422, isic_id#12421, ...]
  File ".../cdm/62b1b45/lakehouse.py", line 111, in silver_metadata
    return joined.groupBy(F.lit(1)).applyInPandas(
```

Spark reads an integer literal in `groupBy` as a column position, so `groupBy(F.lit(1))`
could not be resolved; `gate_results` used the same pattern. Fixed in `4f320a7` by grouping on
a named constant column. `tests/test_pipeline_source.py` inspects the parsed pipeline for that
pattern and for eager Spark actions; on the failing version it flags exactly lines 111 and
158. **No gate or expectation was loosened, skipped or removed**: `src/cdm/gates.py` and
`src/cdm/splits.py` are unchanged between `62b1b45` and `4f320a7`, and both expectation
decorators are untouched. This failed loudly, so it is recorded here rather than in
`silent-failures.md`.

**2. Ingest now reuses stored hashes on reruns.** Reading a file back from a Unity Catalog
volume runs at about six files per second, so re-hashing every landed image made each rerun
cost about an hour. From `4f320a7`, ingest hashes downloaded bytes in memory and, for a file
already in the volume, reuses the SHA-256 recorded in the previous manifest instead of reading
the file again. The rerun in `606896740511548` downloaded 0 images (its ingest task log). The independent check is
the `files_match_manifest` gate: the pipeline computes the SHA-256 of every file in bronze
itself and fails if any differs from the manifest. It passed with 0 mismatches over all
22,657 files.

**3. Pipelines retried deterministic failures.** Databricks retries a failed pipeline update
by default. The first broken demo (run `561727450716511`) made 6 failed updates, the original
plus 5 retries, each repeating the same gate failure, over about 18 minutes. From `c74e24a`
both pipelines set `pipelines.numUpdateRetryAttempts` to 0; the rerun (`161990905473833`) made
exactly 1 failed update and stopped, with its pipeline task taking under 3 minutes.

**4. Smaller issues.** `databricks fs cp` does not create folders inside a volume (deploy
fixed in `62b1b45`). Claude's first estimate of the image total, 26,600, was an addition error;
the per-source numbers were right and the total is 22,657. The Databricks CLI's generic `api
post` returned "Not Found" for the SQL statement endpoint, so the export script calls the REST
endpoint directly with the CLI's login token.

## Findings to carry into Phase 2

- **Compression is not a hidden site signature.** ISIC serves every source at the same JPEG
  quantisation level (29.03 on every image in silver).
- **Resolution is.** HAM10000 (600 px wide) and Barcelona (1,024 px) are fixed sizes; Buenos
  Aires spans 162 to 4,128 px, MSK 639 to 6,828 px and PAD-UFES-20 147 to 3,474 px. All are
  resized to 224 in silver, but the source resolution differs by site.
- **With pretrained ImageNet features, every site scores above HAM10000.** Median Mahalanobis
  score: HAM10000 218.2, MSK 391.0, Buenos Aires 467.3, Barcelona 529.7, PAD-UFES-20 573.5
  (HAM10000's held-out baseline: median 221.4 on test, 216.6 on validation). These use model
  version `imagenet-effb0`, not a fine-tuned model; Phase 2 recomputes them.
- **Skin type is thin**, as expected: it is recorded only for Buenos Aires and PAD-UFES-20,
  with very few images of the darker types (see `gold_feature_stats` in the evidence file).
