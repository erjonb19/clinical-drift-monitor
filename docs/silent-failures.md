# Silent failures

Every entry here is something that **reported success and was wrong**. That is the only
kind of bug this file collects: a crash announces itself, a green run with bad output
gets believed and built on.

Each entry ends with **the check that now catches it**.

---

## 1. Dataverse serves a different metadata file unless asked for the original

**Symptom.** `HAM10000_metadata` downloads from Harvard Dataverse with HTTP 200 and parses
as a table. Its MD5 is `1d13fed9...`, not the published `8f85fb1a...`: Dataverse
"ingested" the upload and, by default, serves its own re-export, tab-separated with every
string quoted. A loader that sniffed the delimiter would have worked on this and silently
diverged from the dataset of record the first time Dataverse changed its export.

**Cause.** Dataverse's access API returns the ingested `.tab` rendering of tabular files
unless the request adds `?format=original`.

**Caught by.** `cdm.data.HAM_FILES` requests `?format=original`, and `cdm.data.fetch`
checks every file's MD5 against the value published by Dataverse and raises `DataError`
on a mismatch, including for files already on disk.

---

## 2. Mahalanobis AUROC of 4.1% reported as a result (ECE 570 notebook)

**Symptom.** HAM10000 vs CIFAR-10 ran to completion and printed an AUROC of 4.1%. An
AUROC that far below 50% means the score ranks almost every OOD image as *more*
in-distribution than the in-distribution images, which is a sign the comparison is
broken, not a finding about the detector.

**Cause.** The in-distribution images were the same 1,002 images a 300-component whitened
PCA had just been fit on. That puts each of them about sqrt(300) from the centre by
construction (printed mean 16.49), while unseen images, CIFAR-10 included, project much
closer (printed mean 6.81). On top of that, in-distribution images were scored against
their *true* class mean and CIFAR-10 against the *nearest*, which is never farther.
[phase0-note.md](phase0-note.md) walks through the cells.

**Caught by.** `MahalanobisDetector.score` takes no labels, so every input is scored
against its nearest class mean; `test_mahalanobis_uses_nearest_class_not_true_class`
checks it against a brute-force minimum. In-distribution scores come from the held-out
test split. `cdm.reproduce` refuses to report an AUROC below 0.5 and exits non-zero.

---

## 3. PathMNIST scored in a different feature space (ECE 570 notebook)

**Symptom.** HAM10000 vs PathMNIST printed an AUROC of 43.4%, with no error.

**Cause.** PathMNIST ran through the ImageNet EfficientNet, not the fine-tuned one, and
PCA was refit on PathMNIST's own training images. Those points were then measured against
class means from HAM10000's PCA space. The two sets of distances were measured in
different spaces, so comparing them means nothing.

**Caught by.** `cdm.reproduce` extracts features for every set with the one fine-tuned
model it just trained. The detector is fit once on training features, and
`test_mahalanobis_statistics_do_not_change_when_scoring` fails if scoring changes the
PCA or class means.

---

## 4. Saved seeds were tied to the repository's latest commit, not the code's

**Symptom.** The first launch of the full Phase 0 run started cleanly, logged its settings
and began training. Nothing was wrong with what it was computing. But every saved seed
carried a run key built from `HEAD`, and a restart only reuses a seed whose key matches.
The next commit of any kind, including a STATUS.md edit or committing the finished seed
files themselves, would have changed `HEAD`. A restart after a crash would then have
refused every finished seed.

This one would not have produced a wrong number. The refusal is loud
(`ResumeMismatchError`). It is logged here because the run looked healthy while carrying
it, and the natural reaction to the refusal ("move it aside and rerun") would have thrown
away up to 10 hours of finished CPU training.

**Cause.** "Same code" was approximated by "same commit". Docs and results live in the
same repository, so the commit changes far more often than the code does.

**Caught by.** Review of the run design two minutes after launch, before anything was
committed; the run was stopped and relaunched. The run key now uses the last commit that
touched `src/` or `pyproject.toml` (`cdm.reproduce.CODE_PATHS`), and the uncommitted-code
guard covers the same paths. `test_resume_reuses_only_a_matching_seed` checks that a
matching key is reused and a changed one is refused. The docs commits made while the run
was training left its code version at `2ccda00`, which confirms the fix on the real run.

---

## 5. Near miss: the Phase 0 run saved its metrics but not its model

**Symptom.** The full 3-seed run finished cleanly (`done`, exit 0) and wrote every metric
it promised. When the CIFAR-10 Mahalanobis result needed checking, the fine-tuned weights
and features behind it did not exist anywhere. `reproduce.py` wrote JSON only. Every number
was real, but none could be re-examined without retraining, which costs about 4 hours per
seed on CPU.

Nothing reported was wrong, so this is a near miss rather than a silent failure. It is
logged because a "finished" run looked complete while missing what an investigation
needs, and the gap only showed when a result had to be checked.

**Cause.** The Phase 0 plan said weights are never committed, but not that they must be
kept. Claude meant to save them beside the data and never implemented it, and no test or
check asked for them.

**Caught by.** `run_seed` now saves the selected weights to
`<data root>/checkpoints/<results folder>/seedN.pt` and records the path and MD5 in the
seed's JSON. `test_checkpoint_is_saved_and_reloads` checks that the file matches its
recorded MD5 and loads into a fresh model. `scripts/phase0_cifar_check.py --checkpoint`
re-extracts features from a saved model.

---

## 6. A fixed seed did not reproduce a run

**Symptom.** Every Phase 0 seed ran with fixed seeds for Python, NumPy, PyTorch and the
data loader, and the results say "only the training seed varies". That reads as "rerun
the command, get the same numbers". Rerunning seed 1 with the same data, split and settings
gave a different model: balanced accuracy 81.0% instead of 76.4%, and CIFAR-10 Mahalanobis
AUROC 95.7% instead of 80.1%. Both numbers fall outside the reported 3-seed range. Nothing
warned that the seed was not controlling the result.

**Cause.** `run_seed` built the model before `train()` called `seed_everything`. Building
the model draws the new classifier head's initial weights, so the head came from whatever
random state the process was in. That state depended on which seeds had already run in
the same process. The reported seed 1 ran after seed 0; the rerun ran it first. The runs
differed from epoch 1 on (loss 1.1328 vs 1.1429). An earlier version of this entry guessed
at CPU threading; reading the diff between the two runs showed the real cause.

A second trap turned up while testing the fix: `torch.save` embeds the file name in the
archive, so identical weights saved as `a.pt` and `b.pt` have different MD5s. A file MD5
cannot answer "same model?".

**Caught by.** `test_same_seed_gives_identical_model_whatever_ran_before` trains seed 0,
then seed 1, then seed 0 again in one process, on tiny images with two data workers. It
requires the two seed 0 runs to have identical weights and metrics, and seed 1 to differ.
It fails when the old order (build, then seed) is put back. The fix, in code `0427b47`:
seed before building the model, fix the CPU thread count, turn on
`torch.use_deterministic_algorithms`, seed DataLoader workers from the seeded generator,
and record a SHA-256 of the weight values with every checkpoint. The reported Phase 0
seeds were not rerun; their run-to-run variation is reported in
[phase0-note.md](phase0-note.md).

## 7. Bordered-image counts measured on a silver format that no longer exists

**What looked right.** The tuning ladder's rung 0 evaluation (`results/phase2/ladder/rung0-baseline.json`,
code `050dd70`) counted "bordered" images (dark corners, `corner_brightness < 20`) per source:
Barcelona 2,264 of 5,000, Buenos Aires 27, HAM10000 2, MSK 1, PAD-UFES-20 0. Those counts were
quoted afterwards as the sites' border rates, including "HAM10000 has almost no bordered images".

**Cause.** Rung 0 ran on the old silver: 224 px, centre-cropped. The crop removes the corners
the test looks at. From rung 1a on, silver stores the full frame at 384 px, so many more images
show dark corners. On v2 silver the same test flags 2,974 of 15,277 training images: Barcelona
2,705, HAM Vienna 212, MSK 33, HAM Queensland 24 (`results/phase2/v2/training/data_summary.json`).
The rung 0 counts were right for the data they measured but were reused as if they described
the current silver. Every per-client bordered and non-bordered score from rung 1a on was
computed on the current silver, so those scores are consistent with each other.

**Caught by.** The v2 data summary, which counts bordered training images per client on the
silver the models actually train on. Bordered counts are now always reported with the silver
format they were measured on. Recomputed on current silver for every source
(`results/phase2/drift/borders.json`, drift run 802132583124668): Barcelona 4,771 of 9,999
(47.7%), Buenos Aires 513 of 1,616 (31.7%, not 27), HAM10000 328 of 10,015 (3.3%), MSK 44 of
3,728 (1.2%), PAD-UFES-20 0 of 2,298. Buenos Aires, a held-out site, is far more bordered
than the old count said, which matters when reading its drift scores.

## 8. A detection look whose results were never saved

**What looked right.** Drift-table run 802132583124668 scored eight of the nine final models,
about eight minutes each, and wrote a scores file per model to the volume. The job was still
"RUNNING" with no error showing.

**Cause.** `np.savez_compressed` wrote straight into the Unity Catalog volume. A zip writer
needs to seek back into the file, which volumes do not support, so each write failed after
about 24 KB, leaving an unreadable file. The per-model result JSON was written after the
scores file, so it was never written either; the error was caught and recorded as a failed
model, and the loop moved on. Each model's detection lock had already been written, so every
model's one detection look was used up with nothing saved. No metric was printed or saved, so
no one saw any detection result. Phase 1 had hit the same volume limit for large files and
works around it with a local copy first; the drift-table notebook did not.

**Caught by.** A one-off status check that listed the volume: score files of 24 KB, no result
JSON, all nine locks present. The run was cancelled during the ninth model. The fix
(`a421c55`) writes each scores file locally, checks it reads back with every array, then
copies it into the volume, and writes the metrics JSON before the scores file, so a failure
saving per-image scores cannot lose the metrics. The dry run had passed because a local disk
can seek; a volume cannot, so only the real run could show it.

Two more changes keep a crash from leaving used-up looks with nothing to show. Each lock now
has a status: `started` before the first test or shift image is scored, `completed` only after
the result JSON and scores file have been read back from the volume; a `started` lock without
a result stops a rerun instead of being repeated silently. And the first model of a job runs
outside the per-model error handler, with its files checked on the volume before any other
model starts. The temporary file uses Python's `tempfile` folder, not a hardcoded `/tmp`. The
nine used-up locks were not deleted: they are archived with a note (run ID, what broke, no
results saved) in `checkpoints/phase2/detection_looks_archive/802132583124668/`, and the
cancelled run's printed output, which has no detection numbers, is in
`results/phase2/drift/cancelled-802132583124668/stdout.txt`.

## 9. Lesion-level splits passed every gate while patients spanned train and test

**What looked right.** The `no_lesion_in_two_splits` gate passed on every pipeline update, and
the tests prove no lesion's images sit in two splits. The splits were reported as leak-free.

**Cause.** Leak-free by lesion is not leak-free by patient. One patient can have several lesions,
and the split assigns lesions, not patients. ISIC records `patient_id` for only some collections.
Among the training clients, only MSK collection 289 does, and there 47 patients have lesions in
more than one split (139 lesions). 30 of the 130 MSK test lesions that carry a patient ID
(55 images) belong to a patient who also has a train lesion. HAM10000, Barcelona and MSK
collection 287 record no patient ID, so patient-level overlap there **cannot be checked at all**.
The overlap is a weaker leak than the same lesion in two splits, but it is real: same skin, often
the same camera session, and the same age and sex. Two places where intervals or splits were
built by lesion at sites with patient IDs are affected:
- The Buenos Aires local recalibration split lesions 50/50, and 199 of its 623 patients have more
  than one lesion, so a patient's lesions can sit in both halves.
- Bootstrap intervals resampled by lesion treat a patient's lesions as independent.

**How much it moved the numbers.** This was measured on the analysis look's saved predictions,
with no new test look. For the shipped ensemble on the pooled client test split, dropping the 55
shared-patient images moves:
- balanced accuracy from 66.46% to 66.09%;
- melanoma sensitivity from 71.62% to 71.72%;
- macro AUROC from 90.90% to 90.76%.

Resampling by patient where known, and by lesion elsewhere, gives a balanced-accuracy interval of
[60.66, 73.01], against [60.24, 72.85] by lesion. On MSK alone (460 test images, 405 without) the
single seeds move by up to 4.2 points (seed 0: 59.04% to 54.80%). That is within the seed spread
there, and dropping images also changes MSK's class mix, so leakage and sample removal cannot be
separated. Evidence: `results/phase2/patient_overlap/patient_overlap.json`.

**Caught by.** The `patients_in_one_split` gate checks every pipeline update. Any patient with
lesions in two splits fails it unless it is one of the 47 documented in
`config/patient_overlap_v1.csv`. The splits were not changed, because re-splitting MSK would move
frozen v1 lesions and invalidate every Phase 2 result. `scripts/patient_overlap.py` recomputes the
counts from silver. From the research extension on:
- intervals resample by patient where ISIC records one;
- test results carry a row without these lesions.

---

## 10. Suppressed cohort counts that could still be worked out

**Symptom.** The cohort service's tests passed, and every count from 1 to 10 in its output
was labelled. Pre-merge reviews of G2 on 2026-10-22 found four ways to get a hidden count
back from one result. Results from the earlier code were released only for the seven
synthetic requests (#8 to #14), under the old labels; each was checked by hand against these
attacks, and none gives a count exactly (the tightest leaves five hidden cells between 1 and 3).
- An attrition step right after a hidden step was shown when nothing dropped in between, so
  `100, suppressed, 95` gave the hidden 95 away.
- When the final count was hidden, the breakdown cells were still shown, and they add up to it.
- Two small cells under a published total (`1, 1, 50` of 52) were each pinned at 1.
- The labels themselves: "<11" (1 to 10) and "suppressed" (11 or more) next to a total pinned
  counts (`1, 11, 100` of 112 makes the "<11" cell exactly 1). Hiding the next smallest cells
  instead could also be undone by anyone who knows the rule.

**Cause.** Each rule was checked against the case it was written for, not against a reader
who knows the rule and every number shown.

**Caught by.** `tests/test_cohort.py::test_no_small_count_can_be_worked_out_from_any_output`
tries every three-cell breakdown with counts up to 24 and every three-step attrition up to 39,
groups them by what is shown, and fails if any count from 1 to 10 is the same in every true
vector behind one output. The rules that pass it: one label for every hidden cell; a step's
drop counts from the last shown step; a breakdown with any small cell is hidden whole, and its
final count too if the hidden cells could still be worked out from it.

---

## 11. Two suppressed cohort results that together gave away a suppressed count

**Symptom.** The cohort service suppressed every result it released: counts from 1 to 10 show
as `<11`, and a second cell is hidden where subtraction would recover the first. Each result
passed that check. On 2026-10-21 two synthetic requests were answered in public issues:
- #8, melanoma by sex and age band: men aged 20 to 29 `<11`, women 16, no other sex row;
- #9, adults with melanoma and a histopathology procedure, by age band: 20 to 29 was 26.

The second cohort sits inside the first, so the band holds at most `<11` + 16 people in #8 and
at least 26 in #9. That makes the men exactly 10. A fresh reviewer found it before any merge.
The data is public ISIC metadata, so no one was exposed, but the service claimed a protection
it did not give.

**Cause.** Suppression looked at one result at a time. Nested cohorts released to different
requesters are the textbook differencing attack, and `phenotypes/README.md` already listed
it as "not detected; the reviewer checks". Here the reviewer was a script, and no one checked.

**Caught by.**
- Before release, `cdm.cohort.guard_releases` compares every number in a new result with every
  number already released for a nested definition (the same or fewer index conditions, an
  age range within, criteria a superset). Releases are recorded in the audit log, with the definition. Two kinds
  of group are hidden:
  - a group where the two results differ by 1 to 10 people;
  - a group where they are equal and one result shows exactly what the other hid.
- `tests/test_cohort.py` reproduces this case and a total that differs by four people.
- The #9 result was removed from the issue and posted again with the guard.
- Overlap that the definitions do not show as nesting is still a reviewer's check, and the README
  says so.

---

## 12. A champion validation score that differs from its committed value (open)

**Symptom.** The retraining run (job run 624636213399909) scored the champion ensemble at
68.45% balanced accuracy on the clients' 3,210 validation images. The same three checkpoints
scored 68.26% in Phase 2 on GPU (`results/phase2/v2/test/final_test.json`, `val_pooled`) and in
the serving run on CPU (`results/phase3/serving/results.json`). The run passed every check it
had: the wheel and file hashes, and each checkpoint's weight hash against the models file it
read from the volume. The promotion decision is not affected (against 68.26% the challenger's
change is -1.40 points, still past the -1.0 limit).

**Cause.** Not found. Candidates: a different image decode or transform path from the Phase 2
evaluation, or GPU arithmetic differences on near-tied predictions (0.19 points is a handful of
images per class). The run did not record the weight hashes it checked or the hash of the
models file, so `decision.json` alone cannot show which weights it scored.

**Caught by.** A fresh review comparing the decision with committed numbers. Not yet caught by
code: a retraining run should record the champion weight hashes and the models file's SHA-256 in
its decision, and refuse when the champion's validation score differs from the committed one by
more than a set tolerance. Until that is built, this entry stays open.

---

## 13. A reload check that named the wrong environment

**Symptom.** `results/phase3/lineage/registered_model_check.json` (job run 8856224820882) says the
registered model was loaded in "a fresh serverless CPU environment (pip requirements from the
model)". The job actually ran in its own environment with the cdm wheel installed, so the
model's packaged code and pins were never tested alone. The check passed while describing a
stronger test than it ran.

**Cause.** The notebook's `loaded_in` text was written for the intended design, not the job as
built.

**Caught by.** A review the same night (commit 00f9da9): the notebook now writes "this job's
environment (the cdm wheel is installed; the model's own pip environment was not rebuilt)", and
the README and `docs/BUILT_VS_PLANNED.md` state the correction. The committed file is logged to
MLflow and kept as it ran. A true isolated reload check is not built.

---

## 14. LLM replies cut off at the token limit, scored as "invalid"

**Symptom.** The first G5 dev run (8 notes, 2026-10-23) gave Claude Haiku 2 of 8 correct on
diagnosis and 1 of 8 on body site, below the keyword baseline (counts read from that run's
output; the rerun overwrote its predictions file, and the 200-token spend entries remain in
`~/.cdm/llm_spend.jsonl`). Six of the eight replies were
empty or stopped mid-word (`"confirmation": "histop`). The run reported success, and the parser
turned each cut reply into "invalid", which scores as a wrong answer.

**Cause.** `MAX_OUTPUT_TOKENS` was 200. Every cut reply had used exactly 200 output tokens while
its visible text was a few dozen, so the model's thinking used up the budget before the answer.
The transport never read `stop_reason`, and the cut replies were cached, so a rerun would have
replayed them.

**Caught by.** The "too bad" rule (EVAL_PLAN section 6): the dev numbers were looked at before
any scoring run. Now `MAX_OUTPUT_TOKENS` is 2048; the transport records `stop_reason` and
`llm_extract` raises `ReplyCutOff` (billed, never parsed or cached) when a reply hits the limit
(`tests/test_extract.py::test_a_reply_cut_off_at_the_token_limit_fails_and_is_not_cached`); and
the cache key holds the output limit, so the old cut replies are never reused. No scoring note
had been run.
