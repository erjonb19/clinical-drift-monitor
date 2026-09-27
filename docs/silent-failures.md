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
