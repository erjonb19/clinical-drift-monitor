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
