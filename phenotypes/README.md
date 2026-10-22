# Phenotype library

Versioned cohort definitions for the governed cohort service (`src/cdm/cohort.py`,
`scripts/cohort.py`). Each definition lives at `<name>/v<version>.yaml`. A definition is
never edited after it is approved: a change is a new version, and needs a new approval.

These cohorts run on **public research images** (HAM10000 and ISIC Archive) in OMOP form,
not on patient records. Concept IDs come from `config/omop_concepts.csv`. Their names stay in
the workspace's concept table, because SNOMED text is not published here.

## Definitions

| Phenotype | Question it answers |
| --- | --- |
| `melanoma_any` | How many people have any melanoma? In situ is listed separately, because the general melanoma concept does not include it. |
| `melanoma_histopathology_adults` | Trial feasibility: adults with melanoma and a histopathology procedure |
| `keratinocyte_cancer_head_neck` | Basal or squamous cell carcinoma with an image of the head or neck |
| `nevus_without_histopathology` | Nevi with no biopsy: the cohort whose labels are weaker |
| `melanoma_under_18` | Pediatric melanoma feasibility. Small by design: it shows suppression |
| `melanoma_dermoscopy` | Proposed for a commercial image export (synthetic issue #14); rejected by the stand-in reviewer |

**Authorship (corrected 2026-10-22).** Claude Code drafted all six definitions. The first four
were written for the library and later used for synthetic requests #8 to #11; the last two were
written for requests #13 and #14. Their v1 files named the owner as author in error; v2 files
say who drafted them and are otherwise identical. Approved files are never
edited, so v1 stays as it was. The stand-in decisions were recorded again for the v2 hashes in
the synthetic log. The owner's own decisions go to `phenotypes/approvals.jsonl`.

## The language

Only these keys are accepted. Anything else is refused before a query is built.

```yaml
name: melanoma_histopathology_adults   # must match the folder
version: 1                             # must match the file name
title: Adults with melanoma and a histopathology procedure
purpose: why the cohort is needed
author: who wrote it
index:                                 # the qualifying condition; age is measured here
  condition_concept_ids: [4162276, 141232, 40391314]
  age_min: 18                          # optional, integer years
  age_max: 90                          # optional
include:                               # each: one key, a list of concept IDs
  - procedure_concept_ids: [43531286]
exclude:
  - anatomic_site_concept_ids: [4263673]
breakdown: [gender_concept_id, age_band]   # optional; only these two
```

Criterion keys:
- `condition_concept_ids`
- `procedure_concept_ids`
- `anatomic_site_concept_ids`
- `modality_concept_ids`
- `gender_concept_ids`

Criteria are person-level, as in OMOP cohort tools. A person who has the index condition
on one lesion and a procedure on another meets both.

## Workflow

1. **Request.** An investigator opens a request issue from the form. The question, cohort,
   data needed and approval status are recorded there.
2. **Define.** The data engineer writes `phenotypes/<name>/v1.yaml`. Check it with
   `python scripts/cohort.py check`, which prints its hash and approval status.
3. **Approve.** A reviewer records a decision on that exact definition:
   `python scripts/cohort.py decide <file> approve --reviewer "<name>" --reason "<why>" --request <issue URL>`.
   - Decisions go to `phenotypes/approvals.jsonl`.
   - `escalate` hands the item to someone else and does not approve it.
   - The synthetic request exercise keeps its stand-in decisions apart, with
     `--approvals results/research/g2_cohort/synthetic_approvals.jsonl`.
4. **Run.** `python scripts/cohort.py run <file> --requester "<role>" --request <issue URL> --engine databricks`.
   - The output is an attrition table (people left after each step) and the breakdown.
   - Counts from 1 to 10 show as `suppressed`. An attrition step is also hidden while its drop
     from the last shown step is from 1 to 10.
   - If any cell of a breakdown is small, every non-zero cell of it is hidden, and so is the
     final count if the hidden cells could still be worked out from it. Ask for a coarser
     breakdown (age band only, say) to see more.
   - Every hidden cell has the same label. Two labels ("under 11", "11 or more") and hiding
     only the next smallest cells were both tried first. A review and a brute-force test
     (`tests/test_cohort.py`) showed that either lets a reader who knows the rule pin a count.
5. **Audit.** Every refusal, decision and run is appended to a hash-chained log.
   `python scripts/cohort.py verify` checks the chain.

## Limits

- Differencing across requests is checked only where the definitions show nesting: the same
  or fewer index conditions, an age range within the other's, and a superset of its criteria.
  Then a
  group that differs from an earlier release by 1 to 10 people, or that would pin a cell the
  other hid, is hidden (`docs/silent-failures.md` 11). Other overlap, such as two different
  site lists, is not detected; a reviewer checks for it when approving similar definitions.
- The guard recomputes earlier releases' true counts from the tables as they are now. If the
  tables are rebuilt with different data, its comparisons are against the new data, not what
  was released; rerun the earlier definitions after a rebuild.
- Results released before 2026-10-22 (the synthetic issues #8 to #14) used two labels, "<11"
  and "suppressed" (11 or more). Every hidden cell is now just "suppressed".
- In this data, a person is a real patient only where ISIC records a patient ID. Elsewhere one
  person is one lesion (`docs/data_dictionary.md`).
- People with no recorded sex have `gender_concept_id` 0, a group of their own in a
  breakdown.
- The audit log's hash chain catches an edited, reordered or deleted line, but not the last
  lines being cut off: nothing outside the log records its head. Git history is the backstop.
- `approvals.jsonl` is not hash-chained. Each decision is also written to the audit log, and
  git history shows who changed the file.
