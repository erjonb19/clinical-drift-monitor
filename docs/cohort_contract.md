# Cohort contract (research Phase F): draft for review

How [governed-clinical-agent](https://github.com/erjonb19/governed-clinical-agent) could
answer cohort questions over this project's OMOP tables. This is a plan only: none of it is
built. Phase F is done when the owner has reviewed this document and the agent repo's plan
notes the item.

It is a portfolio design on public research images (ISIC Archive and HAM10000), not a
clinical system.

## 1. What changed since the first plan

The first plan (`docs/research_extension_plan.md`, Phase F) had the agent query a new view
through parameterized SQL templates. Phase G2 has since built a **governed cohort service**
(`src/cdm/cohort.py`). It already does the hard parts:
- an allowlisted phenotype language, with no SQL;
- approval of each exact definition by its hash;
- small-cell suppression within a result and across nested results;
- a hash-chained audit log.

So the contract below makes the service the **only** way the agent reaches the OMOP tables.
The agent never writes SQL for cohorts. This also removes the need for a new view.

## 2. The tables behind the service

Generic OMOP CDM v5.4 tables, as the service reads them (`OMOP_TABLES` in `cdm.cohort`), in
the workspace as `workspace.cdm.omop_*`:

| Table | Used for | Caveats the agent must repeat when relevant |
| --- | --- | --- |
| `person` | sex (`gender_concept_id`), year of birth | A person is a real patient only where ISIC records a patient ID; elsewhere one person is one lesion |
| `condition_occurrence` | the index diagnosis; age at diagnosis | Dates are placeholders (2000-01-01), so "age" is ISIC's 5-year `age_approx` |
| `procedure_occurrence` | histopathology, imaging procedures | Histopathology is the only confirmation method with a concept |
| `image_occurrence` | body site (`anatomic_site_concept_id`), modality | Modality is concept 0: no DICOM vocabulary was available |

Column meanings and null rates are in [data_dictionary.md](data_dictionary.md).

## 3. The interface

The agent gets two new tools behind its policy engine (`medicare_policy.yaml`-style tiers):

| Tool | Input | Tier | What happens |
| --- | --- | --- | --- |
| `cohort_count` | the name and version of a library phenotype | autonomous | The service runs it if it is approved, and refuses otherwise |
| `cohort_propose` | a structured definition in the phenotype language (index concepts, age range, include/exclude concept lists, breakdown) | gated | It is written as a new version and queued for a human approval by hash. Nothing runs until a reviewer approves it |

- **Plain English to a definition.** The agent maps a question to a library phenotype if one
  fits, or drafts a definition for `cohort_propose`.
- **Concept IDs.** Concept IDs come only from `config/omop_concepts.csv`. An ID outside the map
  is refused (the same rule as G5's extractor).
- **Output.** The output is the service's result: attrition and breakdown, already suppressed.
  The agent may not ask for row-level data; neither tool returns it.

## 4. Groundedness and suppression in answers

The agent's groundedness check (`groundedness.py`) gets two rules for cohort answers:
1. Every number in the answer must equal a number in the service's result for that request.
2. A cell shown as `suppressed` is reported as "suppressed (a small count)". It is never
   estimated, bounded ("fewer than 11"), or derived by subtraction from other cells. A bound is
   itself a disclosure (silent-failures 10).

Both requests and answers are logged: the agent's JSONL audit log records the decision, and the
service's chained log records the run.

## 5. Worked example

"How many melanoma patients over 60 had a lesion on the back?"
- **Diagnosis:** the three melanoma concepts (NOS 4162276, invasive 141232, in situ 40391314).
  In situ must be listed: it is not under the NOS concept.
- **Body site:** back. The map has posterior trunk (4280364) and trunk (4313445); the agent asks
  whether "back" includes "trunk, side unspecified" instead of choosing silently.
- **Age:** `age_approx` is in 5-year bins, so "over 60" means 60 and above or 65 and above.
  The agent shows the choice.
- **"Patients":** at HAM10000 and some ISIC collections a person is a lesion, and the answer
  says so.
- **Result:** no library phenotype fits, so `cohort_propose` drafts one. The answer is "pending
  approval", with the definition's hash. After approval, `cohort_count` runs it.

## 6. What the contract cannot answer

- Anything about dates, time to event, or follow-up: the dates are placeholders.
- True patient counts where ISIC records no patient ID.
- Images or row-level data.
- Overlap between definitions that are not nested in a way the definitions show. The service
  checks nested ones only; a reviewer must catch the others.

## 7. Open questions for review

1. Should library phenotypes run autonomously, or should every agent request be gated?
2. Should `cohort_propose` be allowed at all, or should the agent only pick from the library?
3. Should a concept-ancestor subset (descendants of melanoma, for example) be committed, so
   definitions can say "and descendants"? The Athena download can produce one, as IDs only.
