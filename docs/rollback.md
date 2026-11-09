# Rolling the served model back (backlog item 2)

What serves is whatever version the `champion` alias of
`workspace.cdm.skin_lesion_classifier` points at (owner, 2026-11-06). Every challenger is
registered, promoted or not, with a `model_gen` tag and a `decision` tag (`promoted_by_rule` or
`rejected`), so the previous champion is always still in the registry to roll back to.

**Status:** the script and its checks are written and tested with a fake registry
(`tests/test_registry.py`). The registry holds one version today (v1, `model_gen=v2`, not yet
tagged), so there is nothing to roll back to until item 3's challenger is registered. No move
has been made with this script.

## Steps (the owner, from the laptop, with the repo's virtual environment)

1. See where the alias points: Catalog Explorer → `workspace.cdm.skin_lesion_classifier` →
   Aliases, or the dry run in step 3, which prints it.
2. If the target version has no `model_gen` tag, tag it first (registry v1 is the 3-seed
   ensemble the earlier phases call v2):

   ```bash
   python scripts/move_champion.py --tag 1 model_gen v2 --apply
   ```

3. Plan the rollback, say from v2 back to v1. Nothing changes without `--apply`:

   ```bash
   python scripts/move_champion.py --expect-current 2 --to 1
   ```

   It refuses if the alias is not at v2 (someone moved it), if v1 is missing, untagged or
   marked `rejected`.
4. Apply it. The script reads the alias back, and appends one line to
   `results/phase3/registry/alias_moves.jsonl`:

   ```bash
   python scripts/move_champion.py --expect-current 2 --to 1 --apply
   ```

5. Check the served answers now match v1: run the `cdm-phase3-check-registered-model` job
   (`databricks/check_registered_model_job.json`) with its `bundle` parameter set to v1's
   serving bundle (`/Volumes/workspace/cdm/raw/deploy/serving/d7191f4`). It loads
   `models:/workspace.cdm.skin_lesion_classifier@champion` and fails unless every label matches
   that bundle and every OOD score is within 1e-3.
6. Commit `alias_moves.jsonl` (set its `check_after` to the job's run ID and result) and the
   job's `registered_model_check.json`.

The API container serves a bundle file, not the alias; after a rollback, rebuild or redeploy it
from the same bundle (`serving_bundle` job) so the API and the registry agree.

A promotion is the same move forwards, but it must name the SHA-256 of the decision file that
allowed it (`--decision-sha256`); item 17's `promote_challenger` job is the planned way to do it.
