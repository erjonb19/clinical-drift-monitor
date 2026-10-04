"""Phase 1 data quality gates as plain functions over pandas frames.

The pipeline turns each result into a row of a gate table with an expect-or-fail constraint,
so a violation stops the pipeline update. Keeping the logic here means CI tests exactly the
checks that decide whether a pipeline run fails.

Frames use the silver column names: ``isic_id``, ``source``, ``lesion_id``, ``split``,
``label``, ``isic_label``, ``license``, ``attribution``, ``sha256`` (computed in the
pipeline; null if no file landed) and ``manifest_sha256`` (recorded by the ingest task).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from cdm.splits import HAM, SPLITS, assign_splits, site_labels, split_fingerprint

NEEDS_ATTRIBUTION = ("CC-BY", "CC-BY-NC")


@dataclass(frozen=True)
class Gate:
    name: str
    violations: int
    detail: str


def _examples(values: Iterable[object], limit: int = 5) -> str:
    items = [str(v) for v in values]
    return ", ".join(items[:limit]) + (
        f" (+{len(items) - limit} more)" if len(items) > limit else ""
    )


def latest_joined(
    metadata: pd.DataFrame, manifest: pd.DataFrame, landed: pd.DataFrame
) -> pd.DataFrame:
    """Latest metadata row per image, with the manifest's SHA-256 and the landed file's.

    The pipeline does this step in Spark; tests use this pandas version on the same inputs.
    ``landed`` has ``source``, ``isic_id`` and the ``sha256`` computed from the file in bronze.
    """
    key = ["source", "isic_id"]
    latest = metadata.sort_values("run_id").drop_duplicates(key, keep="last")
    recorded = (
        manifest.sort_values("run_id")
        .drop_duplicates(key, keep="last")[[*key, "sha256"]]
        .rename(columns={"sha256": "manifest_sha256"})
    )
    frame = latest.merge(recorded, on=key, how="left").merge(
        landed[[*key, "sha256"]].drop_duplicates(key), on=key, how="left"
    )
    return frame.reset_index(drop=True)


def roles(config: Mapping[str, Any]) -> dict[str, str]:
    """Source to role: HAM10000 and the configured client sites are ``client``; every other
    site is ``held_out``."""
    sites = config.get("sites", {})
    return {HAM: "client", **{name: spec.get("role", "held_out") for name, spec in sites.items()}}


def frozen_splits(config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Frozen lesion splits (source -> lesion -> split) that the caller loaded into
    ``config["frozen_splits"]["lesions"]`` from the file the config names; empty if none."""
    return dict(config.get("frozen_splits", {}).get("lesions", {}))


def label_rule(config: Mapping[str, Any]) -> list[str]:
    return list(config.get("label_policy", {}).get("unlabelled_at_sites", []))


def with_splits(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Apply the site label rule, then add ``split``, ``client`` and ``role``.

    Shared by the pipeline (inside applyInPandas) and tests. HAM10000's client is its
    institution, from HAM10000's own ``dataset`` column (``ham_dataset``).
    """
    frame = frame.reset_index(drop=True)
    frame = frame.assign(label=site_labels(frame, label_rule(config)))
    role = roles(config)
    clients = [source for source, r in role.items() if r == "client"]
    frame = frame.assign(
        split=assign_splits(
            frame, config["ham10000"]["split_seed"], clients, frozen_splits(config) or None
        )
    )
    by_dataset = config["ham10000"].get("clients_by_dataset", {})
    ham_client = frame.get("ham_dataset", pd.Series(None, index=frame.index)).map(by_dataset)
    client = frame["source"].where(frame["source"] != HAM, ham_client)
    return frame.assign(client=client, role=frame["source"].map(role))


def silver_frame(
    metadata: pd.DataFrame, manifest: pd.DataFrame, landed: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    """The frame the gates check: every ingest run's metadata and manifest rows, plus the
    SHA-256 of each landed file, reduced to the latest row per image, labelled and split."""
    return with_splits(latest_joined(metadata, manifest, landed), config)


def split_computed(frame: pd.DataFrame) -> Gate:
    """The lesion split could be computed (stratification needs enough lesions per class)."""
    bad = frame[frame["split"] == "unsplittable"]["isic_id"]
    return Gate("split_computed", len(bad), _examples(bad))


def frozen_splits_kept(frame: pd.DataFrame, frozen: Mapping[str, Mapping[str, str]]) -> Gate:
    """Every frozen (v1) lesion is still present and every labelled image of it keeps its
    split, so adding data never moves an earlier lesion between train, val and test."""
    if not frozen:
        return Gate("frozen_splits_kept", 0, "no frozen splits configured")
    bad: list[str] = []
    for source, lesions in frozen.items():
        rows = frame[(frame["source"] == source) & frame["label"].notna()]
        got: dict[str, set[str]] = {}
        for lesion, split in zip(rows["lesion_id"], rows["split"], strict=True):
            got.setdefault(str(lesion), set()).add(str(split))
        for lesion, split in lesions.items():
            if lesion not in got:
                bad.append(f"{source}/{lesion} missing")
            elif got[lesion] != {split}:
                bad.append(f"{source}/{lesion} {sorted(got[lesion])} != {split}")
    kept = sum(len(v) for v in frozen.values())
    return Gate("frozen_splits_kept", len(bad), _examples(bad) or f"{kept} lesions kept")


def lesions_in_one_split(frame: pd.DataFrame) -> Gate:
    """No lesion may have images in two of train, val and test."""
    trained = frame[frame["split"].isin(SPLITS)]
    per_lesion = trained.groupby("lesion_id")["split"].nunique()
    leaked = per_lesion[per_lesion > 1].index
    return Gate("no_lesion_in_two_splits", len(leaked), _examples(leaked))


def split_matches_phase0(ham: pd.DataFrame, expected: str) -> Gate:
    """HAM10000's lesion-to-split assignment must be exactly Phase 0's."""
    got = split_fingerprint(ham.rename(columns={"label": "dx"}), ham["split"])
    return Gate(
        "split_fingerprint_matches_phase0", int(got != expected), f"got {got}, expected {expected}"
    )


def class_counts(ham: pd.DataFrame, expected: Mapping[str, int]) -> Gate:
    """HAM10000's image count per class must equal the published counts."""
    counts = ham["label"].value_counts().to_dict()
    wrong = {c: (counts.get(c, 0), n) for c, n in expected.items() if counts.get(c, 0) != n}
    extra = set(counts) - set(expected)
    detail = _examples(f"{c}: got {g}, expected {e}" for c, (g, e) in wrong.items())
    return Gate("class_counts_as_expected", len(wrong) + len(extra), detail or "")


def no_missing_labels(frame: pd.DataFrame) -> Gate:
    """Every image of a trained source has a label: anything not marked ``score``."""
    missing = frame[(frame["split"] != "score") & frame["label"].isna()]["isic_id"]
    return Gate("no_missing_labels", len(missing), _examples(missing))


def no_ham_in_sites(frame: pd.DataFrame) -> Gate:
    """No HAM10000 image may appear as a new-site image: that would be training data
    scored as new data."""
    ham_ids = set(frame.loc[frame["source"] == "ham10000", "isic_id"])
    leaked = frame[(frame["source"] != "ham10000") & frame["isic_id"].isin(ham_ids)]["isic_id"]
    return Gate("no_ham10000_image_in_a_site", len(leaked), _examples(leaked))


def license_policy(frame: pd.DataFrame, allowed: Iterable[str]) -> Gate:
    """Every license is allowed, and every CC-BY or CC-BY-NC image carries an attribution."""
    bad_license = ~frame["license"].isin(list(allowed))
    no_attribution = frame["license"].isin(NEEDS_ATTRIBUTION) & (
        frame["attribution"].isna() | (frame["attribution"].astype(str).str.strip() == "")
    )
    bad = frame[bad_license | no_attribution]["isic_id"]
    return Gate("license_and_attribution", len(bad), _examples(bad))


def mapping_reproduces_ham_labels(ham: pd.DataFrame) -> Gate:
    """The ISIC-to-HAM diagnosis mapping must give each HAM10000 image its HAM10000 label;
    otherwise site labels cannot be trusted either."""
    wrong = ham[ham["isic_label"] != ham["label"]]["isic_id"]
    return Gate("diagnosis_mapping_matches_ham10000", len(wrong), _examples(wrong))


def no_duplicate_files(frame: pd.DataFrame) -> Gate:
    """No two images may be the same file (same SHA-256) under different IDs."""
    dup = frame[frame.duplicated("sha256", keep=False)].sort_values("sha256")["isic_id"]
    return Gate("no_duplicate_files", len(dup), _examples(dup))


def every_image_landed(frame: pd.DataFrame) -> Gate:
    """Every metadata row has a file in bronze. A file that fails to decode still counts as
    landed: it goes to quarantine, which is reported, not failed."""
    missing = frame[frame["sha256"].isna()]["isic_id"]
    return Gate("every_image_landed", len(missing), _examples(missing))


def files_match_manifest(frame: pd.DataFrame) -> Gate:
    """The SHA-256 computed in the pipeline equals the one the ingest task recorded."""
    landed = frame[frame["sha256"].notna()]
    wrong = landed[landed["sha256"] != landed["manifest_sha256"]]["isic_id"]
    return Gate("files_match_manifest", len(wrong), _examples(wrong))


def roles_respected(frame: pd.DataFrame) -> Gate:
    """Held-out sites are never split for training, and every client has train, val and
    test images. Images with no client are left to every_client_image_has_a_client."""
    held = frame[(frame["role"] == "held_out") & (frame["split"] != "score")]["isic_id"]
    clients = frame[frame["role"] == "client"]
    incomplete = [
        str(c) for c, g in clients.groupby("client") if not set(SPLITS) <= set(g["split"])
    ]
    detail = _examples([*held, *(f"client {c} lacks a split" for c in incomplete)])
    return Gate("roles_respected", len(held) + len(incomplete), detail)


def site_label_rule_applied(frame: pd.DataFrame, unlabelled: Iterable[str]) -> Gate:
    """No site image whose diagnosis the label rule excludes still carries a label."""
    bad = frame[
        (frame["source"] != HAM)
        & frame.get("diagnosis_3", pd.Series(None, index=frame.index)).isin(list(unlabelled))
        & frame["label"].notna()
    ]["isic_id"]
    return Gate("site_label_rule_applied", len(bad), _examples(bad))


def every_client_image_has_a_client(frame: pd.DataFrame) -> Gate:
    """Every image of a client source maps to a client (for HAM10000, its institution)."""
    missing = frame[(frame["role"] == "client") & frame["client"].isna()]["isic_id"]
    return Gate("every_client_image_has_a_client", len(missing), _examples(missing))


def run_all(frame: pd.DataFrame, config: Mapping[str, Any]) -> list[Gate]:
    ham_cfg = config["ham10000"]
    assert isinstance(ham_cfg, Mapping)
    allowed = config["license_allowed"]
    assert isinstance(allowed, list)
    ham = frame[frame["source"] == "ham10000"]
    return [
        split_computed(frame),
        lesions_in_one_split(frame),
        split_matches_phase0(ham, str(ham_cfg["split_fingerprint"])),
        class_counts(ham, dict(ham_cfg["class_counts"])),
        no_missing_labels(frame),
        no_ham_in_sites(frame),
        license_policy(frame, allowed),
        mapping_reproduces_ham_labels(ham),
        no_duplicate_files(frame[frame["sha256"].notna()]),
        every_image_landed(frame),
        files_match_manifest(frame),
        roles_respected(frame),
        site_label_rule_applied(frame, label_rule(config)),
        every_client_image_has_a_client(frame),
        frozen_splits_kept(frame, frozen_splits(config)),
    ]
