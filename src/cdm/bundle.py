"""The serving bundle's manifest rules (backlog item 32), shared by cdm.serving's loader and
scripts/deploy_space.py's pre-upload check. No torch import, so the deploy script stays light."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plain_name(name: str) -> str:
    """A bundle file name: one path part, no separators, not '.' or '..'."""
    if not name or name in (".", "..") or "/" in name or "\\" in name or Path(name).name != name:
        raise ValueError(f"{name!r} is not a plain file name in the bundle")
    return name


def check_manifest(manifest: dict[str, Any]) -> dict[str, str]:
    """The manifest's sha256 map, after checking that every name in it and in the seeds is a
    plain file name, that each seed's checkpoint and detector is in the map (so none loads
    unhashed), and that there is at least one seed. Raises ValueError otherwise."""
    hashes: dict[str, str] = manifest["sha256"]
    for name in hashes:
        plain_name(name)
    if not manifest["seeds"]:
        raise ValueError("the bundle manifest lists no seeds")
    for seed in manifest["seeds"]:
        for key in ("checkpoint", "detector"):
            if plain_name(seed[key]) not in hashes:
                raise ValueError(f"{seed[key]} is not listed in the bundle manifest's sha256")
    return hashes
