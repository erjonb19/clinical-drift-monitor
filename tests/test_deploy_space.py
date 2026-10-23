"""The Space upload holds every file the Dockerfile copies (a missing one fails the build)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "deploy_space", REPO / "scripts" / "deploy_space.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_space"] = module
    spec.loader.exec_module(module)
    return module


def test_staged_space_has_every_dockerfile_copy_source(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    target = tmp_path / "space"
    target.mkdir()
    load_script().stage(target, bundle)
    sources = []
    for line in (REPO / "Dockerfile").read_text(encoding="utf-8").splitlines():
        if line.startswith("COPY "):
            sources += line.split()[1:-1]
    assert sources, "the Dockerfile copies nothing?"
    missing = [s for s in sources if not (target / s).exists()]
    assert not missing, f"not staged for the Space: {missing}"


def test_space_readme_carries_notice_licence_and_attribution() -> None:
    readme = " ".join(load_script().SPACE_README.split())
    for text in ("Research demo, not for medical use", "never stored or logged", "rate limited",
                 "license: cc-by-nc-4.0", "Non-commercial research use only",
                 "International Skin Imaging Collaboration (ISIC) Archive"):  # fmt: skip
        assert text in readme, text
