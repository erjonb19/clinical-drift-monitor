"""BiomedCLIP baseline (research Phase B): the pinned model, image and text embeddings, and
zero-shot logits.

BiomedCLIP (Zhang et al. 2023, microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) is a
ViT-B/16 image tower and a PubMedBERT text tower trained on PMC-15M figure-caption pairs. Its
model card says research use only; it is never served here. PMC-15M may include figures of
ISIC or HAM10000 images, so "zero-shot" results cannot be shown to be on unseen images.

The model is loaded from a Hugging Face cache prepared once on a laptop and copied into the
volume with its SHA-256 (serverless may not reach Hugging Face). :func:`check_cache` refuses a
cache whose snapshots are not the pinned revisions. ``open_clip`` is imported only inside
:func:`load`, so CI tests the rest with a stand-in model and no download.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score
from torch import nn

from cdm.splits import CLASSES

Array = NDArray[np.float64]

MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
# Hugging Face commits of the model repository and of the text tower's tokenizer and config.
PINS = {
    "models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224": (
        "9f341de24bfb00180f1b847274256e9b65a3a32e"
    ),
    "models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract": (
        "d673b8835373c6fa116d6d8006b33d48734e305d"
    ),
}
MODEL_VERSION = "biomedclip-9f341de"
CONTEXT_LENGTH = 256

# Plain-English class names for the prompts, in CLASSES order (HAM10000's seven classes).
CLASS_TEXT = {
    "akiec": "actinic keratosis or intraepithelial carcinoma",
    "bcc": "basal cell carcinoma",
    "bkl": "benign keratosis",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic nevus",
    "vasc": "vascular lesion",
}
# Candidate templates. One (or their average) is chosen on validation and frozen in the spec.
TEMPLATES = (
    "{}",
    "a dermoscopy image of {}.",
    "this is a dermoscopy image of {}.",
    "a dermatoscopic photo of a skin lesion: {}.",
    "an image of {}, a type of skin lesion.",
)


def check_cache(hf_home: Path) -> None:
    """Refuse a Hugging Face cache whose model snapshots are not exactly the pinned ones."""
    for repo, revision in PINS.items():
        snapshots = hf_home / "hub" / repo / "snapshots"
        found = sorted(p.name for p in snapshots.iterdir()) if snapshots.is_dir() else []
        if found != [revision]:
            raise ValueError(f"{repo}: snapshots {found}, expected only {revision}")
        ref = (hf_home / "hub" / repo / "refs" / "main").read_text(encoding="utf-8").strip()
        if ref != revision:
            raise ValueError(f"{repo}: refs/main is {ref}, expected {revision}")


def load(device: torch.device) -> tuple[nn.Module, Callable[..., torch.Tensor], Any]:
    """The pinned model in eval mode, its image preprocessing and its tokenizer. Call with
    ``HF_HOME`` pointing at a checked cache and ``HF_HUB_OFFLINE=1``."""
    import open_clip  # the vlm extra; not installed in CI

    model, preprocess = open_clip.create_model_from_pretrained(MODEL)
    tokenizer = open_clip.get_tokenizer(MODEL)
    return model.to(device).eval(), preprocess, tokenizer


def logit_scale(model: nn.Module) -> float:
    return float(model.logit_scale.detach().exp())  # type: ignore[operator]


def _normalize(x: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = x / x.norm(dim=-1, keepdim=True)
    return out


@torch.no_grad()
def image_embeddings(
    model: nn.Module, batches: Iterable[torch.Tensor], device: torch.device
) -> NDArray[np.float32]:
    """L2-normalized image embeddings, in float32 (no autocast, so GPU and CPU agree)."""
    out = [
        _normalize(model.encode_image(x.to(device, non_blocking=True)).float()).cpu().numpy()  # type: ignore[operator]
        for x in batches
    ]
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def text_embeddings(
    model: nn.Module,
    tokenizer: Callable[..., torch.Tensor],
    device: torch.device,
    templates: Sequence[str] = TEMPLATES,
) -> NDArray[np.float32]:
    """Normalized text embeddings, shape (templates, classes, dim), classes in CLASSES order."""
    rows = []
    for template in templates:
        prompts = [template.format(CLASS_TEXT[c]) for c in CLASSES]
        tokens = tokenizer(prompts, context_length=CONTEXT_LENGTH).to(device)
        rows.append(_normalize(model.encode_text(tokens).float()).cpu().numpy())  # type: ignore[operator]
    return np.stack(rows).astype(np.float32)


def class_text(text: NDArray[np.float32], which: Sequence[int]) -> Array:
    """One embedding per class: the chosen templates' embeddings averaged, then renormalized
    (the usual prompt ensemble)."""
    mean = text[list(which)].astype(np.float64).mean(axis=0)
    result: Array = mean / np.linalg.norm(mean, axis=1, keepdims=True)
    return result


def zero_shot_logits(images: NDArray[np.float32], text: Array, scale: float) -> Array:
    """Scaled cosine similarity of each image to each class, the model's own logits."""
    logits: Array = scale * images.astype(np.float64) @ text.T
    return logits


def template_choices(n: int) -> dict[str, list[int]]:
    """The candidates compared on validation: each template alone, and all of them averaged."""
    return {**{f"template_{i}": [i] for i in range(n)}, "all_templates": list(range(n))}


def select_templates(
    val_images: NDArray[np.float32],
    val_labels: NDArray[np.int64],
    text: NDArray[np.float32],
    scale: float,
) -> tuple[str, dict[str, float]]:
    """The choice with the highest validation balanced accuracy (ties: the earlier one), and
    every candidate's score."""
    scores = {
        name: float(
            balanced_accuracy_score(
                val_labels, zero_shot_logits(val_images, class_text(text, which), scale).argmax(1)
            )
        )
        for name, which in template_choices(len(text)).items()
    }
    best = max(scores, key=lambda k: scores[k])
    return best, scores


def clusters(
    patient_ids: Sequence[object], lesion_ids: Sequence[object], image_ids: Sequence[str]
) -> list[str]:
    """The bootstrap's resampling unit per image: the patient where ISIC records one, else the
    lesion, else the image itself."""
    out = []
    for patient, lesion, image in zip(patient_ids, lesion_ids, image_ids, strict=True):
        if isinstance(patient, str) and patient:
            out.append(f"patient:{patient}")
        elif isinstance(lesion, str) and lesion:
            out.append(f"lesion:{lesion}")
        else:
            out.append(f"image:{image}")
    return out


def drift_metrics(
    train_feats: Array,
    train_labels: NDArray[np.int64],
    feats: dict[str, Array],
    logits: dict[str, Array],
    shifts: Sequence[str],
    in_distribution: str = "test",
) -> dict[str, Any]:
    """AUROC and FPR@95TPR of each detector, ``in_distribution`` against each shift.

    Mahalanobis uses PCA sized by Phase 2's method D (95% of training-feature variance), fit
    on training features only; max softmax and energy use ``logits``. Every set goes through
    the same scoring, and higher always means more out-of-distribution.
    """
    from cdm.eval import detection_metrics
    from cdm.ood import MahalanobisDetector, energy_score, msp_score, pca_size_for_variance

    k = pca_size_for_variance(train_feats)
    maha = MahalanobisDetector(k).fit(train_feats, train_labels)
    names = [in_distribution, *shifts]
    scores = {
        name: {
            "mahalanobis": maha.score(feats[name]),
            "max_softmax": msp_score(logits[name]),
            "energy": energy_score(logits[name]),
        }
        for name in names
    }
    table = {
        det: {s: detection_metrics(scores[in_distribution][det], scores[s][det]) for s in shifts}
        for det in scores[in_distribution]
    }
    return {"pca_components_method_d": k, "metrics": table}
