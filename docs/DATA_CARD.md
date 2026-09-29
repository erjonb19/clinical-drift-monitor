# Data card

What data this project uses, where it comes from, and the terms it is used under.
The full card (what the model saw, per-skin-type results, where it fails) is a Phase 5
item; this file starts with the sources so the licensing record exists from the first run.

No image, metadata file or derived array is committed to this repository. Every file is
downloaded by `src/cdm/data.py` and checked against the checksum listed here before use.

## Sources

### HAM10000 (in-distribution: train, validation, test)

| Field | Value |
| --- | --- |
| Source | Harvard Dataverse, [doi:10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T), dataset version 4.0 |
| License | **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**, as stated in the dataset's terms of use on Dataverse |
| What that allows here | Non-commercial use with attribution. This is a personal, non-commercial portfolio project. |
| Attribution | Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Sci. Data* 5, 180161 (2018). [doi:10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161) |
| Contents used | 10,015 dermatoscopic images of 7,470 lesions, 7 diagnostic classes |

Files and checksums (MD5 as published by Dataverse):

| File | Dataverse file id | MD5 |
| --- | --- | --- |
| `HAM10000_metadata.csv` | 4338392 (requested with `?format=original`) | `8f85fb1aa29d80a2797247e434deb79d` |
| `HAM10000_images_part_1.zip` | 3172585 | `4639bfa73ab251610530a97c898e6e46` |
| `HAM10000_images_part_2.zip` | 3172584 | `da43d6cc50f6613013be07e8986b384b` |

Class counts in the metadata (images): nv 6,705 · mel 1,113 · bkl 1,099 · bcc 514 ·
akiec 327 · vasc 142 · df 115.

### PathMNIST at 224 px (near out-of-distribution: histopathology)

| Field | Value |
| --- | --- |
| Source | MedMNIST+ on Zenodo, [record 10519652](https://zenodo.org/records/10519652), file `pathmnist_224.npz`, via the `medmnist` package |
| License | CC BY 4.0 (per the `medmnist` package metadata) |
| Attribution | Yang, J. et al. MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Sci. Data* 10, 41 (2023). Derived from NCT-CRC-HE-100K: Kather, J. N. et al. *PLoS Med.* 16, e1002730 (2019). |
| Contents used | A fixed random subset of the test split, native 224 px (not upscaled from 28 px) |
| MD5 | `2c51a510bcdc9cf8ddb2af93af1eadec` (checked by `medmnist` on download) |

### CIFAR-10 (far out-of-distribution: natural images)

| Field | Value |
| --- | --- |
| Source | torchvision's `CIFAR10` downloader, test split |
| License | No license is stated by the dataset authors; used for research with citation |
| Attribution | Krizhevsky, A. Learning Multiple Layers of Features from Tiny Images. Technical report, University of Toronto (2009). |
| Contents used | A fixed random subset of the test split, 32 px upscaled to 224 px |

## Phase 1 sources: ISIC Archive

From Phase 1 on, all images reach the lakehouse from the ISIC Archive (public S3 bucket
`isic-archive` and its API, no account), because Databricks Free Edition cannot reach Harvard
Dataverse. License and attribution are recorded **per image** from ISIC's metadata, and
every image's license must be CC-0, CC-BY or CC-BY-NC (a pipeline gate). CC-BY and
CC-BY-NC require attribution, which is kept with every row.

| Source | ISIC collections | License(s) | Attribution in ISIC | Used as |
| --- | --- | --- | --- | --- |
| HAM10000 | 212, restricted to the 10,015 HAM10000 image IDs | CC-BY-NC | "MILK study team" | Training, validation, test (Phase 0 split) |
| Barcelona (BCN20000) | 249, capped at 5,000 images by whole lesions | CC-BY-NC | Hospital Clínic de Barcelona | New site |
| Buenos Aires (HIBA) | 251 | CC-BY | Hospital Italiano de Buenos Aires | New site |
| MSK | 287, 289 | CC-0 | "Anonymous" (collections from Memorial Sloan Kettering) | New site |
| PAD-UFES-20 | 406 | CC-BY | Federal University of Espírito Santo (UFES) | New site, smartphone photos |

Sites and the Barcelona selection are defined in `config/sites.json`. The selection is
reproducible from a seed and pinned by a fingerprint of the chosen image IDs; if ISIC's
collection changes, ingestion stops rather than silently choosing different images.

**HAM10000 differs from Phase 0's files.** ISIC serves HAM10000 re-encoded with stronger
JPEG compression (same 600 × 450 pixels, same picture; mean absolute pixel difference of
1.7 to 2.9 on a 0–255 scale in the three images compared). The manifest records our own
byte count and SHA-256 for every file. Labels and lesion IDs still come from
`HAM10000_metadata.csv` (checked against Dataverse's MD5), so the Phase 0 split is
reproduced exactly.

**Diagnoses.** Site labels come from ISIC's diagnosis hierarchy mapped to HAM10000's seven
classes (`cdm.sources.map_diagnosis`). On HAM10000's own images, the mapping reproduces
every HAM10000 label. Images whose diagnosis is missing or outside the seven classes keep
a null label and are used for scoring only.

Attribution for the ISIC Archive as a whole: International Skin Imaging Collaboration (ISIC)
Archive, accessed 2026-09-29 from https://registry.opendata.aws/isic-archive.
