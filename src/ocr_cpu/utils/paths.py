from __future__ import annotations

from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

TRAIN_CSV = RAW_DIR / "train.csv"
TRAIN_LABELS_CSV = RAW_DIR / "train_labels.csv"
TEST_CSV = RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = RAW_DIR / "sample_submission.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = OUTPUTS_DIR / "cache"
DEBUG_IMAGES_DIR = OUTPUTS_DIR / "debug_images"
AUDITS_DIR = OUTPUTS_DIR / "audits"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"


def find_image_dir(root: Path, expected_prefix: str = "img_") -> Path:
    """
    Find the actual image directory inside a possibly nested unzip folder.

    Example supported cases:
    - data/raw/test_images/images/
    - data/raw/test_images/test_images/images/
    - data/raw/train_images/train_images/train_images/
    """

    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"Folder does not exist: {root}")

    candidates = []

    for p in root.rglob("*"):
        if not p.is_dir():
            continue

        jpgs = list(p.glob(f"{expected_prefix}*.jpg"))
        jpegs = list(p.glob(f"{expected_prefix}*.jpeg"))
        pngs = list(p.glob(f"{expected_prefix}*.png"))

        n_images = len(jpgs) + len(jpegs) + len(pngs)

        if n_images > 0:
            candidates.append((n_images, p))

    if not candidates:
        raise FileNotFoundError(f"No image directory found inside: {root}")

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]


def get_train_images_dir() -> Path:
    return find_image_dir(RAW_DIR / "train_images")


def get_test_images_dir() -> Path:
    return find_image_dir(RAW_DIR / "test_images")


def get_image_path(image_id: str, split: str = "test") -> Path:
    """
    Resolve image_id to actual local image path.

    split:
    - "train"
    - "test"
    """

    if split == "train":
        image_dir = get_train_images_dir()
    elif split == "test":
        image_dir = get_test_images_dir()
    else:
        raise ValueError(f"Unknown split: {split}")

    image_id = str(image_id)

    direct = image_dir / image_id
    if direct.exists():
        return direct

    stem = Path(image_id).stem

    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot find image_id={image_id} in split={split}, dir={image_dir}"
    )