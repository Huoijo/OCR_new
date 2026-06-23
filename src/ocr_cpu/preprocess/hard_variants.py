from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class HardPreprocessConfig:
    """
    Config for generating a hard black/white OCR image.

    This is intentionally independent from router.py so we can run tuning
    experiments without editing the production router after every trial.
    """

    config_id: str
    description: str = ""

    # Pre-threshold cleanup / contrast
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    pre_blur: str = "none"  # none | median3 | gaussian3

    # Threshold
    threshold_mode: str = "adaptive"  # adaptive | otsu
    adaptive_method: str = "gaussian"  # gaussian | mean
    block_divisor: float = 28.0
    block_min: int = 21
    block_max: int = 71
    adaptive_C: int = 11

    # Post-threshold cleanup
    morph: str = "none"  # none | open | close
    morph_kernel: int = 2
    morph_iterations: int = 1

    # Invert when black dominates. Paddle/Tesseract usually prefer dark text on light bg.
    invert_if_mean_below: float = 127.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def make_odd(value: int, min_value: int = 3) -> int:
    value = max(int(value), int(min_value))
    if value % 2 == 0:
        value += 1
    return value


def _apply_pre_blur(gray: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return gray
    if mode == "median3":
        return cv.medianBlur(gray, 3)
    if mode == "gaussian3":
        return cv.GaussianBlur(gray, (3, 3), sigmaX=0.4)
    raise ValueError(f"Unknown pre_blur: {mode}")


def _apply_clahe(gray: np.ndarray, clip_limit: float, tile_grid: Tuple[int, int]) -> np.ndarray:
    clahe = cv.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(tile_grid),
    )
    return clahe.apply(gray)


def _apply_threshold(gray: np.ndarray, cfg: HardPreprocessConfig) -> np.ndarray:
    if cfg.threshold_mode == "otsu":
        _, th = cv.threshold(
            gray,
            0,
            255,
            cv.THRESH_BINARY + cv.THRESH_OTSU,
        )
        return th

    if cfg.threshold_mode != "adaptive":
        raise ValueError(f"Unknown threshold_mode: {cfg.threshold_mode}")

    h, w = gray.shape[:2]
    short_side = min(h, w)

    block_size = make_odd(max(cfg.block_min, int(short_side / max(cfg.block_divisor, 1.0))))
    block_size = min(block_size, make_odd(cfg.block_max))

    if cfg.adaptive_method == "gaussian":
        method = cv.ADAPTIVE_THRESH_GAUSSIAN_C
    elif cfg.adaptive_method == "mean":
        method = cv.ADAPTIVE_THRESH_MEAN_C
    else:
        raise ValueError(f"Unknown adaptive_method: {cfg.adaptive_method}")

    th = cv.adaptiveThreshold(
        gray,
        255,
        method,
        cv.THRESH_BINARY,
        block_size,
        float(cfg.adaptive_C),
    )
    return th


def _apply_morph(th: np.ndarray, cfg: HardPreprocessConfig) -> np.ndarray:
    if cfg.morph == "none":
        return th

    k = max(1, int(cfg.morph_kernel))
    kernel = np.ones((k, k), dtype=np.uint8)
    iterations = max(1, int(cfg.morph_iterations))

    if cfg.morph == "open":
        return cv.morphologyEx(th, cv.MORPH_OPEN, kernel, iterations=iterations)
    if cfg.morph == "close":
        return cv.morphologyEx(th, cv.MORPH_CLOSE, kernel, iterations=iterations)

    raise ValueError(f"Unknown morph: {cfg.morph}")


def make_hard_threshold_variant(img_bgr: np.ndarray, cfg: HardPreprocessConfig) -> np.ndarray:
    """
    Convert a BGR image into a hard black/white BGR image according to cfg.
    """

    if img_bgr.ndim == 2:
        gray = img_bgr.copy()
    else:
        gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    gray = _apply_pre_blur(gray, cfg.pre_blur)

    if cfg.use_clahe:
        gray = _apply_clahe(gray, cfg.clahe_clip_limit, cfg.clahe_tile_grid)

    th = _apply_threshold(gray, cfg)
    th = _apply_morph(th, cfg)

    if float(np.mean(th)) < float(cfg.invert_if_mean_below):
        th = 255 - th

    return cv.cvtColor(th, cv.COLOR_GRAY2BGR)


def get_default_hard_configs() -> List[HardPreprocessConfig]:
    """
    A compact first-pass search space.

    Keep this list small enough for CPU tuning. After finding the best family,
    run a second narrower search around the winners.
    """

    return [
        HardPreprocessConfig(
            config_id="current_c11_div28",
            description="Current router hard threshold: CLAHE 2.0 + adaptive gaussian C=11 div=28.",
            use_clahe=True,
            clahe_clip_limit=2.0,
            threshold_mode="adaptive",
            adaptive_method="gaussian",
            block_divisor=28.0,
            adaptive_C=11,
        ),
        HardPreprocessConfig(
            config_id="no_clahe_c11_div28",
            description="No CLAHE before adaptive threshold.",
            use_clahe=False,
            threshold_mode="adaptive",
            adaptive_method="gaussian",
            block_divisor=28.0,
            adaptive_C=11,
        ),
        HardPreprocessConfig(
            config_id="clip1_c11_div28",
            description="Lower CLAHE clip limit.",
            use_clahe=True,
            clahe_clip_limit=1.0,
            threshold_mode="adaptive",
            adaptive_method="gaussian",
            block_divisor=28.0,
            adaptive_C=11,
        ),
        HardPreprocessConfig(
            config_id="clip15_c11_div28",
            description="Mildly lower CLAHE clip limit.",
            use_clahe=True,
            clahe_clip_limit=1.5,
            threshold_mode="adaptive",
            adaptive_method="gaussian",
            block_divisor=28.0,
            adaptive_C=11,
        ),
        HardPreprocessConfig(
            config_id="clip25_c11_div28",
            description="Slightly stronger CLAHE.",
            use_clahe=True,
            clahe_clip_limit=2.5,
            threshold_mode="adaptive",
            adaptive_method="gaussian",
            block_divisor=28.0,
            adaptive_C=11,
        ),
        HardPreprocessConfig(
            config_id="c7_div28",
            description="Lower adaptive C; keeps more dark strokes but may add noise.",
            use_clahe=True,
            clahe_clip_limit=2.0,
            threshold_mode="adaptive",
            adaptive_method="gaussian",
            block_divisor=28.0,
            adaptive_C=7,
        ),
        HardPreprocessConfig(
            config_id="c9_div28",
            description="Slightly lower adaptive C.",
            use_clahe=True,
            clahe_clip_limit=2.0,
            threshold_mode="adaptive",
            adaptive_method="gaussian",
            block_divisor=28.0,
            adaptive_C=9,
        ),
        HardPreprocessConfig(
            config_id="c13_div28",
            description="Slightly higher adaptive C; can reduce black noise.",
            use_clahe=True,
            clahe_clip_limit=2.0,
            threshold_mode="adaptive",
            adaptive_method="gaussian",
            block_divisor=28.0,
            adaptive_C=13,
        ),
        HardPreprocessConfig(
            config_id="c15_div28",
            description="Higher adaptive C; cleaner but can erase weak strokes.",
            use_clahe=True,
            clahe_clip_limit=2.0,
            threshold_mode="adaptive",
            adaptive_method="gaussian",
            block_divisor=28.0,
            adaptive_C=15,
        ),
        HardPreprocessConfig(
            config_id="c17_div28",
            description="Aggressively cleaner adaptive threshold.",
            use_clahe=True,
            clahe_clip_limit=2.0,
            threshold_mode="adaptive",
            adaptive_method="gaussian",
            block_divisor=28.0,
            adaptive_C=17,
        ),
        HardPreprocessConfig(
            config_id="c11_div24",
            description="Smaller local window than current.",
            use_clahe=True,
            clahe_clip_limit=2.0,
            threshold_mode="adaptive",
            adaptive_method="gaussian",
            block_divisor=24.0,
            adaptive_C=11,
        ),
        HardPreprocessConfig(
            config_id="c11_div32",
            description="Larger local window than current.",
            use_clahe=True,
            clahe_clip_limit=2.0,
            threshold_mode="adaptive",
            adaptive_method="gaussian",
            block_divisor=32.0,
            adaptive_C=11,
        ),
        HardPreprocessConfig(
            config_id="c11_div40",
            description="Much larger local window.",
            use_clahe=True,
            clahe_clip_limit=2.0,
            threshold_mode="adaptive",
            adaptive_method="gaussian",
            block_divisor=40.0,
            adaptive_C=11,
        ),
        HardPreprocessConfig(
            config_id="c13_div32",
            description="Larger local window plus slightly cleaner threshold.",
            use_clahe=True,
            clahe_clip_limit=2.0,
            threshold_mode="adaptive",
            adaptive_method="gaussian",
            block_divisor=32.0,
            adaptive_C=13,
        ),
        HardPreprocessConfig(
            config_id="mean_c11_div28",
            description="Adaptive mean instead of gaussian.",
            use_clahe=True,
            clahe_clip_limit=2.0,
            threshold_mode="adaptive",
            adaptive_method="mean",
            block_divisor=28.0,
            adaptive_C=11,
        ),
        HardPreprocessConfig(
            config_id="otsu_clahe",
            description="Otsu global threshold after CLAHE.",
            use_clahe=True,
            clahe_clip_limit=2.0,
            threshold_mode="otsu",
        ),
        HardPreprocessConfig(
            config_id="otsu_no_clahe",
            description="Otsu global threshold without CLAHE.",
            use_clahe=False,
            threshold_mode="otsu",
        ),
        HardPreprocessConfig(
            config_id="median3_current",
            description="Median blur before current threshold; may reduce salt-pepper noise.",
            pre_blur="median3",
            use_clahe=True,
            clahe_clip_limit=2.0,
            threshold_mode="adaptive",
            adaptive_method="gaussian",
            block_divisor=28.0,
            adaptive_C=11,
        ),
        HardPreprocessConfig(
            config_id="median3_c13_div28",
            description="Median blur plus cleaner adaptive threshold.",
            pre_blur="median3",
            use_clahe=True,
            clahe_clip_limit=2.0,
            threshold_mode="adaptive",
            adaptive_method="gaussian",
            block_divisor=28.0,
            adaptive_C=13,
        ),
    ]


def get_hard_config_map() -> Dict[str, HardPreprocessConfig]:
    return {cfg.config_id: cfg for cfg in get_default_hard_configs()}


__all__ = [
    "HardPreprocessConfig",
    "make_hard_threshold_variant",
    "get_default_hard_configs",
    "get_hard_config_map",
]
