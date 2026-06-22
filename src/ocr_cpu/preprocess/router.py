from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

import cv2 as cv
import numpy as np


ImageInput = Union[str, Path, np.ndarray]


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------

@dataclass
class ImageQualityFeatures:
    """
    Lightweight image-quality features used by the preprocessing router.

    These features are intentionally cheap to compute because this project
    targets CPU-only execution.
    """

    width: int
    height: int
    short_side: int
    long_side: int
    aspect_ratio: float

    mean_luma: float
    std_luma: float
    p05_luma: float
    p95_luma: float
    dynamic_range: float

    laplacian_var: float
    edge_density: float
    noise_sigma: float
    background_std: float
    local_contrast: float
    near_binary_ratio: float


@dataclass
class PreprocessDecision:
    """
    Decision returned by the router.

    This object tells the pipeline which preprocessing method should be used
    first, and which fallback should be kept available if OCR quality is bad.
    """

    scale: float = 1.0

    use_clahe: bool = False
    use_median: bool = False
    use_gaussian: bool = False
    use_sharpen: bool = False

    use_hard_fallback: bool = False
    hard_threshold_mode: str = "adaptive"

    primary_variant: str = "raw_resized"
    reason: str = "clean_or_unknown"

    triggered_flags: List[str] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------

def _read_bgr_with_pillow(path: Path) -> np.ndarray:
    """
    Fallback image reader for files OpenCV cannot decode.

    Some dataset images have .jpg extension but are actually animated WebP.
    OpenCV may fail on animated WebP, so we use Pillow and take the first frame.
    """

    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError(
            "OpenCV failed to read this image, and Pillow is not installed. "
            "Install Pillow with: pip install pillow"
        ) from e

    try:
        with Image.open(path) as im:
            # For animated WebP/GIF, use the first frame.
            try:
                im.seek(0)
            except EOFError:
                pass

            im = im.convert("RGB")
            arr_rgb = np.array(im)

            return cv.cvtColor(arr_rgb, cv.COLOR_RGB2BGR)

    except Exception as e:
        raise RuntimeError(
            f"OpenCV and Pillow both failed to read image: {path}"
        ) from e


def _read_bgr(image: ImageInput) -> np.ndarray:
    """
    Read image as BGR ndarray.

    Accepts:
    - file path
    - numpy ndarray in BGR/RGB-like format

    This reader is robust to mislabeled animated WebP files with .jpg extension.
    """

    if isinstance(image, (str, Path)):
        path = Path(image)

        if not path.exists():
            raise FileNotFoundError(f"Image path does not exist: {path}")

        img = cv.imread(str(path), cv.IMREAD_COLOR)

        if img is not None:
            return img

        # Fallback for animated WebP or unusual image headers.
        return _read_bgr_with_pillow(path)

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv.cvtColor(image, cv.COLOR_GRAY2BGR)

        if image.ndim == 3 and image.shape[2] == 3:
            return image.copy()

        if image.ndim == 3 and image.shape[2] == 4:
            return cv.cvtColor(image, cv.COLOR_BGRA2BGR)

    raise TypeError(
        "Unsupported image input. Expected path or numpy ndarray."
    )


# ---------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------

def _safe_downsample_for_analysis(
    img: np.ndarray,
    max_long_side: int = 768,
) -> np.ndarray:
    """
    Downsample image for fast feature extraction.

    The router should not spend too much CPU time analyzing large images.
    This function keeps the image shape similar but limits the long side.
    """

    h, w = img.shape[:2]
    long_side = max(h, w)

    if long_side <= max_long_side:
        return img

    scale = max_long_side / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    return cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)


def _robust_noise_sigma(gray: np.ndarray) -> float:
    """
    Estimate noise level using median absolute deviation of high-frequency residual.

    Idea:
    - Median blur removes small noise.
    - Difference between original and median-blurred image approximates noise.
    - MAD is more robust than normal std.
    """

    gray_f = gray.astype(np.float32)
    med = cv.medianBlur(gray, 3).astype(np.float32)
    residual = gray_f - med

    mad = np.median(np.abs(residual - np.median(residual)))
    sigma = 1.4826 * mad

    return float(sigma)


def _estimate_background_std(gray: np.ndarray) -> float:
    """
    Estimate uneven lighting / background variation.

    High value means the image may have shadows, gradients, or uneven background.
    """

    h, w = gray.shape[:2]
    short_side = min(h, w)

    sigma = max(12.0, short_side / 18.0)
    background = cv.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)

    return float(np.std(background))


def _estimate_local_contrast(gray: np.ndarray) -> float:
    """
    Estimate local contrast after removing smooth background.

    This helps detect whether text strokes are distinguishable from background.
    """

    h, w = gray.shape[:2]
    short_side = min(h, w)

    sigma = max(8.0, short_side / 24.0)
    background = cv.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)

    residual = gray.astype(np.float32) - background.astype(np.float32)

    return float(np.std(residual))


def _compute_edge_density(gray: np.ndarray) -> float:
    """
    Estimate amount of visible edge/text structure.

    Low edge density can mean:
    - blur
    - low contrast
    - very plain image
    """

    med = float(np.median(gray))
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))

    edges = cv.Canny(gray, lower, upper)
    return float(np.mean(edges > 0))


def _compute_near_binary_ratio(gray: np.ndarray) -> float:
    """
    Estimate whether image is already close to black-white / hard-thresholded.

    High value means many pixels are near 0 or near 255.
    """

    dark = gray < 25
    bright = gray > 230

    return float(np.mean(dark | bright))


def _extract_features(img: np.ndarray) -> ImageQualityFeatures:
    """
    Extract lightweight quality features from image.
    """

    h0, w0 = img.shape[:2]

    analysis_img = _safe_downsample_for_analysis(img)
    gray = cv.cvtColor(analysis_img, cv.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    short_side = min(h0, w0)
    long_side = max(h0, w0)

    mean_luma = float(np.mean(gray))
    std_luma = float(np.std(gray))

    p05 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    dynamic_range = p95 - p05

    laplacian_var = float(cv.Laplacian(gray, cv.CV_64F).var())
    edge_density = _compute_edge_density(gray)
    noise_sigma = _robust_noise_sigma(gray)
    background_std = _estimate_background_std(gray)
    local_contrast = _estimate_local_contrast(gray)
    near_binary_ratio = _compute_near_binary_ratio(gray)

    return ImageQualityFeatures(
        width=w0,
        height=h0,
        short_side=short_side,
        long_side=long_side,
        aspect_ratio=float(w0 / max(h0, 1)),

        mean_luma=mean_luma,
        std_luma=std_luma,
        p05_luma=p05,
        p95_luma=p95,
        dynamic_range=dynamic_range,

        laplacian_var=laplacian_var,
        edge_density=edge_density,
        noise_sigma=noise_sigma,
        background_std=background_std,
        local_contrast=local_contrast,
        near_binary_ratio=near_binary_ratio,
    )


# ---------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------

def classify_image_quality(
    image: ImageInput,
    target_short_side: int = 900,
    max_long_side: int = 1600,
    engine: str = "paddle",
) -> PreprocessDecision:
    """
    Decide which preprocessing strategy should be used for an image.

    Parameters
    ----------
    image:
        Image path or BGR numpy image.

    target_short_side:
        If image is too small, resize so the short side approaches this value.

    max_long_side:
        Prevent image from becoming too large after resize.

    engine:
        OCR engine profile. Currently used lightly to keep future flexibility.

    Returns
    -------
    PreprocessDecision
        Decision object containing selected method, flags, scale, and features.
    """

    img = _read_bgr(image)
    f = _extract_features(img)

    flags: List[str] = []

    # --------------------------------------------------------------
    # 1. Size / scale decision
    # --------------------------------------------------------------

    scale = 1.0

    if f.short_side < target_short_side:
        scale = target_short_side / max(float(f.short_side), 1.0)

    if f.long_side * scale > max_long_side:
        scale = max_long_side / max(float(f.long_side), 1.0)

    scale = float(np.clip(scale, 0.5, 2.5))

    if scale > 1.10:
        flags.append("small_image_upscale")
    elif scale < 0.90:
        flags.append("large_image_downscale")

    # --------------------------------------------------------------
    # 2. Brightness / contrast flags
    # --------------------------------------------------------------

    is_dark = f.mean_luma < 75
    is_bright = f.mean_luma > 205
    is_low_contrast = f.std_luma < 38 or f.dynamic_range < 115
    is_very_low_contrast = f.std_luma < 25 or f.dynamic_range < 80

    if is_dark:
        flags.append("dark_image")

    if is_bright:
        flags.append("bright_image")

    if is_low_contrast:
        flags.append("low_contrast")

    if is_very_low_contrast:
        flags.append("very_low_contrast")

    # --------------------------------------------------------------
    # 3. Blur / edge flags
    # --------------------------------------------------------------

    is_blurry = f.laplacian_var < 70 and f.edge_density < 0.075
    is_very_blurry = f.laplacian_var < 35 and f.edge_density < 0.045

    if is_blurry:
        flags.append("blurry")

    if is_very_blurry:
        flags.append("very_blurry")

    # --------------------------------------------------------------
    # 4. Noise / uneven background flags
    # --------------------------------------------------------------

    is_noisy = f.noise_sigma > 7.5
    is_very_noisy = f.noise_sigma > 13.0

    has_uneven_light = f.background_std > 18.0
    has_strong_uneven_light = f.background_std > 28.0

    if is_noisy:
        flags.append("noisy")

    if is_very_noisy:
        flags.append("very_noisy")

    if has_uneven_light:
        flags.append("uneven_light")

    if has_strong_uneven_light:
        flags.append("strong_uneven_light")

    # --------------------------------------------------------------
    # 5. Near-binary image flag
    # --------------------------------------------------------------

    is_near_binary = f.near_binary_ratio > 0.55

    if is_near_binary:
        flags.append("near_binary")

    # --------------------------------------------------------------
    # 6. Decide preprocessing actions
    # --------------------------------------------------------------

    use_clahe = False
    use_median = False
    use_gaussian = False
    use_sharpen = False
    use_hard_fallback = False
    hard_threshold_mode = "adaptive"

    # CLAHE helps when contrast/light is bad.
    # Avoid applying CLAHE too aggressively to near-binary images.
    if (
        is_low_contrast
        or is_dark
        or is_bright
        or has_uneven_light
    ) and not is_near_binary:
        use_clahe = True

    # Median blur is safer for small salt-pepper-like noise.
    if is_noisy:
        use_median = True

    # Gaussian is only used for very noisy cases.
    # It can blur text, so keep it conservative.
    if is_very_noisy and not is_very_blurry:
        use_gaussian = True

    # Sharpen helps small or mildly blurry images.
    # Avoid strong sharpening on very noisy images.
    if (
        is_blurry
        or scale > 1.15
        or use_clahe
    ) and not is_very_noisy:
        use_sharpen = True

    # Hard threshold should be treated as fallback, not default.
    # It may remove Vietnamese diacritics if used too aggressively.
    if (
        is_very_low_contrast
        or has_strong_uneven_light
        or is_near_binary
        or is_very_blurry
    ):
        use_hard_fallback = True

    if has_uneven_light:
        hard_threshold_mode = "adaptive"
    else:
        hard_threshold_mode = "otsu"

    # --------------------------------------------------------------
    # 7. Primary variant selection
    # --------------------------------------------------------------

    if use_clahe or use_median or use_gaussian or use_sharpen:
        primary_variant = "soft_enhanced"
    else:
        primary_variant = "raw_resized"

    reason = "+".join(flags) if flags else "clean_or_unknown"

    return PreprocessDecision(
        scale=scale,

        use_clahe=use_clahe,
        use_median=use_median,
        use_gaussian=use_gaussian,
        use_sharpen=use_sharpen,

        use_hard_fallback=use_hard_fallback,
        hard_threshold_mode=hard_threshold_mode,

        primary_variant=primary_variant,
        reason=reason,

        triggered_flags=flags,
        features={
            "width": float(f.width),
            "height": float(f.height),
            "short_side": float(f.short_side),
            "long_side": float(f.long_side),
            "aspect_ratio": float(f.aspect_ratio),

            "mean_luma": float(f.mean_luma),
            "std_luma": float(f.std_luma),
            "p05_luma": float(f.p05_luma),
            "p95_luma": float(f.p95_luma),
            "dynamic_range": float(f.dynamic_range),

            "laplacian_var": float(f.laplacian_var),
            "edge_density": float(f.edge_density),
            "noise_sigma": float(f.noise_sigma),
            "background_std": float(f.background_std),
            "local_contrast": float(f.local_contrast),
            "near_binary_ratio": float(f.near_binary_ratio),
        },
    )


# ---------------------------------------------------------------------
# Preprocessing transforms
# ---------------------------------------------------------------------

def _resize_by_scale(img: np.ndarray, scale: float) -> np.ndarray:
    """
    Resize image by scale.

    Uses:
    - INTER_CUBIC when upscaling
    - INTER_AREA when downscaling
    """

    if abs(scale - 1.0) < 0.03:
        return img.copy()

    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if scale > 1.0:
        interp = cv.INTER_CUBIC
    else:
        interp = cv.INTER_AREA

    return cv.resize(img, (new_w, new_h), interpolation=interp)


def _apply_clahe_l_channel(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE on L channel in LAB color space.

    This enhances local contrast while preserving color better than
    applying CLAHE directly on grayscale.
    """

    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)

    clahe = cv.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )

    l2 = clahe.apply(l)

    out = cv.merge([l2, a, b])
    out = cv.cvtColor(out, cv.COLOR_LAB2BGR)

    return out


def _apply_unsharp_mask(
    img: np.ndarray,
    amount: float = 0.55,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Mild sharpening using unsharp mask.

    Formula:
    sharpened = image * (1 + amount) - blurred * amount
    """

    blur = cv.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv.addWeighted(img, 1.0 + amount, blur, -amount, 0)

    return sharp


def _make_odd(value: int, min_value: int = 3) -> int:
    """
    Ensure value is odd and at least min_value.
    """

    value = max(int(value), min_value)

    if value % 2 == 0:
        value += 1

    return value


def _make_hard_threshold(
    img: np.ndarray,
    mode: str = "adaptive",
) -> np.ndarray:
    """
    Create a hard black-white fallback image.

    This is useful for badly lit or near-binary images, but it should
    not be the first choice because it can destroy small Vietnamese marks.
    """

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Mild local contrast before thresholding
    clahe = cv.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    gray = clahe.apply(gray)

    if mode == "otsu":
        _, th = cv.threshold(
            gray,
            0,
            255,
            cv.THRESH_BINARY + cv.THRESH_OTSU,
        )
    else:
        h, w = gray.shape[:2]
        short_side = min(h, w)

        block_size = _make_odd(max(21, int(short_side / 28)))
        block_size = min(block_size, 71)

        th = cv.adaptiveThreshold(
            gray,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            block_size,
            11,
        )

    # OCR usually prefers dark text on light background.
    # If black region dominates, invert it.
    if np.mean(th) < 127:
        th = 255 - th

    return cv.cvtColor(th, cv.COLOR_GRAY2BGR)


def _soft_enhance(
    img: np.ndarray,
    decision: PreprocessDecision,
) -> np.ndarray:
    """
    Apply soft enhancement according to router decision.

    Order matters:
    1. Denoise lightly
    2. Improve contrast
    3. Sharpen mildly
    """

    out = img.copy()

    if decision.use_median:
        out = cv.medianBlur(out, 3)

    if decision.use_gaussian:
        out = cv.GaussianBlur(out, (3, 3), sigmaX=0.4)

    if decision.use_clahe:
        out = _apply_clahe_l_channel(
            out,
            clip_limit=2.0,
            tile_grid_size=(8, 8),
        )

    if decision.use_sharpen:
        out = _apply_unsharp_mask(
            out,
            amount=0.50,
            sigma=1.0,
        )

    return out


def apply_preprocess_by_decision(
    image: ImageInput,
    decision: PreprocessDecision,
    engine: str = "paddle",
) -> Dict[str, np.ndarray]:
    """
    Generate preprocessing variants from router decision.

    Returns
    -------
    Dict[str, np.ndarray]
        {
            "raw_resized": image after safe resize,
            "soft_enhanced": resized + soft enhancement,
            "hard_fallback": resized + threshold fallback
        }

    In the actual OCR pipeline, you should run:
    - decision.primary_variant first
    - hard_fallback only if OCR result is bad
    """

    img = _read_bgr(image)

    raw_resized = _resize_by_scale(img, decision.scale)
    soft_enhanced = _soft_enhance(raw_resized, decision)

    hard_fallback = _make_hard_threshold(
        raw_resized,
        mode=decision.hard_threshold_mode,
    )

    return {
        "raw_resized": raw_resized,
        "soft_enhanced": soft_enhanced,
        "hard_fallback": hard_fallback,
    }


__all__ = [
    "ImageQualityFeatures",
    "PreprocessDecision",
    "classify_image_quality",
    "apply_preprocess_by_decision",
]