from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .engine import BaseOCREngine, OCRResult


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class OCRQualityConfig:
    """
    Config for OCR quality gate.

    The thresholds are intentionally conservative.
    They are not final competition metrics.
    They are used to decide whether an OCR result is obviously bad.
    """

    min_boxes: int = 1
    min_text_chars: int = 3
    min_alnum_chars: int = 4

    min_avg_conf: float = 0.35
    very_low_avg_conf: float = 0.18

    long_text_alnum_threshold: int = 20

    max_junk_ratio: float = 0.35
    max_symbol_ratio: float = 0.55
    max_cjk_ratio: float = 0.20
    max_emoji_ratio: float = 0.20

    hard_fallback_penalty: float = 0.04
    bad_result_penalty: float = 0.25

    ultra_short_junk_tokens: Tuple[str, ...] = (
        "心",
        "福",
        "★",
        "O",
        "Y",
        "5",
        "CS",
        "24",
    )

    # If primary is bad, should we also try raw_resized?
    # This is useful when primary_variant is soft_enhanced but soft hurts OCR.
    try_raw_when_primary_bad: bool = True


DEFAULT_OCR_QUALITY_CONFIG = OCRQualityConfig()


# ---------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------

@dataclass
class TextJunkStats:
    """
    Statistics about OCR text quality.
    """

    text_length: int = 0
    alnum_chars: int = 0
    alpha_chars: int = 0
    digit_chars: int = 0
    space_chars: int = 0

    cjk_chars: int = 0
    kana_chars: int = 0
    hangul_chars: int = 0
    emoji_chars: int = 0
    symbol_chars: int = 0
    junk_chars: int = 0

    alnum_ratio: float = 0.0
    cjk_ratio: float = 0.0
    emoji_ratio: float = 0.0
    symbol_ratio: float = 0.0
    junk_ratio: float = 0.0


@dataclass
class OCRQualityReport:
    """
    Quality report for one OCR result.
    """

    is_bad: bool
    score: float
    reasons: List[str]

    text_stats: TextJunkStats
    avg_conf: float
    n_boxes: int
    n_chars: int
    n_alnum: int

    variant_name: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OCRSelection:
    """
    Final selection after quality gate.
    """

    selected_variant: str
    selected_result: OCRResult
    selected_report: OCRQualityReport

    reports: Dict[str, OCRQualityReport]
    results: Dict[str, OCRResult]

    selection_reason: str
    tried_variants: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_variant": self.selected_variant,
            "selected_text": self.selected_result.text,
            "selected_avg_conf": self.selected_result.avg_conf,
            "selected_n_boxes": self.selected_result.n_boxes,
            "selected_score": self.selected_report.score,
            "selection_reason": self.selection_reason,
            "tried_variants": self.tried_variants,
            "reports": {
                k: v.to_dict()
                for k, v in self.reports.items()
            },
        }


# ---------------------------------------------------------------------
# Unicode / junk detection
# ---------------------------------------------------------------------

_ALLOWED_PUNCT = set(
    ".,:;!?%/\\-+_()[]{}#@&'\"`~|=<>"
)


def normalize_text_for_quality(text: str) -> str:
    """
    Normalize OCR text before quality analysis.

    NFC keeps Vietnamese characters in stable composed form.
    """

    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.strip()

    return text


def _char_category(ch: str) -> str:
    try:
        return unicodedata.category(ch)
    except Exception:
        return "Cn"


def _is_cjk(ch: str) -> bool:
    code = ord(ch)

    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
    )


def _is_kana(ch: str) -> bool:
    code = ord(ch)

    return (
        0x3040 <= code <= 0x309F  # Hiragana
        or 0x30A0 <= code <= 0x30FF  # Katakana
        or 0x31F0 <= code <= 0x31FF
    )


def _is_hangul(ch: str) -> bool:
    code = ord(ch)

    return (
        0xAC00 <= code <= 0xD7AF
        or 0x1100 <= code <= 0x11FF
        or 0x3130 <= code <= 0x318F
    )


def _is_emoji(ch: str) -> bool:
    code = ord(ch)

    return (
        0x1F300 <= code <= 0x1FAFF
        or 0x2600 <= code <= 0x27BF
    )


def _is_allowed_punctuation(ch: str) -> bool:
    return ch in _ALLOWED_PUNCT


def _is_symbol_like_junk(ch: str) -> bool:
    """
    Detect symbol-like OCR noise.

    Vietnamese letters with accents are safe because they are alphabetic.
    """

    if ch.isalnum():
        return False

    if ch.isspace():
        return False

    if _is_allowed_punctuation(ch):
        return False

    cat = _char_category(ch)

    if cat.startswith("S"):
        return True

    if cat.startswith("C"):
        return True

    return False


def analyze_text_junk(text: str) -> TextJunkStats:
    """
    Analyze OCR text for junk characters.

    Important:
    - Vietnamese letters are not treated as junk.
    - Chinese/Japanese/Korean characters are treated as suspicious for this task.
    - Emoji/symbol-heavy outputs are suspicious.
    """

    text = normalize_text_for_quality(text)

    stats = TextJunkStats()
    stats.text_length = len(text)

    if stats.text_length == 0:
        return stats

    for ch in text:
        if ch.isalnum():
            stats.alnum_chars += 1

        if ch.isalpha():
            stats.alpha_chars += 1

        if ch.isdigit():
            stats.digit_chars += 1

        if ch.isspace():
            stats.space_chars += 1

        is_cjk = _is_cjk(ch)
        is_kana = _is_kana(ch)
        is_hangul = _is_hangul(ch)
        is_emoji = _is_emoji(ch)
        is_symbol = _is_symbol_like_junk(ch)

        if is_cjk:
            stats.cjk_chars += 1

        if is_kana:
            stats.kana_chars += 1

        if is_hangul:
            stats.hangul_chars += 1

        if is_emoji:
            stats.emoji_chars += 1

        if is_symbol:
            stats.symbol_chars += 1

        if is_cjk or is_kana or is_hangul or is_emoji or is_symbol:
            stats.junk_chars += 1

    denom = max(stats.text_length, 1)

    stats.alnum_ratio = stats.alnum_chars / denom
    stats.cjk_ratio = (
        stats.cjk_chars + stats.kana_chars + stats.hangul_chars
    ) / denom
    stats.emoji_ratio = stats.emoji_chars / denom
    stats.symbol_ratio = stats.symbol_chars / denom
    stats.junk_ratio = stats.junk_chars / denom

    return stats


def _compact_upper(text: str) -> str:
    text = normalize_text_for_quality(text)
    text = re.sub(r"\s+", "", text)
    return text.upper()


# ---------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------

def evaluate_ocr_result(
    result: OCRResult,
    config: OCRQualityConfig = DEFAULT_OCR_QUALITY_CONFIG,
    variant_name: str = "",
) -> OCRQualityReport:
    """
    Evaluate one OCRResult and return detailed quality report.
    """

    reasons: List[str] = []

    if result is None:
        stats = TextJunkStats()
        return OCRQualityReport(
            is_bad=True,
            score=-1.0,
            reasons=["result_is_none"],
            text_stats=stats,
            avg_conf=0.0,
            n_boxes=0,
            n_chars=0,
            n_alnum=0,
            variant_name=variant_name,
            error="result_is_none",
        )

    text = normalize_text_for_quality(result.text)
    stats = analyze_text_junk(text)

    avg_conf = float(result.avg_conf or 0.0)
    n_boxes = int(result.n_boxes or 0)

    if result.error:
        reasons.append("ocr_error")

    if n_boxes < config.min_boxes:
        reasons.append("no_or_too_few_boxes")

    if len(text) < config.min_text_chars:
        reasons.append("text_too_short")

    if stats.alnum_chars < config.min_alnum_chars:
        reasons.append("too_few_alnum_chars")

    if avg_conf < config.very_low_avg_conf:
        reasons.append("very_low_avg_conf")
    elif (
        avg_conf < config.min_avg_conf
        and stats.alnum_chars < config.long_text_alnum_threshold
    ):
        reasons.append("low_avg_conf_short_text")

    if stats.cjk_ratio > config.max_cjk_ratio:
        reasons.append("too_many_cjk_kana_hangul_chars")

    if stats.emoji_ratio > config.max_emoji_ratio:
        reasons.append("too_many_emoji_chars")

    if stats.symbol_ratio > config.max_symbol_ratio:
        reasons.append("too_many_symbol_chars")

    if stats.junk_ratio > config.max_junk_ratio:
        reasons.append("too_much_junk")

    compact = _compact_upper(text)

    if compact in config.ultra_short_junk_tokens:
        reasons.append("ultra_short_known_junk_token")

    if stats.alnum_chars == 0 and len(text) > 0:
        reasons.append("no_alnum_only_symbols")

    score = score_ocr_result(
        result=result,
        config=config,
        variant_name=variant_name,
        precomputed_stats=stats,
        precomputed_reasons=reasons,
    )

    return OCRQualityReport(
        is_bad=len(reasons) > 0,
        score=score,
        reasons=reasons,
        text_stats=stats,
        avg_conf=avg_conf,
        n_boxes=n_boxes,
        n_chars=len(text),
        n_alnum=stats.alnum_chars,
        variant_name=variant_name,
        error=result.error,
    )


def is_bad_ocr_result(
    result: OCRResult,
    config: OCRQualityConfig = DEFAULT_OCR_QUALITY_CONFIG,
) -> bool:
    """
    Boolean shortcut.

    Returns True if OCR result is obviously bad.
    """

    return evaluate_ocr_result(result, config=config).is_bad


def score_ocr_result(
    result: OCRResult,
    config: OCRQualityConfig = DEFAULT_OCR_QUALITY_CONFIG,
    variant_name: str = "",
    precomputed_stats: Optional[TextJunkStats] = None,
    precomputed_reasons: Optional[List[str]] = None,
) -> float:
    """
    Heuristic score for comparing OCR variants.

    Higher is better.

    This score is not the competition metric.
    It is only for choosing between raw / soft / hard OCR outputs.
    """

    if result is None:
        return -1.0

    if result.error:
        return -1.0

    text = normalize_text_for_quality(result.text)

    if precomputed_stats is None:
        stats = analyze_text_junk(text)
    else:
        stats = precomputed_stats

    if precomputed_reasons is None:
        reasons = []
    else:
        reasons = precomputed_reasons

    if len(text) == 0:
        return 0.0

    conf_score = float(np.clip(result.avg_conf or 0.0, 0.0, 1.0))
    length_score = min(stats.alnum_chars / 80.0, 1.0)
    box_score = min((result.n_boxes or 0) / 12.0, 1.0)
    clean_score = 1.0 - min(stats.junk_ratio, 1.0)

    score = (
        0.60 * conf_score
        + 0.20 * length_score
        + 0.10 * box_score
        + 0.10 * clean_score
    )

    if reasons:
        score -= config.bad_result_penalty

    if variant_name == "hard_fallback":
        score -= config.hard_fallback_penalty

    score = float(np.clip(score, -1.0, 1.0))

    return score


# ---------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------

def choose_best_ocr_result(
    results: Mapping[str, OCRResult],
    primary_variant: str,
    config: OCRQualityConfig = DEFAULT_OCR_QUALITY_CONFIG,
) -> OCRSelection:
    """
    Choose final OCR result from already-computed variant results.

    Logic:
    1. If primary variant is good, keep primary.
    2. Otherwise choose the result with highest heuristic score.
    """

    if not results:
        raise ValueError("No OCR results provided.")

    reports: Dict[str, OCRQualityReport] = {}

    for variant_name, result in results.items():
        reports[variant_name] = evaluate_ocr_result(
            result,
            config=config,
            variant_name=variant_name,
        )

    if primary_variant in reports:
        primary_report = reports[primary_variant]

        if not primary_report.is_bad:
            return OCRSelection(
                selected_variant=primary_variant,
                selected_result=results[primary_variant],
                selected_report=primary_report,
                reports=reports,
                results=dict(results),
                selection_reason="primary_variant_is_good",
                tried_variants=list(results.keys()),
            )

    best_variant = max(
        reports.keys(),
        key=lambda name: reports[name].score,
    )

    return OCRSelection(
        selected_variant=best_variant,
        selected_result=results[best_variant],
        selected_report=reports[best_variant],
        reports=reports,
        results=dict(results),
        selection_reason="primary_variant_bad_choose_highest_score",
        tried_variants=list(results.keys()),
    )


def _resolve_primary_variant(
    variants: Mapping[str, Any],
    decision: Any,
) -> str:
    """
    Resolve primary variant from router decision.
    """

    primary = getattr(decision, "primary_variant", None)

    if primary in variants:
        return primary

    if "raw_resized" in variants:
        return "raw_resized"

    return list(variants.keys())[0]


def run_ocr_with_quality_gate(
    variants: Mapping[str, Any],
    decision: Any,
    engine: BaseOCREngine,
    config: OCRQualityConfig = DEFAULT_OCR_QUALITY_CONFIG,
) -> OCRSelection:
    """
    Production-like OCR flow.

    Flow:
    1. Run OCR on router primary_variant.
    2. If primary result is good, stop.
    3. If primary is bad:
       - optionally try raw_resized
       - try hard_fallback only if decision.use_hard_fallback is True
    4. Choose best result among tried variants.

    This avoids running OCR on every variant for every image.
    """

    if not variants:
        raise ValueError("No preprocessing variants provided.")

    primary_variant = _resolve_primary_variant(variants, decision)

    results: Dict[str, OCRResult] = {}

    # 1. Run primary first
    results[primary_variant] = engine.recognize(variants[primary_variant])

    primary_report = evaluate_ocr_result(
        results[primary_variant],
        config=config,
        variant_name=primary_variant,
    )

    # 2. If primary is good, stop immediately
    if not primary_report.is_bad:
        return OCRSelection(
            selected_variant=primary_variant,
            selected_result=results[primary_variant],
            selected_report=primary_report,
            reports={primary_variant: primary_report},
            results=results,
            selection_reason="primary_variant_is_good_no_fallback_needed",
            tried_variants=[primary_variant],
        )

    # 3. Primary is bad, prepare fallbacks
    fallback_order: List[str] = []

    if (
        config.try_raw_when_primary_bad
        and primary_variant != "raw_resized"
        and "raw_resized" in variants
    ):
        fallback_order.append("raw_resized")

    use_hard_fallback = bool(
        getattr(decision, "use_hard_fallback", False)
    )

    if (
        use_hard_fallback
        and primary_variant != "hard_fallback"
        and "hard_fallback" in variants
    ):
        fallback_order.append("hard_fallback")

    # Remove duplicates while preserving order
    seen = set(results.keys())
    fallback_order = [
        name for name in fallback_order
        if not (name in seen or seen.add(name))
    ]

    for variant_name in fallback_order:
        results[variant_name] = engine.recognize(variants[variant_name])

    # 4. Choose best among tried variants
    return choose_best_ocr_result(
        results=results,
        primary_variant=primary_variant,
        config=config,
    )


# ---------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------

def quality_reports_to_rows(
    reports: Mapping[str, OCRQualityReport],
) -> List[Dict[str, Any]]:
    """
    Convert reports to flat rows for CSV/debug table.
    """

    rows = []

    for variant_name, report in reports.items():
        stats = report.text_stats

        rows.append(
            {
                "variant": variant_name,
                "is_bad": report.is_bad,
                "score": report.score,
                "reasons": "|".join(report.reasons),

                "avg_conf": report.avg_conf,
                "n_boxes": report.n_boxes,
                "n_chars": report.n_chars,
                "n_alnum": report.n_alnum,

                "junk_ratio": stats.junk_ratio,
                "symbol_ratio": stats.symbol_ratio,
                "cjk_ratio": stats.cjk_ratio,
                "emoji_ratio": stats.emoji_ratio,

                "error": report.error,
            }
        )

    return rows


__all__ = [
    "OCRQualityConfig",
    "OCRQualityReport",
    "OCRSelection",
    "TextJunkStats",
    "DEFAULT_OCR_QUALITY_CONFIG",
    "normalize_text_for_quality",
    "analyze_text_junk",
    "evaluate_ocr_result",
    "is_bad_ocr_result",
    "score_ocr_result",
    "choose_best_ocr_result",
    "run_ocr_with_quality_gate",
    "quality_reports_to_rows",
]
