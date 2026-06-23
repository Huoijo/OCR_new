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

    # V2: do not punish hard fallback by default.
    # The latest audit showed hard_fallback often wins on uneven-light images.
    hard_fallback_penalty: float = 0.0

    bad_result_penalty: float = 0.25

    # V2 selection policy:
    # - False: compare all available variants by score.
    # - True: keep primary if it is good and close enough to the best score.
    prefer_primary_if_good: bool = False
    primary_keep_margin: float = 0.03

    # Mild priors from current audit:
    # soft_enhanced was over-selected, so give it a tiny penalty.
    # raw_resized is a safe baseline, so give it a tiny bonus.
    soft_enhanced_penalty: float = 0.015
    raw_resized_bonus: float = 0.005

    # Penalize outputs that look like over-reading too much background text.
    excessive_alnum_threshold: int = 180
    excessive_boxes_threshold: int = 40
    max_excessive_text_penalty: float = 0.10
    max_excessive_box_penalty: float = 0.06

    # V3 hard-first policy.
    # The 200-image hard-only audit showed hard_fallback is the strongest
    # default variant. We therefore treat hard_fallback as primary and only
    # fallback when hard has a fatal failure.
    use_hard_first: bool = True
    hard_first_variant: str = "hard_fallback"
    hard_first_fallback_order: Tuple[str, ...] = (
        "raw_resized",
        "soft_enhanced",
    )

    # Reasons that mean the OCR result is unusable and should trigger fallback.
    # CJK/symbol/junk-heavy text is suspicious, but not fatal for OCR text,
    # because some thumbnails contain visible non-Vietnamese characters.
    fatal_reasons: Tuple[str, ...] = (
        "ocr_error",
        "no_or_too_few_boxes",
        "text_too_short",
        "too_few_alnum_chars",
        "very_low_avg_conf",
        "low_avg_conf_short_text",
        "ultra_short_known_junk_token",
        "no_alnum_only_symbols",
    )

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


def is_fatal_quality_report(
    report: OCRQualityReport,
    config: OCRQualityConfig = DEFAULT_OCR_QUALITY_CONFIG,
) -> bool:
    """
    Return True if an OCR result is unusable and should trigger fallback.

    Important:
    - `report.is_bad` can be True for suspicious-but-usable text.
    - Fatal means the result is empty, too short, too low-confidence, or errored.
    - CJK/symbol/junk-heavy text is not fatal by itself for OCR text.
    """

    if report is None:
        return True

    fatal = set(config.fatal_reasons)

    return any(reason in fatal for reason in report.reasons)


def is_fatal_ocr_result(
    result: OCRResult,
    config: OCRQualityConfig = DEFAULT_OCR_QUALITY_CONFIG,
    variant_name: str = "",
) -> bool:
    """
    Boolean shortcut for fatal OCR failure.
    """

    report = evaluate_ocr_result(
        result,
        config=config,
        variant_name=variant_name,
    )

    return is_fatal_quality_report(report, config=config)


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

    # V2: avoid rewarding very long OCR too much.
    # In thumbnail OCR, over-reading background/UI text can hurt CER badly.
    length_score = min(stats.alnum_chars / 60.0, 1.0)
    box_score = min((result.n_boxes or 0) / 10.0, 1.0)
    clean_score = 1.0 - min(stats.junk_ratio, 1.0)

    score = (
        0.45 * conf_score
        + 0.20 * length_score
        + 0.10 * box_score
        + 0.25 * clean_score
    )

    if reasons:
        score -= config.bad_result_penalty

    if variant_name == "hard_fallback":
        score -= config.hard_fallback_penalty

    if variant_name == "soft_enhanced":
        score -= config.soft_enhanced_penalty

    if variant_name == "raw_resized":
        score += config.raw_resized_bonus

    # Penalize extreme over-reading.
    if stats.alnum_chars > config.excessive_alnum_threshold:
        excess = stats.alnum_chars - config.excessive_alnum_threshold
        score -= min(excess / 600.0, config.max_excessive_text_penalty)

    n_boxes = int(result.n_boxes or 0)
    if n_boxes > config.excessive_boxes_threshold:
        excess_boxes = n_boxes - config.excessive_boxes_threshold
        score -= min(excess_boxes / 150.0, config.max_excessive_box_penalty)

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

    V3 logic:
    1. Evaluate all available variants.
    2. If hard_fallback exists and is not fatal, choose hard_fallback.
    3. If hard_fallback is fatal, choose the highest-score non-fatal fallback.
    4. If all variants are fatal, choose the highest score anyway.

    Why:
    - Hard-only audit on 200 samples showed hard_fallback had much lower CER.
    - Some `is_bad` reasons such as CJK/symbol are suspicious but not fatal.
    - We only fallback when hard is truly unusable: empty, too short, very low conf, etc.
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

    hard_variant = config.hard_first_variant

    if config.use_hard_first and hard_variant in reports:
        hard_report = reports[hard_variant]

        if not is_fatal_quality_report(hard_report, config=config):
            return OCRSelection(
                selected_variant=hard_variant,
                selected_result=results[hard_variant],
                selected_report=hard_report,
                reports=reports,
                results=dict(results),
                selection_reason="hard_first_non_fatal",
                tried_variants=list(results.keys()),
            )

    # Fallback: choose among non-fatal variants first.
    candidate_names = [
        name for name, report in reports.items()
        if not is_fatal_quality_report(report, config=config)
    ]

    if not candidate_names:
        candidate_names = list(reports.keys())
        selection_reason = "all_variants_fatal_choose_highest_score"
    else:
        selection_reason = "hard_fatal_choose_highest_score_non_fatal_fallback"

    best_variant = max(
        candidate_names,
        key=lambda name: reports[name].score,
    )

    return OCRSelection(
        selected_variant=best_variant,
        selected_result=results[best_variant],
        selected_report=reports[best_variant],
        reports=reports,
        results=dict(results),
        selection_reason=selection_reason,
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
    Production OCR flow, V3 hard-first.

    Flow:
    1. Run OCR on hard_fallback first if available.
    2. If hard is not fatal, stop and use hard.
    3. If hard is fatal, try raw_resized then soft_enhanced.
    4. Choose best among tried variants.

    This is faster than all-variant audit because most images need only one OCR call.
    """

    if not variants:
        raise ValueError("No preprocessing variants provided.")

    results: Dict[str, OCRResult] = {}
    reports: Dict[str, OCRQualityReport] = {}

    hard_variant = config.hard_first_variant

    if config.use_hard_first and hard_variant in variants:
        first_variant = hard_variant
    else:
        first_variant = _resolve_primary_variant(variants, decision)

    # 1. Run first variant.
    results[first_variant] = engine.recognize(variants[first_variant])
    reports[first_variant] = evaluate_ocr_result(
        results[first_variant],
        config=config,
        variant_name=first_variant,
    )

    # 2. If first variant is non-fatal, stop immediately.
    if not is_fatal_quality_report(reports[first_variant], config=config):
        return OCRSelection(
            selected_variant=first_variant,
            selected_result=results[first_variant],
            selected_report=reports[first_variant],
            reports=reports,
            results=results,
            selection_reason="hard_first_non_fatal_no_fallback_needed"
            if first_variant == hard_variant
            else "primary_non_fatal_no_fallback_needed",
            tried_variants=[first_variant],
        )

    # 3. Fatal failure: try fallbacks.
    fallback_order: List[str] = []

    if config.use_hard_first:
        fallback_order.extend(config.hard_first_fallback_order)
    else:
        if (
            config.try_raw_when_primary_bad
            and first_variant != "raw_resized"
            and "raw_resized" in variants
        ):
            fallback_order.append("raw_resized")

        use_hard_fallback = bool(
            getattr(decision, "use_hard_fallback", False)
        )

        if (
            use_hard_fallback
            and first_variant != "hard_fallback"
            and "hard_fallback" in variants
        ):
            fallback_order.append("hard_fallback")

    # Remove duplicates / missing variants while preserving order.
    seen = set(results.keys())
    fallback_order = [
        name for name in fallback_order
        if name in variants and not (name in seen or seen.add(name))
    ]

    for variant_name in fallback_order:
        results[variant_name] = engine.recognize(variants[variant_name])
        reports[variant_name] = evaluate_ocr_result(
            results[variant_name],
            config=config,
            variant_name=variant_name,
        )

        # Stop early once a fallback is non-fatal.
        if not is_fatal_quality_report(reports[variant_name], config=config):
            return OCRSelection(
                selected_variant=variant_name,
                selected_result=results[variant_name],
                selected_report=reports[variant_name],
                reports=reports,
                results=results,
                selection_reason=f"first_non_fatal_fallback_after_{first_variant}_fatal",
                tried_variants=list(results.keys()),
            )

    # 4. If all tried variants are fatal, choose highest score among tried.
    return choose_best_ocr_result(
        results=results,
        primary_variant=first_variant,
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
                "is_fatal": is_fatal_quality_report(report),
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
    "is_fatal_quality_report",
    "is_fatal_ocr_result",
    "score_ocr_result",
    "choose_best_ocr_result",
    "run_ocr_with_quality_gate",
    "quality_reports_to_rows",
]
