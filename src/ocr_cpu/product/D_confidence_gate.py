from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from .A_candidate import normalize_light, normalize_dedupe_key, tokenize_text
from .B_gazeetteer import fold_key, fold_token_key
from .rules import (
    apply_all_rules,
    is_generic_topic_only,
    should_reject_short_candidate,
)

log = __import__("logging").getLogger("ocr_cpu.product.gate")


# ---------------------------------------------------------------------
# Context lexicons
# ---------------------------------------------------------------------

# These lists are deliberately conservative. They do not decide alone; they
# only act as penalty/bonus evidence in a precision-first gate.
NEGATIVE_CONTEXT_PHRASES: Tuple[str, ...] = (
    # news/legal/crisis context
    "pháp luật",
    "phap luat",
    "khởi tố",
    "khoi to",
    "bắt khẩn cấp",
    "bat khan cap",
    "bắt giam",
    "bat giam",
    "hối lộ",
    "hoi lo",
    "cán bộ",
    "can bo",
    "công an",
    "cong an",
    "công ty cổ phần",
    "cong ty co phan",
    "tổng giám đốc",
    "tong giam doc",
    "giám đốc",
    "giam doc",
    "chi cục thú y",
    "chi cuc thu y",
    "kiểm dịch",
    "kiem dich",
    "virus",
    "vius",
    "viruss",
    "dịch tả",
    "dich ta",
    "tả lợn",
    "ta lon",
    "thịt lợn bệnh",
    "thit lon benh",
    "thu hồi",
    "thu hoi",
    "cảnh báo",
    "canh bao",
    "tranh cãi",
    "tranh cai",
    "xoá bài",
    "xoa bai",
    "bài viết",
    "bai viet",
    "tin nhanh",
    "nguồn",
    "nguon",
    "news",
    "báo",
    "bao",
)

POSITIVE_CONTEXT_PHRASES: Tuple[str, ...] = (
    # product / eating / buying / packaging context
    "review",
    "mukbang",
    "ăn thử",
    "an thu",
    "uống thử",
    "uong thu",
    "ngon",
    "mua",
    "giá",
    "gia",
    "sale",
    "chính hãng",
    "chinh hang",
    "hộp",
    "hop",
    "lon",
    "sữa",
    "sua",
    "pate",
    "patê",
    "bánh mì",
    "banh mi",
    "trà",
    "tra",
    "coffee",
    "vị",
    "vi",
)

# Product-specific token anchors. These help distinguish real product mentions
# from generic company/news mentions. Folded forms are used for matching.
PRODUCT_SPECIFIC_TOKEN_KEYS: Tuple[str, ...] = tuple(
    sorted(
        {
            "pate",
            "cot",
            "den",
            "hai",
            "phong",
            "do",
            "hop",
            "ha",
            "long",
            "halong",
            "canfoco",
            "nan",
            "nestle",
            "optipro",
            "infinipro",
            "a2",
            "sua",
            "highland",
            "highlands",
            "coffee",
            "tra",
            "sen",
            "vang",
            "banh",
            "mi",
            "que",
        }
    )
)

HIGH_TRUST_MATCH_TYPES: Tuple[str, ...] = (
    "exact",
    "surface_exact",
    "folded_exact",
    "folded_surface_exact",
)

FUZZY_PARTIAL_TYPES: Tuple[str, ...] = (
    "fuzzy_partial",
    "folded_fuzzy_partial",
)


# ---------------------------------------------------------------------
# Config / result dataclasses
# ---------------------------------------------------------------------

@dataclass
class GateConfig:
    """
    Confidence gating thresholds.

    Keep defaults conservative. Cell 6 will tune them on validation.
    """

    # Match thresholds from Cell 3.
    fuzzy_threshold: float = 88.0
    strong_threshold: float = 94.0

    # Overall gate score thresholds.
    accept_gate_score: float = 0.72
    accept_weak_gate_score: float = 0.82

    # Ambiguity controls.
    min_margin_for_fuzzy: float = 5.0
    min_margin_for_weak: float = 8.0
    low_margin_threshold: float = 3.0

    # Negative context controls.
    strong_negative_count: int = 3
    moderate_negative_count: int = 2

    # Minimum candidate evidence.
    min_match_score: float = 88.0
    min_folded_token_overlap: int = 2
    min_candidate_token_coverage: float = 0.50

    # Weighting for candidate gate score.
    w_match: float = 0.42
    w_margin: float = 0.14
    w_structure: float = 0.10
    w_source: float = 0.08
    w_ocr: float = 0.08
    w_variant: float = 0.05
    w_position: float = 0.04
    w_prior: float = 0.04
    w_positive_context: float = 0.05

    # Penalties.
    negative_penalty_per_hit: float = 0.06
    partial_match_penalty: float = 0.04
    low_margin_penalty: float = 0.06
    one_variant_no_bonus: bool = True

    # Full-vs-compose (multi-item).
    full_coverage_threshold: float = 0.70   # fraction of full-entry tokens seen in image
    min_corroborating_parts: int = 2        # high-trust atomic items that confirm a full entry
    max_compose_items: int = 4              # cap items when composing from atomics
    compose_join: str = " + "               # competition multi-item separator
    single_ambiguity_gap: float = 0.05      # gate-score gap below which a rival flags ambiguity

    # Debug output.
    top_k_debug_candidates: int = 10

    # Centralized rule registry (product/rules.py) integration.
    use_rules: bool = True               # apply rule override / short-candidate reject / generic gate
    rule_override_priority: int = 90     # a rule hit >= this forces its canonical (beats fuzzy)
    rule_strong_priority: int = 95       # below this, a generic-topic image is blanked


@dataclass
class GateDecision:
    image_id: str
    product_name_candidate: str
    emit_product: bool

    gate_score: float
    gate_decision: str
    gate_reason: str

    selected_candidate_index: Optional[int] = None
    matched_entry_id: str = ""
    matched_display: str = ""
    match_score: float = 0.0
    match_margin: float = 0.0
    match_type: str = ""

    negative_context_count: int = 0
    positive_context_count: int = 0
    negative_context_hits: List[str] = field(default_factory=list)
    positive_context_hits: List[str] = field(default_factory=list)

    # Full-vs-compose / per-entry aggregation outputs.
    compose_mode: str = "none"  # single | multi | none
    chosen_entry_ids: List[str] = field(default_factory=list)
    chosen_displays: List[str] = field(default_factory=list)
    n_matched_candidates: int = 0
    n_distinct_entries: int = 0
    is_ambiguous: bool = False

    top_candidates_json: str = "[]"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _fold_for_context(text: Any) -> str:
    # Put spaces around folded key to simplify substring phrase matching.
    text = fold_key(_safe_str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _phrase_hits(text: str, phrases: Sequence[str]) -> List[str]:
    folded_text = f" {_fold_for_context(text)} "
    hits = []
    for p in phrases:
        fp = _fold_for_context(p)
        if not fp:
            continue
        if f" {fp} " in folded_text or fp in folded_text:
            hits.append(p)
    # Deduplicate while preserving order.
    seen = set()
    out = []
    for h in hits:
        k = _fold_for_context(h)
        if k not in seen:
            seen.add(k)
            out.append(h)
    return out


def _tokens_from_row(row: pd.Series) -> List[str]:
    val = row.get("tokenized", [])
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val]
    return tokenize_text(_safe_str(row.get("clean_text", "")))


def _folded_token_keys_from_row(row: pd.Series) -> List[str]:
    toks = _tokens_from_row(row)
    out = []
    for t in toks:
        k = fold_token_key(t)
        if k:
            out.append(k)
    return sorted(set(out))


def _source_score(row: pd.Series) -> float:
    if "source_score" in row.index:
        return _clamp01(_safe_float(row.get("source_score"), 0.0))

    source = _safe_str(row.get("source", ""))
    mapping = {
        "line": 1.0,
        "whole_line": 0.95,
        "segment": 0.85,
        "ngram": 0.70,
    }
    return mapping.get(source, 0.60)


def _variant_score(row: pd.Series, config: GateConfig) -> float:
    available = _safe_int(row.get("available_variant_count"), 0)
    agreement = _safe_float(row.get("variant_agreement_score"), 0.0)
    support = _safe_int(row.get("variant_support_count"), 1)

    if available < 2 and config.one_variant_no_bonus:
        return 0.0

    if available >= 2:
        return _clamp01(agreement)

    # Fallback: a tiny signal if field exists but available count is missing.
    return 0.15 if support >= 2 else 0.0


def _prior_score(freq: int, max_freq: int) -> float:
    if max_freq <= 0 or freq <= 0:
        return 0.0
    return _clamp01(math.log1p(freq) / math.log1p(max_freq))


# ---------------------------------------------------------------------
# Context scoring
# ---------------------------------------------------------------------


def image_text_from_group(group: pd.DataFrame) -> str:
    """
    Combine candidate texts for context detection.

    We use clean_text/raw_text from candidates because Cell 4 should not depend
    on raw OCR CSV. This is enough to catch news/legal context terms that were
    generated as candidates or line records.
    """

    parts: List[str] = []
    for col in ["clean_text", "raw_text"]:
        if col in group.columns:
            vals = group[col].dropna().astype(str).tolist()
            parts.extend(vals)
    return "\n".join(parts)


def compute_context_evidence(group: pd.DataFrame) -> Dict[str, Any]:
    text = image_text_from_group(group)

    negative_hits = _phrase_hits(text, NEGATIVE_CONTEXT_PHRASES)
    positive_hits = _phrase_hits(text, POSITIVE_CONTEXT_PHRASES)

    return {
        "image_context_text": text,
        "negative_context_hits": negative_hits,
        "positive_context_hits": positive_hits,
        "negative_context_count": len(negative_hits),
        "positive_context_count": len(positive_hits),
    }


# ---------------------------------------------------------------------
# Candidate evidence scoring
# ---------------------------------------------------------------------


def candidate_has_product_specific_evidence(row: pd.Series) -> bool:
    """
    Product-specific evidence means the candidate is not merely a vague brand
    or news/company mention. This is still heuristic; Cell 6 will tune.
    """

    match_type = _safe_str(row.get("match_type", ""))
    if match_type in HIGH_TRUST_MATCH_TYPES:
        return True

    folded_overlap = _safe_int(row.get("match_folded_token_overlap"), 0)
    candidate_cov = _safe_float(row.get("match_candidate_token_coverage"), 0.0)
    score = _safe_float(row.get("match_score"), 0.0)

    if score >= 94.0 and folded_overlap >= 2 and candidate_cov >= 0.50:
        return True

    folded_keys = set(_folded_token_keys_from_row(row))
    anchor_hits = folded_keys.intersection(PRODUCT_SPECIFIC_TOKEN_KEYS)

    # At least two product-specific anchors is a useful signal.
    return len(anchor_hits) >= 2 and score >= 88.0


def compute_candidate_gate_score(
    row: pd.Series,
    context: Mapping[str, Any],
    config: GateConfig,
    max_frequency_prior: int,
) -> Tuple[float, Dict[str, float], List[str]]:
    """
    Combine match evidence and OCR/candidate metadata into a 0..1 gate score.
    """

    reasons: List[str] = []

    match_score = _safe_float(row.get("match_score"), 0.0)
    match_margin = _safe_float(row.get("match_margin"), 0.0)
    match_type = _safe_str(row.get("match_type", ""))

    match_component = _clamp01(match_score / 100.0)
    margin_component = _clamp01(match_margin / 12.0)
    structure_component = _clamp01(_safe_float(row.get("structure_score"), 0.60))
    source_component = _source_score(row)

    ocr_conf = row.get("ocr_conf", None)
    ocr_component = _safe_float(ocr_conf, 0.65)
    if ocr_component <= 0.0:
        ocr_component = 0.55
    ocr_component = _clamp01(ocr_component)

    variant_component = _variant_score(row, config)

    pos = row.get("position_score", None)
    position_component = _safe_float(pos, 0.50)
    if position_component <= 0.0:
        position_component = 0.50
    position_component = _clamp01(position_component)

    freq = _safe_int(row.get("matched_frequency_prior"), 0)
    prior_component = _prior_score(freq, max_frequency_prior)

    positive_count = int(context.get("positive_context_count", 0) or 0)
    positive_component = _clamp01(positive_count / 3.0)

    score = (
        config.w_match * match_component
        + config.w_margin * margin_component
        + config.w_structure * structure_component
        + config.w_source * source_component
        + config.w_ocr * ocr_component
        + config.w_variant * variant_component
        + config.w_position * position_component
        + config.w_prior * prior_component
        + config.w_positive_context * positive_component
    )

    negative_count = int(context.get("negative_context_count", 0) or 0)
    if negative_count > 0:
        penalty = min(0.24, negative_count * config.negative_penalty_per_hit)
        score -= penalty
        reasons.append(f"negative_context_penalty:{penalty:.3f}")

    if match_type in FUZZY_PARTIAL_TYPES:
        score -= config.partial_match_penalty
        reasons.append(f"partial_match_penalty:{config.partial_match_penalty:.3f}")

    if match_margin < config.low_margin_threshold and match_type not in HIGH_TRUST_MATCH_TYPES:
        score -= config.low_margin_penalty
        reasons.append(f"low_margin_penalty:{config.low_margin_penalty:.3f}")

    components = {
        "match": round(match_component, 4),
        "margin": round(margin_component, 4),
        "structure": round(structure_component, 4),
        "source": round(source_component, 4),
        "ocr": round(ocr_component, 4),
        "variant": round(variant_component, 4),
        "position": round(position_component, 4),
        "prior": round(prior_component, 4),
        "positive_context": round(positive_component, 4),
    }

    return round(_clamp01(score), 4), components, reasons


# ---------------------------------------------------------------------
# Image-level decision
# ---------------------------------------------------------------------


def _top_candidates_debug(group: pd.DataFrame, top_k: int) -> str:
    cols = [
        "clean_text",
        "matched_display",
        "matched",
        "is_strong_match",
        "match_score",
        "match_margin",
        "match_type",
        "gate_score_candidate",
        "source",
        "ocr_conf",
        "structure_score",
        "match_debug",
    ]
    cols = [c for c in cols if c in group.columns]

    records = group[cols].head(top_k).to_dict(orient="records")
    return _json_dumps(records)


def _blank_decision(
    image_id: str,
    reason: str,
    context: Optional[Mapping[str, Any]] = None,
    top_candidates_json: str = "[]",
) -> GateDecision:
    context = context or {}
    return GateDecision(
        image_id=str(image_id),
        product_name_candidate="",
        emit_product=False,
        gate_score=0.0,
        gate_decision="blank",
        gate_reason=reason,
        negative_context_count=int(context.get("negative_context_count", 0) or 0),
        positive_context_count=int(context.get("positive_context_count", 0) or 0),
        negative_context_hits=list(context.get("negative_context_hits", []) or []),
        positive_context_hits=list(context.get("positive_context_hits", []) or []),
        top_candidates_json=top_candidates_json,
    )


# ---------------------------------------------------------------------
# Per-entry aggregation + full-vs-compose selection
# ---------------------------------------------------------------------


@dataclass
class EntryEvidence:
    """Aggregated evidence for one gazetteer entry within a single image."""

    entry_id: str
    display: str
    source_class: str
    item_count: int
    frequency_prior: int
    is_multi: bool
    folded_tokens: frozenset
    best_gate_score: float
    best_match_score: float
    best_margin: float
    best_match_type: str
    best_is_strong: bool
    best_has_specific: bool
    support: int
    sources: frozenset
    min_line_index: int
    best_row_index: Any

    @property
    def distinct_sources(self) -> int:
        return len(self.sources)


def _folded_tokens_of(text: Any) -> frozenset:
    return frozenset(k for k in (fold_token_key(t) for t in tokenize_text(_safe_str(text))) if k)


def _aggregate_entries(matched: pd.DataFrame) -> List[EntryEvidence]:
    """
    Group matched candidates by matched_entry_id and combine evidence.

    Multiple candidates (line / segment / ngram, across variants) can point to
    the same entry; we keep the best gate score and accumulate support.
    """

    by_id: Dict[str, Dict[str, Any]] = {}

    for idx, row in matched.iterrows():
        eid = _safe_str(row.get("matched_entry_id", ""))
        if not eid:
            continue

        gs = _safe_float(row.get("gate_score_candidate"), 0.0)
        disp = _safe_str(row.get("matched_display", ""))
        item_count = _safe_int(row.get("matched_item_count"), 1)
        is_multi = item_count >= 2 or (" + " in disp)
        line_index = _safe_int(row.get("line_index"), 10 ** 6)
        source = _safe_str(row.get("source", ""))

        bucket = by_id.get(eid)
        if bucket is None:
            by_id[eid] = {
                "entry_id": eid,
                "display": disp,
                "source_class": _safe_str(row.get("matched_source_class", "")),
                "item_count": item_count,
                "frequency_prior": _safe_int(row.get("matched_frequency_prior"), 0),
                "is_multi": is_multi,
                "folded_tokens": _folded_tokens_of(disp),
                "best_gate_score": gs,
                "best_match_score": _safe_float(row.get("match_score"), 0.0),
                "best_margin": _safe_float(row.get("match_margin"), 0.0),
                "best_match_type": _safe_str(row.get("match_type", "")),
                "best_is_strong": bool(row.get("is_strong_match", False)),
                "best_has_specific": bool(row.get("has_product_specific_evidence", False)),
                "support": 1,
                "sources": {source},
                "min_line_index": line_index,
                "best_row_index": idx,
            }
        else:
            bucket["support"] += 1
            bucket["sources"].add(source)
            bucket["min_line_index"] = min(bucket["min_line_index"], line_index)
            if gs > bucket["best_gate_score"]:
                bucket["best_gate_score"] = gs
                bucket["best_match_score"] = _safe_float(row.get("match_score"), 0.0)
                bucket["best_margin"] = _safe_float(row.get("match_margin"), 0.0)
                bucket["best_match_type"] = _safe_str(row.get("match_type", ""))
                bucket["best_is_strong"] = bool(row.get("is_strong_match", False))
                bucket["best_has_specific"] = bool(row.get("has_product_specific_evidence", False))
                bucket["best_row_index"] = idx

    entries = [
        EntryEvidence(
            entry_id=b["entry_id"],
            display=b["display"],
            source_class=b["source_class"],
            item_count=b["item_count"],
            frequency_prior=b["frequency_prior"],
            is_multi=b["is_multi"],
            folded_tokens=b["folded_tokens"],
            best_gate_score=b["best_gate_score"],
            best_match_score=b["best_match_score"],
            best_margin=b["best_margin"],
            best_match_type=b["best_match_type"],
            best_is_strong=b["best_is_strong"],
            best_has_specific=b["best_has_specific"],
            support=b["support"],
            sources=frozenset(b["sources"]),
            min_line_index=b["min_line_index"],
            best_row_index=b["best_row_index"],
        )
        for b in by_id.values()
    ]
    # Deterministic order.
    entries.sort(key=lambda e: (-e.best_gate_score, -e.best_match_score, e.entry_id))
    return entries


def _entry_qualifies(e: EntryEvidence, config: GateConfig) -> Tuple[bool, str]:
    """Precision-first per-entry gate (exact/surface matches are margin-exempt)."""

    if not e.display:
        return False, "empty_display"
    if e.best_match_score < config.min_match_score:
        return False, "below_min_match"

    high_trust = e.best_match_type in HIGH_TRUST_MATCH_TYPES
    if not high_trust and e.best_margin < config.min_margin_for_fuzzy:
        return False, "fuzzy_low_margin"
    if not e.best_is_strong and e.best_margin < config.min_margin_for_weak:
        return False, "weak_low_margin"

    required = config.accept_gate_score if e.best_is_strong else config.accept_weak_gate_score
    if e.best_gate_score < required:
        return False, "gate_below_threshold"

    return True, "ok"


def _image_folded_token_pool(matched: pd.DataFrame) -> frozenset:
    """Folded tokens actually seen in OCR (union over matched candidate clean_text)."""

    pool: set = set()
    for _, row in matched.iterrows():
        pool |= _folded_tokens_of(row.get("clean_text", ""))
    return frozenset(pool)


def _dedup_entries(entries: Sequence[EntryEvidence]) -> List[EntryEvidence]:
    """Drop entries whose token set is a subset of an already-kept (richer) entry."""

    # Richest first; among equal token sets prefer the canonical with the higher
    # frequency prior (clean Title-Case form, e.g. "Pate Cột Đèn Hải Phòng" over a
    # messy-casing duplicate), then gate score.
    ordered = sorted(
        entries,
        key=lambda e: (len(e.folded_tokens), e.frequency_prior, e.best_gate_score),
        reverse=True,
    )
    kept: List[EntryEvidence] = []
    for e in ordered:
        if not e.folded_tokens:
            continue
        if any(e.folded_tokens <= k.folded_tokens for k in kept):
            continue
        kept.append(e)
    return kept


def _is_ambiguous_single(best: EntryEvidence, qualified: Sequence[EntryEvidence], config: GateConfig) -> bool:
    for e in qualified:
        if e.entry_id == best.entry_id or e.folded_tokens == best.folded_tokens:
            continue
        # A different product (not a subset/superset) with near-equal gate score.
        nested = e.folded_tokens <= best.folded_tokens or best.folded_tokens <= e.folded_tokens
        if not nested and abs(e.best_gate_score - best.best_gate_score) <= config.single_ambiguity_gap:
            return True
    return False


def _select_full_or_compose(
    entries: List[EntryEvidence],
    qualified: List[EntryEvidence],
    pool: frozenset,
    config: GateConfig,
) -> Optional[Dict[str, Any]]:
    """
    Decide what an image emits, precision-first:
      1. a multi-item `full` entry that is either strictly qualified, OR corroborated
         (token coverage high + >=2 high-trust atomic items are subsets of it);
      2. else compose >=2 distinct qualified atomic entries with " + ";
      3. else the single best qualified entry.
    Returns None when nothing qualifies (caller emits blank).
    """

    qids = {e.entry_id for e in qualified}
    high_trust_q = [e for e in qualified if e.best_match_type in HIGH_TRUST_MATCH_TYPES]

    def coverage(e: EntryEvidence) -> float:
        if not e.folded_tokens:
            return 0.0
        return len(e.folded_tokens & pool) / len(e.folded_tokens)

    def n_corroborating(fe: EntryEvidence) -> int:
        return sum(
            1
            for a in high_trust_q
            if a.entry_id != fe.entry_id and a.folded_tokens and a.folded_tokens <= fe.folded_tokens
        )

    # 1. Full multi-item entry.
    acceptable_full: List[Tuple[EntryEvidence, float, int]] = []
    for fe in entries:
        if not fe.is_multi or not fe.folded_tokens:
            continue
        cov = coverage(fe)
        n_corr = n_corroborating(fe)
        strictly_q = fe.entry_id in qids
        corroborated = (
            fe.best_match_score >= config.fuzzy_threshold
            and cov >= config.full_coverage_threshold
            and n_corr >= config.min_corroborating_parts
        )
        if strictly_q or corroborated:
            acceptable_full.append((fe, cov, n_corr))

    if acceptable_full:
        fe, cov, n_corr = sorted(
            acceptable_full,
            key=lambda x: (x[1], len(x[0].folded_tokens), x[0].best_match_score, x[0].entry_id),
            reverse=True,
        )[0]
        return {
            "mode": "single",
            "entries": [fe],
            "is_ambiguous": False,
            "reason": f"full_entry:cov={cov:.2f},parts={n_corr}",
        }

    # 2. Compose distinct qualified atomic entries.
    atoms = _dedup_entries([e for e in qualified if not e.is_multi])
    if len(atoms) >= 2:
        chosen = sorted(atoms, key=lambda e: (e.min_line_index, -e.best_gate_score, e.entry_id))
        chosen = chosen[: config.max_compose_items]
        return {
            "mode": "multi",
            "entries": chosen,
            "is_ambiguous": False,
            "reason": f"compose_{len(chosen)}_atomics",
        }

    # 3. Single best qualified entry. Drop entries whose token set is a subset of
    #    a richer qualified entry first (a brand-only "PATE" is subsumed by the
    #    specific "Pate Cột Đèn Hải Phòng"), then prefer a high-trust exact match
    #    over a fuzzy one. This avoids losing the specific product to a generic
    #    brand whose fuzzy match happened to have a larger (spurious) margin.
    if qualified:
        candidates = _dedup_entries(qualified) or list(qualified)
        best = max(
            candidates,
            key=lambda e: (
                e.best_match_type in HIGH_TRUST_MATCH_TYPES,
                e.best_gate_score,
                e.best_match_score,
                e.frequency_prior,
                e.entry_id,
            ),
        )
        return {
            "mode": "single",
            "entries": [best],
            "is_ambiguous": _is_ambiguous_single(best, candidates, config),
            "reason": "single_best_qualified",
        }

    return None


def _rule_override_decision(
    image_id: str,
    hit: Dict[str, Any],
    context: Mapping[str, Any],
    top_candidates_json: str = "[]",
) -> GateDecision:
    """Force a high-confidence rule hit as the image decision (precision-first)."""
    canonical = hit["canonical"]
    log.info("[Gate] chosen=%s source=%s priority=%s (rule override)",
             canonical, hit.get("source"), hit.get("priority"))
    return GateDecision(
        image_id=str(image_id),
        product_name_candidate=canonical,
        emit_product=True,
        gate_score=1.0,
        gate_decision="emit",
        gate_reason="rule_override:%s|priority=%s|source=%s" % (
            hit.get("rule_name"), hit.get("priority"), hit.get("source")),
        matched_display=canonical,
        match_score=float(hit.get("score", 1.0)) * 100.0,
        match_type="rule:%s" % hit.get("source", "rule"),
        negative_context_count=int(context.get("negative_context_count", 0) or 0),
        positive_context_count=int(context.get("positive_context_count", 0) or 0),
        negative_context_hits=list(context.get("negative_context_hits", []) or []),
        positive_context_hits=list(context.get("positive_context_hits", []) or []),
        compose_mode="single",
        chosen_displays=[canonical],
        top_candidates_json=top_candidates_json,
    )


def _best_rule_hit(full_text: str) -> Optional[Dict[str, Any]]:
    """Top rule hit (after rejecting contextually-bad short candidates), or None."""
    hits = [h for h in apply_all_rules(full_text)
            if not should_reject_short_candidate(h["canonical"], full_text)]
    return hits[0] if hits else None


def gate_one_image(
    image_id: str,
    group: pd.DataFrame,
    config: GateConfig,
    max_frequency_prior: int,
    full_text: str = "",
) -> GateDecision:
    """
    Main image-level gate.

    Input group is all linked candidates for one image_id.
    Output is one decision: product_name_candidate or blank.

    full_text: optional raw OCR text for the image. When empty it is derived
    from candidate texts; it feeds the centralized rule registry (rules.py).
    """

    if group is None or group.empty:
        return _blank_decision(str(image_id), "no_candidates")

    context = compute_context_evidence(group)

    # ---- Centralized rule registry: precision-first override ----
    rule_text = full_text or context.get("image_context_text", "") or image_text_from_group(group)
    rule_hit = _best_rule_hit(rule_text) if config.use_rules else None
    if rule_hit is not None:
        log.info("[Rule] hit=%s canonical=%s priority=%s",
                 rule_hit.get("rule_name"), rule_hit.get("canonical"), rule_hit.get("priority"))
    if rule_hit is not None and int(rule_hit.get("priority", 0)) >= config.rule_override_priority:
        top_json = _top_candidates_debug(group.copy(), config.top_k_debug_candidates)
        return _rule_override_decision(str(image_id), rule_hit, context, top_json)

    matched = group[group.get("matched", False).fillna(False).astype(bool)].copy()
    if matched.empty:
        top_json = _top_candidates_debug(group.copy(), config.top_k_debug_candidates)
        return _blank_decision(
            str(image_id),
            "no_matched_candidates",
            context=context,
            top_candidates_json=top_json,
        )

    # Compute gate score for each matched candidate.
    gate_scores = []
    gate_components = []
    gate_reasons = []

    for _, row in matched.iterrows():
        score, components, reasons = compute_candidate_gate_score(
            row=row,
            context=context,
            config=config,
            max_frequency_prior=max_frequency_prior,
        )
        gate_scores.append(score)
        gate_components.append(components)
        gate_reasons.append("|".join(reasons))

    matched["gate_score_candidate"] = gate_scores
    matched["gate_components_json"] = [_json_dumps(x) for x in gate_components]
    matched["gate_candidate_penalties"] = gate_reasons

    # Product-specific evidence flag.
    matched["has_product_specific_evidence"] = matched.apply(candidate_has_product_specific_evidence, axis=1)

    # Debug table (matched first, ranked by gate score) + leftover candidates.
    debug_sort = [
        c for c in ["gate_score_candidate", "is_strong_match", "match_score", "match_margin"]
        if c in matched.columns
    ]
    debug_matched = matched.sort_values(debug_sort, ascending=False) if debug_sort else matched
    debug_group = pd.concat(
        [debug_matched, group[~group.index.isin(matched.index)].copy()], axis=0
    )
    top_json = _top_candidates_debug(debug_group, config.top_k_debug_candidates)

    neg_count = int(context.get("negative_context_count", 0) or 0)
    n_matched = int(len(matched))

    # ---- Per-entry aggregation + full-vs-compose selection ----
    entries = _aggregate_entries(matched)
    n_distinct = len(entries)

    qualified = [e for e in entries if _entry_qualifies(e, config)[0]]
    pool = _image_folded_token_pool(matched)
    selection = _select_full_or_compose(entries, qualified, pool, config)

    if selection is None:
        return _blank_decision(str(image_id), "no_qualified_entry", context, top_json)

    chosen: List[EntryEvidence] = selection["entries"]
    primary = chosen[0]

    # Negative-context precision-first blanks (image level).
    any_specific = any(e.best_has_specific for e in chosen)
    if neg_count >= config.strong_negative_count and not any_specific:
        return _blank_decision(
            str(image_id),
            f"strong_negative_context_without_specific_product_evidence:{neg_count}",
            context,
            top_json,
        )
    if (
        neg_count >= config.moderate_negative_count
        and primary.best_match_type in FUZZY_PARTIAL_TYPES
        and primary.best_margin < 10.0
    ):
        return _blank_decision(
            str(image_id),
            "negative_context_partial_match_low_margin",
            context,
            top_json,
        )

    # Generic topic-only precision guard: a news/topic headline with only a
    # weak fuzzy match and no strong rule should blank (precision-first).
    if (
        config.use_rules
        and is_generic_topic_only(rule_text)
        and not (rule_hit and int(rule_hit.get("priority", 0)) >= config.rule_strong_priority)
        and primary.best_match_type in FUZZY_PARTIAL_TYPES
        and not any_specific
    ):
        log.info("[Gate] generic topic only + weak fuzzy -> blank (image=%s)", image_id)
        return _blank_decision(
            str(image_id),
            "generic_topic_only_weak_fuzzy",
            context,
            top_json,
        )

    # Compose final product_name candidate (Cell 5 still does Title Case / brand fix).
    if selection["mode"] == "multi":
        product_name = config.compose_join.join(e.display for e in chosen)
        compose_mode = "multi"
        image_conf = min(e.best_gate_score for e in chosen)
        rep_match_type = "compose_multi"
        rep_match_score = min(e.best_match_score for e in chosen)
        rep_margin = min(e.best_margin for e in chosen)
    else:
        product_name = primary.display
        compose_mode = "single"
        image_conf = primary.best_gate_score
        rep_match_type = primary.best_match_type
        rep_match_score = primary.best_match_score
        rep_margin = primary.best_margin

    reason_parts = [
        f"accepted:{selection['reason']}",
        f"mode={compose_mode}",
        f"match_score={rep_match_score:.2f}",
        f"gate_score={image_conf:.3f}",
    ]
    if neg_count:
        reason_parts.append(f"negative_context={neg_count}")
    if selection.get("is_ambiguous"):
        reason_parts.append("ambiguous=True")
    if any_specific:
        reason_parts.append("specific_product_evidence=True")

    sel_idx = primary.best_row_index if isinstance(primary.best_row_index, int) else None

    return GateDecision(
        image_id=str(image_id),
        product_name_candidate=product_name,
        emit_product=True,
        gate_score=round(image_conf, 4),
        gate_decision="emit",
        gate_reason="|".join(reason_parts),
        selected_candidate_index=sel_idx,
        matched_entry_id=primary.entry_id,
        matched_display=primary.display,
        match_score=round(rep_match_score, 4),
        match_margin=round(rep_margin, 4),
        match_type=rep_match_type,
        negative_context_count=neg_count,
        positive_context_count=int(context.get("positive_context_count", 0) or 0),
        negative_context_hits=list(context.get("negative_context_hits", []) or []),
        positive_context_hits=list(context.get("positive_context_hits", []) or []),
        compose_mode=compose_mode,
        chosen_entry_ids=[e.entry_id for e in chosen],
        chosen_displays=[e.display for e in chosen],
        n_matched_candidates=n_matched,
        n_distinct_entries=n_distinct,
        is_ambiguous=bool(selection.get("is_ambiguous", False)),
        top_candidates_json=top_json,
    )


# ---------------------------------------------------------------------
# Public API: apply gating to linked candidates
# ---------------------------------------------------------------------


def apply_confidence_gating(
    linked_candidate_df: pd.DataFrame,
    config: Optional[GateConfig] = None,
    show_progress: bool = False,
    image_text_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """
    Main function of Cell 4.

    Input:
        linked_candidate_df from Cell 3.

    Output:
        decision_df with 1 row per image_id:
            image_id, product_name_candidate, emit_product, gate_score, ...

    This is still not the final submission layer; Cell 5 will normalize/merge
    product_name_candidate into image-level output schema.
    """

    config = config or GateConfig()

    if linked_candidate_df is None or linked_candidate_df.empty:
        return pd.DataFrame(columns=list(GateDecision.__annotations__.keys()))

    if "image_id" not in linked_candidate_df.columns:
        raise ValueError("linked_candidate_df must contain image_id column")

    df = linked_candidate_df.copy()

    # If columns are missing because Cell 3 had no matches, fill safe defaults.
    if "matched_frequency_prior" not in df.columns:
        df["matched_frequency_prior"] = 0

    max_prior = int(pd.to_numeric(df["matched_frequency_prior"], errors="coerce").fillna(0).max())

    # Build per-image raw OCR text for the rule registry. Prefer an explicit
    # map (E_predict passes one), else recover from any ocr/selected text column.
    text_map: Dict[str, str] = {str(k): str(v) for k, v in (image_text_map or {}).items()}
    if not text_map:
        for col in ("ocr_text", "selected_text"):
            if col in df.columns:
                for img, sub in df.groupby("image_id", sort=False):
                    vals = [v for v in sub[col].dropna().astype(str).tolist() if v.strip()]
                    if vals:
                        text_map[str(img)] = max(vals, key=len)
                break

    decisions: List[GateDecision] = []
    grouped = df.groupby("image_id", sort=False)
    iterator = grouped

    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(grouped, total=df["image_id"].nunique(), desc="Confidence gating")
        except Exception:
            pass

    for image_id, group in iterator:
        decisions.append(
            gate_one_image(
                image_id=str(image_id),
                group=group,
                config=config,
                max_frequency_prior=max_prior,
                full_text=text_map.get(str(image_id), ""),
            )
        )

    # Rescue: an image whose OCR is so noisy it produced no candidate group can
    # still be emitted by a strong rule (precision-first). Only possible when the
    # caller supplies image_text_map covering all target images (orchestrator job).
    if config.use_rules and text_map:
        decided = {d.image_id for d in decisions}
        for img, full_text in text_map.items():
            if str(img) in decided or not full_text:
                continue
            rule_hit = _best_rule_hit(full_text)
            if rule_hit and int(rule_hit.get("priority", 0)) >= config.rule_override_priority:
                log.info("[Rule] rescue image=%s hit=%s priority=%s",
                         img, rule_hit.get("rule_name"), rule_hit.get("priority"))
                decisions.append(_rule_override_decision(str(img), rule_hit, {}))

    return pd.DataFrame([d.to_dict() for d in decisions])


# ---------------------------------------------------------------------
# Convenience merge / debug / summary
# ---------------------------------------------------------------------


def decisions_to_product_frame(decision_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert decision_df to minimal image_id/product_name frame.
    """

    if decision_df is None or decision_df.empty:
        return pd.DataFrame(columns=["image_id", "product_name"])

    out = decision_df[["image_id", "product_name_candidate", "emit_product"]].copy()
    out["product_name"] = out.apply(
        lambda r: r["product_name_candidate"] if bool(r["emit_product"]) else "",
        axis=1,
    )
    return out[["image_id", "product_name"]]


def merge_product_decisions(
    ocr_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    ocr_text_col: str = "ocr_text",
) -> pd.DataFrame:
    """
    Create a submission-like frame: image_id, ocr_text, product_name.

    This is a convenience for local inspection. Final normalization remains Cell 5.
    """

    product_df = decisions_to_product_frame(decision_df)

    base_cols = ["image_id"]
    if ocr_text_col in ocr_df.columns:
        base_cols.append(ocr_text_col)

    out = ocr_df[base_cols].copy().drop_duplicates("image_id")
    out = out.merge(product_df, on="image_id", how="left")
    out["product_name"] = out["product_name"].fillna("")

    if ocr_text_col in out.columns and ocr_text_col != "ocr_text":
        out = out.rename(columns={ocr_text_col: "ocr_text"})
    elif "ocr_text" not in out.columns:
        out["ocr_text"] = ""

    return out[["image_id", "ocr_text", "product_name"]]


def debug_gate_for_image(
    decision_df: pd.DataFrame,
    image_id: str,
) -> pd.DataFrame:
    if decision_df is None or decision_df.empty:
        return pd.DataFrame()
    out = decision_df[decision_df["image_id"].astype(str) == str(image_id)].copy()
    return out.reset_index(drop=True)


def summarize_gating(decision_df: pd.DataFrame) -> Dict[str, Any]:
    if decision_df is None or decision_df.empty:
        return {
            "n_images": 0,
            "n_emit": 0,
            "product_fill_rate": 0.0,
        }

    emit = decision_df["emit_product"].fillna(False).astype(bool)
    score = pd.to_numeric(decision_df.get("gate_score", pd.Series(dtype=float)), errors="coerce")

    decision_counts = (
        decision_df["gate_decision"]
        .fillna("")
        .astype(str)
        .value_counts()
        .to_dict()
        if "gate_decision" in decision_df.columns
        else {}
    )

    reason_counts = (
        decision_df["gate_reason"]
        .fillna("")
        .astype(str)
        .value_counts()
        .head(30)
        .to_dict()
        if "gate_reason" in decision_df.columns
        else {}
    )

    product_counts = (
        decision_df.loc[emit, "product_name_candidate"]
        .fillna("")
        .astype(str)
        .value_counts()
        .head(30)
        .to_dict()
        if "product_name_candidate" in decision_df.columns
        else {}
    )

    return {
        "n_images": int(len(decision_df)),
        "n_emit": int(emit.sum()),
        "n_blank": int((~emit).sum()),
        "product_fill_rate": float(emit.mean()),
        "gate_score_mean": float(score.mean()) if len(score.dropna()) else None,
        "gate_score_median": float(score.median()) if len(score.dropna()) else None,
        "gate_decision_counts": decision_counts,
        "top_gate_reasons": reason_counts,
        "top_products": product_counts,
    }


__all__ = [
    "GateConfig",
    "GateDecision",
    "EntryEvidence",
    "compute_context_evidence",
    "candidate_has_product_specific_evidence",
    "compute_candidate_gate_score",
    "gate_one_image",
    "apply_confidence_gating",
    "decisions_to_product_frame",
    "merge_product_decisions",
    "debug_gate_for_image",
    "summarize_gating",
]
