from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from .A_candidate import normalize_dedupe_key, normalize_light, tokenize_text
from .B_gazeetteer import (
    GazetteerEntry,
    ProductGazetteer,
    fold_key,
    fold_token_key,
)


# ---------------------------------------------------------------------
# Optional rapidfuzz backend
# ---------------------------------------------------------------------

try:
    from rapidfuzz import fuzz as _rf_fuzz

    RAPIDFUZZ_AVAILABLE = True
except Exception:
    _rf_fuzz = None
    RAPIDFUZZ_AVAILABLE = False


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DEFAULT_FUZZY_THRESHOLD = 88.0
DEFAULT_STRONG_THRESHOLD = 94.0

# Score constants for deterministic exact-ish matches.
SCORE_EXACT = 100.0
SCORE_SURFACE_EXACT = 99.0
SCORE_FOLDED_EXACT = 98.0
SCORE_FOLDED_SURFACE_EXACT = 97.0

# Fuzzy score caps. Folded fuzzy is lossy, so it is capped below exact/folded-exact.
CAP_DIA_TOKEN_SET = 96.0
CAP_DIA_PARTIAL = 95.0
CAP_DIA_RATIO = 94.0
CAP_FOLD_TOKEN_SET = 94.0
CAP_FOLD_PARTIAL = 93.0
CAP_FOLD_RATIO = 92.0


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


def _get(row: Union[pd.Series, Mapping[str, Any]], key: str, default: Any = None) -> Any:
    if hasattr(row, "get"):
        return row.get(key, default)
    return default


def _parse_tokenized(value: Any, fallback_text: str = "") -> List[str]:
    """
    Candidate DataFrame keeps tokenized as a list in memory. If saved/reloaded
    from CSV, it can become a string like "['DO', 'HOP']". This helper accepts
    both forms.
    """

    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if _safe_str(x)]

    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    return [str(x) for x in parsed if _safe_str(x)]
            except Exception:
                pass
        # Fall back to tokenizing the string itself.
        toks = tokenize_text(text)
        if toks:
            return toks

    return tokenize_text(fallback_text)


def _token_keys(tokens: Sequence[Any]) -> List[str]:
    out = []
    for t in tokens:
        key = normalize_dedupe_key(_safe_str(t))
        if key:
            out.append(key)
    return sorted(set(out))


def _folded_token_keys(tokens: Sequence[Any]) -> List[str]:
    out = []
    for t in tokens:
        key = fold_token_key(_safe_str(t))
        if key:
            out.append(key)
    return sorted(set(out))


def _ratio(a: str, b: str) -> float:
    a = _safe_str(a)
    b = _safe_str(b)
    if not a or not b:
        return 0.0
    if RAPIDFUZZ_AVAILABLE:
        return float(_rf_fuzz.ratio(a, b))
    return 100.0 * SequenceMatcher(None, a, b).ratio()


def _partial_ratio(a: str, b: str) -> float:
    a = _safe_str(a)
    b = _safe_str(b)
    if not a or not b:
        return 0.0
    if RAPIDFUZZ_AVAILABLE:
        return float(_rf_fuzz.partial_ratio(a, b))

    # Lightweight fallback: compare the shorter string against windows of the longer.
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) == 0:
        return 0.0
    if len(short) == len(long):
        return _ratio(short, long)

    best = 0.0
    width = len(short)
    for start in range(0, max(1, len(long) - width + 1)):
        window = long[start : start + width]
        best = max(best, _ratio(short, window))
    return best


def _token_sort_ratio(a: str, b: str) -> float:
    a_toks = sorted(tokenize_text(a))
    b_toks = sorted(tokenize_text(b))
    return _ratio(" ".join(a_toks), " ".join(b_toks))


def _token_set_ratio(a: str, b: str) -> float:
    a = _safe_str(a)
    b = _safe_str(b)
    if not a or not b:
        return 0.0
    if RAPIDFUZZ_AVAILABLE:
        return float(_rf_fuzz.token_set_ratio(a, b))

    a_set = set(tokenize_text(a))
    b_set = set(tokenize_text(b))
    if not a_set or not b_set:
        return 0.0

    common = sorted(a_set & b_set)
    a_only = sorted(a_set - b_set)
    b_only = sorted(b_set - a_set)

    common_s = " ".join(common)
    a_combined = " ".join(common + a_only)
    b_combined = " ".join(common + b_only)

    return max(
        _ratio(common_s, a_combined),
        _ratio(common_s, b_combined),
        _ratio(a_combined, b_combined),
    )


def _cap(score: float, cap: float) -> float:
    return float(max(0.0, min(float(score), float(cap))))


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------

@dataclass
class CandidateTextFeatures:
    clean_text: str
    normalized_key: str
    folded_key: str
    tokenized: List[str]
    token_keys: List[str]
    folded_token_keys: List[str]

    @property
    def token_count(self) -> int:
        return len(self.tokenized)


@dataclass
class MatchResult:
    entry_id: str
    canonical_display: str
    canonical_key: str
    source_class: str
    frequency_prior: int
    item_count: int

    score: float
    match_type: str
    debug: str

    # Additional diagnostics useful for Cell 4.
    token_overlap: int = 0
    folded_token_overlap: int = 0
    entry_token_coverage: float = 0.0
    candidate_token_coverage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------
# Candidate feature extraction
# ---------------------------------------------------------------------


def candidate_features(candidate_row: Union[pd.Series, Mapping[str, Any]]) -> CandidateTextFeatures:
    clean_text = normalize_light(_safe_str(_get(candidate_row, "clean_text", "")))

    normalized_key = _safe_str(_get(candidate_row, "normalized_key", ""))
    if not normalized_key:
        normalized_key = normalize_dedupe_key(clean_text)

    folded_key = fold_key(clean_text) if clean_text else fold_key(normalized_key)

    tokenized = _parse_tokenized(
        _get(candidate_row, "tokenized", []),
        fallback_text=clean_text,
    )

    token_keys = _token_keys(tokenized)
    folded_token_keys = _folded_token_keys(tokenized)

    return CandidateTextFeatures(
        clean_text=clean_text,
        normalized_key=normalized_key,
        folded_key=folded_key,
        tokenized=tokenized,
        token_keys=token_keys,
        folded_token_keys=folded_token_keys,
    )


# ---------------------------------------------------------------------
# Scoring one candidate-entry pair
# ---------------------------------------------------------------------


def _surface_keys(entry: GazetteerEntry) -> List[str]:
    return [normalize_dedupe_key(v) for v in entry.surface_variants if normalize_dedupe_key(v)]


def _score_exact_and_surface(
    cf: CandidateTextFeatures,
    entry: GazetteerEntry,
) -> Optional[MatchResult]:
    """
    Handle deterministic exact/surface matches before fuzzy scoring.
    """

    surface_keys = set(_surface_keys(entry))
    folded_surface_keys = set(getattr(entry, "folded_surface_keys", []) or [])

    if cf.normalized_key and cf.normalized_key == entry.canonical_key:
        return _make_result(
            entry=entry,
            score=SCORE_EXACT,
            match_type="exact",
            debug="candidate normalized_key equals entry canonical_key",
            cf=cf,
        )

    if cf.normalized_key and cf.normalized_key in surface_keys:
        return _make_result(
            entry=entry,
            score=SCORE_SURFACE_EXACT,
            match_type="surface_exact",
            debug="candidate normalized_key equals an observed surface variant",
            cf=cf,
        )

    if cf.folded_key and cf.folded_key == entry.folded_key:
        return _make_result(
            entry=entry,
            score=SCORE_FOLDED_EXACT,
            match_type="folded_exact",
            debug="candidate folded_key equals entry folded_key",
            cf=cf,
        )

    if cf.folded_key and cf.folded_key in folded_surface_keys:
        return _make_result(
            entry=entry,
            score=SCORE_FOLDED_SURFACE_EXACT,
            match_type="folded_surface_exact",
            debug="candidate folded_key equals an observed folded surface variant",
            cf=cf,
        )

    return None


def _make_result(
    entry: GazetteerEntry,
    score: float,
    match_type: str,
    debug: str,
    cf: CandidateTextFeatures,
) -> MatchResult:
    entry_token_set = set(entry.token_keys)
    entry_folded_token_set = set(entry.folded_token_keys)
    cand_token_set = set(cf.token_keys)
    cand_folded_token_set = set(cf.folded_token_keys)

    token_overlap = len(cand_token_set & entry_token_set)
    folded_overlap = len(cand_folded_token_set & entry_folded_token_set)

    entry_cov = folded_overlap / max(1, len(entry_folded_token_set))
    cand_cov = folded_overlap / max(1, len(cand_folded_token_set))

    return MatchResult(
        entry_id=entry.entry_id,
        canonical_display=entry.canonical_display,
        canonical_key=entry.canonical_key,
        source_class=entry.source_class,
        frequency_prior=int(entry.frequency_prior),
        item_count=int(entry.item_count),
        score=round(float(score), 4),
        match_type=match_type,
        debug=debug,
        token_overlap=int(token_overlap),
        folded_token_overlap=int(folded_overlap),
        entry_token_coverage=round(float(entry_cov), 4),
        candidate_token_coverage=round(float(cand_cov), 4),
    )


def _score_fuzzy(
    cf: CandidateTextFeatures,
    entry: GazetteerEntry,
) -> MatchResult:
    """
    Score fuzzy similarity between one candidate and one gazetteer entry.

    Diacritic-aware scores are preferred over folded scores. Folded scores are
    useful because PaddleOCR often drops Vietnamese accents, but folded matching
    is lossy and should remain below exact/folded-exact matches.
    """

    c_key = cf.normalized_key
    e_key = entry.canonical_key
    c_fold = cf.folded_key
    e_fold = entry.folded_key

    scored: List[Tuple[float, str, str]] = []

    if c_key and e_key:
        scored.append((_cap(_token_set_ratio(c_key, e_key), CAP_DIA_TOKEN_SET), "fuzzy_token_set", "diacritic token_set_ratio"))
        scored.append((_cap(_partial_ratio(c_key, e_key), CAP_DIA_PARTIAL), "fuzzy_partial", "diacritic partial_ratio"))
        scored.append((_cap(_token_sort_ratio(c_key, e_key), CAP_DIA_RATIO), "fuzzy_token_sort", "diacritic token_sort_ratio"))
        scored.append((_cap(_ratio(c_key, e_key), CAP_DIA_RATIO), "fuzzy_ratio", "diacritic ratio"))

    if c_fold and e_fold:
        scored.append((_cap(_token_set_ratio(c_fold, e_fold), CAP_FOLD_TOKEN_SET), "folded_fuzzy_token_set", "folded token_set_ratio"))
        scored.append((_cap(_partial_ratio(c_fold, e_fold), CAP_FOLD_PARTIAL), "folded_fuzzy_partial", "folded partial_ratio"))
        scored.append((_cap(_token_sort_ratio(c_fold, e_fold), CAP_FOLD_RATIO), "folded_fuzzy_token_sort", "folded token_sort_ratio"))
        scored.append((_cap(_ratio(c_fold, e_fold), CAP_FOLD_RATIO), "folded_fuzzy_ratio", "folded ratio"))

    if not scored:
        return _make_result(entry, 0.0, "no_score", "empty candidate or entry key", cf)

    # One-token candidates are often ambiguous: "NAN", "PATE", "LONG".
    # Exact matches above can still score high, but fuzzy-only one-token matches
    # should not pass the default threshold by themselves.
    best_score, best_type, best_debug = max(scored, key=lambda x: x[0])

    if cf.token_count <= 1:
        best_score = min(best_score, 84.0)
        best_debug += " | capped_one_token_candidate"

    # Very short folded text like "ha long" should be linkable, but ambiguity
    # is represented by a low margin later. Do not cap two-token candidates here.

    return _make_result(
        entry=entry,
        score=best_score,
        match_type=best_type,
        debug=best_debug,
        cf=cf,
    )


def score_candidate_entry(
    candidate_row: Union[pd.Series, Mapping[str, Any]],
    entry: GazetteerEntry,
) -> MatchResult:
    """
    Public scorer for one candidate row against one GazetteerEntry.

    This function does not apply the fuzzy threshold. It only returns the best
    match type + score for this candidate-entry pair.
    """

    cf = candidate_features(candidate_row)

    exact = _score_exact_and_surface(cf, entry)
    if exact is not None:
        return exact

    return _score_fuzzy(cf, entry)


# ---------------------------------------------------------------------
# Link one candidate row
# ---------------------------------------------------------------------


def _empty_link_result(reason: str = "no_candidate_pool") -> Dict[str, Any]:
    return {
        "matched": False,
        "is_strong_match": False,
        "matched_entry_id": "",
        "matched_display": "",
        "matched_key": "",
        "matched_source_class": "",
        "matched_frequency_prior": 0,
        "matched_item_count": 0,
        "match_score": 0.0,
        "second_match_score": 0.0,
        "match_margin": 0.0,
        "match_type": "no_match",
        "candidate_pool_size": 0,
        "top_matches_json": "[]",
        "match_debug": reason,
        "match_token_overlap": 0,
        "match_folded_token_overlap": 0,
        "match_entry_token_coverage": 0.0,
        "match_candidate_token_coverage": 0.0,
    }


def link_one_candidate(
    candidate_row: Union[pd.Series, Mapping[str, Any]],
    gazetteer: ProductGazetteer,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
    max_pool_entries: int = 80,
    top_k_matches: int = 5,
) -> Dict[str, Any]:
    """
    Link one candidate to the best gazetteer entry, if score is high enough.

    Important: this does NOT decide the final product_name for the image.
    It only enriches a candidate row with match metadata for Cell 4.
    """

    entries = gazetteer.lookup_candidates(candidate_row, max_entries=max_pool_entries)
    if not entries:
        return _empty_link_result("gazetteer.lookup_candidates returned empty pool")

    scored: List[MatchResult] = [score_candidate_entry(candidate_row, e) for e in entries]

    # Ranking: score first, then token coverage, then frequency prior as tie-breaker.
    scored.sort(
        key=lambda r: (
            r.score,
            r.folded_token_overlap,
            r.candidate_token_coverage,
            r.entry_token_coverage,
            r.frequency_prior,
            r.canonical_display,
        ),
        reverse=True,
    )

    best = scored[0]
    second_score = scored[1].score if len(scored) >= 2 else 0.0
    margin = best.score - second_score

    matched = bool(best.score >= float(fuzzy_threshold))
    is_strong = bool(best.score >= float(strong_threshold))

    top_records = [r.to_dict() for r in scored[:top_k_matches]]

    if not matched:
        debug = f"best_score_below_threshold:{best.score:.2f}<{float(fuzzy_threshold):.2f}; best={best.canonical_display}; {best.debug}"
    else:
        debug = best.debug

    return {
        "matched": matched,
        "is_strong_match": is_strong,
        "matched_entry_id": best.entry_id if matched else "",
        "matched_display": best.canonical_display if matched else "",
        "matched_key": best.canonical_key if matched else "",
        "matched_source_class": best.source_class if matched else "",
        "matched_frequency_prior": best.frequency_prior if matched else 0,
        "matched_item_count": best.item_count if matched else 0,
        "match_score": round(float(best.score), 4),
        "second_match_score": round(float(second_score), 4),
        "match_margin": round(float(margin), 4),
        "match_type": best.match_type if matched else "no_match",
        "candidate_pool_size": len(entries),
        "top_matches_json": _json_dumps(top_records),
        "match_debug": debug,
        "match_token_overlap": int(best.token_overlap),
        "match_folded_token_overlap": int(best.folded_token_overlap),
        "match_entry_token_coverage": float(best.entry_token_coverage),
        "match_candidate_token_coverage": float(best.candidate_token_coverage),
    }


# ---------------------------------------------------------------------
# Link full candidate DataFrame
# ---------------------------------------------------------------------


def link_candidates_with_gazetteer(
    candidate_df: pd.DataFrame,
    gazetteer: ProductGazetteer,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
    max_pool_entries: int = 80,
    top_k_matches: int = 5,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Main function of Cell 3.

    Input:
        candidate_df from Cell 1
        gazetteer from Cell 2

    Output:
        linked_candidate_df = candidate_df + match columns

    This function does not decide final product_name. Cell 4 will group by
    image_id and decide blank/fill using these match columns plus context.
    """

    if candidate_df is None or candidate_df.empty:
        out = candidate_df.copy() if candidate_df is not None else pd.DataFrame()
        for k, v in _empty_link_result("empty candidate_df").items():
            out[k] = v
        return out

    rows = []
    iterator = candidate_df.iterrows()

    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, total=len(candidate_df), desc="Fuzzy linking candidates")
        except Exception:
            pass

    for _, row in iterator:
        rows.append(
            link_one_candidate(
                candidate_row=row,
                gazetteer=gazetteer,
                fuzzy_threshold=fuzzy_threshold,
                strong_threshold=strong_threshold,
                max_pool_entries=max_pool_entries,
                top_k_matches=top_k_matches,
            )
        )

    match_df = pd.DataFrame(rows, index=candidate_df.index)

    # Cell 1's ProductCandidate already carries a `matched` column (placeholder
    # default False). Drop any column Cell 3 is about to (re)write so concat does
    # not produce duplicate labels — Cell 3 is the authoritative writer of `matched`.
    base = candidate_df.copy()
    overlap = [c for c in match_df.columns if c in base.columns]
    if overlap:
        base = base.drop(columns=overlap)
    out = pd.concat([base, match_df], axis=1)

    # Keep matched candidates near the top for quick manual inspection.
    sort_cols = [
        "image_id",
        "matched",
        "is_strong_match",
        "match_score",
        "match_margin",
        "source_priority",
        "structure_score",
        "ocr_conf",
    ]
    sort_cols = [c for c in sort_cols if c in out.columns]

    if sort_cols:
        ascending = []
        for c in sort_cols:
            if c == "image_id":
                ascending.append(True)
            else:
                ascending.append(False)
        out = out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    return out


# ---------------------------------------------------------------------
# Debug / summaries
# ---------------------------------------------------------------------


def debug_matches_for_image(
    linked_candidate_df: pd.DataFrame,
    image_id: str,
    only_matched: bool = False,
    top_k: int = 30,
) -> pd.DataFrame:
    """
    Human-readable debug table for one image after Cell 3.
    """

    if linked_candidate_df is None or linked_candidate_df.empty:
        return pd.DataFrame()

    out = linked_candidate_df[linked_candidate_df["image_id"].astype(str) == str(image_id)].copy()

    if only_matched:
        out = out[out["matched"] == True]

    sort_cols = [
        "matched",
        "is_strong_match",
        "match_score",
        "match_margin",
        "source_priority",
        "structure_score",
        "ocr_conf",
    ]
    sort_cols = [c for c in sort_cols if c in out.columns]

    if sort_cols:
        out = out.sort_values(sort_cols, ascending=False)

    cols = [
        "image_id",
        "variant",
        "source",
        "line_index",
        "clean_text",
        "matched",
        "is_strong_match",
        "matched_display",
        "match_score",
        "second_match_score",
        "match_margin",
        "match_type",
        "candidate_pool_size",
        "matched_frequency_prior",
        "match_folded_token_overlap",
        "match_candidate_token_coverage",
        "structure_score",
        "ocr_conf",
        "match_debug",
    ]
    cols = [c for c in cols if c in out.columns]

    return out[cols].head(top_k).reset_index(drop=True)


def summarize_linking(linked_candidate_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Lightweight summary for Cell 3 output.
    """

    if linked_candidate_df is None or linked_candidate_df.empty:
        return {
            "n_candidates": 0,
            "n_matched_candidates": 0,
            "match_rate": 0.0,
        }

    n = len(linked_candidate_df)
    matched = linked_candidate_df["matched"].fillna(False).astype(bool)
    strong = linked_candidate_df.get("is_strong_match", pd.Series(False, index=linked_candidate_df.index)).fillna(False).astype(bool)

    by_type = (
        linked_candidate_df.loc[matched, "match_type"]
        .fillna("")
        .astype(str)
        .value_counts()
        .head(20)
        .to_dict()
    )

    by_display = (
        linked_candidate_df.loc[matched, "matched_display"]
        .fillna("")
        .astype(str)
        .value_counts()
        .head(20)
        .to_dict()
    )

    score = pd.to_numeric(linked_candidate_df.get("match_score", pd.Series(dtype=float)), errors="coerce")
    margin = pd.to_numeric(linked_candidate_df.get("match_margin", pd.Series(dtype=float)), errors="coerce")

    return {
        "n_candidates": int(n),
        "n_images": int(linked_candidate_df["image_id"].nunique()) if "image_id" in linked_candidate_df.columns else None,
        "n_matched_candidates": int(matched.sum()),
        "n_strong_candidates": int(strong.sum()),
        "match_rate": float(matched.mean()),
        "strong_rate": float(strong.mean()),
        "match_score_mean": float(score.mean()) if len(score.dropna()) else None,
        "match_score_median": float(score.median()) if len(score.dropna()) else None,
        "match_margin_mean": float(margin.mean()) if len(margin.dropna()) else None,
        "match_margin_median": float(margin.median()) if len(margin.dropna()) else None,
        "match_type_counts": by_type,
        "top_matched_displays": by_display,
        "rapidfuzz_available": RAPIDFUZZ_AVAILABLE,
    }


def sanity_probe_texts(
    gazetteer: ProductGazetteer,
    texts: Optional[List[str]] = None,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
) -> pd.DataFrame:
    """
    Quick text-only sanity probe for Cell 3 without candidate_df.
    Useful examples:
        DO HOP HA LONG -> Đồ Hộp Hạ Long
        PATE COT DEN -> Pate Cột Đèn...
    """

    if texts is None:
        texts = [
            "DO HOP HA LONG",
            "ĐỒ HỘP HẠ LONG",
            "HALONG CANFOCO",
            "PATE COT DEN",
            "PATE CỘT ĐÈN HẢI PHÒNG",
            "NESTLE NAN OPTIPRO",
            "NAN INFINIPRO A2",
            "HIGHLANDS TRA SEN VANG",
            "BANH MI QUE",
            "HA LONG",
        ]

    rows = []
    for i, text in enumerate(texts):
        candidate = {
            "image_id": f"probe_{i:03d}",
            "clean_text": text,
            "normalized_key": normalize_dedupe_key(text),
            "tokenized": tokenize_text(text),
        }
        link = link_one_candidate(
            candidate_row=candidate,
            gazetteer=gazetteer,
            fuzzy_threshold=fuzzy_threshold,
            strong_threshold=strong_threshold,
        )
        rows.append({"probe_text": text, **link})

    return pd.DataFrame(rows)


__all__ = [
    "RAPIDFUZZ_AVAILABLE",
    "DEFAULT_FUZZY_THRESHOLD",
    "DEFAULT_STRONG_THRESHOLD",
    "CandidateTextFeatures",
    "MatchResult",
    "candidate_features",
    "score_candidate_entry",
    "link_one_candidate",
    "link_candidates_with_gazetteer",
    "debug_matches_for_image",
    "summarize_linking",
    "sanity_probe_texts",
]
