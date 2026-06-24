import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from ..ocr.A_candidate_sp import selection_to_product_input_fields
from ..ocr.quality import OCRSelection
from .rules import should_reject_short_candidate


def _row_full_text(row: pd.Series) -> str:
    """Best-effort raw OCR text for a candidate row (for contextual rule checks)."""
    for col in ("ocr_text", "selected_text", "raw_text"):
        val = _safe_str(row.get(col)) if hasattr(row, "get") else ""
        if val.strip():
            return val
    return ""


def keep_candidate(candidate: str, full_text: str) -> bool:
    """Drop empty / contextually-bad short candidates (CP handled by context)."""
    if not str(candidate).strip():
        return False
    if should_reject_short_candidate(candidate, full_text):
        return False
    return True

VARIANT_COLUMN_MAP = {
    "selected": {
        "text": "selected_text",
        "conf": "selected_conf",
        "boxes": "selected_boxes",
        "lines_json": "selected_lines_json",
    },
    "raw": {
        "text": "raw_text",
        "conf": "raw_conf",
        "boxes": "raw_boxes",
        "lines_json": "raw_lines_json",
    },
    "soft": {
        "text": "soft_text",
        "conf": "soft_conf",
        "boxes": "soft_boxes",
        "lines_json": "soft_lines_json",
    },
    "hard": {
        "text": "hard_text",
        "conf": "hard_conf",
        "boxes": "hard_boxes",
        "lines_json": "hard_lines_json",
    },
}

SOURCE_PRIORITY = {
    "line": 4,
    "whole_line": 3,
    "segment": 2,
    "ngram": 1,
}

SOURCE_SCORE_MAP = {
    "line": 1.00,
    "whole_line": 0.95,
    "segment": 0.85,
    "ngram": 0.70,
}

TOKEN_PATTERN = re.compile(
    r"[0-9A-Za-zÀ-ÖØ-öø-ỹĐđ]+(?:[._/-][0-9A-Za-zÀ-ÖØ-öø-ỹĐđ]+)*",
    flags=re.UNICODE,
)

HARD_SPLIT_RE = re.compile(r"[|/:;,\u2022\u2023\u25E6\u2043\u2219•·]+")
BRACKET_SPLIT_RE = re.compile(r"[\(\)\[\]\{\}]")
PLUS_SPLIT_RE = re.compile(r"\s*\+\s*")
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\uFEFF]")


@dataclass
class ProductCandidate:
    image_id: str
    variant: str
    source: str
    raw_text: str
    clean_text: str
    tokenized: List[str]
    ngram_span: Optional[Tuple[int, int]]
    line_index: Optional[int]
    ocr_conf: Optional[float]
    bbox_coords: Optional[Tuple[int, int, int, int]] = None
    position_score: Optional[float] = None
    structure_score: float = 0.0
    matched: bool = False

    variant_support_count: int = 1
    variant_agreement_score: float = 0.0
    source_score: float = 0.0
    normalized_key: str = ""
    available_variant_count: int = 0
    debug_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def is_emoji_or_symbol(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    if 0x1F300 <= code <= 0x1FAFF:
        return True
    if 0x2600 <= code <= 0x27BF:
        return True
    cat = unicodedata.category(ch)
    return cat in {"So", "Sk"}


def remove_emoji_and_control(text: str) -> str:
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("C") and ch not in {"\n", "\t", " "}:
            continue
        if is_emoji_or_symbol(ch):
            continue
        out.append(ch)
    return "".join(out)


def normalize_light(text: str) -> str:
    text = _safe_str(text)
    text = unicodedata.normalize("NFC", text)
    text = ZERO_WIDTH_RE.sub(" ", text)
    text = remove_emoji_and_control(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ ]*\n[ ]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize_text(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text)


def detokenize(tokens: Sequence[str]) -> str:
    return " ".join(tokens).strip()


def normalize_dedupe_key(text: str) -> str:
    text = normalize_light(text).replace("\n", " ").lower()
    text = re.sub(r"[^\w\sÀ-ÖØ-öø-ỹđ]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_filler_tokens_from_txt(path: Optional[str]) -> Set[str]:
    if not path:
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def strip_filler_tokens(tokens: List[str], filler_tokens: Optional[Set[str]]) -> List[str]:
    if not filler_tokens:
        return tokens
    return [tok for tok in tokens if tok.lower() not in filler_tokens]


def text_ratios(text: str) -> Dict[str, float]:
    if not text:
        return {
            "digit_ratio": 1.0,
            "alpha_ratio": 0.0,
            "punct_ratio": 0.0,
            "emoji_ratio": 0.0,
        }
    n = len(text)
    digits = sum(ch.isdigit() for ch in text)
    alpha = sum(ch.isalpha() for ch in text)
    punct = sum(unicodedata.category(ch).startswith("P") for ch in text)
    emoji = sum(is_emoji_or_symbol(ch) for ch in text)
    return {
        "digit_ratio": digits / n,
        "alpha_ratio": alpha / n,
        "punct_ratio": punct / n,
        "emoji_ratio": emoji / n,
    }


def structural_filter_reason(
    clean_text: str,
    tokenized: List[str],
    min_chars: int = 2,
    max_tokens: int = 12,
) -> Optional[str]:
    stripped = clean_text.strip()
    if not stripped:
        return "empty"
    if len(stripped) < min_chars:
        return "too_short"
    if len(tokenized) == 0:
        return "no_tokens"
    if len(tokenized) > max_tokens:
        return "too_many_tokens"
    if all(tok.isdigit() for tok in tokenized):
        return "numeric_only"
    if not any(any(ch.isalpha() for ch in tok) for tok in tokenized):
        return "no_alpha"

    ratios = text_ratios(stripped)
    if ratios["emoji_ratio"] > 0.20:
        return "emoji_heavy"
    if ratios["punct_ratio"] > 0.60:
        return "punct_heavy"
    return None


def compute_structure_score(clean_text: str, tokenized: List[str], source: str) -> float:
    reason = structural_filter_reason(clean_text, tokenized)
    if reason:
        penalty_map = {
            "empty": 0.0,
            "too_short": 0.25,
            "no_tokens": 0.0,
            "too_many_tokens": 0.55,
            "numeric_only": 0.15,
            "no_alpha": 0.10,
            "emoji_heavy": 0.20,
            "punct_heavy": 0.20,
        }
        return penalty_map.get(reason, 0.25)

    score = 0.55
    n_tok = len(tokenized)

    if 1 <= n_tok <= 4:
        score += 0.18
    elif 5 <= n_tok <= 8:
        score += 0.10
    else:
        score -= 0.05

    avg_tok_len = sum(len(t) for t in tokenized) / max(1, n_tok)
    if 2 <= avg_tok_len <= 10:
        score += 0.08

    ratios = text_ratios(clean_text)
    if ratios["alpha_ratio"] >= 0.45:
        score += 0.10
    if ratios["digit_ratio"] > 0.50:
        score -= 0.15
    if source == "ngram" and n_tok == 1:
        score -= 0.08

    return round(max(0.0, min(1.0, score)), 4)


def compute_position_score(
    bbox_coords: Optional[Tuple[int, int, int, int]],
    image_dims: Optional[Tuple[int, int]] = None,
) -> Optional[float]:
    if bbox_coords is None or image_dims is None:
        return None

    x, y, w, h = bbox_coords
    img_w, img_h = image_dims
    if img_w <= 0 or img_h <= 0 or w <= 0 or h <= 0:
        return None

    cx = x + w / 2.0
    cy = y + h / 2.0
    dx = abs(cx - img_w / 2.0) / max(1.0, img_w / 2.0)
    dy = abs(cy - img_h / 2.0) / max(1.0, img_h / 2.0)

    center_score = max(0.0, 1.0 - 0.5 * (dx + dy))
    area_score = min(1.0, (w * h) / max(1.0, img_w * img_h) * 8.0)
    top_half_bonus = 1.0 if cy <= img_h * 0.65 else 0.8

    score = 0.50 * center_score + 0.35 * area_score + 0.15 * top_half_bonus
    return round(max(0.0, min(1.0, score)), 4)


def _polygon_to_xywh(poly: Any) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(poly, list) or not poly:
        return None

    # Project contract: serialized line boxes are already (x, y, w, h).
    # Both OCRLine.box and PaddleOCREngine._box_to_xywh emit (x, y, w, h),
    # so a flat 4-number box must NOT be re-interpreted as (x1, y1, x2, y2).
    if len(poly) == 4 and not isinstance(poly[0], list):
        x, y, w, h = poly
        return (int(x), int(y), int(w), int(h))

    if isinstance(poly[0], list):
        xs = [pt[0] for pt in poly if isinstance(pt, list) and len(pt) >= 2]
        ys = [pt[1] for pt in poly if isinstance(pt, list) and len(pt) >= 2]
        if xs and ys:
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

    return None


def parse_lines_json(value: Any) -> Optional[List[Dict[str, Any]]]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    parsed = value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            parsed = json.loads(value)
        except Exception:
            return None

    if isinstance(parsed, dict):
        parsed = parsed.get("lines", parsed)

    if not isinstance(parsed, list):
        return None

    out = []
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            continue
        text = _safe_str(item.get("text"))
        conf = _safe_float(item.get("conf"))
        box = item.get("box") or item.get("bbox") or item.get("bbox_coords")
        box = _polygon_to_xywh(box)
        out.append(
            {
                "text": text,
                "conf": conf,
                "box": box,
                "line_index": item.get("line_num", idx),
            }
        )
    return out or None


def line_records_from_variant(
    row: pd.Series,
    variant_name: str,
    cols: Dict[str, str],
    image_dims: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    text = normalize_light(row.get(cols["text"], ""))
    conf = _safe_float(row.get(cols.get("conf")))
    lines = parse_lines_json(row.get(cols.get("lines_json")))

    records = []
    if lines:
        for item in lines:
            line_text = normalize_light(item["text"])
            if not line_text:
                continue
            box = item.get("box")
            records.append(
                {
                    "variant": variant_name,
                    "source": "line",
                    "raw_text": item["text"],
                    "clean_text": line_text,
                    "line_index": item.get("line_index"),
                    "ocr_conf": item.get("conf") if item.get("conf") is not None else conf,
                    "bbox_coords": box,
                    "position_score": compute_position_score(box, image_dims),
                }
            )
        return records

    for i, piece in enumerate(text.split("\n")):
        piece = normalize_light(piece)
        if not piece:
            continue
        records.append(
            {
                "variant": variant_name,
                "source": "whole_line",
                "raw_text": piece,
                "clean_text": piece,
                "line_index": i,
                "ocr_conf": conf,
                "bbox_coords": None,
                "position_score": None,
            }
        )
    return records


def split_segments(text: str) -> List[str]:
    text = normalize_light(text).replace("\n", " ")
    if not text:
        return []

    work = BRACKET_SPLIT_RE.sub(" | ", text)
    hard_parts = [p.strip() for p in HARD_SPLIT_RE.split(work) if p.strip()]
    if not hard_parts:
        hard_parts = [text]

    segments = []
    for part in hard_parts:
        segments.append(part)
        if "+" in part:
            plus_parts = [p.strip() for p in PLUS_SPLIT_RE.split(part) if p.strip()]
            segments.extend(plus_parts)
            if len(plus_parts) >= 2:
                for left, right in zip(plus_parts[:-1], plus_parts[1:]):
                    segments.append(f"{left} + {right}")

    seen = set()
    out = []
    for seg in segments:
        key = normalize_light(seg).replace("\n", " ")
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def dedup_rank(c: ProductCandidate) -> Tuple:
    return (
        SOURCE_PRIORITY.get(c.source, 0),
        min(len(c.tokenized), 8),
        0.0 if c.ocr_conf is None else c.ocr_conf,
        len(c.clean_text),
    )

def cross_line_ngram_records(
    line_records: List[Dict[str, Any]],
    filler_tokens: Optional[Set[str]] = None,
    min_n: int = 2,
    max_n: int = 6,
    window_lines: int = 2,
    max_flat_tokens: int = 60,
) -> List[Dict[str, Any]]:
    """
    Sinh ngram span vắt qua ranh giới các dòng OCR liền kề.

    PaddleOCR hay tách một tên sản phẩm thành 2 dòng vật lý
    (vd dòng A '... patê côt đèn' + dòng B 'Hài Phòng ...'). Candidate
    per-line không bao giờ ghép lại nên full-key không hình thành. Ở đây
    ta nối token của `window_lines` dòng liên tiếp rồi trượt cửa sổ ngram,
    CHỈ giữ những cửa sổ thực sự cắt qua ít nhất một ranh giới dòng
    (để không trùng với ngram trong-dòng đã sinh ở chỗ khác).
    """
    # Token mỗi dòng (đã strip filler), theo thứ tự đọc.
    per_line: List[Tuple[int, List[str], Dict[str, Any]]] = []
    for rec in line_records:
        toks = strip_filler_tokens(tokenize_text(rec["clean_text"]), filler_tokens)
        if toks:
            li = rec["line_index"] if rec["line_index"] is not None else len(per_line)
            per_line.append((li, toks, rec))
    per_line.sort(key=lambda x: x[0])

    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for i in range(len(per_line)):
        window = per_line[i : i + window_lines]
        if len(window) < 2:
            continue

        flat: List[str] = []
        boundaries: List[int] = []          # vị trí trong flat nơi một dòng mới bắt đầu
        base_rec = window[0][2]
        for (_, toks, _rec) in window:
            if flat:
                boundaries.append(len(flat))
            flat.extend(toks)
        if len(flat) > max_flat_tokens:     # an toàn, tránh bùng nổ
            continue

        upper = min(max_n, len(flat))
        for n in range(min_n, upper + 1):
            for start in range(0, len(flat) - n + 1):
                end = start + n
                # chỉ giữ cửa sổ cắt qua ranh giới dòng
                if not any(start < b < end for b in boundaries):
                    continue
                ng = flat[start:end]
                ng_clean = detokenize(ng)
                if structural_filter_reason(ng_clean, ng, max_tokens=max_n) is not None:
                    continue
                key = normalize_dedupe_key(ng_clean)
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append({"ng": list(ng), "ng_clean": ng_clean, "key": key, "rec": base_rec})
    return out

def generate_candidates_for_row(
    row: pd.Series,
    filler_tokens: Optional[Set[str]] = None,
    max_ngram: int = 4,
    max_tokens_for_ngram_source: int = 8,
    image_dims: Optional[Tuple[int, int]] = None,
    ngram_all_lines: bool = False,
    max_line_tokens_for_ngram: int = 40,
    cross_line_ngrams: bool = True,        # NEW — fix A
    max_cross_line_window: int = 2,        # NEW — số dòng liền kề gộp
    max_cross_line_ngram: int = 6,
) -> List[ProductCandidate]:
    image_id = _safe_str(row.get("image_id"))

    available_variants = [
        variant_name
        for variant_name, cols in VARIANT_COLUMN_MAP.items()
        if normalize_light(row.get(cols["text"], ""))
    ]
    available_variant_count = len(available_variants)

    raw_candidates: List[ProductCandidate] = []

    for variant_name, cols in VARIANT_COLUMN_MAP.items():
        base_text = normalize_light(row.get(cols["text"], ""))
        if not base_text:
            continue

        line_records = line_records_from_variant(
            row=row,
            variant_name=variant_name,
            cols=cols,
            image_dims=image_dims,
        )

        for rec in line_records:
            line_tokens = strip_filler_tokens(tokenize_text(rec["clean_text"]), filler_tokens)
            line_clean = detokenize(line_tokens) if line_tokens else rec["clean_text"]

            reason = structural_filter_reason(line_clean, line_tokens)
            if reason is None:
                raw_candidates.append(
                    ProductCandidate(
                        image_id=image_id,
                        variant=variant_name,
                        source=rec["source"],
                        raw_text=rec["raw_text"],
                        clean_text=line_clean,
                        tokenized=line_tokens,
                        ngram_span=None,
                        line_index=rec["line_index"],
                        ocr_conf=rec["ocr_conf"],
                        bbox_coords=rec["bbox_coords"],
                        position_score=rec["position_score"],
                        structure_score=compute_structure_score(line_clean, line_tokens, rec["source"]),
                        matched=False,
                        source_score=SOURCE_SCORE_MAP[rec["source"]],
                        normalized_key=normalize_dedupe_key(line_clean),
                        available_variant_count=available_variant_count,
                    )
                )

            has_real_segment = False
            for seg in split_segments(rec["clean_text"]):
                if seg == rec["clean_text"]:
                    continue
                has_real_segment = True

                seg_tokens = strip_filler_tokens(tokenize_text(seg), filler_tokens)
                seg_clean = detokenize(seg_tokens) if seg_tokens else seg
                reason = structural_filter_reason(seg_clean, seg_tokens)
                if reason is not None:
                    continue

                raw_candidates.append(
                    ProductCandidate(
                        image_id=image_id,
                        variant=variant_name,
                        source="segment",
                        raw_text=seg,
                        clean_text=seg_clean,
                        tokenized=seg_tokens,
                        ngram_span=None,
                        line_index=rec["line_index"],
                        ocr_conf=rec["ocr_conf"],
                        bbox_coords=rec["bbox_coords"],
                        position_score=rec["position_score"],
                        structure_score=compute_structure_score(seg_clean, seg_tokens, "segment"),
                        matched=False,
                        source_score=SOURCE_SCORE_MAP["segment"],
                        normalized_key=normalize_dedupe_key(seg_clean),
                        available_variant_count=available_variant_count,
                    )
                )

                if seg_tokens and len(seg_tokens) <= max_tokens_for_ngram_source:
                    upper_n = min(max_ngram, len(seg_tokens))
                    for n in range(1, upper_n + 1):
                        for start in range(0, len(seg_tokens) - n + 1):
                            ng = seg_tokens[start : start + n]
                            ng_clean = detokenize(ng)

                            reason = structural_filter_reason(
                                ng_clean,
                                ng,
                                max_tokens=max_ngram,
                            )
                            if reason is not None:
                                continue

                            raw_candidates.append(
                                ProductCandidate(
                                    image_id=image_id,
                                    variant=variant_name,
                                    source="ngram",
                                    raw_text=ng_clean,
                                    clean_text=ng_clean,
                                    tokenized=list(ng),
                                    ngram_span=(start, start + n),
                                    line_index=rec["line_index"],
                                    ocr_conf=rec["ocr_conf"],
                                    bbox_coords=rec["bbox_coords"],
                                    position_score=rec["position_score"],
                                    structure_score=compute_structure_score(ng_clean, ng, "ngram"),
                                    matched=False,
                                    source_score=SOURCE_SCORE_MAP["ngram"],
                                    normalized_key=normalize_dedupe_key(ng_clean),
                                    available_variant_count=available_variant_count,
                                )
                            )

            # Ngram-over-all-lines: when a line has no separator-based segments,
            # the brand is often a clean sub-span buried in a noisy line
            # (e.g. "Cô gái ăn patê ct đèn" -> brand "patê ct đèn"). Slide token
            # windows over the line so that sub-span becomes a candidate the
            # gazetteer fuzzy-linker can hit. Skipped when real segments exist
            # (those are already covered by segment-ngrams above, with boundaries).
            if (
                ngram_all_lines
                and not has_real_segment
                and 2 <= len(line_tokens) <= max_line_tokens_for_ngram
            ):
                upper_n = min(max_ngram, len(line_tokens))
                for n in range(2, upper_n + 1):
                    for start in range(0, len(line_tokens) - n + 1):
                        ng = line_tokens[start : start + n]
                        ng_clean = detokenize(ng)

                        if ng_clean == line_clean:
                            continue

                        reason = structural_filter_reason(ng_clean, ng, max_tokens=max_ngram)
                        if reason is not None:
                            continue

                        raw_candidates.append(
                            ProductCandidate(
                                image_id=image_id,
                                variant=variant_name,
                                source="ngram",
                                raw_text=ng_clean,
                                clean_text=ng_clean,
                                tokenized=list(ng),
                                ngram_span=(start, start + n),
                                line_index=rec["line_index"],
                                ocr_conf=rec["ocr_conf"],
                                bbox_coords=rec["bbox_coords"],
                                position_score=rec["position_score"],
                                structure_score=compute_structure_score(ng_clean, ng, "ngram"),
                                matched=False,
                                source_score=SOURCE_SCORE_MAP["ngram"],
                                normalized_key=normalize_dedupe_key(ng_clean),
                                available_variant_count=available_variant_count,
                            )
                        )

        # Cross-line ngrams: ghép token các dòng liền kề để bắt tên SP bị OCR
        # tách dòng (vd 'patê côt đèn' | 'Hài Phòng' -> 'patê côt đèn Hài Phòng').
        if cross_line_ngrams and len(line_records) >= 2:
            for item in cross_line_ngram_records(
                line_records,
                filler_tokens=filler_tokens,
                min_n=2,
                max_n=max_cross_line_ngram,
                window_lines=max_cross_line_window,
            ):
                rec = item["rec"]
                ng = item["ng"]
                ng_clean = item["ng_clean"]
                raw_candidates.append(
                    ProductCandidate(
                        image_id=image_id,
                        variant=variant_name,
                        source="ngram",
                        raw_text=ng_clean,
                        clean_text=ng_clean,
                        tokenized=list(ng),
                        ngram_span=None,
                        line_index=rec["line_index"],
                        ocr_conf=rec["ocr_conf"],
                        bbox_coords=None,
                        position_score=rec["position_score"],
                        structure_score=compute_structure_score(ng_clean, ng, "ngram"),
                        matched=False,
                        source_score=SOURCE_SCORE_MAP["ngram"],
                        normalized_key=item["key"],
                        available_variant_count=available_variant_count,
                    )
                )

    best_by_key: Dict[str, ProductCandidate] = {}
    
    variant_sets: Dict[str, Set[str]] = {}

    for cand in raw_candidates:
        key = cand.normalized_key
        if not key:
            continue

        variant_sets.setdefault(key, set()).add(cand.variant)
        if key not in best_by_key or dedup_rank(cand) > dedup_rank(best_by_key[key]):
            best_by_key[key] = cand

    out: List[ProductCandidate] = []
    for key, cand in best_by_key.items():
        support = len(variant_sets[key])
        cand.variant_support_count = support

        # Cross-variant agreement is only meaningful when >= 2 variants exist.
        # With a single variant, support/count == 1.0 would falsely signal
        # "all variants agree". Use 0.0 as an explicit "not applicable" sentinel
        # so Cell 4 (gating) does not award a consensus bonus that wasn't earned.
        # available_variant_count is kept so downstream can tell apart
        # "0.0 = single-variant N/A" from a real multi-variant score (always > 0).
        if available_variant_count >= 2:
            cand.variant_agreement_score = round(support / available_variant_count, 4)
        else:
            cand.variant_agreement_score = 0.0

        out.append(cand)

    # Contextual filter: drop standalone legal/noisy short candidates
    # (CTCP/Công ty/Cổ phần ...; CP only when context isn't a true CP product).
    full_text = _row_full_text(row)
    if full_text:
        out = [c for c in out if keep_candidate(c.clean_text, full_text)]

    out.sort(
        key=lambda c: (
            SOURCE_PRIORITY.get(c.source, 0),
            min(len(c.tokenized), 8),
            c.variant_agreement_score,
            c.structure_score,
            0.0 if c.ocr_conf is None else c.ocr_conf,
            len(c.clean_text),
        ),
        reverse=True,
    )

    return out


def candidates_to_dataframe(candidates: List[ProductCandidate]) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame(columns=list(ProductCandidate.__annotations__.keys()))

    df = pd.DataFrame.from_records([c.to_dict() for c in candidates])
    df["token_count"] = df["tokenized"].apply(len)
    df["char_count"] = df["clean_text"].astype(str).str.len()
    df["source_priority"] = df["source"].map(SOURCE_PRIORITY).fillna(0).astype(int)
    return df


def candidates_to_json_records(candidates: List[ProductCandidate]) -> List[Dict[str, Any]]:
    return [c.to_dict() for c in candidates]


def generate_candidate_dataframe(
    ocr_df: pd.DataFrame,
    filler_tokens: Optional[Set[str]] = None,
    max_ngram: int = 4,
    max_tokens_for_ngram_source: int = 8,
    image_width_col: Optional[str] = None,
    image_height_col: Optional[str] = None,
    ngram_all_lines: bool = False,
    max_line_tokens_for_ngram: int = 40,
    cross_line_ngrams: bool = True,
    max_cross_line_window: int = 2,
    max_cross_line_ngram: int = 6,
) -> pd.DataFrame:
    all_candidates: List[ProductCandidate] = []

    for _, row in ocr_df.iterrows():
        image_dims = None
        if image_width_col and image_height_col:
            try:
                iw = row.get(image_width_col)
                ih = row.get(image_height_col)
                if pd.notna(iw) and pd.notna(ih):
                    image_dims = (int(iw), int(ih))
            except Exception:
                image_dims = None

        all_candidates.extend(
            generate_candidates_for_row(
                row=row,
                filler_tokens=filler_tokens,
                max_ngram=max_ngram,
                max_tokens_for_ngram_source=max_tokens_for_ngram_source,
                image_dims=image_dims,
                ngram_all_lines=ngram_all_lines,
                max_line_tokens_for_ngram=max_line_tokens_for_ngram,
                cross_line_ngrams=cross_line_ngrams,
                max_cross_line_window=max_cross_line_window,
                max_cross_line_ngram=max_cross_line_ngram,
            )
        )

    return candidates_to_dataframe(all_candidates)


# ---------------------------------------------------------------------
# Bridge: OCR pipeline (engine + quality) -> product candidate input
# ---------------------------------------------------------------------

def selections_to_ocr_dataframe(
    selections: Iterable[Tuple[str, OCRSelection]],
    include_variants: bool = True,
) -> pd.DataFrame:
    """
    Build the OCR input DataFrame consumed by this module directly from the
    OCR pipeline output, without going through a CSV file.

    Parameters
    ----------
    selections:
        Iterable of (image_id, OCRSelection) pairs. OCRSelection is the result
        of ocr.quality.run_ocr_with_quality_gate / choose_best_ocr_result.

    include_variants:
        If True, also flatten raw/soft/hard variant columns (when present).

    Returns
    -------
    DataFrame with columns expected by VARIANT_COLUMN_MAP
    (image_id, selected_text/conf/boxes/lines_json, soft_*, raw_*, hard_*, ...).
    Missing variant columns are filled with NaN, which the candidate
    generator already tolerates.
    """

    rows: List[Dict[str, Any]] = [
        selection_to_product_input_fields(
            selection,
            image_id=image_id,
            include_variants=include_variants,
        )
        for image_id, selection in selections
    ]

    if not rows:
        return pd.DataFrame(columns=["image_id"])

    return pd.DataFrame(rows)


def selections_to_candidate_dataframe(
    selections: Iterable[Tuple[str, OCRSelection]],
    filler_tokens: Optional[Set[str]] = None,
    max_ngram: int = 4,
    max_tokens_for_ngram_source: int = 8,
    include_variants: bool = True,
    image_width_col: Optional[str] = None,
    image_height_col: Optional[str] = None,
    ngram_all_lines: bool = False,
    max_line_tokens_for_ngram: int = 40,
    cross_line_ngrams: bool = True,
    max_cross_line_window: int = 2,
    max_cross_line_ngram: int = 6,
) -> pd.DataFrame:
    """
    One-shot bridge: OCR pipeline output -> product candidate DataFrame.

    Equivalent to:
        ocr_df = selections_to_ocr_dataframe(selections)
        generate_candidate_dataframe(ocr_df, ...)
    """

    ocr_df = selections_to_ocr_dataframe(
        selections,
        include_variants=include_variants,
    )

    return generate_candidate_dataframe(
        ocr_df,
        filler_tokens=filler_tokens,
        max_ngram=max_ngram,
        max_tokens_for_ngram_source=max_tokens_for_ngram_source,
        image_width_col=image_width_col,
        image_height_col=image_height_col,
        ngram_all_lines=ngram_all_lines,
        max_line_tokens_for_ngram=max_line_tokens_for_ngram,
        cross_line_ngrams=cross_line_ngrams,
        max_cross_line_window=max_cross_line_window,
        max_cross_line_ngram=max_cross_line_ngram,
    )


def debug_candidates_for_image(
    candidate_df: pd.DataFrame,
    image_id: str,
    min_token_count: int = 2,
    top_k: int = 20,
) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df.copy()

    out = candidate_df[candidate_df["image_id"] == image_id].copy()
    if min_token_count is not None:
        out = out[out["token_count"] >= min_token_count]

    out = out.sort_values(
        ["source_priority", "token_count", "variant_agreement_score", "structure_score", "ocr_conf", "char_count"],
        ascending=False,
    )

    cols = [
        "image_id",
        "variant",
        "source",
        "line_index",
        "clean_text",
        "token_count",
        "ocr_conf",
        "variant_support_count",
        "variant_agreement_score",
        "structure_score",
        "position_score",
        "bbox_coords",
        "ngram_span",
        "normalized_key",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].head(top_k).reset_index(drop=True)
