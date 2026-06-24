from __future__ import annotations

import ast
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

try:
    from .B_gazeetteer import ProductGazetteer
except Exception:
    from .B_gazetteer import ProductGazetteer

from .rules import resolve_product_name


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class FinalPredictorConfig:
    """
    Cell 5 config.

    Cell 5 should not re-decide product vs blank. It only resolves canonical
    display strings from Cell 4 decisions and builds the final submission frame.
    """

    joiner: str = " + "
    output_product_col: str = "product_name"
    output_ocr_col: str = "ocr_text"

    # If True, Cell 5 uses gazetteer.entry_by_id for canonical displays whenever
    # chosen_entry_ids are present. This is the safest path.
    prefer_gazetteer_display: bool = True

    # If true, remove duplicate product displays while preserving order.
    deduplicate_items: bool = True

    # If true, remove atomic items already contained in a chosen full multi-item
    # display. Example: ["Highlands Coffee Trà Sen Vàng + Bánh Mì Que", "Bánh Mì Que"]
    # -> ["Highlands Coffee Trà Sen Vàng + Bánh Mì Que"]
    drop_items_covered_by_full: bool = True

    # For final submission, unknown/blank product must be empty string.
    blank_product_value: str = ""

    # Optional: output an empty brand_name column for external templates that expect it.
    include_brand_name: bool = False


# ---------------------------------------------------------------------
# Safe parsing helpers
# ---------------------------------------------------------------------


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x)
    if s.strip().lower() in {"nan", "none", "null"}:
        return ""
    return s


def _safe_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def _parse_list_like(value: Any) -> List[str]:
    """
    Parse list fields that may be:
    - a Python list from in-memory DataFrame
    - JSON list string
    - Python repr list string
    - a plain string fallback
    """

    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [_safe_str(x) for x in value if _safe_str(x)]

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "[]"}:
        return []

    # JSON list.
    if s.startswith("[") and s.endswith("]"):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [_safe_str(x) for x in obj if _safe_str(x)]
        except Exception:
            pass

        # Python repr list.
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple, set)):
                return [_safe_str(x) for x in obj if _safe_str(x)]
        except Exception:
            pass

    # Fallback: one display string.
    return [_safe_str(s)] if _safe_str(s) else []


# ---------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------


def normalize_output_text(text: Any) -> str:
    """
    Minimal output text cleanup.

    Do NOT over-normalize product_name here. Gazetteer canonical display already
    carries correct casing/diacritics. This only fixes whitespace and Unicode.
    """

    s = _safe_str(text)
    if not s:
        return ""

    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\+\s*", " + ", s)
    s = re.sub(r"\n+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_ocr_output_text(text: Any) -> str:
    """
    OCR text output can keep newlines or be flattened depending on submission
    preference. We keep it simple and stable: normalize Unicode and trim spaces,
    but preserve newlines when present.
    """

    s = _safe_str(text)
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"[ ]*\n[ ]*", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# WHITELIST (robust), not a blacklist. We list what is ALLOWED in real
# Vietnamese/Latin text and drop everything else, so any junk script/symbol we
# have never seen before (a new emoji, Cyrillic, Thai, Arabic, CJK, box-drawing)
# is removed automatically -- no maintenance, no "new char slips through" bug.
#
# Allowed (after NFC):
#   \x20-\x7E  ASCII printable  -> a-z A-Z 0-9 space + all common punctuation
#   00C0-024F  Latin-1 Suppl + Latin Extended-A/B  -> Vietnamese a/d-bar/o-horn...
#   1E00-1EFF  Latin Extended Additional           -> a-dot a-hook a-acute ... y
#   0300-036F  combining diacritics (safety net for un-composable decomposed VN)
#   2010-2027  common typographic punctuation (dashes, curly quotes, ellipsis, bullet)
#   20AB       Vietnamese dong sign
#   whitespace (collapsed afterwards)
_KEEP_RE = re.compile(
    "[^"
    "\x20-\x7e"
    "\u00c0-\u024f"
    "\u1e00-\u1eff"
    "\u0300-\u036f"
    "\u2010-\u2027"
    "\u20ab"
    r"\s"
    "]+"
)

# A "real" Vietnamese/Latin word: >= 3 letters in a row (ASCII or precomposed VN).
_HAS_WORD_RE = re.compile(r"[0-9A-Za-zÀ-ɏḀ-ỿ]{3,}")


def clean_ocr_text_for_output(text: Any, drop_junk_only: bool = True) -> str:
    """
    Optional, opt-in cleanup of the submission `ocr_text` column.

    1. NFC + collapse to a single line (so decomposed Vietnamese composes first,
       never strip a standalone tone mark).
    2. WHITELIST filter: keep only ASCII + Latin/Vietnamese letters + common
       punctuation; drop every other script/symbol. Future-proof -- unseen junk
       (new emoji, CJK, Cyrillic, Thai...) is removed without code changes.
    3. If drop_junk_only and what remains has NO real word (no >=3-letter run),
       blank it — the image was OCR garbage (e.g. 'D 5 DD a EA D 8 O 5 G O O D').

    NOTE: train ground-truth ocr_text is noisy and a few rows legitimately contain
    CJK, so this is OFF by default in the pipeline; enable + measure CER on train
    (Cell 6) before trusting it for the final submission.
    """
    s = normalize_ocr_output_text(text)
    if not s:
        return ""
    s = s.replace("\n", " ")
    s = _KEEP_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if drop_junk_only and not _HAS_WORD_RE.search(s):
        return ""
    return s


def _split_plus_items(display: str) -> List[str]:
    s = normalize_output_text(display)
    if not s:
        return []
    return [p.strip() for p in re.split(r"\s+\+\s+", s) if p.strip()]


def _display_key(display: str) -> str:
    """
    Lightweight key for deduping output display. Keeps accents folded only for
    dedup behavior, not for output value.
    """

    s = normalize_output_text(display).lower()
    s = unicodedata.normalize("NFD", s).replace("đ", "d")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^0-9a-zA-ZÀ-ÖØ-öø-ỹĐđ\s+]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        clean = normalize_output_text(item)
        key = _display_key(clean)
        if not clean or not key or key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def _drop_items_covered_by_full(items: Sequence[str]) -> List[str]:
    """
    If a full multi-item display is present, remove atomic items already covered
    inside that full display.
    """

    cleaned = [normalize_output_text(x) for x in items if normalize_output_text(x)]
    if not cleaned:
        return []

    full_items = [x for x in cleaned if " + " in x]
    if not full_items:
        return cleaned

    covered_keys = set()
    for full in full_items:
        for part in _split_plus_items(full):
            covered_keys.add(_display_key(part))

    out = []
    for x in cleaned:
        if " + " in x:
            out.append(x)
            continue
        if _display_key(x) in covered_keys:
            continue
        out.append(x)
    return out


# ---------------------------------------------------------------------
# Gazetteer resolution
# ---------------------------------------------------------------------


def _get_entry_display(gazetteer: Optional[ProductGazetteer], entry_id: str) -> str:
    if gazetteer is None or not entry_id:
        return ""
    entry = getattr(gazetteer, "entry_by_id", {}).get(entry_id)
    if entry is None:
        return ""
    return normalize_output_text(getattr(entry, "canonical_display", ""))


def resolve_chosen_displays(
    decision_row: Union[pd.Series, Mapping[str, Any]],
    gazetteer: Optional[ProductGazetteer] = None,
    config: Optional[FinalPredictorConfig] = None,
) -> List[str]:
    """
    Resolve canonical displays from a Cell 4 decision row.

    Priority:
    1. chosen_entry_ids + gazetteer canonical_display
    2. chosen_displays / chosen_displays_json from decision_df
    3. product_name_candidate / matched_display fallback
    """

    config = config or FinalPredictorConfig()
    get = decision_row.get if hasattr(decision_row, "get") else lambda k, d=None: d

    displays: List[str] = []

    chosen_ids = _parse_list_like(get("chosen_entry_ids", []))
    if not chosen_ids:
        chosen_ids = _parse_list_like(get("chosen_entry_ids_json", []))

    if config.prefer_gazetteer_display and gazetteer is not None and chosen_ids:
        for eid in chosen_ids:
            display = _get_entry_display(gazetteer, eid)
            if display:
                displays.append(display)

    if not displays:
        displays = _parse_list_like(get("chosen_displays", []))
    if not displays:
        displays = _parse_list_like(get("chosen_displays_json", []))

    if not displays:
        fallback = _safe_str(get("product_name_candidate", "")) or _safe_str(get("matched_display", ""))
        if fallback:
            displays = [fallback]

    displays = [normalize_output_text(x) for x in displays if normalize_output_text(x)]

    if config.deduplicate_items:
        displays = _dedupe_preserve_order(displays)

    if config.drop_items_covered_by_full:
        displays = _drop_items_covered_by_full(displays)
        if config.deduplicate_items:
            displays = _dedupe_preserve_order(displays)

    return displays


# ---------------------------------------------------------------------
# Final product construction
# ---------------------------------------------------------------------


def build_final_product_name(
    decision_row: Union[pd.Series, Mapping[str, Any]],
    gazetteer: Optional[ProductGazetteer] = None,
    config: Optional[FinalPredictorConfig] = None,
) -> str:
    """
    Build final product_name for one image from Cell 4 decision.

    This is the core of Cell 5. It does not re-score or re-gate; it only
    resolves the chosen canonical displays and joins multi-item outputs.
    """

    config = config or FinalPredictorConfig()
    get = decision_row.get if hasattr(decision_row, "get") else lambda k, d=None: d

    if not _safe_bool(get("emit_product", False)):
        return config.blank_product_value

    compose_mode = _safe_str(get("compose_mode", "")).lower()
    displays = resolve_chosen_displays(decision_row, gazetteer=gazetteer, config=config)

    if not displays:
        return config.blank_product_value

    # Full/single entry may already contain " + ". Keep it as one canonical string.
    if compose_mode in {"single", "full"} and len(displays) == 1:
        return normalize_output_text(displays[0])

    # For multi compose, join individual canonical displays.
    # If a display already contains +, keep it and avoid adding covered atomics.
    if config.drop_items_covered_by_full:
        displays = _drop_items_covered_by_full(displays)

    displays = _dedupe_preserve_order(displays) if config.deduplicate_items else list(displays)
    return normalize_output_text(config.joiner.join(displays))


def predict_product_name_from_text(
    ocr_text: str,
    extra_candidates: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> str:
    """
    Rule-first single-text orchestration entry point.

    Applies the centralized rule registry (product/rules.py) to one OCR string
    and returns the final, output-normalized product_name ("" for blank). This
    is the lightweight path used for debugging / unit tests; the full batch flow
    is A -> C -> D (which already applies the same rules via the gate) -> E.
    """
    name, _hit = resolve_product_name(
        ocr_text,
        extra_candidates=list(extra_candidates) if extra_candidates else None,
        verbose=verbose,
    )
    return normalize_output_text(name)


def finalize_product_predictions(
    decision_df: pd.DataFrame,
    gazetteer: Optional[ProductGazetteer] = None,
    config: Optional[FinalPredictorConfig] = None,
) -> pd.DataFrame:
    """
    Convert Cell 4 decision_df into image_id/product_name final predictions.
    """

    config = config or FinalPredictorConfig()

    if decision_df is None or decision_df.empty:
        return pd.DataFrame(columns=["image_id", config.output_product_col])

    if "image_id" not in decision_df.columns:
        raise ValueError("decision_df must contain image_id column")

    rows = []
    for _, row in decision_df.iterrows():
        product_name = build_final_product_name(row, gazetteer=gazetteer, config=config)
        rows.append(
            {
                "image_id": _safe_str(row.get("image_id", "")),
                config.output_product_col: product_name,
            }
        )

    out = pd.DataFrame(rows)
    out[config.output_product_col] = out[config.output_product_col].fillna("").astype(str)
    return out


# ---------------------------------------------------------------------
# OCR output merge
# ---------------------------------------------------------------------


def select_ocr_text_column(ocr_df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    """
    Pick OCR text column from engine output.

    Supports both current hard-only output and richer selected/raw/soft/hard output.
    """

    if preferred and preferred in ocr_df.columns:
        return preferred

    for col in ["ocr_text", "selected_text", "hard_text", "soft_text", "raw_text"]:
        if col in ocr_df.columns:
            return col

    raise ValueError(
        "Cannot find OCR text column. Expected one of: "
        "ocr_text, selected_text, hard_text, soft_text, raw_text"
    )


def build_final_submission(
    ocr_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    gazetteer: Optional[ProductGazetteer] = None,
    config: Optional[FinalPredictorConfig] = None,
    ocr_text_col: Optional[str] = None,
    sample_submission_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Build final submission-like DataFrame.

    Output default:
        image_id, ocr_text, product_name

    If sample_submission_path is provided, row order follows that file.
    """

    config = config or FinalPredictorConfig()

    if ocr_df is None or ocr_df.empty:
        raise ValueError("ocr_df is empty")
    if "image_id" not in ocr_df.columns:
        raise ValueError("ocr_df must contain image_id column")

    text_col = select_ocr_text_column(ocr_df, preferred=ocr_text_col)

    base = ocr_df[["image_id", text_col]].copy()
    base = base.drop_duplicates("image_id", keep="first")
    base = base.rename(columns={text_col: config.output_ocr_col})
    base[config.output_ocr_col] = base[config.output_ocr_col].apply(normalize_ocr_output_text)

    product_df = finalize_product_predictions(decision_df, gazetteer=gazetteer, config=config)

    out = base.merge(product_df, on="image_id", how="left")
    out[config.output_product_col] = out[config.output_product_col].fillna(config.blank_product_value).astype(str)

    if config.include_brand_name and "brand_name" not in out.columns:
        out.insert(2, "brand_name", "")

    # Respect sample submission order if available.
    if sample_submission_path:
        sample_path = Path(sample_submission_path)
        if sample_path.exists():
            sample = pd.read_csv(sample_path)
            if "image_id" in sample.columns:
                ordered = sample[["image_id"]].copy()
                out = ordered.merge(out, on="image_id", how="left")
                out[config.output_ocr_col] = out[config.output_ocr_col].fillna("")
                out[config.output_product_col] = out[config.output_product_col].fillna(config.blank_product_value)
                if config.include_brand_name and "brand_name" not in out.columns:
                    out.insert(2, "brand_name", "")

    cols = ["image_id", config.output_ocr_col]
    if config.include_brand_name:
        cols.append("brand_name")
    cols.append(config.output_product_col)

    return out[cols]


# ---------------------------------------------------------------------
# Validation / debug helpers
# ---------------------------------------------------------------------


def validate_final_submission(
    submission_df: pd.DataFrame,
    required_cols: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if required_cols is None:
        required_cols = ["image_id", "ocr_text", "product_name"]

    missing = [c for c in required_cols if c not in submission_df.columns]
    n_rows = 0 if submission_df is None else len(submission_df)

    result = {
        "n_rows": int(n_rows),
        "missing_columns": missing,
        "duplicate_image_id_count": 0,
        "blank_ocr_count": None,
        "blank_product_count": None,
        "product_fill_rate": None,
    }

    if submission_df is None or submission_df.empty:
        return result

    if "image_id" in submission_df.columns:
        result["duplicate_image_id_count"] = int(submission_df["image_id"].duplicated().sum())

    if "ocr_text" in submission_df.columns:
        result["blank_ocr_count"] = int(submission_df["ocr_text"].fillna("").astype(str).str.strip().eq("").sum())

    if "product_name" in submission_df.columns:
        blank_product = submission_df["product_name"].fillna("").astype(str).str.strip().eq("")
        result["blank_product_count"] = int(blank_product.sum())
        result["product_fill_rate"] = float((~blank_product).mean())

    return result


def summarize_final_predictions(submission_df: pd.DataFrame) -> Dict[str, Any]:
    if submission_df is None or submission_df.empty:
        return {"n_rows": 0}

    summary = validate_final_submission(submission_df)

    if "product_name" in submission_df.columns:
        top_products = (
            submission_df["product_name"]
            .fillna("")
            .astype(str)
            .loc[lambda s: s.str.strip() != ""]
            .value_counts()
            .head(30)
            .to_dict()
        )
        summary["top_products"] = top_products

    return summary


def debug_final_prediction(
    image_id: str,
    decision_df: pd.DataFrame,
    submission_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"image_id": str(image_id)}

    if decision_df is not None and not decision_df.empty and "image_id" in decision_df.columns:
        rows = decision_df[decision_df["image_id"].astype(str) == str(image_id)]
        if not rows.empty:
            out["decision"] = rows.iloc[0].to_dict()

    if submission_df is not None and not submission_df.empty and "image_id" in submission_df.columns:
        rows = submission_df[submission_df["image_id"].astype(str) == str(image_id)]
        if not rows.empty:
            out["submission"] = rows.iloc[0].to_dict()

    return out


__all__ = [
    "FinalPredictorConfig",
    "normalize_output_text",
    "normalize_ocr_output_text",
    "resolve_chosen_displays",
    "build_final_product_name",
    "finalize_product_predictions",
    "select_ocr_text_column",
    "build_final_submission",
    "validate_final_submission",
    
    "summarize_final_predictions",
    "debug_final_prediction",
]
