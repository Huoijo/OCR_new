from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import pandas as pd

from .A_candidate import (
    normalize_light,
    normalize_dedupe_key,
    tokenize_text,
    generate_candidate_dataframe,
)


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

PLUS_SPLIT_RE = re.compile(r"\s+\+\s+")

# A small protection list so auto-filler mining never removes important
# product/brand tokens even if they appear often in text-no-product rows.
# All keys here use normalize_dedupe_key(token).
PROTECTED_PRODUCT_TOKEN_KEYS: Set[str] = {
    "pate",
    "patê",
    "cot",
    "cột",
    "den",
    "đèn",
    "ha",
    "hạ",
    "long",
    "halong",
    "canfoco",
    "nan",
    "nestle",
    "nestlé",
    "optipro",
    "infinipro",
    "a2",
    "highland",
    "highlands",
    "coffee",
    "tra",
    "trà",
    "sen",
    "vang",
    "vàng",
    "sua",
    "sữa",
    "banh",
    "bánh",
    "mi",
    "mì",
    "que",
}


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def _norm_token_key(token: Any) -> str:
    """
    Normalize a single token for token-index/filler decisions.

    This intentionally reuses normalize_dedupe_key from Candidate Generation
    to keep Cell 1 and Cell 2 aligned.
    """

    return normalize_dedupe_key(_safe_str(token))


def _token_lower(token: Any) -> str:
    return normalize_light(_safe_str(token)).lower().strip()


# ---------------------------------------------------------------------
# Diacritic folding
#
# Real PaddleOCR output frequently drops Vietnamese diacritics
# (e.g. "ĐỒ HỘP HẠ LONG" -> "DO HOP HA LONG"). Diacritic-sensitive indices
# would then fail to retrieve the correct entry. We keep the diacritic-aware
# indices for precision AND add diacritic-folded indices for robust retrieval;
# disambiguation between folded collisions is left to Cell 4 (uses frequency).
# ---------------------------------------------------------------------

# đ/Đ do not decompose under NFD, so map them explicitly.
_VI_FOLD_MAP = {ord("đ"): "d", ord("Đ"): "D"}


def fold_text(text: Any) -> str:
    """ASCII-fold Vietnamese text: drop diacritics, keep base letters."""

    text = normalize_light(_safe_str(text)).translate(_VI_FOLD_MAP)
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", text)


def fold_key(text: Any) -> str:
    """Diacritic-folded counterpart of normalize_dedupe_key (the match key)."""

    return normalize_dedupe_key(fold_text(text))


def fold_token_key(token: Any) -> str:
    """Diacritic-folded counterpart of _norm_token_key (single token)."""

    return fold_key(_safe_str(token))


def _stable_entry_id(canonical_key: str, source_hint: str = "entry") -> str:
    raw = f"{source_hint}:{canonical_key}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _is_valid_filler_token(token: str) -> bool:
    token = _token_lower(token)
    if not token:
        return False
    if len(token) < 2:
        return False
    if token.isdigit():
        return False
    if not any(ch.isalpha() for ch in token):
        return False
    return True


def _json_default(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---------------------------------------------------------------------
# Product name splitting / canonical entry aggregation
# ---------------------------------------------------------------------


def split_product_items(product_name: Any) -> List[str]:
    """
    Split multi-item product labels on the competition separator " + ".

    Example:
        "Highlands Coffee Trà Sen Vàng + Bánh Mì Que"
        -> ["Highlands Coffee Trà Sen Vàng", "Bánh Mì Que"]

    We intentionally require spaces around + to avoid breaking model names
    or formulas that may contain plus-like characters.
    """

    text = normalize_light(_safe_str(product_name)).replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    parts = [p.strip() for p in PLUS_SPLIT_RE.split(text) if p.strip()]
    return parts or [text]


def iter_product_surfaces(
    product_name: Any,
    include_full: bool = True,
    include_atomic: bool = True,
) -> Iterable[Tuple[str, str, int]]:
    """
    Yield (surface, source_class, item_count) from one product_name label.

    source_class:
    - full: original full product_name, including " + " if present.
    - atomic: one item split from a multi-item product_name.
    """

    full = normalize_light(_safe_str(product_name)).replace("\n", " ")
    full = re.sub(r"\s+", " ", full).strip()
    if not full:
        return

    items = split_product_items(full)
    item_count = len(items)

    if include_full:
        yield full, "full", item_count

    if include_atomic and item_count >= 2:
        for item in items:
            yield item, "atomic", 1


def _choose_canonical_display(display_counts: Counter) -> str:
    """
    Deterministically choose canonical display among variants with same key.

    Priority:
    1. Highest frequency in train labels.
    2. More alphabetic characters.
    3. Longer text.
    4. Lexicographic order for reproducibility.
    """

    if not display_counts:
        return ""

    def rank(item: Tuple[str, int]) -> Tuple[int, int, int, str]:
        display, count = item
        alpha = sum(ch.isalpha() for ch in display)
        return (count, alpha, len(display), display)

    return max(display_counts.items(), key=rank)[0]


def _infer_brand_anchor(display: str, max_tokens: int = 2) -> str:
    """
    Lightweight brand anchor heuristic.

    This is intentionally conservative. It is only metadata for Cell 3/4,
    not a final product decision.
    """

    toks = tokenize_text(display)
    toks = [_token_lower(t) for t in toks if _token_lower(t)]
    if not toks:
        return ""
    return " ".join(toks[:max_tokens])


# ---------------------------------------------------------------------
# Gazetteer dataclasses
# ---------------------------------------------------------------------

@dataclass
class GazetteerEntry:
    entry_id: str
    canonical_display: str
    canonical_key: str
    folded_key: str
    token_keys: List[str]
    folded_token_keys: List[str]
    source_class: str
    frequency_prior: int = 1
    item_count: int = 1
    brand_anchor: str = ""
    surface_variants: List[str] = field(default_factory=list)
    folded_surface_keys: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProductGazetteer:
    entries: List[GazetteerEntry]
    exact_index: Dict[str, str]
    token_inverted_index: Dict[str, List[str]]
    surface_index: Dict[str, List[str]]
    folded_exact_index: Dict[str, List[str]]
    folded_token_inverted_index: Dict[str, List[str]]
    folded_surface_index: Dict[str, List[str]]
    entry_by_id: Dict[str, GazetteerEntry]
    filler_tokens: Set[str]
    ambiguous_anchors: Dict[str, Set[str]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def enrich_with_rule_aliases(self, aliases: Optional[Dict[str, str]] = None) -> int:
        """
        Add hand-curated rule aliases (product/rules.py) as extra surface lookups.

        For each {normalized_alias: canonical}, resolve the existing entry whose
        canonical matches, then register the alias (diacritic + folded form) in the
        surface indices so a candidate equal to the alias retrieves that entry.
        Purely additive; never removes/merges entries. Returns #surfaces added.
        """
        try:
            from .rules import get_rule_aliases
        except Exception:
            return 0
        aliases = aliases if aliases is not None else get_rule_aliases()
        added = 0
        for alias_norm, canonical in aliases.items():
            ck = normalize_dedupe_key(canonical)
            eid = self.exact_index.get(ck)
            if not eid:
                fids = self.folded_exact_index.get(fold_key(canonical), [])
                eid = fids[0] if fids else None
            if not eid or eid not in self.entry_by_id:
                continue
            dk = normalize_dedupe_key(alias_norm)
            fk = fold_key(alias_norm)
            for idx, key in ((self.surface_index, dk), (self.folded_surface_index, fk)):
                if not key:
                    continue
                lst = idx.setdefault(key, [])
                if eid not in lst:
                    lst.append(eid)
                    added += 1
        if added:
            self.metadata["n_rule_alias_surfaces_added"] = int(
                self.metadata.get("n_rule_alias_surfaces_added", 0)) + added
        return added

    def lookup_candidates(
        self,
        candidate_row: Union[pd.Series, Mapping[str, Any]],
        max_entries: int = 80,
    ) -> List[GazetteerEntry]:
        """
        Return plausible gazetteer entries for a candidate row.

        This is NOT fuzzy matching yet. It only narrows the search space for
        Cell 3 via 6 lookups: exact / surface / token, each in both
        diacritic-aware and diacritic-folded form (robust to OCR dropping dấu).
        """

        get = candidate_row.get if hasattr(candidate_row, "get") else lambda k, d=None: d

        clean_text = _safe_str(get("clean_text", ""))
        normalized_key = _safe_str(get("normalized_key", ""))
        if not normalized_key:
            normalized_key = normalize_dedupe_key(clean_text)

        # Folded key from clean_text if available, else fold the normalized key.
        folded_key_ = fold_key(clean_text) if clean_text else fold_key(normalized_key)

        tokenized = get("tokenized", [])
        if isinstance(tokenized, str):
            # Handles CSV-reloaded candidate_df where tokenized may become string.
            tokenized = tokenize_text(tokenized)
        if not isinstance(tokenized, (list, tuple)):
            tokenized = []

        token_keys = [t for t in (_norm_token_key(x) for x in tokenized) if t]
        folded_token_keys = [t for t in (fold_token_key(x) for x in tokenized) if t]

        entry_ids: Set[str] = set()

        # 1-2. Exact canonical match (diacritic, then folded).
        if normalized_key in self.exact_index:
            entry_ids.add(self.exact_index[normalized_key])
        for eid in self.folded_exact_index.get(folded_key_, []):
            entry_ids.add(eid)

        # 3-4. Observed OCR surface match (diacritic, then folded).
        for eid in self.surface_index.get(normalized_key, []):
            entry_ids.add(eid)
        for eid in self.folded_surface_index.get(folded_key_, []):
            entry_ids.add(eid)

        # 5-6. Token inverted lookup (diacritic, then folded).
        for tok in token_keys:
            for eid in self.token_inverted_index.get(tok, []):
                entry_ids.add(eid)
        for tok in folded_token_keys:
            for eid in self.folded_token_inverted_index.get(tok, []):
                entry_ids.add(eid)

        entries = [self.entry_by_id[eid] for eid in entry_ids if eid in self.entry_by_id]

        cand_tok = set(token_keys)
        cand_ftok = set(folded_token_keys)

        # Ordering prefers diacritic-correct evidence over folded (fold is lossy),
        # then token overlap, then frequency prior. Cell 4 still disambiguates ties.
        def rank(entry: GazetteerEntry) -> Tuple[int, int, int, int, int, str]:
            exact_dia = int(entry.canonical_key == normalized_key)
            surf_dia = int(normalized_key in {normalize_dedupe_key(v) for v in entry.surface_variants})
            exact_fold = int(entry.folded_key == folded_key_)
            surf_fold = int(folded_key_ in set(entry.folded_surface_keys))
            overlap_dia = len(cand_tok.intersection(entry.token_keys))
            overlap_fold = len(cand_ftok.intersection(entry.folded_token_keys))
            return (
                2 * exact_dia + surf_dia,      # diacritic-exact strongest
                2 * exact_fold + surf_fold,    # folded-exact next
                overlap_dia,                   # diacritic token overlap
                overlap_fold,                  # folded token overlap
                int(entry.frequency_prior),
                entry.canonical_display,
            )

        entries.sort(key=rank, reverse=True)
        return entries[:max_entries]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "exact_index": self.exact_index,
            "token_inverted_index": self.token_inverted_index,
            "surface_index": self.surface_index,
            "folded_exact_index": self.folded_exact_index,
            "folded_token_inverted_index": self.folded_token_inverted_index,
            "folded_surface_index": self.folded_surface_index,
            "filler_tokens": sorted(self.filler_tokens),
            "ambiguous_anchors": {k: sorted(v) for k, v in self.ambiguous_anchors.items()},
            "metadata": self.metadata,
        }

    def save_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ProductGazetteer":
        path = Path(path)
        obj = json.loads(path.read_text(encoding="utf-8"))

        entries = [GazetteerEntry(**e) for e in obj.get("entries", [])]
        entry_by_id = {e.entry_id: e for e in entries}

        return cls(
            entries=entries,
            exact_index=dict(obj.get("exact_index", {})),
            token_inverted_index={k: list(v) for k, v in obj.get("token_inverted_index", {}).items()},
            surface_index={k: list(v) for k, v in obj.get("surface_index", {}).items()},
            folded_exact_index={k: list(v) for k, v in obj.get("folded_exact_index", {}).items()},
            folded_token_inverted_index={k: list(v) for k, v in obj.get("folded_token_inverted_index", {}).items()},
            folded_surface_index={k: list(v) for k, v in obj.get("folded_surface_index", {}).items()},
            entry_by_id=entry_by_id,
            filler_tokens=set(obj.get("filler_tokens", [])),
            ambiguous_anchors={k: set(v) for k, v in obj.get("ambiguous_anchors", {}).items()},
            metadata=dict(obj.get("metadata", {})),
        )


# ---------------------------------------------------------------------
# Build entries / indices
# ---------------------------------------------------------------------


def _build_entries_from_product_names(
    product_names: Sequence[Any],
    include_full: bool = True,
    include_atomic: bool = True,
) -> List[GazetteerEntry]:
    """
    Build deterministic GazetteerEntry objects from product_name labels.
    """

    buckets: Dict[str, Dict[str, Any]] = {}

    for product_name in product_names:
        for surface, source_class, item_count in iter_product_surfaces(
            product_name,
            include_full=include_full,
            include_atomic=include_atomic,
        ):
            canonical_key = normalize_dedupe_key(surface)
            if not canonical_key:
                continue

            bucket = buckets.setdefault(
                canonical_key,
                {
                    "display_counts": Counter(),
                    "source_classes": Counter(),
                    "frequency_prior": 0,
                    "item_count": item_count,
                },
            )

            bucket["display_counts"][surface] += 1
            bucket["source_classes"][source_class] += 1
            bucket["frequency_prior"] += 1
            bucket["item_count"] = max(bucket["item_count"], item_count)

    entries: List[GazetteerEntry] = []

    for canonical_key in sorted(buckets.keys()):
        bucket = buckets[canonical_key]
        canonical_display = _choose_canonical_display(bucket["display_counts"])
        tokens = tokenize_text(canonical_display)
        token_keys = sorted({_norm_token_key(t) for t in tokens if _norm_token_key(t)})
        folded_token_keys = sorted({fold_token_key(t) for t in tokens if fold_token_key(t)})

        source_classes = bucket["source_classes"]
        if len(source_classes) == 1:
            source_class = next(iter(source_classes.keys()))
        else:
            source_class = "mixed"

        entries.append(
            GazetteerEntry(
                entry_id=_stable_entry_id(canonical_key),
                canonical_display=canonical_display,
                canonical_key=canonical_key,
                folded_key=fold_key(canonical_display),
                token_keys=token_keys,
                folded_token_keys=folded_token_keys,
                source_class=source_class,
                frequency_prior=int(bucket["frequency_prior"]),
                item_count=int(bucket["item_count"]),
                brand_anchor=_infer_brand_anchor(canonical_display),
                surface_variants=[],
            )
        )

    return entries


def _build_indices(
    entries: Sequence[GazetteerEntry],
    ambiguous_max_df_ratio: float = 0.10,
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]], Dict[str, Set[str]]]:
    exact_index: Dict[str, str] = {}
    token_index: Dict[str, Set[str]] = defaultdict(set)
    folded_exact: Dict[str, Set[str]] = defaultdict(set)
    folded_token_index: Dict[str, Set[str]] = defaultdict(set)
    anchors: Dict[str, Set[str]] = defaultdict(set)

    for e in entries:
        # Diacritic-aware exact key is unique per entry (canonical_key dedup),
        # but folded keys COLLIDE across entries, so folded_exact maps to a list.
        exact_index[e.canonical_key] = e.entry_id
        if e.folded_key:
            folded_exact[e.folded_key].add(e.entry_id)

        for tok in e.token_keys:
            if tok:
                token_index[tok].add(e.entry_id)
        for tok in e.folded_token_keys:
            if tok:
                folded_token_index[tok].add(e.entry_id)

        if e.brand_anchor:
            anchor_key = fold_key(e.brand_anchor)
            if anchor_key:
                anchors[anchor_key].add(e.entry_id)

    token_inverted_index = {k: sorted(v) for k, v in sorted(token_index.items())}
    folded_exact_index = {k: sorted(v) for k, v in sorted(folded_exact.items())}
    folded_token_inverted_index = {k: sorted(v) for k, v in sorted(folded_token_index.items())}

    ambiguous_anchors = _build_ambiguous_anchors(entries, anchors, ambiguous_max_df_ratio)

    return (
        exact_index,
        token_inverted_index,
        folded_exact_index,
        folded_token_inverted_index,
        ambiguous_anchors,
    )


def _build_ambiguous_anchors(
    entries: Sequence[GazetteerEntry],
    folded_anchor_groups: Mapping[str, Set[str]],
    max_df_ratio: float,
) -> Dict[str, Set[str]]:
    """
    Flag entry groups that an ambiguous surface could resolve to.

    Two signals:
    - "anchor:<folded brand anchor>"  -> entries sharing a brand anchor
      (diacritic variants merge because the anchor key is folded).
    - "tok:<a>|<b>"  -> entries sharing >= 2 folded tokens (a, b).
      Generic tokens (document frequency > max_df_ratio of entries) are skipped
      so groups stay specific (e.g. "ha|long") instead of exploding on "sua"/"pate".
    """

    ambiguous: Dict[str, Set[str]] = {
        f"anchor:{k}": set(v) for k, v in folded_anchor_groups.items() if len(v) >= 2
    }

    n = max(1, len(entries))
    df: Counter = Counter()
    for e in entries:
        for tok in set(e.folded_token_keys):
            if tok:
                df[tok] += 1

    max_df = max(2, int(max_df_ratio * n))

    pair_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for e in entries:
        ftoks = sorted({t for t in e.folded_token_keys if t and df[t] <= max_df})
        for i in range(len(ftoks)):
            for j in range(i + 1, len(ftoks)):
                pair_index[(ftoks[i], ftoks[j])].add(e.entry_id)

    for (a, b), ids in pair_index.items():
        if len(ids) >= 2:
            ambiguous[f"tok:{a}|{b}"] = set(ids)

    return ambiguous


# ---------------------------------------------------------------------
# Surface variant mining from train OCR text
# ---------------------------------------------------------------------


def _simple_similarity(a: str, b: str) -> float:
    """
    Deterministic stdlib similarity. Cell 3 can use rapidfuzz later;
    Cell 2 avoids adding another dependency.
    """

    from difflib import SequenceMatcher

    a = normalize_dedupe_key(a)
    b = normalize_dedupe_key(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _best_ocr_surface_for_product(
    ocr_text: Any,
    product_surface: str,
    min_score: float = 0.72,
    max_extra_tokens: int = 4,
) -> Optional[str]:
    """
    Find the best token-window in train ocr_text that resembles product_surface.

    This mines real OCR spelling variants without using test labels.
    """

    ocr_clean = normalize_light(_safe_str(ocr_text)).replace("\n", " ")
    product_clean = normalize_light(product_surface).replace("\n", " ")
    if not ocr_clean or not product_clean:
        return None

    ocr_tokens = tokenize_text(ocr_clean)
    product_tokens = tokenize_text(product_clean)
    if not ocr_tokens or not product_tokens:
        return None

    target_n = len(product_tokens)
    min_n = max(1, target_n - 2)
    max_n = min(len(ocr_tokens), target_n + max_extra_tokens)

    best_surface = None
    best_score = 0.0

    for n in range(min_n, max_n + 1):
        for start in range(0, len(ocr_tokens) - n + 1):
            window = " ".join(ocr_tokens[start : start + n])
            score = _simple_similarity(window, product_clean)
            if score > best_score:
                best_score = score
                best_surface = window

    if best_surface and best_score >= min_score:
        return best_surface

    return None


def _mine_surface_variants(
    entries: List[GazetteerEntry],
    text_with_product_df: pd.DataFrame,
    max_variants_per_entry: int = 40,
    min_score: float = 0.72,
) -> None:
    """
    Mutates entries by adding observed OCR surface variants from train rows.
    """

    entry_by_key = {e.canonical_key: e for e in entries}
    variant_counts: Dict[str, Counter] = defaultdict(Counter)

    for _, row in text_with_product_df.iterrows():
        ocr_text = row.get("ocr_text", "")
        product_name = row.get("product_name", "")

        for surface, _, _ in iter_product_surfaces(product_name, include_full=True, include_atomic=True):
            key = normalize_dedupe_key(surface)
            if key not in entry_by_key:
                continue

            found = _best_ocr_surface_for_product(
                ocr_text=ocr_text,
                product_surface=surface,
                min_score=min_score,
            )
            if found:
                variant_key = normalize_dedupe_key(found)
                if variant_key and variant_key != key:
                    variant_counts[key][found] += 1

    for e in entries:
        counts = variant_counts.get(e.canonical_key, Counter())
        ranked = sorted(counts.items(), key=lambda kv: (kv[1], len(kv[0]), kv[0]), reverse=True)
        e.surface_variants = [surface for surface, _ in ranked[:max_variants_per_entry]]


def _build_surface_index(
    entries: Sequence[GazetteerEntry],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Build diacritic and folded surface indices, and populate
    entry.folded_surface_keys as a side effect (used by lookup ranking).
    """

    idx: Dict[str, Set[str]] = defaultdict(set)
    fidx: Dict[str, Set[str]] = defaultdict(set)

    for e in entries:
        folded_keys: Set[str] = set()
        for surface in e.surface_variants:
            key = normalize_dedupe_key(surface)
            if key:
                idx[key].add(e.entry_id)
            fkey = fold_key(surface)
            if fkey:
                fidx[fkey].add(e.entry_id)
                folded_keys.add(fkey)
        e.folded_surface_keys = sorted(folded_keys)

    surface_index = {k: sorted(v) for k, v in sorted(idx.items())}
    folded_surface_index = {k: sorted(v) for k, v in sorted(fidx.items())}
    return surface_index, folded_surface_index


# ---------------------------------------------------------------------
# Filler token mining
# ---------------------------------------------------------------------


def _product_token_keys_from_entries(entries: Sequence[GazetteerEntry]) -> Set[str]:
    """
    Collect all token keys that must never be stripped as filler.

    Includes both diacritic and folded forms so a brand token survives even
    when the candidate (or filler mining) sees a de-accented spelling.
    """

    out: Set[str] = set(PROTECTED_PRODUCT_TOKEN_KEYS)
    out.update(fold_token_key(t) for t in PROTECTED_PRODUCT_TOKEN_KEYS)
    for e in entries:
        out.update(e.token_keys)
        out.update(e.folded_token_keys)
        for surface in e.surface_variants:
            for t in tokenize_text(surface):
                out.add(_norm_token_key(t))
                out.add(fold_token_key(t))
    return {x for x in out if x}


def _read_manual_filler_tokens(path: Optional[Union[str, Path]]) -> Set[str]:
    if not path:
        return set()

    p = Path(path)
    if not p.exists():
        return set()

    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        token = _token_lower(line.strip())
        if token:
            out.add(token)
    return out


def build_filler_tokens(
    train_df: pd.DataFrame,
    entries: Sequence[GazetteerEntry],
    manual_filler_path: Optional[Union[str, Path]] = None,
    min_count: int = 8,
    top_k: int = 300,
    max_product_row_ratio: float = 0.02,
) -> Set[str]:
    """
    Mine high-precision filler tokens from text-no-product rows.

    A token is considered filler when:
    - it appears often in ocr_text where product_name is blank;
    - it is not a product/brand token according to gazetteer entries;
    - it appears very rarely inside product_name-positive rows.
    """

    df = train_df.copy()
    df["ocr_text"] = df.get("ocr_text", "").fillna("").astype(str)
    df["product_name"] = df.get("product_name", "").fillna("").astype(str)

    text_no_product = df[(df["ocr_text"].str.strip() != "") & (df["product_name"].str.strip() == "")]
    text_with_product = df[(df["ocr_text"].str.strip() != "") & (df["product_name"].str.strip() != "")]

    product_token_keys = _product_token_keys_from_entries(entries)

    no_product_counts: Counter = Counter()
    product_row_counts: Counter = Counter()

    for text in text_no_product["ocr_text"].tolist():
        toks = {_token_lower(t) for t in tokenize_text(text) if _is_valid_filler_token(t)}
        no_product_counts.update(toks)

    for text in text_with_product["ocr_text"].tolist():
        toks = {_token_lower(t) for t in tokenize_text(text) if _is_valid_filler_token(t)}
        product_row_counts.update(toks)

    n_product_rows = max(1, len(text_with_product))

    candidates = []
    for tok, cnt in no_product_counts.items():
        tok_key = _norm_token_key(tok)
        if not tok_key:
            continue
        if tok_key in product_token_keys or fold_token_key(tok) in product_token_keys:
            continue
        if cnt < min_count:
            continue

        product_ratio = product_row_counts.get(tok, 0) / n_product_rows
        if product_ratio > max_product_row_ratio:
            continue

        candidates.append((tok, cnt, product_ratio))

    candidates.sort(key=lambda x: (x[1], -x[2], x[0]), reverse=True)

    auto_fillers = {tok for tok, _, _ in candidates[:top_k]}
    manual = _read_manual_filler_tokens(manual_filler_path)

    # Manual fillers are still protected from accidentally stripping brand tokens.
    out = set()
    for tok in auto_fillers | manual:
        tok_key = _norm_token_key(tok)
        tok_fold = fold_token_key(tok)
        if (
            tok_key
            and tok_key not in product_token_keys
            and tok_fold not in product_token_keys
            and _is_valid_filler_token(tok)
        ):
            out.add(tok)

    return out


# ---------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------


def build_product_gazetteer(
    train_labels_path: Union[str, Path],
    manual_filler_path: Optional[Union[str, Path]] = None,
    include_full: bool = True,
    include_atomic: bool = True,
    mine_surface_variants: bool = True,
    surface_min_score: float = 0.72,
    filler_min_count: int = 8,
    filler_top_k: int = 300,
    filler_max_product_row_ratio: float = 0.02,
    cache_json_path: Optional[Union[str, Path]] = None,
    apply_rule_aliases: bool = True,
) -> ProductGazetteer:
    """
    Build Cell 2 ProductGazetteer from train_labels.csv.

    Required columns:
    - image_id
    - ocr_text
    - product_name
    """

    path = Path(train_labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find train_labels.csv: {path}")

    df = pd.read_csv(path)

    required = {"image_id", "ocr_text", "product_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train_labels.csv missing required columns: {missing}. Found: {list(df.columns)}")

    df = df.copy()
    df["ocr_text"] = df["ocr_text"].fillna("").astype(str)
    df["product_name"] = df["product_name"].fillna("").astype(str)

    all_product = df[df["product_name"].str.strip() != ""].copy()
    text_with_product = df[(df["ocr_text"].str.strip() != "") & (df["product_name"].str.strip() != "")].copy()
    text_no_product = df[(df["ocr_text"].str.strip() != "") & (df["product_name"].str.strip() == "")].copy()
    blank = df[df["ocr_text"].str.strip() == ""].copy()

    # Vocabulary comes from EVERY labelled product_name (even rows whose ocr_text
    # is blank) so the gazetteer knows all known products. Surface-variant mining,
    # which needs OCR text to align against, uses only text_with_product rows.
    entries = _build_entries_from_product_names(
        product_names=all_product["product_name"].tolist(),
        include_full=include_full,
        include_atomic=include_atomic,
    )

    if mine_surface_variants:
        _mine_surface_variants(
            entries=entries,
            text_with_product_df=text_with_product,
            min_score=surface_min_score,
        )

    (
        exact_index,
        token_inverted_index,
        folded_exact_index,
        folded_token_inverted_index,
        ambiguous_anchors,
    ) = _build_indices(entries)
    surface_index, folded_surface_index = _build_surface_index(entries)
    entry_by_id = {e.entry_id: e for e in entries}

    filler_tokens = build_filler_tokens(
        train_df=df,
        entries=entries,
        manual_filler_path=manual_filler_path,
        min_count=filler_min_count,
        top_k=filler_top_k,
        max_product_row_ratio=filler_max_product_row_ratio,
    )

    metadata = {
        "train_labels_path": str(path),
        "n_rows": int(len(df)),
        "n_blank_ocr": int(len(blank)),
        "n_text_no_product": int(len(text_no_product)),
        "n_text_with_product": int(len(text_with_product)),
        "n_all_product_rows": int(len(all_product)),
        "n_entries": int(len(entries)),
        "n_exact_index": int(len(exact_index)),
        "n_token_index": int(len(token_inverted_index)),
        "n_surface_index": int(len(surface_index)),
        "n_folded_exact_index": int(len(folded_exact_index)),
        "n_folded_token_index": int(len(folded_token_inverted_index)),
        "n_folded_surface_index": int(len(folded_surface_index)),
        "n_ambiguous_anchors": int(len(ambiguous_anchors)),
        "n_filler_tokens": int(len(filler_tokens)),
        "include_full": bool(include_full),
        "include_atomic": bool(include_atomic),
        "mine_surface_variants": bool(mine_surface_variants),
        "surface_min_score": float(surface_min_score),
        "filler_min_count": int(filler_min_count),
        "filler_top_k": int(filler_top_k),
        "filler_max_product_row_ratio": float(filler_max_product_row_ratio),
    }

    gazetteer = ProductGazetteer(
        entries=entries,
        exact_index=exact_index,
        token_inverted_index=token_inverted_index,
        surface_index=surface_index,
        folded_exact_index=folded_exact_index,
        folded_token_inverted_index=folded_token_inverted_index,
        folded_surface_index=folded_surface_index,
        entry_by_id=entry_by_id,
        filler_tokens=filler_tokens,
        ambiguous_anchors=ambiguous_anchors,
        metadata=metadata,
    )

    if apply_rule_aliases:
        gazetteer.enrich_with_rule_aliases()

    if cache_json_path:
        gazetteer.save_json(cache_json_path)

    return gazetteer


# ---------------------------------------------------------------------
# Rerun Candidate Generation with filler tokens
# ---------------------------------------------------------------------


def rerun_candidates_with_gazetteer_fillers(
    ocr_df: pd.DataFrame,
    gazetteer: ProductGazetteer,
    max_ngram: int = 4,
    max_tokens_for_ngram_source: int = 8,
    image_width_col: Optional[str] = None,
    image_height_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Step 10: rerun Cell 1 Candidate Generation with Cell 2 filler tokens.
    """

    return generate_candidate_dataframe(
        ocr_df=ocr_df,
        filler_tokens=gazetteer.filler_tokens,
        max_ngram=max_ngram,
        max_tokens_for_ngram_source=max_tokens_for_ngram_source,
        image_width_col=image_width_col,
        image_height_col=image_height_col,
    )


# ---------------------------------------------------------------------
# Sanity checks / debug helpers
# ---------------------------------------------------------------------


def summarize_gazetteer(gazetteer: ProductGazetteer, top_k: int = 20) -> Dict[str, Any]:
    top_entries = sorted(
        gazetteer.entries,
        key=lambda e: (e.frequency_prior, e.canonical_display),
        reverse=True,
    )[:top_k]

    return {
        **gazetteer.metadata,
        "top_entries": [
            {
                "canonical_display": e.canonical_display,
                "frequency_prior": e.frequency_prior,
                "source_class": e.source_class,
                "surface_variants": e.surface_variants[:5],
            }
            for e in top_entries
        ],
        "ambiguous_anchor_count": len(gazetteer.ambiguous_anchors),
        "sample_ambiguous_anchors": {
            k: sorted(v)[:8]
            for k, v in list(sorted(gazetteer.ambiguous_anchors.items()))[:10]
        },
        "sample_filler_tokens": sorted(gazetteer.filler_tokens)[:top_k],
    }


def product_coverage_check(train_labels_path: Union[str, Path], gazetteer: ProductGazetteer) -> Dict[str, Any]:
    df = pd.read_csv(train_labels_path)
    df["product_name"] = df.get("product_name", "").fillna("").astype(str)
    product_names = df.loc[df["product_name"].str.strip() != "", "product_name"].tolist()

    missing = []
    total_surfaces = 0
    for product_name in product_names:
        for surface, _, _ in iter_product_surfaces(product_name, include_full=True, include_atomic=True):
            total_surfaces += 1
            key = normalize_dedupe_key(surface)
            if key and key not in gazetteer.exact_index:
                missing.append(surface)

    return {
        "total_surfaces": int(total_surfaces),
        "missing_count": int(len(missing)),
        "coverage": 1.0 - (len(missing) / max(1, total_surfaces)),
        "missing_examples": sorted(set(missing))[:20],
    }


def filler_safety_check(gazetteer: ProductGazetteer) -> Dict[str, Any]:
    product_token_keys = _product_token_keys_from_entries(gazetteer.entries)
    bad = []
    for tok in gazetteer.filler_tokens:
        if _norm_token_key(tok) in product_token_keys:
            bad.append(tok)

    return {
        "bad_filler_count": len(bad),
        "bad_filler_examples": sorted(bad)[:50],
    }


__all__ = [
    "GazetteerEntry",
    "ProductGazetteer",
    "fold_text",
    "fold_key",
    "fold_token_key",
    "split_product_items",
    "build_product_gazetteer",
    "rerun_candidates_with_gazetteer_fillers",
    "summarize_gazetteer",
    "product_coverage_check",
    "filler_safety_check",
]
