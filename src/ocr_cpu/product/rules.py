from __future__ import annotations

"""
Centralized, context-aware rule registry for product_name extraction.

This module is the single place where hand-curated, high-confidence product
rules live. A/B/C/D/E import helpers from here instead of spreading fragile
if/else chains across files.

Design contract (every rule hit is a dict):
    {
        "candidate": "...",        # what triggered the hit (text / pattern label)
        "canonical": "...",        # canonical product_name to emit
        "score": 1.0,              # 0..1 confidence of the hit
        "priority": 95,            # integer, higher wins
        "source": "exact_rule",    # context_rule / exact_rule / alias_rule / conflict_rule
        "rule_name": "...",        # stable id of the rule that fired
        "reason": "...",           # human readable why
    }

Final resolution order (NEVER longest-candidate-alone):
    priority desc  >  score desc  >  len(canonical) desc

All canonicals below were validated against the real
`the-2nd-ura-hackathon/train_labels.csv`. Two canonicals that do NOT exist in
train were corrected per project decision:
  - Highlands sen vàng + trà vải -> "Highlands Coffee trà sen vàng, trà vải"
    (train only ever uses the comma/lowercase surface for this combo; the
    " + " joiner surface only exists for Bánh Mì Que and Americano Vải)
  - "Halong Canfoco Pate Cột Đèn" -> "Halong Canfoco Pate Cột Đèn Hải Phòng"
    (the bare form has 0 occurrences in train)
"""

import logging
import re
import unicodedata
from typing import Dict, List, Optional

log = logging.getLogger("ocr_cpu.product.rules")


# =====================================================================
# 2. Core normalization helpers
# =====================================================================

def strip_accents(s: str) -> str:
    """Remove Vietnamese diacritics, keep base latin letters (đ/Đ -> d/D)."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", s)


def normalize_rule_text(s: str) -> str:
    """
    Lowercase, remove accents, keep only latin letters/digits, collapse spaces.

    'Công ty Cổ phần Đồ hộp Hạ Long!!!' -> 'cong ty co phan do hop ha long'
    """
    s = strip_accents(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


# =====================================================================
# 3. CP — context-sensitive (NOT globally blacklisted; CP is a real label)
# =====================================================================

# Standalone tokens that are NEVER a valid product_name on their own.
# NOTE: "cp" is deliberately NOT here — it is handled contextually.
BAD_STANDALONE_CANDIDATES = {
    "ctcp",
    "jsc",
    "tnhh",
    "cong ty",
    "congty",
    "co phan",
    "nha may",
    "tong giam doc",
    "giam doc",
}

# Contexts where standalone CP is the real CP product/brand.
CP_PRODUCT_POSITIVE_PATTERNS = [
    r"(^|\s)cp\s+vi\s+",
    r"(^|\s)cp\s+ngon\b",
    r"(^|\s)cp\s+hoi\b",
    r"(^|\s)cp\s+bo\b",
    r"(^|\s)cp\s+ga\b",
    r"(^|\s)cp\s+heo\b",
    r"(^|\s)cp\s+xuc\s+xich\b",
]

# Contexts where standalone CP is almost surely legal-abbrev / OCR noise.
CP_DANGEROUS_CONTEXT_PATTERNS = [
    r"\bcong ty\s+cp\b",
    r"\bctcp\b",
    r"\bco phan\b",
    r"\bcong ty co phan\b",
    r"\bdo hop ha long\b",
    r"\bha long\b",
    r"\bhp ha long\b",
    r"\bdo hp\b",
    r"\bcap nhat\b",
    r"\bvu\b",
    r"\bvirus\b",
    r"\btieu diet\b",
    r"\btieu huy\b",
    r"\bpate\b",
    r"\bdoi tac\b",
]


def is_cp_product_context(full_text: str) -> bool:
    """True only when CP is likely the real CP product/brand."""
    norm = normalize_rule_text(full_text)
    has_positive = any(re.search(p, norm) for p in CP_PRODUCT_POSITIVE_PATTERNS)
    if not has_positive:
        return False
    # Even with a positive cue, abort if a dangerous legal/news context co-occurs.
    has_danger = any(re.search(p, norm) for p in CP_DANGEROUS_CONTEXT_PATTERNS)
    return not has_danger


def should_reject_short_candidate(candidate: str, full_text: str) -> bool:
    """
    Reject legal/noisy short candidates contextually.

    - CTCP / JSC / TNHH / Công ty / Cổ phần / Nhà máy ... -> always reject standalone.
    - CP -> reject only when context is NOT a true CP product context.
    - Anything longer/normal -> keep.
    """
    cand = normalize_rule_text(candidate)
    if not cand:
        return True
    if cand in BAD_STANDALONE_CANDIDATES:
        return True
    if cand == "cp":
        return not is_cp_product_context(full_text)
    return False


# =====================================================================
# Rule groups (each pattern matched against normalized text)
# =====================================================================

# ---- 4. Hạ Long legal-company -> commercial brand --------------------
HALONG_LEGAL_RULES = [
    {
        "name": "halong_legal_company_to_brand",
        "priority": 96,
        "canonical": "Đồ hộp Hạ Long",
        "patterns": [
            r"\bcong ty co phan\b.*\bdo hop ha long\b",
            r"\bcong ty cp\b.*\bdo hop ha long\b",
            r"\bctcp\b.*\bdo hop ha long\b",
            r"\bcong ty\b.*\bdo hop ha long\b",
            r"\bnha may\b.*\bdo hop ha long\b",
        ],
    }
]

# ---- 4. Hạ Long noisy-OCR / news context ----------------------------
HALONG_CONTEXT_RULES = [
    {
        "name": "halong_news_context",
        "priority": 92,
        "canonical": "Đồ hộp Hạ Long",
        "patterns": [
            r"\bha long\b.*\bvirus\b",
            r"\bvirus\b.*\bha long\b",
            r"\bha long\b.*\btieu diet\b",
            r"\btieu diet\b.*\bha long\b",
            r"\bha long\b.*\btieu huy\b",
            r"\btieu huy\b.*\bha long\b",
            r"\bha long\b.*\bcap nhat\b",
            r"\bcap nhat\b.*\bha long\b",
            r"\bha long\b.*\bvu\b",
            r"\bvu\b.*\bha long\b",
            r"\bdo hop\b.*\bha long\b",
            r"\bdo hp\b.*\bha long\b",
            r"\bhp ha long\b",
        ],
    }
]

# ---- 5. Pate Cột Đèn / Halong Canfoco (exact, very high priority) ----
# canonical "Pate Cột Đèn Hải Phòng" (39), "Pate Cột Đèn" (14) exist in train.
# "Halong Canfoco Pate Cột Đèn Hải Phòng" (3) exists; bare form does NOT -> corrected.
#
# OCR-tolerant "Hải Phòng": PaddleOCR often drops a letter ("Hải"->"Hi") or the
# space ("haiphong"). After normalize_rule_text everything is accent-folded ascii,
# so we accept h + optional a + optional i + optional space + phong.
_HAI_PHONG = r"h[a]?i?\s*phong"
PATE_RULES = [
    {
        "name": "pate_cot_den_hai_phong",
        "priority": 100,
        "canonical": "Pate Cột Đèn Hải Phòng",
        "patterns": [
            r"\bpat[eê]\b.*\bcot\b.*\bden\b.*\b" + _HAI_PHONG + r"\b",
        ],
    },
    {
        "name": "halong_canfoco_pate_cot_den",
        "priority": 99,
        "canonical": "Halong Canfoco Pate Cột Đèn Hải Phòng",
        "patterns": [
            r"\bhalong\b.*\bcanfoco\b.*\bpate\b.*\bcot\b.*\bden\b",
            r"\bha long\b.*\bcanfoco\b.*\bpate\b.*\bcot\b.*\bden\b",
        ],
    },
    {
        "name": "pate_cot_den",
        "priority": 98,
        "canonical": "Pate Cột Đèn",
        "patterns": [
            r"\bpat[eê]\b.*\bcot\b.*\bden\b",
            r"\bpate\b.*\bcot\b.*\bden\b",
        ],
    },
]

# ---- 6. Highlands / The Coffee House conflict + drinks ---------------
COFFEE_HALONG_CONFLICT_RULES = [
    {
        "name": "highlands_partner_of_halong",
        "priority": 99,
        "canonical": "Đồ hộp Hạ Long",
        "patterns": [
            r"\bhighlands coffee\b.*\bdoi tac\b.*\bdo hop ha long\b",
            r"\bhighlands coffee\b.*\bcong ty co phan\b.*\bdo hop ha long\b",
            r"\bhighlands\b.*\bdoi tac\b.*\bha long\b",
        ],
    }
]

# canonical corrected to the real train surface (comma/lowercase) per decision.
HIGHLANDS_DRINK_RULES = [
    {
        "name": "highlands_tra_sen_vang_tra_vai",
        "priority": 100,
        "canonical": "Highlands Coffee trà sen vàng, trà vải",
        "patterns": [
            r"\bhighlands\b.*\btra sen vang\b.*\btra vai\b",
            r"\btra sen vang\b.*\btra vai\b.*\bhighlands\b",
        ],
    },
    {
        "name": "highlands_tra_sen_vang_banh_mi_que",
        "priority": 100,
        "canonical": "Highlands Coffee Trà Sen Vàng + Bánh Mì Que",
        "patterns": [
            r"\bhighlands\b.*\btra sen vang\b.*\bbanh mi que\b",
        ],
    },
]

THE_COFFEE_HOUSE_RULES = [
    {
        "name": "tch_tra_phuc_kien_sen_vai_americano_vai",
        "priority": 100,
        "canonical": "The Coffee House Trà Phúc Kiến Sen Vải + Americano Vải",
        "patterns": [
            r"\bthe coffee house\b.*\btra phuc kien sen vai\b.*\bamericano vai\b",
            r"\btra phuc kien sen vai\b.*\bamericano vai\b",
        ],
    }
]

# ---- 7. Nestlé / NAN / milk (model-specific beats generic brand) -----
NESTLE_NAN_RULES = [
    {
        "name": "nan_infinipro_a2",
        "priority": 100,
        "canonical": "Nestlé NAN INFINIPRO A2",
        "patterns": [
            r"\bnan\b.*\binfinipro\b.*\ba2\b",
            r"\bnestle\b.*\bnan\b.*\binfinipro\b.*\ba2\b",
        ],
    },
    {
        "name": "nan_optipro_plus",
        "priority": 99,
        "canonical": "Nestlé NAN OPTIPRO PLUS",
        "patterns": [
            r"\bnan\b.*\boptipro\b.*\bplus\b",
            r"\bnestle\b.*\bnan\b.*\boptipro\b.*\bplus\b",
        ],
    },
    {
        "name": "nestle_beba",
        "priority": 96,
        "canonical": "Nestlé BEBA",
        "patterns": [
            r"\bnestle\b.*\bbeba\b",
            r"\bbeba\b",
        ],
    },
    {
        "name": "sua_nan_generic",
        "priority": 75,
        "canonical": "sữa NAN",
        "patterns": [
            r"\bsua\b.*\bnan\b",
            r"\bnan\b.*\bsua\b",
        ],
    },
    {
        "name": "sua_nestle_generic",
        "priority": 70,
        "canonical": "sữa Nestle",
        "patterns": [
            r"\bsua\b.*\bnestle\b",
            r"\bnestle\b.*\bsua\b",
        ],
    },
]

# ---- 8. Misc exact rules for smaller product groups -----------------
MISC_EXACT_RULES = [
    {
        "name": "chinsu_tuong_ot_cay_bung_vi_tom",
        "priority": 100,
        "canonical": "CHIN-SU tương ớt cay bùng vị tôm",
        "patterns": [r"\bchin su\b.*\btuong ot\b.*\bcay bung\b.*\bvi tom\b"],
    },
    {
        "name": "tuong_ot_chinsu",
        "priority": 90,
        "canonical": "tương ớt Chinsu",
        "patterns": [
            r"\bchin su\b.*\btuong ot\b",
            r"\btuong ot\b.*\bchin su\b",
        ],
    },
    {
        "name": "pate_gan_vissan_3_bong_mai",
        "priority": 98,
        "canonical": "PATE GAN VISSAN 3 BÔNG MAI",
        "patterns": [r"\bpate gan\b.*\bvissan\b.*\b3 bong mai\b"],
    },
    {
        "name": "pate_gan_vissan",
        "priority": 95,
        "canonical": "PATE GAN VISSAN",
        "patterns": [
            r"\bpate gan\b.*\bvissan\b",
            r"\bvissan\b.*\bpate gan\b",
        ],
    },
    {
        "name": "acnes_vitamin_cleanser",
        "priority": 96,
        "canonical": "Acnes Vitamin Cleanser",
        "patterns": [r"\bacnes\b.*\bvitamin\b.*\bcleanser\b"],
    },
    {
        "name": "aptamil_profutura",
        "priority": 96,
        "canonical": "Aptamil Profutura",
        "patterns": [r"\baptamil\b.*\bprofutura\b"],
    },
    {
        "name": "hipp_combiotic",
        "priority": 96,
        "canonical": "HiPP COMBIOTIC",
        "patterns": [r"\bhipp\b.*\bcombiotic\b"],
    },
    {
        "name": "similac_total_protection",
        "priority": 96,
        "canonical": "Similac Total Protection",
        "patterns": [r"\bsimilac\b.*\btotal\b.*\bprotection\b"],
    },
    {
        "name": "nutifood_grow_plus",
        "priority": 96,
        "canonical": "Nutifood Grow PLUS+",
        "patterns": [r"\bnutifood\b.*\bgrow\b.*\bplus\b"],
    },
]

# ---- 9. Generic / topic-only (usually blank unless strong rule) ------
GENERIC_TOPIC_PATTERNS = [
    r"\bhe luy\b",
    r"\bde che\b",
    r"\b130 tan thit\b",
    r"\bthit benh\b",
    r"\bdich ta lon chau phi\b",
    r"\bsua bi thu hoi\b",
    r"\bdung nhap san pham nao\b",
    r"\bchinh thuc len tieng\b",
]


# Strength below which a generic-topic OCR is forced blank.
STRONG_RULE_PRIORITY = 95


# =====================================================================
# 10. Rule application helpers
# =====================================================================

def _make_hit(canonical: str, priority: int, source: str, rule_name: str,
              reason: str, candidate: str = "", score: float = 1.0) -> Dict:
    return {
        "candidate": candidate or canonical,
        "canonical": canonical,
        "score": float(score),
        "priority": int(priority),
        "source": source,
        "rule_name": rule_name,
        "reason": reason,
    }


def apply_rule_group(full_text: str, rules: List[dict], source: str = "exact_rule") -> List[Dict]:
    """Run one rule group against normalized text; one hit per matching rule."""
    norm = normalize_rule_text(full_text)
    hits: List[Dict] = []
    for rule in rules:
        for pat in rule["patterns"]:
            if re.search(pat, norm):
                hits.append(_make_hit(
                    canonical=rule["canonical"],
                    priority=rule["priority"],
                    source=source,
                    rule_name=rule["name"],
                    reason="matched_pattern:%s" % pat,
                ))
                break  # one hit per rule is enough
    return hits


def is_generic_topic_only(full_text: str) -> bool:
    """True if OCR looks like a topic/news headline with no explicit product."""
    norm = normalize_rule_text(full_text)
    return any(re.search(p, norm) for p in GENERIC_TOPIC_PATTERNS)


def _cp_context_hits(full_text: str) -> List[Dict]:
    """Emit a standalone-CP product hit only in a true CP product context."""
    if is_cp_product_context(full_text):
        return [_make_hit(
            canonical="CP",
            priority=85,
            source="context_rule",
            rule_name="cp_product_context",
            reason="cp_positive_context_no_danger",
        )]
    return []


def apply_exact_rules(full_text: str) -> List[Dict]:
    """Pate, Highlands drinks, The Coffee House, Nestlé/NAN, misc products."""
    hits: List[Dict] = []
    hits += apply_rule_group(full_text, PATE_RULES, source="exact_rule")
    hits += apply_rule_group(full_text, HIGHLANDS_DRINK_RULES, source="exact_rule")
    hits += apply_rule_group(full_text, THE_COFFEE_HOUSE_RULES, source="exact_rule")
    hits += apply_rule_group(full_text, NESTLE_NAN_RULES, source="exact_rule")
    hits += apply_rule_group(full_text, MISC_EXACT_RULES, source="exact_rule")
    return hits


def apply_context_rules(full_text: str) -> List[Dict]:
    """Hạ Long legal-company + Hạ Long news/noisy-OCR context, plus CP context."""
    hits: List[Dict] = []
    hits += apply_rule_group(full_text, HALONG_LEGAL_RULES, source="context_rule")
    hits += apply_rule_group(full_text, HALONG_CONTEXT_RULES, source="context_rule")
    hits += _cp_context_hits(full_text)
    return hits


def apply_conflict_rules(full_text: str, matches: Optional[List[dict]] = None) -> List[Dict]:
    """Forced conflict decisions (Highlands partner-of-Hạ Long, etc.)."""
    return apply_rule_group(full_text, COFFEE_HALONG_CONFLICT_RULES, source="conflict_rule")


def _sort_key(hit: Dict):
    return (
        int(hit.get("priority", 0)),
        float(hit.get("score", 0.0)),
        len(hit.get("canonical", "")),
    )


def apply_all_rules(full_text: str) -> List[Dict]:
    """
    Combine conflict + exact + context rules, dedupe by (canonical, rule_name),
    sort by priority desc, score desc, canonical length desc.
    """
    hits: List[Dict] = []
    hits += apply_conflict_rules(full_text)
    hits += apply_exact_rules(full_text)
    hits += apply_context_rules(full_text)

    seen = set()
    deduped: List[Dict] = []
    for h in hits:
        key = (h["canonical"], h["rule_name"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(h)

    deduped.sort(key=_sort_key, reverse=True)
    return deduped


# =====================================================================
# Final resolver (used by D / E and the self-test)
# =====================================================================

def resolve_product_name(full_text: str, extra_candidates: Optional[List[str]] = None,
                         verbose: bool = False):
    """
    Resolve the final product_name from OCR text using the rule registry.

    Returns (product_name, winning_hit_or_None). product_name is "" for blank.
    `extra_candidates` lets the real pipeline inject candidate strings (e.g. a
    gazetteer-linked display) so they can compete with rule hits.
    """
    hits = list(apply_all_rules(full_text))

    for cand in (extra_candidates or []):
        cand = str(cand).strip()
        if not cand:
            continue
        hits.append(_make_hit(
            canonical=cand,
            priority=80,
            source="pipeline_candidate",
            rule_name="pipeline_candidate",
            reason="injected_from_pipeline",
            score=0.9,
        ))

    # Drop contextually-rejected short/legal candidates.
    valid = [h for h in hits if not should_reject_short_candidate(h["canonical"], full_text)]

    if verbose:
        for h in hits:
            log.info("[Rule] hit=%s canonical=%s priority=%s%s",
                     h["rule_name"], h["canonical"], h["priority"],
                     "" if h in valid else " (rejected_short)")

    if not valid:
        if verbose:
            log.info("[Gate] no valid candidate -> blank")
        return "", None

    valid.sort(key=_sort_key, reverse=True)
    best = valid[0]

    if is_generic_topic_only(full_text) and best["priority"] < STRONG_RULE_PRIORITY:
        if verbose:
            log.info("[Gate] generic topic only, best priority %s < %s -> blank",
                     best["priority"], STRONG_RULE_PRIORITY)
        return "", None

    if verbose:
        log.info("[Gate] chosen=%s source=%s priority=%s",
                 best["canonical"], best["source"], best["priority"])
    return best["canonical"], best


# =====================================================================
# Alias mapping for gazetteer / fuzzy linker (B & C)
# =====================================================================

# Hand-curated normalized-alias -> canonical. Keys MUST be normalize_rule_text form.
_ALIAS_RULES: Dict[str, str] = {
    # Hạ Long legal variants -> commercial brand
    "cong ty co phan do hop ha long": "Đồ hộp Hạ Long",
    "cong ty cp do hop ha long": "Đồ hộp Hạ Long",
    "ctcp do hop ha long": "Đồ hộp Hạ Long",
    "cong ty do hop ha long": "Đồ hộp Hạ Long",
    "nha may do hop ha long": "Đồ hộp Hạ Long",
    "do hop ha long": "Đồ hộp Hạ Long",
    # Pate Cột Đèn family
    "pate cot den hai phong": "Pate Cột Đèn Hải Phòng",
    "pate cot den": "Pate Cột Đèn",
    "halong canfoco pate cot den hai phong": "Halong Canfoco Pate Cột Đèn Hải Phòng",
    # Nestlé / NAN models
    "nan optipro plus": "Nestlé NAN OPTIPRO PLUS",
    "nan infinipro a2": "Nestlé NAN INFINIPRO A2",
}


def get_rule_aliases() -> Dict[str, str]:
    """
    Return {normalized_alias: canonical} for B_gazetteer / C_fuzzy_linker.

    Also includes every rule canonical mapped from its own normalized form, so
    a candidate equal to a canonical resolves instantly.
    """
    aliases: Dict[str, str] = dict(_ALIAS_RULES)
    for group in (PATE_RULES, HIGHLANDS_DRINK_RULES, THE_COFFEE_HOUSE_RULES,
                  NESTLE_NAN_RULES, MISC_EXACT_RULES, HALONG_LEGAL_RULES,
                  HALONG_CONTEXT_RULES, COFFEE_HALONG_CONFLICT_RULES):
        for rule in group:
            canon = rule["canonical"]
            aliases.setdefault(normalize_rule_text(canon), canon)
    return aliases
