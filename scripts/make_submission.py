from __future__ import annotations

"""
End-to-end submission builder: OCR the whole test set, then run the product
stage (Cell 1 -> Cell 5) and write submission.csv.

Pipeline
    images -> OCR (router + engine + quality gate)  -> ocr_df (cached)
           -> Cell 1  candidate generation           (generate_candidate_dataframe)
           -> Cell 2  gazetteer                        (build/load + rule aliases)
           -> Cell 3  fuzzy link                       (link_candidates_with_gazetteer)
           -> Cell 4  confidence gate + rules          (apply_confidence_gating)
           -> Cell 5  final predictor + submission     (build_final_submission)

The OCR stage is cached to outputs/ so the (slow) OCR runs once and the product
stage can be re-iterated quickly. Re-OCR with --rebuild-ocr.

Examples
    python scripts/make_submission.py                       # public test -> outputs/submission.csv
    python scripts/make_submission.py --split phase2         # private    -> outputs/submission_private.csv
    python scripts/make_submission.py --limit 50             # quick smoke on first 50 images
    python scripts/make_submission.py --rebuild-ocr          # ignore OCR cache
    python scripts/make_submission.py --no-rules             # ablation: disable rules.py integration
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_cpu.preprocess.router import classify_image_quality, apply_preprocess_by_decision  # noqa: E402
from ocr_cpu.ocr.engine import create_ocr_engine  # noqa: E402
from ocr_cpu.ocr.quality import OCRQualityConfig, run_ocr_with_quality_gate  # noqa: E402
from ocr_cpu.product.A_candidate import (  # noqa: E402
    selections_to_ocr_dataframe,
    generate_candidate_dataframe,
)
from ocr_cpu.product.B_gazeetteer import build_product_gazetteer, ProductGazetteer  # noqa: E402
from ocr_cpu.product.C_fuzzy_linker import link_candidates_with_gazetteer  # noqa: E402
from ocr_cpu.product.D_confidence_gate import apply_confidence_gating, GateConfig  # noqa: E402
from ocr_cpu.product.E_predict import (  # noqa: E402
    build_final_submission,
    validate_final_submission,
    clean_ocr_text_for_output,
    FinalPredictorConfig,
)

DATA = ROOT / "the-2nd-ura-hackathon"
TRAIN_LABELS = DATA / "train_labels.csv"
GAZETTEER_JSON = ROOT / "outputs" / "gazetteer" / "product_gazetteer.json"

# Per-split layout: test csv, sample submission (for row order + schema), image dir.
SPLITS = {
    "test": {
        "test_csv": DATA / "test.csv",
        "sample": DATA / "sample_submission.csv",
        "image_dir": DATA / "test_images",
        "output": ROOT / "outputs" / "submission.csv",
        "ocr_cache": ROOT / "outputs" / "ocr_cache_test.csv",
        "brand_name": False,
    },
    "phase2": {
        "test_csv": DATA / "phase2_dataset" / "phase2_dataset" / "private_test.csv",
        "sample": DATA / "phase2_dataset" / "phase2_dataset" / "sample_submission_private.csv",
        "image_dir": DATA / "phase2_dataset" / "phase2_dataset" / "images",
        "output": ROOT / "outputs" / "submission_private.csv",
        "ocr_cache": ROOT / "outputs" / "ocr_cache_phase2.csv",
        "brand_name": True,
    },
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _progress(iterable, total: int, desc: str):
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def load_image_ids(test_csv: Path, sample: Path, limit: Optional[int]) -> List[str]:
    src = test_csv if test_csv.exists() else sample
    if not src.exists():
        raise FileNotFoundError(f"Cannot find test list: {test_csv} or {sample}")
    df = pd.read_csv(src)
    if "image_id" not in df.columns:
        raise ValueError(f"{src} has no image_id column (found {list(df.columns)})")
    ids = df["image_id"].astype(str).tolist()
    if limit:
        ids = ids[:limit]
    return ids


def index_images(image_dir: Path) -> Dict[str, Path]:
    """Map image stem -> path for every image under image_dir (recursive)."""
    idx: Dict[str, Path] = {}
    if not image_dir.exists():
        return idx
    for p in image_dir.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS and p.stem not in idx:
            idx[p.stem] = p
    return idx


def get_gazetteer(rebuild: bool) -> ProductGazetteer:
    if not rebuild and GAZETTEER_JSON.exists():
        print(f"[Cell 2] load cache: {GAZETTEER_JSON}")
        gz = ProductGazetteer.from_json(GAZETTEER_JSON)
        # Cached JSON predates rule aliases -> enrich in memory (additive, safe).
        added = gz.enrich_with_rule_aliases()
        if added:
            print(f"[Cell 2] enriched gazetteer with {added} rule-alias surfaces")
        return gz
    GAZETTEER_JSON.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Cell 2] build from {TRAIN_LABELS} (~1-2 min)...")
    return build_product_gazetteer(TRAIN_LABELS, cache_json_path=GAZETTEER_JSON)


def run_ocr_stage(
    image_ids: List[str],
    image_dir: Path,
    cache_path: Path,
    backend: str,
    lang: str,
    rebuild: bool,
) -> pd.DataFrame:
    """Return ocr_df (image_id + ocr_text + variant columns). Cached to disk."""
    if not rebuild and cache_path.exists():
        print(f"[OCR] load cache: {cache_path}")
        ocr_df = pd.read_csv(cache_path)
        have = set(ocr_df["image_id"].astype(str)) if "image_id" in ocr_df.columns else set()
        missing = [i for i in image_ids if i not in have]
        if not missing:
            return ocr_df
        print(f"[OCR] cache missing {len(missing)} images -> OCR only those")
        new_df = _ocr_images(missing, image_dir, backend, lang)
        ocr_df = pd.concat([ocr_df, new_df], ignore_index=True)
        ocr_df.to_csv(cache_path, index=False)
        return ocr_df

    ocr_df = _ocr_images(image_ids, image_dir, backend, lang)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ocr_df.to_csv(cache_path, index=False)
    print(f"[OCR] cached -> {cache_path}")
    return ocr_df


def _ocr_images(image_ids: List[str], image_dir: Path, backend: str, lang: str) -> pd.DataFrame:
    img_index = index_images(image_dir)
    engine = create_ocr_engine(backend=backend, lang=lang)
    quality_cfg = OCRQualityConfig()

    selections: List[Tuple[str, object]] = []
    n_fail = 0
    for image_id in _progress(image_ids, total=len(image_ids), desc="OCR"):
        path = img_index.get(image_id)
        if path is None:
            n_fail += 1
            continue
        try:
            decision = classify_image_quality(path)
            variants = apply_preprocess_by_decision(path, decision=decision)
            selection = run_ocr_with_quality_gate(variants, decision, engine, quality_cfg)
            selections.append((image_id, selection))
        except Exception as exc:  # one bad image must not kill the whole run
            n_fail += 1
            print(f"[OCR] FAIL {image_id}: {exc}")

    if n_fail:
        print(f"[OCR] {n_fail} images failed / not found (will be blank in submission)")
    if not selections:
        return pd.DataFrame(columns=["image_id", "ocr_text"])
    return selections_to_ocr_dataframe(selections)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Build competition submission (OCR -> Cell 1-5).")
    ap.add_argument("--split", default="test", choices=list(SPLITS.keys()))
    ap.add_argument("--backend", default="paddle", choices=["paddle", "tesseract"])
    ap.add_argument("--lang", default="vi")
    ap.add_argument("--limit", type=int, default=None, help="only process first N images (smoke test)")
    ap.add_argument("--output", default=None, help="override output csv path")
    ap.add_argument("--rebuild-gazetteer", action="store_true")
    ap.add_argument("--rebuild-ocr", action="store_true", help="ignore OCR cache and re-OCR")
    ap.add_argument("--no-rules", action="store_true", help="ablation: disable rules.py integration")
    ap.add_argument("--no-clean-ocr", dest="clean_ocr", action="store_false", default=True,
                    help="disable ocr_text junk cleanup (CJK/symbol strip + blank junk-only rows)")
    args = ap.parse_args()

    cfg = SPLITS[args.split]
    out_path = Path(args.output) if args.output else cfg["output"]
    t0 = time.time()

    image_ids = load_image_ids(cfg["test_csv"], cfg["sample"], args.limit)
    print(f"[Submission] split={args.split} images={len(image_ids)} output={out_path}")

    # Cell 2 first: the gazetteer also supplies filler tokens for Cell 1.
    gz = get_gazetteer(rebuild=args.rebuild_gazetteer)

    # OCR (cached).
    ocr_df = run_ocr_stage(
        image_ids, cfg["image_dir"], cfg["ocr_cache"],
        backend=args.backend, lang=args.lang, rebuild=args.rebuild_ocr,
    )
    # Keep only the requested ids (cache may hold more); test.csv order is restored
    # later by build_final_submission via sample_submission_path.
    ocr_df = ocr_df[ocr_df["image_id"].astype(str).isin(set(image_ids))].copy()
    image_text_map = {
        str(r["image_id"]): str(r.get("ocr_text", "") or "")
        for _, r in ocr_df.iterrows()
    }

    # Cell 1: candidates (with gazetteer filler tokens).
    cand_df = generate_candidate_dataframe(ocr_df, filler_tokens=gz.filler_tokens)
    n_imgs = ocr_df["image_id"].nunique() if not ocr_df.empty else 0
    print(f"[Cell 1] candidates: {len(cand_df)} from {n_imgs} images")

    # Cell 3: fuzzy link (rules drop noisy short candidates via image_text_map).
    linked = link_candidates_with_gazetteer(
        cand_df, gz, show_progress=True, image_text_map=image_text_map,
    )
    n_matched = int(linked["matched"].fillna(False).astype(bool).sum()) if "matched" in linked.columns else 0
    print(f"[Cell 3] matched candidates: {n_matched}")

    # Cell 4: confidence gate + rule override / rescue (image_text_map covers all ids).
    gate_cfg = GateConfig(use_rules=not args.no_rules)
    decision_df = apply_confidence_gating(
        linked, gate_cfg, show_progress=True, image_text_map=image_text_map,
    )
    n_emit = int((decision_df["emit_product"] == True).sum()) if "emit_product" in decision_df.columns else 0  # noqa: E712
    print(f"[Cell 4] emit={n_emit} / {len(decision_df)} decisions")

    # Cell 5: final submission (sample order + phase schema).
    final_cfg = FinalPredictorConfig(include_brand_name=cfg["brand_name"])
    submission = build_final_submission(
        ocr_df, decision_df, gazetteer=gz, config=final_cfg,
        sample_submission_path=cfg["sample"],
    )

    # ocr_text output hygiene. Train ground-truth ocr_text is single-line
    # (space-joined), so flatten newlines either way. With --clean-ocr (default)
    # also strip hallucinated junk scripts (CJK/Kana/symbols) and blank rows with
    # no real word. Junk cleanup is opt-out because a few train GT rows legitimately
    # contain CJK -> measure CER on train (Cell 6) before fully trusting it.
    if "ocr_text" in submission.columns:
        if args.clean_ocr:
            n_before = int(submission["ocr_text"].fillna("").astype(str).str.strip().eq("").sum())
            submission["ocr_text"] = submission["ocr_text"].apply(clean_ocr_text_for_output)
            n_after = int(submission["ocr_text"].fillna("").astype(str).str.strip().eq("").sum())
            print(f"[clean] ocr_text junk cleanup: blanked {n_after - n_before} junk-only rows")
        else:
            submission["ocr_text"] = (
                submission["ocr_text"].fillna("").astype(str)
                .str.replace(r"\s+", " ", regex=True).str.strip()
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False, encoding="utf-8")

    # Validate structure. Expected row count is the official sample submission
    # size (build_final_submission emits every sample row); --limit only changes
    # how many get a non-blank prediction, not the row count.
    required = ["image_id", "ocr_text", "product_name"]
    if cfg["brand_name"]:
        required.insert(2, "brand_name")
    report = validate_final_submission(submission, required_cols=required)
    expected_rows = report["n_rows"]
    if cfg["sample"].exists():
        expected_rows = len(pd.read_csv(cfg["sample"]))
    elif cfg["test_csv"].exists():
        expected_rows = len(pd.read_csv(cfg["test_csv"]))
    print("\n" + "=" * 60)
    print(f"WROTE {out_path}")
    print(f"  rows               : {report['n_rows']}")
    print(f"  missing columns    : {report['missing_columns']}")
    print(f"  duplicate image_id : {report['duplicate_image_id_count']}")
    print(f"  blank product_name : {report['blank_product_count']}")
    print(f"  product fill rate  : {report['product_fill_rate']}")
    print(f"  elapsed            : {time.time() - t0:.1f}s")
    print("=" * 60)

    ok = (not report["missing_columns"]) and report["duplicate_image_id_count"] == 0 \
        and report["n_rows"] == expected_rows
    if not ok:
        print(f"[WARN] structural check failed: rows={report['n_rows']} expected={expected_rows}, "
              f"dups={report['duplicate_image_id_count']}, missing={report['missing_columns']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
