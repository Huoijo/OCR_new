from __future__ import annotations

"""
Full Cell 1->5 demo on ONE real image, ending with product_name.

Chain:
    image -> OCR (router+engine+quality)  -> OCRSelection
          -> Cell 1 candidates
          -> Cell 2 gazetteer (cache)
          -> Cell 3 fuzzy link
          -> Cell 4 confidence gate (decision)
          -> Cell 5 final predictor (image_id, ocr_text, product_name)

Usage:
    python scripts/demo_predict_one_image.py --image img_0011.jpg
    python scripts/demo_predict_one_image.py --image img_0006.jpg --rebuild-gazetteer
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_cpu.preprocess.router import classify_image_quality, apply_preprocess_by_decision  # noqa: E402
from ocr_cpu.ocr.engine import create_ocr_engine  # noqa: E402
from ocr_cpu.ocr.quality import OCRQualityConfig, run_ocr_with_quality_gate  # noqa: E402
from ocr_cpu.ocr.A_candidate_sp import selection_to_product_input_fields  # noqa: E402
from ocr_cpu.product.A_candidate import selections_to_candidate_dataframe  # noqa: E402
from ocr_cpu.product.B_gazeetteer import build_product_gazetteer, ProductGazetteer  # noqa: E402
from ocr_cpu.product.C_fuzzy_linker import link_candidates_with_gazetteer  # noqa: E402
from ocr_cpu.product.D_confidence_gate import apply_confidence_gating, GateConfig  # noqa: E402
from ocr_cpu.product.E_predict import build_final_submission, FinalPredictorConfig  # noqa: E402

TRAIN_LABELS = ROOT / "the-2nd-ura-hackathon" / "train_labels.csv"
GAZETTEER_JSON = ROOT / "outputs" / "gazetteer" / "product_gazetteer.json"


def resolve_image(image_arg: str, split: str) -> Path:
    p = Path(image_arg)
    if p.exists():
        return p.resolve()
    search = ROOT / "the-2nd-ura-hackathon" / f"{split}_images"
    stem = Path(image_arg).stem
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        hits = list(search.rglob(f"{stem}{ext}"))
        if hits:
            return sorted(hits)[0].resolve()
    raise FileNotFoundError(f"Cannot find image: {image_arg}")


def get_gazetteer(rebuild: bool) -> ProductGazetteer:
    if not rebuild and GAZETTEER_JSON.exists():
        print(f"[Cell 2] load cache: {GAZETTEER_JSON}")
        return ProductGazetteer.from_json(GAZETTEER_JSON)
    print(f"[Cell 2] build từ {TRAIN_LABELS} (~1-2 phút)...")
    return build_product_gazetteer(TRAIN_LABELS, cache_json_path=GAZETTEER_JSON)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="image_id hoặc path, vd img_0011.jpg")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--backend", default="paddle", choices=["paddle", "tesseract"])
    ap.add_argument("--lang", default="vi")
    ap.add_argument("--rebuild-gazetteer", action="store_true")
    args = ap.parse_args()

    image_path = resolve_image(args.image, args.split)
    image_id = image_path.stem
    print(f"Image: {image_path}\n")

    # OCR
    decision = classify_image_quality(image_path)
    variants = apply_preprocess_by_decision(image_path, decision=decision)
    engine = create_ocr_engine(backend=args.backend, lang=args.lang)
    selection = run_ocr_with_quality_gate(variants, decision, engine, OCRQualityConfig())
    print(f"[OCR] variant={selection.selected_variant}")
    print("[OCR] text:")
    print(selection.selected_result.text or "(empty)")

    # Cell 1
    cand_df = selections_to_candidate_dataframe([(image_id, selection)])
    print(f"\n[Cell 1] candidates: {len(cand_df)}")

    # Cell 2
    gz = get_gazetteer(rebuild=args.rebuild_gazetteer)

    # Cell 3
    linked = link_candidates_with_gazetteer(cand_df, gz)
    n_matched = int(linked["matched"].fillna(False).astype(bool).sum()) if "matched" in linked.columns else 0
    print(f"[Cell 3] matched candidates: {n_matched}")

    # Cell 4
    decision_df = apply_confidence_gating(linked, GateConfig())
    d = decision_df.iloc[0]
    print(f"[Cell 4] decision={d['gate_decision']}  compose_mode={d['compose_mode']}  "
          f"gate_score={d['gate_score']}  chosen={list(d['chosen_displays'])}")
    print(f"[Cell 4] reason: {d['gate_reason']}")

    # Cell 5
    ocr_df = pd.DataFrame([{
        "image_id": image_id,
        "ocr_text": selection_to_product_input_fields(selection, image_id=image_id)["ocr_text"],
    }])
    submission = build_final_submission(ocr_df, decision_df, gazetteer=gz, config=FinalPredictorConfig())

    print("\n" + "=" * 60)
    print("FINAL (Cell 5):")
    row = submission.iloc[0]
    print(f"  image_id     : {row['image_id']}")
    print(f"  product_name : {row['product_name']!r}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
