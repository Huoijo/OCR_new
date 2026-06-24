from __future__ import annotations

"""
Time ONE image end-to-end, splitting the two phases clearly:

  [OCR]      router + engine + quality gate              -> OCRSelection
  [PRODUCT]  Cell 1 -> Cell 3 -> Cell 4 -> Cell 5         -> product_name

One-time setup (engine init, gazetteer load) is timed separately so it is not
counted against per-image latency.

Usage
    python scripts/time_one_image.py --image img_0019.jpg
    python scripts/time_one_image.py --image img_0019.jpg --split train --repeat 3
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

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
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def resolve_image(image_arg: str, split: str) -> Path:
    p = Path(image_arg)
    if p.exists():
        return p.resolve()
    search = ROOT / "the-2nd-ura-hackathon" / f"{split}_images"
    stem = Path(image_arg).stem
    for ext in IMAGE_EXTS:
        hits = list(search.rglob(f"{stem}{ext}"))
        if hits:
            return sorted(hits)[0].resolve()
    raise FileNotFoundError(f"Cannot find image: {image_arg} under {search}")


def get_gazetteer(rebuild: bool) -> ProductGazetteer:
    if not rebuild and GAZETTEER_JSON.exists():
        gz = ProductGazetteer.from_json(GAZETTEER_JSON)
        gz.enrich_with_rule_aliases()
        return gz
    return build_product_gazetteer(TRAIN_LABELS, cache_json_path=GAZETTEER_JSON)


def run_ocr(image_path: Path, engine, quality_cfg):
    """Return (selection, ocr_seconds)."""
    t0 = perf_counter()
    decision = classify_image_quality(image_path)
    variants = apply_preprocess_by_decision(image_path, decision=decision)
    selection = run_ocr_with_quality_gate(variants, decision, engine, quality_cfg)
    return selection, perf_counter() - t0


def run_product(image_id: str, selection, gz) -> tuple[str, dict]:
    """Cell 1 -> Cell 5 for one image. Return (product_name, per-cell seconds)."""
    times: dict = {}

    t = perf_counter()
    cand_df = selections_to_candidate_dataframe([(image_id, selection)])
    times["cell1_candidates"] = perf_counter() - t

    ocr_text = selection_to_product_input_fields(selection, image_id=image_id)["ocr_text"]
    image_text_map = {image_id: ocr_text}

    t = perf_counter()
    linked = link_candidates_with_gazetteer(cand_df, gz, image_text_map=image_text_map)
    times["cell3_link"] = perf_counter() - t

    t = perf_counter()
    decision_df = apply_confidence_gating(linked, GateConfig(), image_text_map=image_text_map)
    times["cell4_gate"] = perf_counter() - t

    t = perf_counter()
    ocr_df = pd.DataFrame([{"image_id": image_id, "ocr_text": ocr_text}])
    submission = build_final_submission(ocr_df, decision_df, gazetteer=gz, config=FinalPredictorConfig())
    times["cell5_final"] = perf_counter() - t

    product_name = str(submission.iloc[0]["product_name"])
    return product_name, times


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="image_id or path, e.g. img_0019.jpg")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--backend", default="paddle", choices=["paddle", "tesseract"])
    ap.add_argument("--lang", default="vi")
    ap.add_argument("--repeat", type=int, default=1, help="repeat OCR+product N times (warm timing)")
    ap.add_argument("--rebuild-gazetteer", action="store_true")
    args = ap.parse_args()

    image_path = resolve_image(args.image, args.split)
    image_id = image_path.stem

    # ---- one-time setup (NOT per-image latency) ----
    t = perf_counter()
    engine = create_ocr_engine(backend=args.backend, lang=args.lang)
    t_engine = perf_counter() - t

    t = perf_counter()
    gz = get_gazetteer(rebuild=args.rebuild_gazetteer)
    t_gz = perf_counter() - t

    quality_cfg = OCRQualityConfig()

    print("=" * 64)
    print(f"Image: {image_path.name}  (id={image_id})")
    print(f"[setup] engine init     : {t_engine:8.3f}s  (one-time)")
    print(f"[setup] gazetteer load  : {t_gz:8.3f}s  (one-time)")
    print("=" * 64)

    ocr_times, prod_times, total_times = [], [], []
    product_name = ""
    for i in range(max(1, args.repeat)):
        selection, t_ocr = run_ocr(image_path, engine, quality_cfg)
        product_name, cell_times = run_product(image_id, selection, gz)
        t_prod = sum(cell_times.values())
        ocr_times.append(t_ocr)
        prod_times.append(t_prod)
        total_times.append(t_ocr + t_prod)

        print(f"\n--- run {i + 1}/{args.repeat} ---")
        print(f"[OCR]      router+engine+quality : {t_ocr:8.3f}s")
        print(f"[PRODUCT]  Cell 1 candidates     : {cell_times['cell1_candidates']:8.3f}s")
        print(f"[PRODUCT]  Cell 3 fuzzy link     : {cell_times['cell3_link']:8.3f}s")
        print(f"[PRODUCT]  Cell 4 gate (+rules)  : {cell_times['cell4_gate']:8.3f}s")
        print(f"[PRODUCT]  Cell 5 final          : {cell_times['cell5_final']:8.3f}s")
        print(f"[PRODUCT]  subtotal              : {t_prod:8.3f}s")
        print(f"[TOTAL]    OCR + PRODUCT          : {t_ocr + t_prod:8.3f}s")

    def avg(xs):
        return sum(xs) / len(xs)

    print("\n" + "=" * 64)
    print(f"product_name : {product_name!r}")
    print(f"avg OCR      : {avg(ocr_times):8.3f}s")
    print(f"avg PRODUCT  : {avg(prod_times):8.3f}s   (OCRSelection -> product_name)")
    print(f"avg TOTAL    : {avg(total_times):8.3f}s   (excl. one-time setup)")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
