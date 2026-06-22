from src.ocr_cpu.preprocess.router import (
    classify_image_quality,
    apply_preprocess_by_decision,
)
from src.ocr_cpu.ocr.engine import create_ocr_engine
from src.ocr_cpu.ocr.quality import run_ocr_with_quality_gate

img_path = "data/raw/test_images/test_images/images/img_2934.jpg"

decision = classify_image_quality(
    img_path,
    target_short_side=900,
    max_long_side=1600,
    engine="tesseract",
)

variants = apply_preprocess_by_decision(
    img_path,
    decision=decision,
    engine="tesseract",
)

engine = create_ocr_engine(
    backend="tesseract",
    lang="eng",
    psm=11,
    oem=3,
    timeout=10,
    min_token_conf=0.0,
)

selection = run_ocr_with_quality_gate(
    variants=variants,
    decision=decision,
    engine=engine,
)

print("SELECTED VARIANT:", selection.selected_variant)
print("SELECTION REASON:", selection.selection_reason)
print("TEXT:")
print(selection.selected_result.text)

print("\nREPORTS:")
for variant, report in selection.reports.items():
    print("=" * 80)
    print("variant:", variant)
    print("is_bad:", report.is_bad)
    print("score:", report.score)
    print("reasons:", report.reasons)
    print("avg_conf:", report.avg_conf)
    print("n_boxes:", report.n_boxes)