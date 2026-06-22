from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import cv2 as cv


# ---------------------------------------------------------------------
# Project import setup
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_cpu.preprocess.router import (  # noqa: E402
    classify_image_quality,
    apply_preprocess_by_decision,
)
from ocr_cpu.ocr.engine import (  # noqa: E402
    OCRResult,
    create_ocr_engine,
    draw_ocr_boxes,
)
from ocr_cpu.ocr.quality import (  # noqa: E402
    OCRQualityConfig,
    choose_best_ocr_result,
    quality_reports_to_rows,
    run_ocr_with_quality_gate,
)


# ---------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]


def resolve_image_path(image_arg: str, split: str = "test") -> Path:
    """
    Resolve image argument to a real local file path.

    Supports:
    - absolute path
    - relative path
    - image_id like img_2934.jpg
    - image stem like img_2934
    """

    raw = Path(image_arg).expanduser()

    # 1. Direct absolute/relative path
    if raw.exists():
        return raw.resolve()

    # 2. Relative to project root
    root_relative = ROOT / image_arg
    if root_relative.exists():
        return root_relative.resolve()

    # 3. Try project path resolver if available
    try:
        from ocr_cpu.utils.paths import get_image_path

        return get_image_path(image_arg, split=split).resolve()
    except Exception:
        pass

    # 4. Fallback recursive search inside data/raw
    search_root = ROOT / "data" / "raw"

    if not search_root.exists():
        raise FileNotFoundError(
            f"Cannot find image={image_arg}. Also data/raw does not exist."
        )

    candidates = []

    image_path = Path(image_arg)
    stem = image_path.stem
    suffix = image_path.suffix.lower()

    if suffix:
        patterns = [image_path.name]
    else:
        patterns = [f"{stem}{ext}" for ext in IMAGE_EXTS]

    for pattern in patterns:
        candidates.extend(search_root.rglob(pattern))

    if not candidates:
        raise FileNotFoundError(
            f"Cannot resolve image: {image_arg}. "
            f"Searched inside: {search_root}"
        )

    candidates = sorted(candidates)
    return candidates[0].resolve()


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------

def print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def print_subheader(title: str) -> None:
    print("\n" + "-" * 88)
    print(title)
    print("-" * 88)


def print_kv(key: str, value: Any, width: int = 26) -> None:
    print(f"{key:<{width}}: {value}")


def shorten(text: str, limit: int = 120) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\n", " / ")

    if len(text) <= limit:
        return text

    return text[: limit - 3] + "..."


def result_to_dict(result: OCRResult) -> Dict[str, Any]:
    return {
        "text": result.text,
        "avg_conf": result.avg_conf,
        "n_boxes": result.n_boxes,
        "latency_ms": result.latency_ms,
        "backend": result.backend,
        "lang": result.lang,
        "psm": result.psm,
        "oem": result.oem,
        "image_shape": result.image_shape,
        "error": result.error,
        "lines": [
            {
                "text": line.text,
                "conf": line.conf,
                "box": line.box,
                "block_num": line.block_num,
                "par_num": line.par_num,
                "line_num": line.line_num,
            }
            for line in result.lines
        ],
    }


def save_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv.imwrite(str(path), image)

    if not ok:
        raise RuntimeError(f"Failed to save image: {path}")


def save_debug_outputs(
    image_path: Path,
    variants: Mapping[str, Any],
    selection,
    out_dir: Path,
) -> Dict[str, str]:
    """
    Save preprocessed variants, selected overlay, and JSON report.
    """

    image_stem = image_path.stem
    image_out_dir = out_dir / image_stem
    image_out_dir.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, str] = {}

    for variant_name, img in variants.items():
        out_path = image_out_dir / f"{variant_name}.png"
        save_image(out_path, img)
        saved[f"variant_{variant_name}"] = str(out_path)

    selected_overlay = draw_ocr_boxes(
        variants[selection.selected_variant],
        selection.selected_result,
        min_conf=0.0,
        show_text=True,
    )

    overlay_path = image_out_dir / f"selected_overlay__{selection.selected_variant}.png"
    save_image(overlay_path, selected_overlay)
    saved["selected_overlay"] = str(overlay_path)

    report_path = image_out_dir / "report.json"

    payload = {
        "image_path": str(image_path),
        "selected_variant": selection.selected_variant,
        "selection_reason": selection.selection_reason,
        "selected_text": selection.selected_result.text,
        "tried_variants": selection.tried_variants,
        "reports": {
            name: report.to_dict()
            for name, report in selection.reports.items()
        },
        "results": {
            name: result_to_dict(result)
            for name, result in selection.results.items()
        },
    }

    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    saved["report_json"] = str(report_path)

    return saved


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def run_one_image_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run the complete OCR pipeline for one image.

    image
    -> router
    -> preprocess variants
    -> OCR engine
    -> quality gate
    -> selected final OCR
    """

    image_path = resolve_image_path(args.image, split=args.split)

    decision = classify_image_quality(
        image_path,
        target_short_side=args.target_short_side,
        max_long_side=args.max_long_side,
        engine=args.router_engine,
    )

    variants = apply_preprocess_by_decision(
        image_path,
        decision=decision,
        engine=args.router_engine,
    )

    engine = create_ocr_engine(
        backend=args.backend,
        lang=args.lang,
        psm=args.psm,
        oem=args.oem,
        timeout=args.timeout,
        min_token_conf=args.min_token_conf,
    )

    quality_config = OCRQualityConfig(
        min_avg_conf=args.min_avg_conf,
        min_alnum_chars=args.min_alnum_chars,
        try_raw_when_primary_bad=not args.no_try_raw_when_primary_bad,
    )

    if args.run_all_variants:
        results: Dict[str, OCRResult] = {}

        for variant_name, img in variants.items():
            results[variant_name] = engine.recognize(img)

        selection = choose_best_ocr_result(
            results=results,
            primary_variant=decision.primary_variant,
            config=quality_config,
        )

    else:
        selection = run_ocr_with_quality_gate(
            variants=variants,
            decision=decision,
            engine=engine,
            config=quality_config,
        )

    saved_outputs: Dict[str, str] = {}

    if args.save_debug:
        saved_outputs = save_debug_outputs(
            image_path=image_path,
            variants=variants,
            selection=selection,
            out_dir=Path(args.out_dir),
        )

    payload = {
        "image_path": str(image_path),
        "decision": asdict(decision),
        "run_all_variants": args.run_all_variants,
        "selected_variant": selection.selected_variant,
        "selection_reason": selection.selection_reason,
        "selected_text": selection.selected_result.text,
        "tried_variants": selection.tried_variants,
        "reports": {
            name: report.to_dict()
            for name, report in selection.reports.items()
        },
        "results": {
            name: result_to_dict(result)
            for name, result in selection.results.items()
        },
        "saved_outputs": saved_outputs,
    }

    if args.json_out:
        json_out = Path(args.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return payload


def print_pipeline_report(payload: Dict[str, Any]) -> None:
    """
    Print human-readable pipeline report.
    """

    decision = payload["decision"]
    reports = payload["reports"]
    results = payload["results"]

    print_header("ONE IMAGE OCR PIPELINE DEBUG")

    print_subheader("IMAGE")
    print_kv("image_path", payload["image_path"])

    print_subheader("ROUTER DECISION")
    print_kv("primary_variant", decision.get("primary_variant"))
    print_kv("reason", decision.get("reason"))
    print_kv("scale", decision.get("scale"))
    print_kv("use_clahe", decision.get("use_clahe"))
    print_kv("use_median", decision.get("use_median"))
    print_kv("use_gaussian", decision.get("use_gaussian"))
    print_kv("use_sharpen", decision.get("use_sharpen"))
    print_kv("use_hard_fallback", decision.get("use_hard_fallback"))
    print_kv("hard_threshold_mode", decision.get("hard_threshold_mode"))

    print_subheader("TRIED OCR VARIANTS")
    for variant_name in payload["tried_variants"]:
        result = results[variant_name]
        report = reports[variant_name]

        print(f"\n[{variant_name}]")
        print_kv("is_bad", report["is_bad"])
        print_kv("score", round(float(report["score"]), 4))
        print_kv("reasons", report["reasons"])
        print_kv("avg_conf", round(float(result["avg_conf"]), 4))
        print_kv("n_boxes", result["n_boxes"])
        print_kv("latency_ms", round(float(result["latency_ms"]), 2))
        print_kv("error", result["error"])
        print_kv("text_preview", shorten(result["text"], 160))

    print_subheader("QUALITY REPORT TABLE")
    rows = []

    for variant_name, report in reports.items():
        rows.append(
            {
                "variant": variant_name,
                "is_bad": report["is_bad"],
                "score": round(float(report["score"]), 4),
                "reasons": "|".join(report["reasons"]),
                "avg_conf": round(float(report["avg_conf"]), 4),
                "n_boxes": report["n_boxes"],
                "n_chars": report["n_chars"],
                "n_alnum": report["n_alnum"],
            }
        )

    if rows:
        columns = [
            "variant",
            "is_bad",
            "score",
            "avg_conf",
            "n_boxes",
            "n_chars",
            "n_alnum",
            "reasons",
        ]

        widths = {
            col: max(
                len(str(col)),
                max(len(str(row.get(col, ""))) for row in rows),
            )
            for col in columns
        }

        header = " | ".join(f"{col:<{widths[col]}}" for col in columns)
        sep = "-+-".join("-" * widths[col] for col in columns)

        print(header)
        print(sep)

        for row in rows:
            print(
                " | ".join(
                    f"{str(row.get(col, '')):<{widths[col]}}"
                    for col in columns
                )
            )

    print_subheader("FINAL SELECTION")
    print_kv("selected_variant", payload["selected_variant"])
    print_kv("selection_reason", payload["selection_reason"])

    print("\nSELECTED OCR TEXT:")
    print(payload["selected_text"])

    if payload.get("saved_outputs"):
        print_subheader("SAVED DEBUG OUTPUTS")
        for k, v in payload["saved_outputs"].items():
            print_kv(k, v)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug one image through the full OCR pipeline.",
    )

    parser.add_argument(
        "--image",
        required=True,
        help=(
            "Image path or image_id. Examples: "
            "data/raw/test_images/test_images/images/img_2934.jpg or img_2934.jpg"
        ),
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Split used when resolving image_id.",
    )

    parser.add_argument(
        "--router-engine",
        default="tesseract",
        choices=["tesseract", "paddle", "easyocr", "vietocr"],
        help="Engine profile for preprocessing router.",
    )

    parser.add_argument(
        "--backend",
        default="tesseract",
        choices=["tesseract", "paddle"],
        help="OCR backend.",
    )

    parser.add_argument(
        "--lang",
        default="eng",
        help="Tesseract language. Use eng now, vie+eng after installing vie tessdata.",
    )

    parser.add_argument(
        "--psm",
        type=int,
        default=11,
        help="Tesseract page segmentation mode.",
    )

    parser.add_argument(
        "--oem",
        type=int,
        default=3,
        help="Tesseract OCR engine mode.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="OCR timeout per image in seconds.",
    )

    parser.add_argument(
        "--min-token-conf",
        type=float,
        default=0.0,
        help="Drop OCR tokens below this normalized confidence.",
    )

    parser.add_argument(
        "--target-short-side",
        type=int,
        default=900,
        help="Target short side used by preprocessing router.",
    )

    parser.add_argument(
        "--max-long-side",
        type=int,
        default=1600,
        help="Max long side after preprocessing resize.",
    )

    parser.add_argument(
        "--min-avg-conf",
        type=float,
        default=0.35,
        help="Quality gate avg confidence threshold.",
    )

    parser.add_argument(
        "--min-alnum-chars",
        type=int,
        default=4,
        help="Quality gate minimum alphanumeric characters.",
    )

    parser.add_argument(
        "--no-try-raw-when-primary-bad",
        action="store_true",
        help="Do not try raw_resized when primary variant is bad.",
    )

    parser.add_argument(
        "--run-all-variants",
        action="store_true",
        help=(
            "Run OCR on raw_resized, soft_enhanced, and hard_fallback. "
            "Useful for debugging but slower."
        ),
    )

    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save preprocess variants, overlay image, and JSON report.",
    )

    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "outputs" / "debug_images"),
        help="Output folder for debug images.",
    )

    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to save JSON report.",
    )

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    payload = run_one_image_pipeline(args)
    print_pipeline_report(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
