from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm


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
from ocr_cpu.ocr.engine import create_ocr_engine, OCRResult  # noqa: E402
from ocr_cpu.ocr.quality import (  # noqa: E402
    OCRQualityConfig,
    choose_best_ocr_result,
    evaluate_ocr_result,
)


# ---------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------

def resolve_train_labels_path(path_arg: str) -> Path:
    p = Path(path_arg).expanduser()

    if p.exists():
        return p.resolve()

    p2 = ROOT / path_arg

    if p2.exists():
        return p2.resolve()

    default = ROOT / "data" / "raw" / "train_labels.csv"

    if default.exists():
        return default.resolve()

    raise FileNotFoundError(f"Cannot find train_labels.csv: {path_arg}")


def resolve_image_path(image_id: str, split: str = "train") -> Path:
    """
    Resolve image_id to actual local image path.

    Supports nested dataset structure like:
    data/raw/train_images/train_images/train_images/img_xxxx.jpg
    """

    try:
        from ocr_cpu.utils.paths import get_image_path

        return get_image_path(image_id, split=split).resolve()
    except Exception:
        pass

    raw_dir = ROOT / "data" / "raw"

    if split == "train":
        search_root = raw_dir / "train_images"
    elif split == "test":
        search_root = raw_dir / "test_images"
    else:
        search_root = raw_dir

    image_id = str(image_id)
    stem = Path(image_id).stem
    suffix = Path(image_id).suffix

    patterns = []

    if suffix:
        patterns.append(image_id)
    else:
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            patterns.append(f"{stem}{ext}")

    candidates = []

    for pattern in patterns:
        candidates.extend(search_root.rglob(pattern))

    if not candidates:
        raise FileNotFoundError(
            f"Cannot find image_id={image_id} inside {search_root}"
        )

    return sorted(candidates)[0].resolve()


# ---------------------------------------------------------------------
# CER
# ---------------------------------------------------------------------

def normalize_for_cer(text: Any, lower: bool = False) -> str:
    """
    Normalize text before CER.

    Keep Vietnamese diacritics.
    Normalize spaces.
    """

    if text is None:
        text = ""

    try:
        if pd.isna(text):
            text = ""
    except Exception:
        pass

    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()

    if lower:
        text = text.lower()

    return text


def levenshtein_distance(a: str, b: str) -> int:
    """
    Standard Levenshtein distance.
    """

    if a == b:
        return 0

    if len(a) == 0:
        return len(b)

    if len(b) == 0:
        return len(a)

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        current = [i]

        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if ca == cb else 1)

            current.append(
                min(insert_cost, delete_cost, replace_cost)
            )

        previous = current

    return previous[-1]


def cer(pred: Any, gt: Any, lower: bool = False) -> float:
    """
    Character Error Rate.

    CER = edit_distance(pred, gt) / len(gt)
    """

    pred = normalize_for_cer(pred, lower=lower)
    gt = normalize_for_cer(gt, lower=lower)

    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0

    return levenshtein_distance(pred, gt) / len(gt)


# ---------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------

VARIANT_TO_PREFIX = {
    "raw_resized": "raw",
    "soft_enhanced": "soft",
    "hard_fallback": "hard",
}


def result_to_text(result: Optional[OCRResult]) -> str:
    if result is None:
        return ""

    return result.text or ""


def result_to_conf(result: Optional[OCRResult]) -> float:
    if result is None:
        return 0.0

    return float(result.avg_conf or 0.0)


def result_to_latency(result: Optional[OCRResult]) -> float:
    if result is None:
        return 0.0

    return float(result.latency_ms or 0.0)


def result_to_boxes(result: Optional[OCRResult]) -> int:
    if result is None:
        return 0

    return int(result.n_boxes or 0)


def result_to_error(result: Optional[OCRResult]) -> str:
    if result is None:
        return "missing_result"

    return str(result.error or "")


# ---------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------

def load_train_subset(
    train_labels_path: Path,
    limit: int,
    sample_mode: str,
    seed: int,
    include_blank_gt: bool,
) -> pd.DataFrame:
    df = pd.read_csv(train_labels_path)

    required_cols = {"image_id", "ocr_text"}

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"train_labels.csv missing columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    if not include_blank_gt:
        mask = df["ocr_text"].fillna("").astype(str).str.strip() != ""
        df = df[mask].copy()

    if sample_mode == "head":
        out = df.head(limit).copy()
    elif sample_mode == "tail":
        out = df.tail(limit).copy()
    elif sample_mode == "random":
        n = min(limit, len(df))
        out = df.sample(n=n, random_state=seed).copy()
    else:
        raise ValueError(f"Unknown sample_mode: {sample_mode}")

    out = out.reset_index(drop=True)

    return out


# ---------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------

def audit_one_image(
    row: pd.Series,
    engine,
    quality_config: OCRQualityConfig,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    image_id = str(row["image_id"])
    gt_text = row.get("ocr_text", "")

    start_total = time.perf_counter()

    base: Dict[str, Any] = {
        "image_id": image_id,
        "gt_ocr_text": gt_text,
        "error": "",
    }

    try:
        image_path = resolve_image_path(image_id, split="train")

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

        results: Dict[str, OCRResult] = {}

        # For audit we intentionally run all variants.
        # This gives evidence for tuning router.py.
        for variant_name, img in variants.items():
            results[variant_name] = engine.recognize(img)

        # Production-like selection:
        # If hard_fallback is not allowed by router, do not let selector choose it.
        selectable_results = dict(results)

        if not bool(decision.use_hard_fallback):
            selectable_results.pop("hard_fallback", None)

        selection = choose_best_ocr_result(
            results=selectable_results,
            primary_variant=decision.primary_variant,
            config=quality_config,
        )

        row_out: Dict[str, Any] = {
            **base,
            "image_path": str(image_path),

            "router_reason": decision.reason,
            "primary_variant": decision.primary_variant,
            "scale": decision.scale,
            "use_clahe": decision.use_clahe,
            "use_median": decision.use_median,
            "use_gaussian": decision.use_gaussian,
            "use_sharpen": decision.use_sharpen,
            "use_hard_fallback": decision.use_hard_fallback,
            "hard_threshold_mode": decision.hard_threshold_mode,

            "selected_variant": selection.selected_variant,
            "selected_text": selection.selected_result.text,
            "selected_conf": selection.selected_result.avg_conf,
            "selected_boxes": selection.selected_result.n_boxes,
            "selected_score": selection.selected_report.score,
            "selected_reasons": "|".join(selection.selected_report.reasons),
            "selected_cer": cer(
                selection.selected_result.text,
                gt_text,
                lower=args.cer_lower,
            ),
            "selection_reason": selection.selection_reason,
        }

        variant_cers: Dict[str, float] = {}

        for variant_name, result in results.items():
            prefix = VARIANT_TO_PREFIX.get(variant_name, variant_name)

            report = evaluate_ocr_result(
                result,
                config=quality_config,
                variant_name=variant_name,
            )

            variant_cer = cer(
                result.text,
                gt_text,
                lower=args.cer_lower,
            )

            variant_cers[variant_name] = variant_cer

            row_out[f"{prefix}_text"] = result.text
            row_out[f"{prefix}_conf"] = result.avg_conf
            row_out[f"{prefix}_boxes"] = result.n_boxes
            row_out[f"{prefix}_latency_ms"] = result.latency_ms
            row_out[f"{prefix}_cer"] = variant_cer
            row_out[f"{prefix}_is_bad"] = report.is_bad
            row_out[f"{prefix}_score"] = report.score
            row_out[f"{prefix}_reasons"] = "|".join(report.reasons)
            row_out[f"{prefix}_error"] = result.error or ""

        best_cer_variant = min(
            variant_cers.keys(),
            key=lambda name: variant_cers[name],
        )

        row_out["oracle_best_cer_variant"] = best_cer_variant
        row_out["oracle_best_cer"] = variant_cers[best_cer_variant]
        row_out["selected_is_oracle_best"] = (
            selection.selected_variant == best_cer_variant
        )

        if decision.primary_variant in variant_cers:
            row_out["primary_cer"] = variant_cers[decision.primary_variant]
        else:
            row_out["primary_cer"] = None

        row_out["latency_ms"] = sum(
            result_to_latency(r)
            for r in results.values()
        )

        row_out["total_wall_ms"] = (
            time.perf_counter() - start_total
        ) * 1000.0

        return row_out

    except Exception as e:
        return {
            **base,
            "error": str(e),
            "latency_ms": 0.0,
            "total_wall_ms": (
                time.perf_counter() - start_total
            ) * 1000.0,
        }


def summarize_audit(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    summary["n_rows"] = int(len(df))
    summary["n_errors"] = int(df["error"].fillna("").astype(str).ne("").sum())

    cer_cols = [
        "raw_cer",
        "soft_cer",
        "hard_cer",
        "selected_cer",
        "primary_cer",
        "oracle_best_cer",
    ]

    for col in cer_cols:
        if col in df.columns:
            valid = pd.to_numeric(df[col], errors="coerce").dropna()

            if len(valid) > 0:
                summary[f"{col}_mean"] = float(valid.mean())
                summary[f"{col}_median"] = float(valid.median())
                summary[f"{col}_p90"] = float(valid.quantile(0.90))
            else:
                summary[f"{col}_mean"] = None
                summary[f"{col}_median"] = None
                summary[f"{col}_p90"] = None

    count_cols = [
        "primary_variant",
        "selected_variant",
        "oracle_best_cer_variant",
        "router_reason",
    ]

    for col in count_cols:
        if col in df.columns:
            summary[f"{col}_counts"] = (
                df[col]
                .fillna("")
                .astype(str)
                .value_counts()
                .head(20)
                .to_dict()
            )

    if "selected_is_oracle_best" in df.columns:
        summary["selected_is_oracle_best_rate"] = float(
            df["selected_is_oracle_best"].fillna(False).mean()
        )

    if "latency_ms" in df.columns:
        latency = pd.to_numeric(df["latency_ms"], errors="coerce").dropna()

        if len(latency) > 0:
            summary["latency_ms_mean"] = float(latency.mean())
            summary["latency_ms_median"] = float(latency.median())
            summary["latency_ms_total"] = float(latency.sum())

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print("AUDIT SUMMARY")
    print("=" * 88)

    scalar_keys = [
        k for k, v in summary.items()
        if not isinstance(v, dict)
    ]

    for k in scalar_keys:
        print(f"{k:<36}: {summary[k]}")

    dict_keys = [
        k for k, v in summary.items()
        if isinstance(v, dict)
    ]

    for k in dict_keys:
        print("\n" + "-" * 88)
        print(k)
        print("-" * 88)

        for kk, vv in summary[k].items():
            print(f"{kk:<48}: {vv}")


def run_audit(args: argparse.Namespace) -> None:
    train_labels_path = resolve_train_labels_path(args.train_labels)

    df_subset = load_train_subset(
        train_labels_path=train_labels_path,
        limit=args.limit,
        sample_mode=args.sample_mode,
        seed=args.seed,
        include_blank_gt=args.include_blank_gt,
    )

    print(f"Train labels: {train_labels_path}")
    print(f"Audit rows:   {len(df_subset)}")
    print(f"Sample mode:  {args.sample_mode}")
    print(f"Lang:         {args.lang}")
    print(f"PSM:          {args.psm}")
    print(f"Output:       {args.out}")

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
        try_raw_when_primary_bad=True,
    )

    rows: List[Dict[str, Any]] = []

    iterator = tqdm(
        df_subset.iterrows(),
        total=len(df_subset),
        desc="Auditing train subset",
    )

    for _, row in iterator:
        audit_row = audit_one_image(
            row=row,
            engine=engine,
            quality_config=quality_config,
            args=args,
        )

        rows.append(audit_row)

    out_df = pd.DataFrame(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(
        out_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = summarize_audit(out_df)

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_summary(summary)

    print("\n" + "=" * 88)
    print("SAVED")
    print("=" * 88)
    print(f"Audit CSV:    {out_path}")
    print(f"Summary JSON: {summary_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit OCR preprocessing variants on train subset.",
    )

    parser.add_argument(
        "--train-labels",
        default=str(ROOT / "data" / "raw" / "train_labels.csv"),
        help="Path to train_labels.csv.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of train samples to audit.",
    )

    parser.add_argument(
        "--sample-mode",
        choices=["random", "head", "tail"],
        default="random",
        help="How to select subset.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample-mode=random.",
    )

    parser.add_argument(
        "--include-blank-gt",
        action="store_true",
        help="Include rows where ground truth ocr_text is blank.",
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
        help="Tesseract PSM.",
    )

    parser.add_argument(
        "--oem",
        type=int,
        default=3,
        help="Tesseract OEM.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="OCR timeout per image.",
    )

    parser.add_argument(
        "--min-token-conf",
        type=float,
        default=0.0,
        help="Drop OCR tokens below this normalized confidence.",
    )

    parser.add_argument(
        "--router-engine",
        default="tesseract",
        choices=["tesseract", "paddle", "easyocr", "vietocr"],
        help="Router engine profile.",
    )

    parser.add_argument(
        "--target-short-side",
        type=int,
        default=900,
        help="Router target short side.",
    )

    parser.add_argument(
        "--max-long-side",
        type=int,
        default=1600,
        help="Router max long side.",
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
        "--cer-lower",
        action="store_true",
        help="Lowercase pred/gt before CER.",
    )

    parser.add_argument(
        "--out",
        default=str(ROOT / "outputs" / "audits" / "train_audit_50.csv"),
        help="Output audit CSV.",
    )

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_audit(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
