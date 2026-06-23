from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------
# Project import setup
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_cpu.ocr.engine import create_ocr_engine  # noqa: E402
from ocr_cpu.preprocess.router import (  # noqa: E402
    _read_bgr,
    classify_image_quality,
)
from ocr_cpu.preprocess.hard_variants import (  # noqa: E402
    HardPreprocessConfig,
    get_default_hard_configs,
    get_hard_config_map,
    make_hard_threshold_variant,
)


# ---------------------------------------------------------------------
# Worker globals
# ---------------------------------------------------------------------

_WORKER_ARGS = None
_WORKER_ENGINE = None
_WORKER_CONFIG = None


def set_single_process_thread_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ---------------------------------------------------------------------
# Text / CER
# ---------------------------------------------------------------------

def normalize_for_cer(text: Any, lower: bool = False) -> str:
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
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (0 if ca == cb else 1),
                )
            )

        previous = current

    return previous[-1]


def cer(pred: Any, gt: Any, lower: bool = False) -> float:
    pred = normalize_for_cer(pred, lower=lower)
    gt = normalize_for_cer(gt, lower=lower)

    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0

    return levenshtein_distance(pred, gt) / len(gt)


# ---------------------------------------------------------------------
# Image path / sample preparation
# ---------------------------------------------------------------------

def resolve_train_image_path(image_id: str) -> Path:
    try:
        from ocr_cpu.utils.paths import get_image_path

        return get_image_path(image_id, split="train").resolve()
    except Exception:
        pass

    raw_dir = ROOT / "data" / "raw" / "train_images"

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
        candidates.extend(raw_dir.rglob(pattern))

    if not candidates:
        raise FileNotFoundError(f"Cannot find train image: {image_id}")

    return sorted(candidates)[0].resolve()


def prepare_tuning_sample(args: argparse.Namespace) -> pd.DataFrame:
    labels = pd.read_csv(args.train_labels)

    required = {"image_id", "ocr_text"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Missing columns in train labels: {missing}")

    labels = labels.copy()
    labels["ocr_text"] = labels["ocr_text"].fillna("").astype(str)

    if args.audit_csv:
        audit = pd.read_csv(args.audit_csv)
        audit = audit.copy()

        audit["hard_cer"] = pd.to_numeric(audit.get("hard_cer"), errors="coerce")
        audit["hard_boxes"] = pd.to_numeric(audit.get("hard_boxes"), errors="coerce")
        audit["hard_text_len"] = audit.get("hard_text", "").fillna("").astype(str).str.len()
        audit["error_str"] = audit.get("error", "").fillna("").astype(str)

        mask = audit["error_str"].str.strip().eq("")
        mask &= audit["hard_cer"].notna()
        mask &= audit["hard_cer"] >= float(args.min_base_cer)
        mask &= audit["hard_cer"] <= float(args.max_base_cer)
        mask &= audit["hard_boxes"] >= int(args.min_boxes)
        mask &= audit["hard_boxes"] <= int(args.max_boxes)
        mask &= audit["hard_text_len"] <= int(args.max_text_len)

        sample = audit[mask].copy()

        # Merge the canonical GT from labels to avoid relying on audit column naming.
        sample = sample.drop(columns=[c for c in ["ocr_text", "gt_ocr_text_from_label"] if c in sample.columns], errors="ignore")
        sample = sample.merge(labels[["image_id", "ocr_text"]], on="image_id", how="left")

        if "image_path" not in sample.columns:
            sample["image_path"] = ""

    else:
        sample = labels[labels["ocr_text"].str.strip().ne("")].copy()
        sample["image_path"] = ""

    if args.sample_mode == "random":
        sample = sample.sample(n=min(args.limit, len(sample)), random_state=args.seed)
    elif args.sample_mode == "head":
        sample = sample.head(args.limit)
    elif args.sample_mode == "tail":
        sample = sample.tail(args.limit)
    else:
        raise ValueError(f"Unknown sample mode: {args.sample_mode}")

    sample = sample.reset_index(drop=True)
    sample.insert(0, "tune_idx", np.arange(1, len(sample) + 1))

    return sample


# ---------------------------------------------------------------------
# Resize helper
# ---------------------------------------------------------------------

def resize_by_scale(img: np.ndarray, scale: float) -> np.ndarray:
    if abs(float(scale) - 1.0) < 0.03:
        return img.copy()

    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv.INTER_CUBIC if scale > 1.0 else cv.INTER_AREA
    return cv.resize(img, (new_w, new_h), interpolation=interp)


# ---------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------

def init_worker(args_dict: Dict[str, Any], cfg_dict: Dict[str, Any]) -> None:
    global _WORKER_ARGS
    global _WORKER_ENGINE
    global _WORKER_CONFIG

    set_single_process_thread_env()

    _WORKER_ARGS = SimpleNamespace(**args_dict)
    _WORKER_CONFIG = HardPreprocessConfig(**cfg_dict)

    _WORKER_ENGINE = create_ocr_engine(
        backend=_WORKER_ARGS.backend,
        lang=_WORKER_ARGS.lang,
        device=_WORKER_ARGS.device,
        psm=_WORKER_ARGS.psm,
        oem=_WORKER_ARGS.oem,
        timeout=_WORKER_ARGS.timeout,
        min_token_conf=_WORKER_ARGS.min_token_conf,
    )

    print(
        f"[worker pid={os.getpid()}] initialized config={_WORKER_CONFIG.config_id} "
        f"backend={_WORKER_ARGS.backend} lang={_WORKER_ARGS.lang}",
        flush=True,
    )


def worker_task(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    global _WORKER_ARGS
    global _WORKER_ENGINE
    global _WORKER_CONFIG

    start = time.perf_counter()

    tune_idx = int(row_dict.get("tune_idx", -1))
    image_id = str(row_dict.get("image_id", ""))
    gt_text = row_dict.get("ocr_text", "")

    try:
        image_path_raw = str(row_dict.get("image_path", "") or "")
        if image_path_raw and Path(image_path_raw).exists():
            image_path = Path(image_path_raw)
        else:
            image_path = resolve_train_image_path(image_id)

        img = _read_bgr(image_path)

        decision = classify_image_quality(
            img,
            target_short_side=_WORKER_ARGS.target_short_side,
            max_long_side=_WORKER_ARGS.max_long_side,
            engine=_WORKER_ARGS.router_engine,
        )

        raw_resized = resize_by_scale(img, decision.scale)
        hard_img = make_hard_threshold_variant(raw_resized, _WORKER_CONFIG)

        result = _WORKER_ENGINE.recognize(hard_img)

        pred_text = result.text or ""
        pred_len = len(str(pred_text))
        pred_boxes = int(result.n_boxes or 0)
        pred_conf = float(result.avg_conf or 0.0)
        pred_cer = cer(pred_text, gt_text, lower=_WORKER_ARGS.cer_lower)

        no_detect = pred_boxes == 0 or len(str(pred_text).strip()) == 0
        overread = pred_boxes >= _WORKER_ARGS.overread_boxes or pred_len >= _WORKER_ARGS.overread_text_len

        return {
            "config_id": _WORKER_CONFIG.config_id,
            "tune_idx": tune_idx,
            "image_id": image_id,
            "image_path": str(image_path),
            "gt_ocr_text": gt_text,
            "pred_ocr_text": pred_text,
            "cer": pred_cer,
            "conf": pred_conf,
            "boxes": pred_boxes,
            "text_len": pred_len,
            "no_detect": bool(no_detect),
            "overread": bool(overread),
            "router_reason": decision.reason,
            "scale": decision.scale,
            "latency_ms": result.latency_ms,
            "wall_ms": (time.perf_counter() - start) * 1000.0,
            "error": result.error or "",
            "exception": "",
        }

    except Exception as e:
        return {
            "config_id": _WORKER_CONFIG.config_id if _WORKER_CONFIG else "",
            "tune_idx": tune_idx,
            "image_id": image_id,
            "gt_ocr_text": gt_text,
            "pred_ocr_text": "",
            "cer": math.nan,
            "conf": 0.0,
            "boxes": 0,
            "text_len": 0,
            "no_detect": True,
            "overread": False,
            "router_reason": "",
            "scale": math.nan,
            "latency_ms": 0.0,
            "wall_ms": (time.perf_counter() - start) * 1000.0,
            "error": repr(e),
            "exception": traceback.format_exc(),
        }


# ---------------------------------------------------------------------
# Metrics / objective
# ---------------------------------------------------------------------

def safe_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.fillna(False).astype(bool).mean())


def summarize_config(df: pd.DataFrame, cfg: HardPreprocessConfig) -> Dict[str, Any]:
    cer_s = pd.to_numeric(df["cer"], errors="coerce").dropna()

    n = int(len(df))
    n_valid = int(len(cer_s))
    error_rate = float(df["error"].fillna("").astype(str).str.strip().ne("").mean()) if n else 0.0
    no_detect_rate = safe_rate(df["no_detect"])
    overread_rate = safe_rate(df["overread"])

    median = float(cer_s.median()) if n_valid else math.inf
    mean = float(cer_s.mean()) if n_valid else math.inf
    p90 = float(cer_s.quantile(0.90)) if n_valid else math.inf
    p95 = float(cer_s.quantile(0.95)) if n_valid else math.inf
    max_cer = float(cer_s.max()) if n_valid else math.inf

    # Optimize for mainstream performance while punishing fatal/unstable behavior.
    objective = (
        median
        + 0.50 * p90
        + 2.00 * no_detect_rate
        + 1.50 * overread_rate
        + 3.00 * error_rate
    )

    return {
        "config_id": cfg.config_id,
        "description": cfg.description,
        "objective": objective,
        "n_rows": n,
        "n_valid": n_valid,
        "cer_mean": mean,
        "cer_median": median,
        "cer_p90": p90,
        "cer_p95": p95,
        "cer_max": max_cer,
        "no_detect_rate": no_detect_rate,
        "overread_rate": overread_rate,
        "error_rate": error_rate,
        "conf_mean": float(pd.to_numeric(df["conf"], errors="coerce").mean()),
        "boxes_mean": float(pd.to_numeric(df["boxes"], errors="coerce").mean()),
        "latency_ms_mean": float(pd.to_numeric(df["latency_ms"], errors="coerce").mean()),
        "config_json": json.dumps(cfg.to_dict(), ensure_ascii=False),
    }


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

def select_configs(args: argparse.Namespace) -> List[HardPreprocessConfig]:
    config_map = get_hard_config_map()

    if args.config_ids.strip():
        ids = [x.strip() for x in re.split(r"[,\s]+", args.config_ids.strip()) if x.strip()]
        missing = [x for x in ids if x not in config_map]
        if missing:
            raise ValueError(f"Unknown config IDs: {missing}. Available: {sorted(config_map)}")
        return [config_map[x] for x in ids]

    return get_default_hard_configs()


def write_sample_file(sample: pd.DataFrame, out_dir: Path) -> None:
    sample_path = out_dir / "tuning_sample.csv"
    cols = [c for c in ["tune_idx", "image_id", "image_path", "ocr_text", "hard_cer", "hard_boxes", "hard_text_len", "router_reason"] if c in sample.columns]
    sample[cols].to_csv(sample_path, index=False, encoding="utf-8-sig")
    print(f"Saved tuning sample: {sample_path}")


def run_one_config(args: argparse.Namespace, cfg: HardPreprocessConfig, sample: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    detail_path = out_dir / f"detail__{cfg.config_id}.csv"

    if detail_path.exists() and not args.force:
        print(f"[SKIP] Existing detail found for {cfg.config_id}: {detail_path}")
        detail = pd.read_csv(detail_path)
        return detail, summarize_config(detail, cfg)

    print("=" * 88)
    print(f"CONFIG_START: {cfg.config_id}")
    print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))
    print("=" * 88)

    rows: List[Dict[str, Any]] = []
    tasks = [row.to_dict() for _, row in sample.iterrows()]

    args_dict = vars(args).copy()
    cfg_dict = cfg.to_dict()

    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=init_worker,
        initargs=(args_dict, cfg_dict),
    ) as executor:
        futures = [executor.submit(worker_task, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Tuning {cfg.config_id}"):
            row = future.result()
            rows.append(row)

            if len(rows) % args.checkpoint_every == 0:
                tmp = pd.DataFrame(rows).sort_values("tune_idx")
                tmp.to_csv(detail_path.with_suffix(".partial.csv"), index=False, encoding="utf-8-sig")

    detail = pd.DataFrame(rows)
    if "tune_idx" in detail.columns:
        detail = detail.sort_values("tune_idx")

    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary = summarize_config(detail, cfg)

    print(f"CONFIG_DONE: {cfg.config_id}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return detail, summary


def run(args: argparse.Namespace) -> None:
    set_single_process_thread_env()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = prepare_tuning_sample(args)
    if len(sample) == 0:
        raise ValueError("Tuning sample is empty. Relax your filters or provide a different audit CSV.")

    configs = select_configs(args)

    print("=" * 88)
    print("HARD PREPROCESS TUNING START")
    print("=" * 88)
    print(f"sample rows:  {len(sample)}")
    print(f"configs:      {len(configs)}")
    print(f"workers:      {args.num_workers}")
    print(f"backend:      {args.backend}")
    print(f"lang:         {args.lang}")
    print(f"out_dir:      {out_dir}")
    print("=" * 88)

    write_sample_file(sample, out_dir)

    summaries: List[Dict[str, Any]] = []
    leaderboard_path = out_dir / "leaderboard.csv"

    for cfg in configs:
        _, summary = run_one_config(args, cfg, sample, out_dir)
        summaries.append(summary)

        leaderboard = pd.DataFrame(summaries)
        leaderboard = leaderboard.sort_values(["objective", "cer_p90", "cer_median"], ascending=[True, True, True])
        leaderboard.to_csv(leaderboard_path, index=False, encoding="utf-8-sig")

        print("=" * 88)
        print("CURRENT LEADERBOARD")
        print("=" * 88)
        cols = [
            "config_id",
            "objective",
            "cer_median",
            "cer_p90",
            "cer_p95",
            "cer_mean",
            "no_detect_rate",
            "overread_rate",
            "error_rate",
        ]
        print(leaderboard[cols].head(15).to_string(index=False))

    final = pd.DataFrame(summaries)
    final = final.sort_values(["objective", "cer_p90", "cer_median"], ascending=[True, True, True])
    final.to_csv(leaderboard_path, index=False, encoding="utf-8-sig")

    best_path = out_dir / "best_config.json"
    best = final.iloc[0].to_dict()
    best_path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 88)
    print("TUNING DONE")
    print("=" * 88)
    print(f"Leaderboard: {leaderboard_path}")
    print(f"Best config: {best_path}")
    print(final.head(10).to_string(index=False))


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-tune hard preprocessing configs using OCR CER feedback.")

    parser.add_argument("--train-labels", default=str(ROOT / "data" / "raw" / "train_labels.csv"))
    parser.add_argument("--audit-csv", default="", help="Existing hard-only audit CSV used to pick mainstream tuning samples.")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "tuning" / "hard_preprocess"))

    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--sample-mode", choices=["random", "head", "tail"], default="random")
    parser.add_argument("--seed", type=int, default=42)

    # Filter for mainstream samples from an existing audit CSV.
    parser.add_argument("--min-base-cer", type=float, default=0.0)
    parser.add_argument("--max-base-cer", type=float, default=1.5)
    parser.add_argument("--min-boxes", type=int, default=1)
    parser.add_argument("--max-boxes", type=int, default=29)
    parser.add_argument("--max-text-len", type=int, default=499)

    # OCR
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--backend", default="paddle", choices=["paddle", "tesseract"])
    parser.add_argument("--lang", default="vi")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--psm", type=int, default=11)
    parser.add_argument("--oem", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--min-token-conf", type=float, default=0.0)

    # Router scale profile
    parser.add_argument("--router-engine", default="paddle", choices=["paddle", "tesseract", "easyocr", "vietocr"])
    parser.add_argument("--target-short-side", type=int, default=900)
    parser.add_argument("--max-long-side", type=int, default=1600)

    # Failure/instability proxy thresholds
    parser.add_argument("--overread-boxes", type=int, default=30)
    parser.add_argument("--overread-text-len", type=int, default=500)
    parser.add_argument("--cer-lower", action="store_true")

    # Config selection / bookkeeping
    parser.add_argument("--config-ids", default="", help="Optional comma/space-separated config IDs to run. Default: all built-in configs.")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--force", action="store_true", help="Re-run configs even if detail CSV exists.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
