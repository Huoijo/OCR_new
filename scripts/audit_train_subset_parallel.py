from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------
# Project import setup
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from scripts.audit_train_subset import (  # noqa: E402
    audit_one_image,
    load_train_subset,
    print_summary,
    resolve_train_labels_path,
    summarize_audit,
)
from ocr_cpu.ocr.engine import create_ocr_engine  # noqa: E402
from ocr_cpu.ocr.quality import OCRQualityConfig  # noqa: E402


# ---------------------------------------------------------------------
# Worker globals
# ---------------------------------------------------------------------

_WORKER_ENGINE = None
_WORKER_ARGS = None
_WORKER_QUALITY_CONFIG = None


def _set_single_process_thread_env() -> None:
    """
    Prevent every Paddle worker from creating too many internal threads.

    This is important when running multiple processes on a laptop CPU.
    """

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def init_worker(args_dict: Dict[str, Any]) -> None:
    """
    Initialize Paddle/Tesseract engine once per worker process.
    """

    global _WORKER_ENGINE
    global _WORKER_ARGS
    global _WORKER_QUALITY_CONFIG

    _set_single_process_thread_env()

    args = SimpleNamespace(**args_dict)
    _WORKER_ARGS = args

    _WORKER_ENGINE = create_ocr_engine(
        backend=args.backend,
        lang=args.lang,
        device=args.device,
        psm=args.psm,
        oem=args.oem,
        timeout=args.timeout,
        min_token_conf=args.min_token_conf,
    )

    _WORKER_QUALITY_CONFIG = OCRQualityConfig(
        min_avg_conf=args.min_avg_conf,
        min_alnum_chars=args.min_alnum_chars,
        try_raw_when_primary_bad=True,
    )

    print(
        f"[worker pid={os.getpid()}] initialized "
        f"backend={args.backend} lang={args.lang}",
        flush=True,
    )


def worker_task(payload: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process one image in a worker.

    Returns one audit row.
    """

    global _WORKER_ENGINE
    global _WORKER_ARGS
    global _WORKER_QUALITY_CONFIG

    idx, row_dict = payload
    image_id = str(row_dict.get("image_id", ""))

    try:
        row = pd.Series(row_dict)

        out = audit_one_image(
            row=row,
            engine=_WORKER_ENGINE,
            quality_config=_WORKER_QUALITY_CONFIG,
            args=_WORKER_ARGS,
        )

        out["idx"] = idx
        out["worker_pid"] = os.getpid()

        return out

    except Exception as e:
        return {
            "idx": idx,
            "image_id": image_id,
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "worker_pid": os.getpid(),
        }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def write_checkpoint(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    df = pd.DataFrame(rows)
    df = df.sort_values("idx") if "idx" in df.columns else df

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def run_parallel_audit(args: argparse.Namespace) -> None:
    _set_single_process_thread_env()

    train_labels_path = resolve_train_labels_path(args.train_labels)

    df_subset = load_train_subset(
        train_labels_path=train_labels_path,
        limit=args.limit,
        sample_mode=args.sample_mode,
        seed=args.seed,
        include_blank_gt=args.include_blank_gt,
    )

    tasks = [
        (i + 1, row.to_dict())
        for i, (_, row) in enumerate(df_subset.iterrows())
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    partial_path = out_path.with_suffix(".partial.csv")
    summary_path = out_path.with_suffix(".summary.json")

    args_dict = vars(args).copy()

    print("=" * 88)
    print("PARALLEL AUDIT START")
    print("=" * 88)
    print(f"train_labels: {train_labels_path}")
    print(f"rows:         {len(tasks)}")
    print(f"workers:      {args.num_workers}")
    print(f"backend:      {args.backend}")
    print(f"lang:         {args.lang}")
    print(f"out:          {out_path}")
    print(f"partial:      {partial_path}")
    print("=" * 88)

    rows: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=init_worker,
        initargs=(args_dict,),
    ) as executor:
        futures = [
            executor.submit(worker_task, task)
            for task in tasks
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Parallel auditing",
        ):
            row = future.result()
            rows.append(row)

            idx = row.get("idx")
            image_id = row.get("image_id")
            selected_variant = row.get("selected_variant")
            selected_cer = row.get("selected_cer")
            oracle = row.get("oracle_best_cer_variant")
            error = row.get("error", "")

            if error:
                print(
                    f"[{idx}/{len(tasks)}] FAIL image_id={image_id} error={error}",
                    flush=True,
                )
            else:
                print(
                    f"[{idx}/{len(tasks)}] DONE image_id={image_id} "
                    f"selected={selected_variant} selected_cer={selected_cer} "
                    f"oracle={oracle}",
                    flush=True,
                )

            write_checkpoint(rows, partial_path)

    out_df = pd.DataFrame(rows)

    if "idx" in out_df.columns:
        out_df = out_df.sort_values("idx")

    out_df.to_csv(
        out_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = summarize_audit(out_df)

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_summary(summary)

    print("=" * 88)
    print("SAVED")
    print("=" * 88)
    print(f"Audit CSV:    {out_path}")
    print(f"Partial CSV:  {partial_path}")
    print(f"Summary JSON: {summary_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel audit OCR preprocessing variants on train subset.",
    )

    parser.add_argument(
        "--train-labels",
        default=str(ROOT / "data" / "raw" / "train_labels.csv"),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--sample-mode",
        choices=["random", "head", "tail"],
        default="random",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--include-blank-gt",
        action="store_true",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Start with 2 on Mac M1.",
    )

    parser.add_argument(
        "--backend",
        default="paddle",
        choices=["paddle", "tesseract"],
    )

    parser.add_argument(
        "--lang",
        default="vi",
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--psm",
        type=int,
        default=11,
    )

    parser.add_argument(
        "--oem",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--min-token-conf",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--router-engine",
        default="paddle",
        choices=["paddle", "tesseract", "easyocr", "vietocr"],
    )

    parser.add_argument(
        "--target-short-side",
        type=int,
        default=900,
    )

    parser.add_argument(
        "--max-long-side",
        type=int,
        default=1600,
    )

    parser.add_argument(
        "--min-avg-conf",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--min-alnum-chars",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--cer-lower",
        action="store_true",
    )

    parser.add_argument(
        "--out",
        default=str(ROOT / "outputs" / "audits" / "train_audit_parallel.csv"),
    )

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_parallel_audit(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
