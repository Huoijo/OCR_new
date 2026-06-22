from __future__ import annotations

import re
import sys
import time
import logging
import traceback
from datetime import datetime
import unicodedata
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2 as cv
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# Project import
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_cpu.preprocess.router import (
    classify_image_quality,
    apply_preprocess_by_decision,
)
from ocr_cpu.ocr.engine import (
    OCRResult,
    create_ocr_engine,
    draw_ocr_boxes,
    read_bgr,
)
from ocr_cpu.ocr.quality import (
    OCRQualityConfig,
    choose_best_ocr_result,
    evaluate_ocr_result,
)


# ---------------------------------------------------------------------
# CER
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
# Data / path
# ---------------------------------------------------------------------

def resolve_image_path(image_id: str, split: str = "train") -> Path:
    try:
        from ocr_cpu.utils.paths import get_image_path

        return get_image_path(image_id, split=split).resolve()
    except Exception:
        pass

    raw_dir = ROOT / "data" / "raw"

    if split == "train":
        search_root = raw_dir / "train_images"
    else:
        search_root = raw_dir / "test_images"

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
        raise FileNotFoundError(f"Cannot find image_id={image_id} in {search_root}")

    return sorted(candidates)[0].resolve()


def load_train_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"image_id", "ocr_text"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


def sample_train_df(
    df: pd.DataFrame,
    limit: int,
    sample_mode: str,
    seed: int,
    include_blank_gt: bool,
) -> pd.DataFrame:
    if not include_blank_gt:
        mask = df["ocr_text"].fillna("").astype(str).str.strip() != ""
        df = df[mask].copy()

    if sample_mode == "head":
        out = df.head(limit)
    elif sample_mode == "tail":
        out = df.tail(limit)
    elif sample_mode == "random":
        out = df.sample(n=min(limit, len(df)), random_state=seed)
    else:
        raise ValueError(f"Unknown sample_mode: {sample_mode}")

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------
# Engine cache
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_cached_engine(
    backend: str,
    lang: str,
    device: str,
    psm: int,
    oem: int,
    timeout: float,
    min_token_conf: float,
):
    if backend == "paddle":
        return create_ocr_engine(
            backend="paddle",
            lang=lang,
            device=device,
            timeout=timeout,
            min_token_conf=min_token_conf,
        )

    return create_ocr_engine(
        backend="tesseract",
        lang=lang,
        psm=psm,
        oem=oem,
        timeout=timeout,
        min_token_conf=min_token_conf,
    )


# ---------------------------------------------------------------------
# Audit one image
# ---------------------------------------------------------------------

VARIANT_PREFIX = {
    "raw_resized": "raw",
    "soft_enhanced": "soft",
    "hard_fallback": "hard",
}


def audit_one_image(
    image_id: str,
    gt_text: str,
    engine,
    quality_config: OCRQualityConfig,
    router_engine: str,
    target_short_side: int,
    max_long_side: int,
    cer_lower: bool,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    start = time.perf_counter()

    image_path = resolve_image_path(image_id, split="train")

    decision = classify_image_quality(
        image_path,
        target_short_side=target_short_side,
        max_long_side=max_long_side,
        engine=router_engine,
    )

    variants = apply_preprocess_by_decision(
        image_path,
        decision=decision,
        engine=router_engine,
    )

    results: Dict[str, OCRResult] = {}

    for variant_name, img in variants.items():
        results[variant_name] = engine.recognize(img)

    selectable_results = dict(results)

    if not bool(decision.use_hard_fallback):
        selectable_results.pop("hard_fallback", None)

    selection = choose_best_ocr_result(
        results=selectable_results,
        primary_variant=decision.primary_variant,
        config=quality_config,
    )

    row: Dict[str, Any] = {
        "image_id": image_id,
        "gt_ocr_text": gt_text,
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
        "selected_cer": cer(selection.selected_result.text, gt_text, lower=cer_lower),
        "selection_reason": selection.selection_reason,
    }

    variant_cers = {}

    for variant_name, result in results.items():
        prefix = VARIANT_PREFIX.get(variant_name, variant_name)

        report = evaluate_ocr_result(
            result,
            config=quality_config,
            variant_name=variant_name,
        )

        v_cer = cer(result.text, gt_text, lower=cer_lower)
        variant_cers[variant_name] = v_cer

        row[f"{prefix}_text"] = result.text
        row[f"{prefix}_conf"] = result.avg_conf
        row[f"{prefix}_boxes"] = result.n_boxes
        row[f"{prefix}_latency_ms"] = result.latency_ms
        row[f"{prefix}_cer"] = v_cer
        row[f"{prefix}_is_bad"] = report.is_bad
        row[f"{prefix}_score"] = report.score
        row[f"{prefix}_reasons"] = "|".join(report.reasons)
        row[f"{prefix}_error"] = result.error or ""

    oracle_best = min(variant_cers.keys(), key=lambda x: variant_cers[x])

    row["oracle_best_cer_variant"] = oracle_best
    row["oracle_best_cer"] = variant_cers[oracle_best]
    row["selected_is_oracle_best"] = selection.selected_variant == oracle_best

    if decision.primary_variant in variant_cers:
        row["primary_cer"] = variant_cers[decision.primary_variant]
    else:
        row["primary_cer"] = None

    row["latency_ms"] = sum(float(r.latency_ms or 0.0) for r in results.values())
    row["total_wall_ms"] = (time.perf_counter() - start) * 1000.0
    row["error"] = ""

    detail = {
        "image_id": image_id,
        "gt_text": gt_text,
        "image_path": image_path,
        "decision": decision,
        "results": results,
        "selection": selection,
        "row": row,
    }

    return row, detail


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def summarize_df(df: pd.DataFrame) -> Dict[str, Any]:
    summary = {}

    for col in [
        "raw_cer",
        "soft_cer",
        "hard_cer",
        "selected_cer",
        "primary_cer",
        "oracle_best_cer",
    ]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()

            if len(s) > 0:
                summary[col] = {
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "p90": float(s.quantile(0.90)),
                }

    for col in [
        "primary_variant",
        "selected_variant",
        "oracle_best_cer_variant",
        "router_reason",
    ]:
        if col in df.columns:
            summary[col] = df[col].fillna("").astype(str).value_counts().head(10).to_dict()

    if "selected_is_oracle_best" in df.columns:
        summary["selected_is_oracle_best_rate"] = float(
            df["selected_is_oracle_best"].fillna(False).mean()
        )

    if "latency_ms" in df.columns:
        s = pd.to_numeric(df["latency_ms"], errors="coerce").dropna()
        summary["latency_ms"] = {
            "mean": float(s.mean()) if len(s) else None,
            "median": float(s.median()) if len(s) else None,
            "total": float(s.sum()) if len(s) else None,
        }

    return summary


def make_label(row: pd.Series) -> str:
    image_id = row.get("image_id", "")
    selected = row.get("selected_variant", "")
    oracle = row.get("oracle_best_cer_variant", "")
    cer_value = row.get("selected_cer", None)

    try:
        cer_str = f"{float(cer_value):.3f}"
    except Exception:
        cer_str = "NA"

    return f"{image_id} | selected={selected} | CER={cer_str} | oracle={oracle}"



# ---------------------------------------------------------------------
# Run logging / checkpoint helpers
# ---------------------------------------------------------------------

def build_run_id(
    backend: str,
    lang: str,
    limit: int,
    seed: int,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_backend = str(backend).replace("/", "_")
    safe_lang = str(lang).replace("/", "_").replace("+", "plus")

    return f"batch_audit_ui_{timestamp}_{safe_backend}_{safe_lang}_n{limit}_seed{seed}"


def setup_run_logger(run_id: str):
    """
    Create a logger that writes both to terminal and a log file.

    The terminal log helps while Streamlit is running.
    The file log survives if the browser state disappears.
    """

    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{run_id}.log"

    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        log_path,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger, log_path


def write_checkpoint_csv(
    rows: List[Dict[str, Any]],
    path: Path,
) -> None:
    """
    Write partial audit CSV after every image.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_float_for_log(value: Any, default: float = -1.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def log_audit_row(
    logger,
    idx: int,
    total: int,
    row: Dict[str, Any],
) -> None:
    """
    Log one finished audit row.
    """

    image_id = row.get("image_id", "")
    selected_variant = row.get("selected_variant", "")
    selected_cer = safe_float_for_log(row.get("selected_cer"))
    oracle_variant = row.get("oracle_best_cer_variant", "")
    oracle_cer = safe_float_for_log(row.get("oracle_best_cer"))
    latency_ms = safe_float_for_log(row.get("latency_ms"))
    error = row.get("error", "")

    if error:
        logger.info(
            "[%d/%d] IMAGE_DONE_WITH_ERROR image_id=%s error=%s",
            idx,
            total,
            image_id,
            error,
        )
    else:
        logger.info(
            "[%d/%d] IMAGE_DONE image_id=%s selected=%s selected_cer=%.4f "
            "oracle=%s oracle_cer=%.4f latency_ms=%.2f",
            idx,
            total,
            image_id,
            selected_variant,
            selected_cer,
            oracle_variant,
            oracle_cer,
            latency_ms,
        )


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Batch OCR Audit UI",
    layout="wide",
)

st.title("Batch OCR Audit UI")
st.caption(
    "Run OCR audit on a small train subset, compare raw/soft/hard variants, "
    "and inspect each image visually."
)


# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------

st.sidebar.header("Dataset")

train_labels_path = st.sidebar.text_input(
    "train_labels.csv",
    value=str(ROOT / "data" / "raw" / "train_labels.csv"),
)

limit = st.sidebar.slider(
    "Batch size",
    min_value=1,
    max_value=50,
    value=5,
    step=1,
)

sample_mode = st.sidebar.selectbox(
    "Sample mode",
    ["random", "head", "tail"],
    index=0,
)

seed = st.sidebar.number_input(
    "Seed",
    min_value=0,
    max_value=999999,
    value=42,
    step=1,
)

include_blank_gt = st.sidebar.checkbox(
    "Include blank ground-truth OCR",
    value=False,
)

cer_lower = st.sidebar.checkbox(
    "Lowercase before CER",
    value=False,
)


st.sidebar.header("OCR engine")

backend = st.sidebar.selectbox(
    "Backend",
    ["paddle", "tesseract"],
    index=0,
)

if backend == "paddle":
    lang_default = "vi"
else:
    lang_default = "eng"

lang = st.sidebar.text_input(
    "Language",
    value=lang_default,
)

device = st.sidebar.text_input(
    "Device",
    value="cpu",
)

psm = st.sidebar.selectbox(
    "Tesseract PSM",
    [11, 6, 12, 3, 4, 7],
    index=0,
)

oem = st.sidebar.selectbox(
    "Tesseract OEM",
    [3, 1],
    index=0,
)

timeout = st.sidebar.slider(
    "Timeout per image",
    min_value=1.0,
    max_value=60.0,
    value=30.0,
    step=1.0,
)

min_token_conf = st.sidebar.slider(
    "min_token_conf",
    min_value=0.0,
    max_value=0.9,
    value=0.0,
    step=0.05,
)


st.sidebar.header("Router")

router_engine = st.sidebar.selectbox(
    "Router engine profile",
    ["paddle", "tesseract", "easyocr", "vietocr"],
    index=0,
)

target_short_side = st.sidebar.slider(
    "target_short_side",
    min_value=500,
    max_value=1600,
    value=900,
    step=50,
)

max_long_side = st.sidebar.slider(
    "max_long_side",
    min_value=800,
    max_value=2400,
    value=1600,
    step=100,
)


st.sidebar.header("Quality gate")

min_avg_conf = st.sidebar.slider(
    "min_avg_conf",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.05,
)

min_alnum_chars = st.sidebar.slider(
    "min_alnum_chars",
    min_value=0,
    max_value=30,
    value=4,
    step=1,
)


run_button = st.sidebar.button("Run batch audit")


# ---------------------------------------------------------------------
# Run batch
# ---------------------------------------------------------------------

if run_button:
    run_id = build_run_id(
        backend=backend,
        lang=lang,
        limit=int(limit),
        seed=int(seed),
    )

    logger, log_path = setup_run_logger(run_id)

    audits_dir = ROOT / "outputs" / "audits"
    audits_dir.mkdir(parents=True, exist_ok=True)

    partial_csv_path = audits_dir / f"{run_id}.partial.csv"
    final_csv_path = audits_dir / f"{run_id}.final.csv"

    logger.info("=" * 88)
    logger.info("RUN_START run_id=%s", run_id)
    logger.info("backend=%s lang=%s device=%s router_engine=%s", backend, lang, device, router_engine)
    logger.info("limit=%s sample_mode=%s seed=%s", limit, sample_mode, seed)
    logger.info("target_short_side=%s max_long_side=%s", target_short_side, max_long_side)
    logger.info("partial_csv=%s", partial_csv_path)
    logger.info("final_csv=%s", final_csv_path)
    logger.info("log_path=%s", log_path)
    logger.info("=" * 88)

    train_labels = load_train_labels(Path(train_labels_path))
    subset = sample_train_df(
        train_labels,
        limit=int(limit),
        sample_mode=sample_mode,
        seed=int(seed),
        include_blank_gt=include_blank_gt,
    )

    logger.info("Loaded train_labels rows=%d selected_subset=%d", len(train_labels), len(subset))

    engine = get_cached_engine(
        backend=backend,
        lang=lang,
        device=device,
        psm=int(psm),
        oem=int(oem),
        timeout=float(timeout),
        min_token_conf=float(min_token_conf),
    )

    logger.info("Engine initialized backend=%s lang=%s", backend, lang)

    quality_config = OCRQualityConfig(
        min_avg_conf=float(min_avg_conf),
        min_alnum_chars=int(min_alnum_chars),
        try_raw_when_primary_bad=True,
    )

    rows = []
    details = {}

    progress = st.progress(0)
    status = st.empty()

    total = len(subset)

    for i, row in subset.iterrows():
        idx = i + 1
        image_id = str(row["image_id"])
        gt_text = row.get("ocr_text", "")

        status.write(f"Running {idx}/{total}: `{image_id}`")
        logger.info("[%d/%d] IMAGE_START image_id=%s", idx, total, image_id)

        image_start = time.perf_counter()

        try:
            audit_row, detail = audit_one_image(
                image_id=image_id,
                gt_text=gt_text,
                engine=engine,
                quality_config=quality_config,
                router_engine=router_engine,
                target_short_side=int(target_short_side),
                max_long_side=int(max_long_side),
                cer_lower=cer_lower,
            )

            audit_row["run_id"] = run_id
            audit_row["idx"] = idx

            image_wall_ms = (time.perf_counter() - image_start) * 1000.0
            audit_row["image_wall_ms"] = image_wall_ms

            log_audit_row(
                logger=logger,
                idx=idx,
                total=total,
                row=audit_row,
            )

        except Exception as e:
            tb = traceback.format_exc()

            logger.error(
                "[%d/%d] IMAGE_FAIL image_id=%s error=%s",
                idx,
                total,
                image_id,
                repr(e),
            )
            logger.error("TRACEBACK\n%s", tb)

            audit_row = {
                "run_id": run_id,
                "idx": idx,
                "image_id": image_id,
                "gt_ocr_text": gt_text,
                "error": repr(e),
                "traceback": tb,
                "image_wall_ms": (time.perf_counter() - image_start) * 1000.0,
            }

            detail = {
                "image_id": image_id,
                "gt_text": gt_text,
                "error": repr(e),
                "traceback": tb,
            }

        rows.append(audit_row)
        details[image_id] = detail

        # Write checkpoint after every image.
        write_checkpoint_csv(rows, partial_csv_path)
        logger.info(
            "[%d/%d] CHECKPOINT_WRITTEN rows=%d path=%s",
            idx,
            total,
            len(rows),
            partial_csv_path,
        )

        progress.progress(idx / total)

    audit_df = pd.DataFrame(rows)

    audit_df.to_csv(
        final_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info("RUN_DONE rows=%d final_csv=%s", len(audit_df), final_csv_path)
    logger.info("RUN_DONE log_path=%s", log_path)
    logger.info("=" * 88)

    st.session_state["batch_audit_df"] = audit_df
    st.session_state["batch_audit_details"] = details
    st.session_state["batch_audit_run_info"] = {
        "run_id": run_id,
        "log_path": str(log_path),
        "partial_csv_path": str(partial_csv_path),
        "final_csv_path": str(final_csv_path),
    }

    status.write("Done.")


# ---------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------

if "batch_audit_df" not in st.session_state:
    st.info("Choose settings in the sidebar, then click **Run batch audit**.")
    st.stop()

audit_df: pd.DataFrame = st.session_state["batch_audit_df"]
details: Dict[str, Any] = st.session_state["batch_audit_details"]

run_info = st.session_state.get("batch_audit_run_info")

if run_info:
    st.success(
        "Batch audit finished. Files were saved to outputs/logs and outputs/audits."
    )
    st.code(
        "\n".join(
            [
                f"run_id: {run_info.get('run_id')}",
                f"log_path: {run_info.get('log_path')}",
                f"partial_csv_path: {run_info.get('partial_csv_path')}",
                f"final_csv_path: {run_info.get('final_csv_path')}",
            ]
        )
    )

summary = summarize_df(audit_df)

st.header("Batch summary")

m1, m2, m3, m4 = st.columns(4)

if "selected_cer" in audit_df.columns:
    m1.metric(
        "selected CER mean",
        round(float(pd.to_numeric(audit_df["selected_cer"], errors="coerce").mean()), 4),
    )

if "oracle_best_cer" in audit_df.columns:
    m2.metric(
        "oracle CER mean",
        round(float(pd.to_numeric(audit_df["oracle_best_cer"], errors="coerce").mean()), 4),
    )

if "selected_is_oracle_best" in audit_df.columns:
    m3.metric(
        "selected = oracle rate",
        round(float(audit_df["selected_is_oracle_best"].fillna(False).mean()), 4),
    )

if "latency_ms" in audit_df.columns:
    m4.metric(
        "avg latency ms",
        round(float(pd.to_numeric(audit_df["latency_ms"], errors="coerce").mean()), 2),
    )

with st.expander("Full summary JSON", expanded=False):
    st.json(summary)


st.header("Audit table")

display_cols = [
    "image_id",
    "router_reason",
    "primary_variant",
    "raw_cer",
    "soft_cer",
    "hard_cer",
    "selected_variant",
    "selected_cer",
    "oracle_best_cer_variant",
    "oracle_best_cer",
    "selected_is_oracle_best",
    "latency_ms",
    "error",
]

existing_cols = [c for c in display_cols if c in audit_df.columns]

sort_worst = st.checkbox(
    "Sort by worst selected CER",
    value=True,
)

table_df = audit_df.copy()

if sort_worst and "selected_cer" in table_df.columns:
    table_df["_selected_cer_sort"] = pd.to_numeric(
        table_df["selected_cer"],
        errors="coerce",
    )
    table_df = table_df.sort_values(
        "_selected_cer_sort",
        ascending=False,
    ).drop(columns=["_selected_cer_sort"])

st.dataframe(
    table_df[existing_cols],
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download audit CSV",
    data=audit_df.to_csv(index=False).encode("utf-8-sig"),
    file_name="batch_audit_ui.csv",
    mime="text/csv",
)


# ---------------------------------------------------------------------
# Select one image for visual inspection
# ---------------------------------------------------------------------

st.header("Visual inspection")

labels = []

for _, row in table_df.iterrows():
    labels.append(make_label(row))

selected_label = st.selectbox(
    "Select image",
    labels,
    index=0,
)

selected_image_id = selected_label.split(" | ")[0]
detail = details.get(selected_image_id)

if not detail:
    st.error("No detail found for selected image.")
    st.stop()

if detail.get("error"):
    st.error(detail["error"])
    st.stop()

image_path = Path(detail["image_path"])
gt_text = detail["gt_text"]
decision = detail["decision"]
results: Dict[str, OCRResult] = detail["results"]
selection = detail["selection"]

original = read_bgr(image_path)
variants = apply_preprocess_by_decision(
    original,
    decision=decision,
    engine=router_engine,
)

row = detail["row"]

st.subheader(f"Image: {selected_image_id}")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Primary", row.get("primary_variant"))
c2.metric("Selected", row.get("selected_variant"))
c3.metric("Selected CER", round(float(row.get("selected_cer", 0.0)), 4))
c4.metric("Oracle", row.get("oracle_best_cer_variant"))

st.write("Router reason:", f"`{row.get('router_reason')}`")

img_cols = st.columns(4)

with img_cols[0]:
    st.image(
        original,
        channels="BGR",
        width="stretch",
        caption="original",
    )

for col, variant_name in zip(
    img_cols[1:],
    ["raw_resized", "soft_enhanced", "hard_fallback"],
):
    with col:
        if variant_name in variants:
            st.image(
                variants[variant_name],
                channels="BGR",
                width="stretch",
                caption=variant_name,
            )


st.subheader("Ground truth vs selected OCR")

left, right = st.columns(2)

with left:
    st.text_area(
        "Ground truth OCR text",
        value=str(gt_text),
        height=180,
    )

with right:
    st.text_area(
        f"Selected OCR text: {selection.selected_variant}",
        value=selection.selected_result.text,
        height=180,
    )


st.subheader("OCR by variant")

for variant_name in ["raw_resized", "soft_enhanced", "hard_fallback"]:
    if variant_name not in results:
        continue

    result = results[variant_name]
    prefix = VARIANT_PREFIX[variant_name]

    with st.expander(
        f"{variant_name} | CER={row.get(prefix + '_cer')} | conf={result.avg_conf:.3f} | boxes={result.n_boxes}",
        expanded=(variant_name == selection.selected_variant),
    ):
        if result.error:
            st.error(result.error)

        st.text_area(
            f"OCR text: {variant_name}",
            value=result.text,
            height=140,
            key=f"text_{selected_image_id}_{variant_name}",
        )

        line_rows = []

        for line in result.lines:
            line_rows.append(
                {
                    "text": line.text,
                    "conf": round(float(line.conf), 4),
                    "box": line.box,
                    "line_num": line.line_num,
                }
            )

        if line_rows:
            st.dataframe(
                pd.DataFrame(line_rows),
                use_container_width=True,
                hide_index=True,
            )


st.subheader("OCR box overlay")

overlay_variant = st.selectbox(
    "Overlay variant",
    ["raw_resized", "soft_enhanced", "hard_fallback"],
    index=["raw_resized", "soft_enhanced", "hard_fallback"].index(selection.selected_variant)
    if selection.selected_variant in ["raw_resized", "soft_enhanced", "hard_fallback"]
    else 0,
)

if overlay_variant in variants and overlay_variant in results:
    overlay = draw_ocr_boxes(
        variants[overlay_variant],
        results[overlay_variant],
        min_conf=0.0,
        show_text=True,
    )

    st.image(
        overlay,
        channels="BGR",
        width="stretch",
        caption=f"overlay: {overlay_variant}",
    )


st.subheader("Router details")

st.json(asdict(decision))
