from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

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

try:
    from ocr_cpu.preprocess.router import (
        classify_image_quality,
        apply_preprocess_by_decision,
    )
except Exception as e:
    st.error(
        "Cannot import preprocessing router.\n\n"
        "Expected functions:\n"
        "- classify_image_quality\n"
        "- apply_preprocess_by_decision\n\n"
        "Please check:\n"
        "src/ocr_cpu/preprocess/router.py"
    )
    st.exception(e)
    st.stop()

try:
    from ocr_cpu.ocr.engine import (
        OCRResult,
        create_ocr_engine,
        draw_ocr_boxes,
    )
except Exception as e:
    st.error(
        "Cannot import OCR engine.\n\n"
        "Expected file:\n"
        "src/ocr_cpu/ocr/engine.py"
    )
    st.exception(e)
    st.stop()


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []

    return sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    )


def guess_default_image_dir() -> str:
    """
    Guess useful default image directory for this project.

    Supports the user's current nested dataset structure:
    data/raw/test_images/test_images/images/
    """

    candidates = [
        ROOT / "data" / "raw" / "test_images" / "test_images" / "images",
        ROOT / "data" / "raw" / "test_images" / "images",
        ROOT / "data" / "raw" / "test_images",
        ROOT / "data" / "raw" / "train_images" / "train_images" / "train_images",
        ROOT / "data" / "raw" / "train_images" / "train_images",
        ROOT / "data" / "raw" / "train_images",
    ]

    for p in candidates:
        if p.exists() and len(list_images(p)) > 0:
            return str(p)

    return str(ROOT / "data" / "raw" / "test_images")


@st.cache_data(show_spinner=False)
def read_image_bgr(path_str: str) -> np.ndarray:
    img = cv.imread(path_str, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path_str}")
    return img


def read_uploaded_bgr(uploaded_file) -> Optional[np.ndarray]:
    if uploaded_file is None:
        return None

    data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    img = cv.imdecode(data, cv.IMREAD_COLOR)

    if img is None:
        return None

    return img


def summarize_image(img: np.ndarray) -> Dict[str, object]:
    h, w = img.shape[:2]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    return {
        "shape": str(img.shape),
        "width": w,
        "height": h,
        "mean": round(float(np.mean(gray)), 3),
        "std": round(float(np.std(gray)), 3),
        "min": int(np.min(gray)),
        "max": int(np.max(gray)),
    }


def save_debug_image(
    image_id: str,
    variant_name: str,
    img: np.ndarray,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_image_id = Path(image_id).stem
    out_path = out_dir / f"{safe_image_id}__{variant_name}.png"

    ok = cv.imwrite(str(out_path), img)
    if not ok:
        raise RuntimeError(f"Failed to save image: {out_path}")

    return out_path


def decision_to_dataframe(decision) -> pd.DataFrame:
    d = asdict(decision)
    rows = []

    for k, v in d.items():
        rows.append(
            {
                "field": k,
                "value": v,
            }
        )

    return pd.DataFrame(rows)


def params_summary(decision) -> pd.DataFrame:
    fields = [
        "scale",
        "use_clahe",
        "use_median",
        "use_gaussian",
        "use_sharpen",
        "use_hard_fallback",
        "hard_threshold_mode",
        "primary_variant",
        "reason",
    ]

    d = asdict(decision)
    return pd.DataFrame(
        [
            {
                "param": k,
                "value": d.get(k),
            }
            for k in fields
        ]
    )


# ---------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_cached_ocr_engine(
    backend: str,
    lang: str,
    psm: int,
    oem: int,
    timeout: float,
    min_token_conf: float,
):
    """
    Cache OCR engine so Streamlit does not recreate it on every rerun.
    """

    return create_ocr_engine(
        backend=backend,
        lang=lang,
        psm=psm,
        oem=oem,
        timeout=timeout,
        min_token_conf=min_token_conf,
    )


def count_alnum(text: str) -> int:
    return sum(ch.isalnum() for ch in str(text))


def is_bad_ocr_result(
    result: OCRResult,
    min_chars: int = 4,
    min_conf: float = 0.35,
    min_boxes: int = 1,
) -> bool:
    """
    Lightweight OCR quality gate.

    This is intentionally conservative.
    It only decides whether a result is obviously weak.
    """

    if result is None:
        return True

    if result.error:
        return True

    text = str(result.text or "").strip()

    if not text:
        return True

    if result.n_boxes < min_boxes:
        return True

    n_alnum = count_alnum(text)

    if n_alnum < min_chars:
        return True

    # If confidence is low but text is long, keep it for review.
    if result.avg_conf < min_conf and n_alnum < 20:
        return True

    return False


def score_ocr_result(result: OCRResult) -> float:
    """
    Simple heuristic score for comparing OCR variants.

    Higher is better.

    Score uses:
    - avg_conf
    - amount of readable text
    - number of detected boxes

    This is not final evaluation.
    It is just for UI/debug selection.
    """

    if result is None or result.error:
        return -1.0

    text = str(result.text or "").strip()
    n_alnum = count_alnum(text)

    if n_alnum == 0:
        return 0.0

    conf_score = float(result.avg_conf)
    length_score = min(n_alnum / 80.0, 1.0)
    box_score = min(result.n_boxes / 12.0, 1.0)

    score = (
        0.65 * conf_score
        + 0.25 * length_score
        + 0.10 * box_score
    )

    if is_bad_ocr_result(result):
        score *= 0.65

    return float(score)


def select_final_ocr_result(
    results: Dict[str, OCRResult],
    primary_variant: str,
) -> tuple[str, OCRResult, str]:
    """
    Select final OCR result among variants.

    Strategy:
    1. If primary variant is not bad, keep it.
    2. Otherwise, choose variant with highest heuristic score.
    """

    if not results:
        raise ValueError("No OCR results to select from.")

    if primary_variant in results:
        primary_result = results[primary_variant]

        if not is_bad_ocr_result(primary_result):
            return (
                primary_variant,
                primary_result,
                "primary_variant_is_good",
            )

    best_name = max(
        results.keys(),
        key=lambda name: score_ocr_result(results[name]),
    )

    return (
        best_name,
        results[best_name],
        "primary_variant_bad_choose_best_score",
    )


def ocr_results_to_dataframe(results: Dict[str, OCRResult]) -> pd.DataFrame:
    rows = []

    for name, r in results.items():
        text = str(r.text or "").replace("\n", " / ")
        preview = text[:160]

        rows.append(
            {
                "variant": name,
                "bad?": is_bad_ocr_result(r),
                "score": round(score_ocr_result(r), 4),
                "avg_conf": round(float(r.avg_conf), 4),
                "n_boxes": int(r.n_boxes),
                "latency_ms": round(float(r.latency_ms), 2),
                "n_chars": len(str(r.text or "")),
                "n_alnum": count_alnum(str(r.text or "")),
                "error": r.error,
                "text_preview": preview,
            }
        )

    return pd.DataFrame(rows)


def make_ocr_state_key(
    image_id: str,
    decision,
    backend: str,
    lang: str,
    psm: int,
    oem: int,
) -> str:
    payload = {
        "image_id": image_id,
        "decision": asdict(decision),
        "backend": backend,
        "lang": lang,
        "psm": psm,
        "oem": oem,
    }

    return json.dumps(payload, sort_keys=True, default=str)


def run_ocr_on_variants(
    variants: Dict[str, np.ndarray],
    backend: str,
    lang: str,
    psm: int,
    oem: int,
    timeout: float,
    min_token_conf: float,
) -> Dict[str, OCRResult]:
    engine = get_cached_ocr_engine(
        backend=backend,
        lang=lang,
        psm=psm,
        oem=oem,
        timeout=timeout,
        min_token_conf=min_token_conf,
    )

    results: Dict[str, OCRResult] = {}

    for name, img in variants.items():
        results[name] = engine.recognize(img)

    return results


# ---------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="OCR Preprocess Debug UI",
    layout="wide",
)

st.title("OCR Preprocess Debug UI")
st.caption(
    "Compare original image, preprocessing variants, router decision, "
    "OCR result, confidence, latency, and selected final text."
)


# ---------------------------------------------------------------------
# Sidebar: input controls
# ---------------------------------------------------------------------

st.sidebar.header("Input")

mode = st.sidebar.radio(
    "Input mode",
    ["Folder", "Upload one image"],
    index=0,
)

image_id = "uploaded_image"
bgr: Optional[np.ndarray] = None
selected_path: Optional[Path] = None

if mode == "Folder":
    folder_str = st.sidebar.text_input(
        "Image folder",
        value=guess_default_image_dir(),
    )

    folder = Path(folder_str).expanduser()
    image_paths = list_images(folder)

    if not image_paths:
        st.warning(f"No images found in: {folder}")
        st.stop()

    idx = st.sidebar.slider(
        "Image index",
        min_value=0,
        max_value=len(image_paths) - 1,
        value=0,
        step=1,
    )

    selected_path = image_paths[idx]
    image_id = selected_path.name
    bgr = read_image_bgr(str(selected_path))

    st.sidebar.write(f"Total images: **{len(image_paths)}**")
    st.sidebar.write(f"Selected: `{selected_path.name}`")

else:
    uploaded = st.sidebar.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
    )

    bgr = read_uploaded_bgr(uploaded)

    if bgr is None:
        st.info("Please upload an image.")
        st.stop()

    image_id = uploaded.name if uploaded is not None else "uploaded_image"


# ---------------------------------------------------------------------
# Sidebar: router controls
# ---------------------------------------------------------------------

st.sidebar.header("Router config")

router_engine_profile = st.sidebar.selectbox(
    "Router engine profile",
    ["paddle", "easyocr", "vietocr", "tesseract"],
    index=3,
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


# ---------------------------------------------------------------------
# Router decision
# ---------------------------------------------------------------------

decision = classify_image_quality(
    bgr,
    target_short_side=target_short_side,
    max_long_side=max_long_side,
    engine=router_engine_profile,
)


st.sidebar.header("Manual preprocess override")

with st.sidebar.expander("Override selected methods", expanded=False):
    decision.use_clahe = st.checkbox(
        "use_clahe",
        value=bool(decision.use_clahe),
    )

    decision.use_median = st.checkbox(
        "use_median",
        value=bool(decision.use_median),
    )

    decision.use_gaussian = st.checkbox(
        "use_gaussian",
        value=bool(decision.use_gaussian),
    )

    decision.use_sharpen = st.checkbox(
        "use_sharpen",
        value=bool(decision.use_sharpen),
    )

    decision.use_hard_fallback = st.checkbox(
        "use_hard_fallback",
        value=bool(decision.use_hard_fallback),
    )

    decision.hard_threshold_mode = st.selectbox(
        "hard_threshold_mode",
        ["adaptive", "otsu"],
        index=0 if decision.hard_threshold_mode == "adaptive" else 1,
    )

    decision.scale = st.slider(
        "scale",
        min_value=0.5,
        max_value=2.5,
        value=float(decision.scale),
        step=0.05,
    )


variants = apply_preprocess_by_decision(
    bgr,
    decision=decision,
    engine=router_engine_profile,
)


variant_names = list(variants.keys())

if decision.primary_variant in variant_names:
    default_variant_idx = variant_names.index(decision.primary_variant)
else:
    default_variant_idx = 0

selected_variant = st.sidebar.selectbox(
    "Variant to inspect",
    variant_names,
    index=default_variant_idx,
)

selected_img = variants[selected_variant]


# ---------------------------------------------------------------------
# Sidebar: OCR controls
# ---------------------------------------------------------------------

st.sidebar.header("OCR config")

enable_ocr = st.sidebar.checkbox(
    "Enable OCR debug",
    value=True,
)

ocr_backend = st.sidebar.selectbox(
    "OCR backend",
    ["tesseract"],
    index=0,
    disabled=not enable_ocr,
)

# Your local available languages currently look like:
# ['eng', 'osd', 'snum']
# So default to eng. Change to vie+eng after installing Vietnamese tessdata.
ocr_lang = st.sidebar.text_input(
    "Tesseract lang",
    value="eng",
    disabled=not enable_ocr,
)

ocr_psm = st.sidebar.selectbox(
    "Tesseract PSM",
    [11, 6, 12, 3, 4, 7],
    index=0,
    disabled=not enable_ocr,
)

ocr_oem = st.sidebar.selectbox(
    "Tesseract OEM",
    [3, 1],
    index=0,
    disabled=not enable_ocr,
)

ocr_timeout = st.sidebar.slider(
    "OCR timeout / image",
    min_value=1.0,
    max_value=30.0,
    value=10.0,
    step=1.0,
    disabled=not enable_ocr,
)

min_token_conf = st.sidebar.slider(
    "min_token_conf",
    min_value=0.0,
    max_value=0.9,
    value=0.0,
    step=0.05,
    disabled=not enable_ocr,
)

current_ocr_key = make_ocr_state_key(
    image_id=image_id,
    decision=decision,
    backend=ocr_backend,
    lang=ocr_lang,
    psm=int(ocr_psm),
    oem=int(ocr_oem),
)

if st.session_state.get("ocr_state_key") != current_ocr_key:
    st.session_state["ocr_state_key"] = current_ocr_key
    st.session_state.pop("ocr_results", None)

run_ocr_button = st.sidebar.button(
    "Run OCR on all variants",
    disabled=not enable_ocr,
)


if run_ocr_button:
    with st.spinner("Running OCR on raw_resized / soft_enhanced / hard_fallback..."):
        st.session_state["ocr_results"] = run_ocr_on_variants(
            variants=variants,
            backend=ocr_backend,
            lang=ocr_lang,
            psm=int(ocr_psm),
            oem=int(ocr_oem),
            timeout=float(ocr_timeout),
            min_token_conf=float(min_token_conf),
        )


# ---------------------------------------------------------------------
# Main layout: image comparison
# ---------------------------------------------------------------------

top_left, top_right = st.columns([1, 1])

with top_left:
    st.subheader("Original")
    st.image(
        bgr,
        channels="BGR",
        width="stretch",
        caption=f"Original: {image_id}",
    )

with top_right:
    st.subheader(f"Preprocessed: {selected_variant}")
    st.image(
        selected_img,
        channels="BGR",
        width="stretch",
        caption=f"Variant: {selected_variant}",
    )


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Decision summary",
        "All variants",
        "OCR debug",
        "Image stats",
        "Save debug",
    ]
)


with tab1:
    st.subheader("Chosen preprocessing route")

    c1, c2, c3 = st.columns(3)

    c1.metric("Primary variant", decision.primary_variant)
    c2.metric("Reason", decision.reason)
    c3.metric("Scale", round(float(decision.scale), 3))

    st.write("Current preprocessing params")
    st.dataframe(
        params_summary(decision),
        use_container_width=True,
        hide_index=True,
    )

    st.write("Full router decision")
    st.dataframe(
        decision_to_dataframe(decision),
        use_container_width=True,
        hide_index=True,
    )


with tab2:
    st.subheader("All preprocessing variants")

    cols = st.columns(len(variants))

    for col, (name, img) in zip(cols, variants.items()):
        with col:
            st.image(
                img,
                channels="BGR",
                width="stretch",
                caption=name,
            )


with tab3:
    st.subheader("OCR debug")

    if not enable_ocr:
        st.info("Enable OCR debug in the sidebar.")
    else:
        results: Optional[Dict[str, OCRResult]] = st.session_state.get("ocr_results")

        if not results:
            st.info("Click **Run OCR on all variants** in the sidebar.")
        else:
            df = ocr_results_to_dataframe(results)

            st.write("OCR summary table")
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )

            final_variant, final_result, final_reason = select_final_ocr_result(
                results=results,
                primary_variant=decision.primary_variant,
            )

            st.divider()

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Final variant", final_variant)
            c2.metric("Final avg_conf", round(float(final_result.avg_conf), 4))
            c3.metric("Final n_boxes", int(final_result.n_boxes))
            c4.metric("Final latency ms", round(float(final_result.latency_ms), 2))

            st.write(f"Selection reason: `{final_reason}`")

            st.text_area(
                "Selected final OCR text",
                value=final_result.text,
                height=160,
            )

            st.divider()

            st.subheader("OCR text by variant")

            for name, r in results.items():
                with st.expander(
                    f"{name} | conf={r.avg_conf:.3f} | boxes={r.n_boxes} | latency={r.latency_ms:.1f} ms",
                    expanded=(name == final_variant),
                ):
                    if r.error:
                        st.error(r.error)

                    st.text_area(
                        f"OCR text: {name}",
                        value=r.text,
                        height=150,
                        key=f"ocr_text_{name}",
                    )

                    line_rows = []

                    for line in r.lines:
                        line_rows.append(
                            {
                                "text": line.text,
                                "conf": round(float(line.conf), 4),
                                "box": line.box,
                                "block": line.block_num,
                                "par": line.par_num,
                                "line": line.line_num,
                            }
                        )

                    if line_rows:
                        st.dataframe(
                            pd.DataFrame(line_rows),
                            use_container_width=True,
                            hide_index=True,
                        )

            st.divider()

            st.subheader("OCR box overlay")

            overlay_variant = st.selectbox(
                "Variant for box overlay",
                list(results.keys()),
                index=list(results.keys()).index(final_variant),
            )

            overlay_img = draw_ocr_boxes(
                variants[overlay_variant],
                results[overlay_variant],
                min_conf=0.0,
                show_text=True,
            )

            st.image(
                overlay_img,
                channels="BGR",
                width="stretch",
                caption=f"OCR boxes: {overlay_variant}",
            )

            st.divider()

            st.subheader("Export current OCR audit row")

            audit_rows = []

            for name, r in results.items():
                audit_rows.append(
                    {
                        "image_id": image_id,
                        "router_reason": decision.reason,
                        "primary_variant": decision.primary_variant,
                        "variant": name,
                        "is_bad": is_bad_ocr_result(r),
                        "score": score_ocr_result(r),
                        "avg_conf": r.avg_conf,
                        "n_boxes": r.n_boxes,
                        "latency_ms": r.latency_ms,
                        "text": r.text,
                        "error": r.error,
                        "final_variant": final_variant,
                        "final_text": final_result.text,
                        "final_reason": final_reason,
                    }
                )

            audit_df = pd.DataFrame(audit_rows)

            st.download_button(
                label="Download current OCR audit CSV",
                data=audit_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{Path(image_id).stem}_ocr_audit.csv",
                mime="text/csv",
            )


with tab4:
    st.subheader("Image statistics")

    left, right = st.columns(2)

    with left:
        st.write("Original stats")
        st.json(summarize_image(bgr))

    with right:
        st.write(f"Selected variant stats: {selected_variant}")
        st.json(summarize_image(selected_img))


with tab5:
    st.subheader("Save debug image")

    out_dir_str = st.text_input(
        "Output debug folder",
        value=str(ROOT / "outputs" / "debug_images"),
    )

    out_dir = Path(out_dir_str).expanduser()

    if st.button("Save selected variant"):
        out_path = save_debug_image(
            image_id=image_id,
            variant_name=selected_variant,
            img=selected_img,
            out_dir=out_dir,
        )

        st.success(f"Saved: {out_path}")

    if st.button("Save all variants"):
        saved = []

        for name, img in variants.items():
            out_path = save_debug_image(
                image_id=image_id,
                variant_name=name,
                img=img,
                out_dir=out_dir,
            )
            saved.append(str(out_path))

        st.success(f"Saved {len(saved)} files.")
        st.write(saved)
