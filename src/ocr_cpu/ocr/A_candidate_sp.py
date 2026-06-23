from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .engine import OCRResult
from .quality import OCRSelection


VARIANT_TO_PREFIX = {
    "raw_resized": "raw",
    "soft_enhanced": "soft",
    "hard_fallback": "hard",
}


def result_to_lines_json(result: Optional[OCRResult]) -> str:
    """
    Serialize line-level OCR output for downstream product candidate generation.

    Output example:
    [
      {"text": "...", "conf": 0.91, "box": [x, y, w, h], "line_num": 0}
    ]
    """

    if result is None:
        return "[]"

    rows = []

    for line in result.lines or []:
        rows.append(
            {
                "text": line.text or "",
                "conf": float(line.conf or 0.0),
                "box": list(line.box) if line.box is not None else [0, 0, 0, 0],
                "block_num": int(line.block_num),
                "par_num": int(line.par_num),
                "line_num": int(line.line_num),
            }
        )

    return json.dumps(rows, ensure_ascii=False)


def result_to_flat_fields(
    result: Optional[OCRResult],
    prefix: str,
) -> Dict[str, Any]:
    """
    Flatten one OCRResult into CSV-friendly columns.
    """

    if result is None:
        return {
            f"{prefix}_text": "",
            f"{prefix}_conf": 0.0,
            f"{prefix}_boxes": 0,
            f"{prefix}_lines_json": "[]",
            f"{prefix}_error": "missing_result",
        }

    return {
        f"{prefix}_text": result.text or "",
        f"{prefix}_conf": float(result.avg_conf or 0.0),
        f"{prefix}_boxes": int(result.n_boxes or 0),
        f"{prefix}_lines_json": result_to_lines_json(result),
        f"{prefix}_error": result.error or "",
    }


def selection_to_product_input_fields(
    selection: OCRSelection,
    image_id: Optional[str] = None,
    include_variants: bool = True,
) -> Dict[str, Any]:
    """
    Convert OCRSelection into the standard row format expected by product module.

    Main aliases:
    - ocr_text / ocr_conf / ocr_boxes / ocr_lines_json
      always point to selected_result.

    image_id:
        Optional image identifier. The product module
        (product/A_candidate.py) reads row["image_id"], so pass it here to
        produce a self-contained row. Defaults to "" when not provided.
    """

    selected = selection.selected_result

    out: Dict[str, Any] = {
        "image_id": "" if image_id is None else str(image_id),
        "selected_variant": selection.selected_variant,
        "selection_reason": selection.selection_reason,

        "ocr_text": selected.text or "",
        "ocr_conf": float(selected.avg_conf or 0.0),
        "ocr_boxes": int(selected.n_boxes or 0),
        "ocr_lines_json": result_to_lines_json(selected),

        "selected_text": selected.text or "",
        "selected_conf": float(selected.avg_conf or 0.0),
        "selected_boxes": int(selected.n_boxes or 0),
        "selected_lines_json": result_to_lines_json(selected),
    }

    if include_variants:
        for variant_name, result in selection.results.items():
            prefix = VARIANT_TO_PREFIX.get(variant_name, variant_name)
            out.update(result_to_flat_fields(result, prefix=prefix))

    return out