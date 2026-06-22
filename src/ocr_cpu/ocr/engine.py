from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2 as cv
import numpy as np


ImageInput = Union[str, Path, np.ndarray]
Box = Tuple[int, int, int, int]  # x, y, w, h


# ---------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------

@dataclass
class OCRLine:
    """
    One OCR text line.

    box format:
        (x, y, w, h)

    conf:
        normalized confidence in range [0, 1].
    """

    text: str
    conf: float
    box: Box

    block_num: int = -1
    par_num: int = -1
    line_num: int = -1


@dataclass
class OCRResult:
    """
    Standard OCR output used by the whole project.

    All OCR engines should eventually return this format.
    This allows us to swap Tesseract / PaddleOCR / EasyOCR later without
    changing the rest of the pipeline.
    """

    text: str = ""
    lines: List[OCRLine] = field(default_factory=list)

    boxes: List[Box] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)

    avg_conf: float = 0.0
    n_boxes: int = 0
    latency_ms: float = 0.0

    backend: str = "unknown"
    lang: str = ""
    psm: int = -1
    oem: int = -1

    image_shape: Optional[Tuple[int, int, int]] = None

    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to serializable dict.
        Useful for audit CSV / JSON.
        """

        d = asdict(self)
        return d


# ---------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------

def _read_bgr_with_pillow(path: Path) -> np.ndarray:
    """
    Fallback image reader for files OpenCV cannot decode.

    Some dataset images have .jpg extension but are actually animated WebP.
    OpenCV may fail on animated WebP, so we use Pillow and take the first frame.
    """

    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError(
            "OpenCV failed to read this image, and Pillow is not installed. "
            "Install Pillow with: pip install pillow"
        ) from e

    try:
        with Image.open(path) as im:
            # For animated WebP/GIF, use the first frame.
            try:
                im.seek(0)
            except EOFError:
                pass

            im = im.convert("RGB")
            arr_rgb = np.array(im)

            return cv.cvtColor(arr_rgb, cv.COLOR_RGB2BGR)

    except Exception as e:
        raise RuntimeError(
            f"OpenCV and Pillow both failed to read image: {path}"
        ) from e


def read_bgr(image: ImageInput) -> np.ndarray:
    """
    Read image as BGR ndarray.

    Accepts:
    - image path
    - numpy ndarray

    This reader is robust to mislabeled animated WebP files with .jpg extension.
    """

    if isinstance(image, (str, Path)):
        path = Path(image)

        if not path.exists():
            raise FileNotFoundError(f"Image path does not exist: {path}")

        img = cv.imread(str(path), cv.IMREAD_COLOR)

        if img is not None:
            return img

        # Fallback for animated WebP or unusual image headers.
        return _read_bgr_with_pillow(path)

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv.cvtColor(image, cv.COLOR_GRAY2BGR)

        if image.ndim == 3 and image.shape[2] == 3:
            return image.copy()

        if image.ndim == 3 and image.shape[2] == 4:
            return cv.cvtColor(image, cv.COLOR_BGRA2BGR)

    raise TypeError(
        "Unsupported image input. Expected path or numpy ndarray."
    )


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV BGR image to RGB.

    pytesseract assumes RGB input for numpy images.
    """

    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def _safe_float(value: Any, default: float = -1.0) -> float:
    """
    Convert OCR confidence to float safely.
    """

    try:
        return float(value)
    except Exception:
        return default


def _normalize_conf(conf: float) -> float:
    """
    Tesseract confidence is usually in [0, 100].
    Convert it to [0, 1].
    """

    if conf < 0:
        return -1.0

    if conf > 1.0:
        return float(conf / 100.0)

    return float(conf)


def _clean_token(text: Any) -> str:
    """
    Basic token clean for OCR output.
    """

    if text is None:
        return ""

    return str(text).strip()


def _union_boxes(boxes: List[Box]) -> Box:
    """
    Compute union box from many token boxes.
    """

    if not boxes:
        return (0, 0, 0, 0)

    xs = [b[0] for b in boxes]
    ys = [b[1] for b in boxes]
    x2s = [b[0] + b[2] for b in boxes]
    y2s = [b[1] + b[3] for b in boxes]

    x1 = int(min(xs))
    y1 = int(min(ys))
    x2 = int(max(x2s))
    y2 = int(max(y2s))

    return (x1, y1, x2 - x1, y2 - y1)


# ---------------------------------------------------------------------
# Base engine
# ---------------------------------------------------------------------

class BaseOCREngine:
    """
    Base OCR engine interface.

    Every backend should implement:
        recognize(image) -> OCRResult
    """

    backend_name: str = "base"

    def recognize(self, image: ImageInput) -> OCRResult:
        raise NotImplementedError


# ---------------------------------------------------------------------
# Tesseract engine
# ---------------------------------------------------------------------

class TesseractOCREngine(BaseOCREngine):
    """
    Lightweight Tesseract OCR wrapper.

    Good for:
    - local CPU debugging
    - fast vertical slice
    - checking whether preprocessing improves OCR

    Not guaranteed to be the strongest final competition engine.
    """

    backend_name = "tesseract"

    def __init__(
        self,
        lang: str = "vie+eng",
        psm: int = 11,
        oem: int = 3,
        timeout: float = 10.0,
        min_token_conf: float = 0.0,
        tessdata_dir: Optional[str] = None,
        tesseract_cmd: Optional[str] = None,
        extra_config: str = "",
    ) -> None:
        """
        Parameters
        ----------
        lang:
            Tesseract language string.
            For Vietnamese + English, use "vie+eng".

        psm:
            Page segmentation mode.
            11 = sparse text, useful for thumbnails / posters.
            6 = assume a uniform block of text.

        oem:
            OCR engine mode.
            3 = default engine mode.

        timeout:
            Max seconds for one OCR call.

        min_token_conf:
            Drop tokens below this normalized confidence.
            0.0 means keep all non-negative confidence tokens.

        tessdata_dir:
            Optional custom tessdata path.

        tesseract_cmd:
            Optional full path to tesseract binary.

        extra_config:
            Extra raw Tesseract config string.
        """

        self.lang = lang
        self.psm = int(psm)
        self.oem = int(oem)
        self.timeout = float(timeout)
        self.min_token_conf = float(min_token_conf)
        self.tessdata_dir = tessdata_dir
        self.extra_config = extra_config

        try:
            import pytesseract
            from pytesseract import Output
        except Exception as e:
            raise ImportError(
                "pytesseract is not installed. Run: pip install pytesseract"
            ) from e

        self.pytesseract = pytesseract
        self.Output = Output

        if tesseract_cmd:
            self.pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def _build_config(self) -> str:
        """
        Build Tesseract config string.
        """

        parts = [
            f"--oem {self.oem}",
            f"--psm {self.psm}",
        ]

        if self.tessdata_dir:
            parts.append(f'--tessdata-dir "{self.tessdata_dir}"')

        if self.extra_config:
            parts.append(self.extra_config)

        return " ".join(parts)

    def available_languages(self) -> List[str]:
        """
        Return available Tesseract languages.
        """

        return list(self.pytesseract.get_languages(config=""))

    def recognize(self, image: ImageInput) -> OCRResult:
        """
        Run OCR on one image and return standardized OCRResult.
        """

        start = time.perf_counter()

        img_bgr = read_bgr(image)
        img_rgb = bgr_to_rgb(img_bgr)

        config = self._build_config()

        try:
            data = self.pytesseract.image_to_data(
                img_rgb,
                lang=self.lang,
                config=config,
                output_type=self.Output.DICT,
                timeout=self.timeout,
            )

            result = self._parse_tesseract_data(
                data=data,
                image_shape=img_bgr.shape,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )

            return result

        except Exception as e:
            return OCRResult(
                text="",
                lines=[],
                boxes=[],
                confidences=[],
                avg_conf=0.0,
                n_boxes=0,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                backend=self.backend_name,
                lang=self.lang,
                psm=self.psm,
                oem=self.oem,
                image_shape=img_bgr.shape,
                error=str(e),
                raw={},
            )

    def _parse_tesseract_data(
        self,
        data: Dict[str, List[Any]],
        image_shape: Tuple[int, int, int],
        latency_ms: float,
    ) -> OCRResult:
        """
        Parse pytesseract image_to_data output.

        Tesseract returns token-level data.
        We group tokens by line.
        """

        n = len(data.get("text", []))

        grouped: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

        token_boxes: List[Box] = []
        token_confs: List[float] = []

        for i in range(n):
            token = _clean_token(data["text"][i])

            if not token:
                continue

            raw_conf = _safe_float(data.get("conf", ["-1"] * n)[i])
            conf = _normalize_conf(raw_conf)

            if conf < 0:
                continue

            if conf < self.min_token_conf:
                continue

            x = int(data.get("left", [0] * n)[i])
            y = int(data.get("top", [0] * n)[i])
            w = int(data.get("width", [0] * n)[i])
            h = int(data.get("height", [0] * n)[i])

            box = (x, y, w, h)

            block_num = int(data.get("block_num", [-1] * n)[i])
            par_num = int(data.get("par_num", [-1] * n)[i])
            line_num = int(data.get("line_num", [-1] * n)[i])

            key = (block_num, par_num, line_num)

            if key not in grouped:
                grouped[key] = {
                    "tokens": [],
                    "confs": [],
                    "boxes": [],
                    "block_num": block_num,
                    "par_num": par_num,
                    "line_num": line_num,
                }

            grouped[key]["tokens"].append(token)
            grouped[key]["confs"].append(conf)
            grouped[key]["boxes"].append(box)

            token_boxes.append(box)
            token_confs.append(conf)

        lines: List[OCRLine] = []

        for key in sorted(grouped.keys()):
            item = grouped[key]

            line_text = " ".join(item["tokens"]).strip()
            line_conf = float(np.mean(item["confs"])) if item["confs"] else 0.0
            line_box = _union_boxes(item["boxes"])

            if not line_text:
                continue

            lines.append(
                OCRLine(
                    text=line_text,
                    conf=line_conf,
                    box=line_box,
                    block_num=item["block_num"],
                    par_num=item["par_num"],
                    line_num=item["line_num"],
                )
            )

        full_text = "\n".join(line.text for line in lines).strip()

        avg_conf = float(np.mean(token_confs)) if token_confs else 0.0

        return OCRResult(
            text=full_text,
            lines=lines,
            boxes=token_boxes,
            confidences=token_confs,
            avg_conf=avg_conf,
            n_boxes=len(token_boxes),
            latency_ms=float(latency_ms),
            backend=self.backend_name,
            lang=self.lang,
            psm=self.psm,
            oem=self.oem,
            image_shape=image_shape,
            error=None,
            raw={
                "num_raw_items": n,
                "config": self._build_config(),
            },
        )



# ---------------------------------------------------------------------
# PaddleOCR engine
# ---------------------------------------------------------------------

class PaddleOCREngine(BaseOCREngine):
    """
    PaddleOCR wrapper with the same output format as TesseractOCREngine.

    This backend is intended for multilingual OCR, especially Vietnamese/Latin
    text in scene images. The wrapper is intentionally defensive because
    PaddleOCR output format differs across versions.
    """

    backend_name = "paddle"

    def __init__(
        self,
        lang: str = "vi",
        device: str = "cpu",
        timeout: float = 30.0,
        min_token_conf: float = 0.0,
        use_angle_cls: bool = False,
        show_log: bool = False,
        enable_mkldnn: Optional[bool] = None,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = False,
        psm: Optional[int] = None,
        oem: Optional[int] = None,
        **extra_kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        lang:
            PaddleOCR language code. For Vietnamese, try "vi".

        device:
            Device string. For this project on Mac/local CPU, use "cpu".

        timeout:
            Kept for interface compatibility. PaddleOCR does not provide a
            simple per-call timeout here, but we store it in the result metadata.

        min_token_conf:
            Drop detected text lines below this confidence.

        use_angle_cls, show_log:
            Compatibility parameters for older PaddleOCR APIs.

        enable_mkldnn, use_doc_orientation_classify, use_doc_unwarping,
        use_textline_orientation:
            Compatibility parameters for newer PaddleOCR APIs.

        psm, oem:
            Ignored. They exist only so the same CLI can pass Tesseract-style
            arguments without breaking the Paddle backend.
        """

        self.lang = lang
        self.device = device
        self.timeout = float(timeout)
        self.min_token_conf = float(min_token_conf)
        self.use_angle_cls = bool(use_angle_cls)
        self.show_log = bool(show_log)
        self.enable_mkldnn = enable_mkldnn
        self.use_doc_orientation_classify = bool(use_doc_orientation_classify)
        self.use_doc_unwarping = bool(use_doc_unwarping)
        self.use_textline_orientation = bool(use_textline_orientation)
        self.extra_kwargs = dict(extra_kwargs)

        try:
            from paddleocr import PaddleOCR
        except Exception as e:
            raise ImportError(
                "PaddleOCR is not installed. Install with: "
                "pip install paddlepaddle paddleocr"
            ) from e

        self.PaddleOCR = PaddleOCR
        self.ocr = self._init_paddleocr()

    def _init_paddleocr(self):
        """
        Initialize PaddleOCR with several compatible argument sets.

        PaddleOCR has changed constructor arguments across versions, so we try
        the newer style first, then fallback to older style.
        """

        attempts: List[Dict[str, Any]] = []

        # Newer PaddleOCR 3.x style seen in recent PP-OCR pipelines.
        v3_kwargs: Dict[str, Any] = {
            "lang": self.lang,
            "use_doc_orientation_classify": self.use_doc_orientation_classify,
            "use_doc_unwarping": self.use_doc_unwarping,
            "use_textline_orientation": self.use_textline_orientation,
        }

        if self.device:
            v3_kwargs["device"] = self.device

        if self.enable_mkldnn is not None:
            v3_kwargs["enable_mkldnn"] = self.enable_mkldnn

        attempts.append(v3_kwargs)

        # Same but without device, for versions that do not accept device.
        v3_no_device = dict(v3_kwargs)
        v3_no_device.pop("device", None)
        attempts.append(v3_no_device)

        # Older PaddleOCR 2.x style.
        attempts.append(
            {
                "lang": self.lang,
                "use_angle_cls": self.use_angle_cls,
                "show_log": self.show_log,
            }
        )

        attempts.append(
            {
                "lang": self.lang,
                "use_angle_cls": self.use_angle_cls,
            }
        )

        # Minimal fallback.
        attempts.append({"lang": self.lang})

        errors: List[str] = []

        for kwargs in attempts:
            merged = dict(kwargs)
            merged.update(self.extra_kwargs)

            try:
                return self.PaddleOCR(**merged)
            except TypeError as e:
                errors.append(f"kwargs={merged} -> {e}")
                continue
            except Exception as e:
                errors.append(f"kwargs={merged} -> {e}")
                continue

        raise RuntimeError(
            "Failed to initialize PaddleOCR. Attempts:\n" + "\n".join(errors)
        )

    def recognize(self, image: ImageInput) -> OCRResult:
        """
        Run PaddleOCR on one image and return standardized OCRResult.
        """

        start = time.perf_counter()
        img_bgr = read_bgr(image)

        try:
            raw = self._run_paddle(img_bgr)
            result = self._parse_paddle_output(
                raw=raw,
                image_shape=img_bgr.shape,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
            return result

        except Exception as e:
            return OCRResult(
                text="",
                lines=[],
                boxes=[],
                confidences=[],
                avg_conf=0.0,
                n_boxes=0,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                backend=self.backend_name,
                lang=self.lang,
                psm=-1,
                oem=-1,
                image_shape=img_bgr.shape,
                error=str(e),
                raw={},
            )

    def _run_paddle(self, img_bgr: np.ndarray) -> Any:
        """
        Run PaddleOCR using whichever API exists in the installed version.
        """

        # Older API. Usually returns: [[ [box, (text, score)], ... ]]
        if hasattr(self.ocr, "ocr"):
            try:
                return self.ocr.ocr(img_bgr, cls=self.use_angle_cls)
            except TypeError:
                return self.ocr.ocr(img_bgr)

        # Newer API. Some versions use predict(input=...).
        if hasattr(self.ocr, "predict"):
            try:
                return self.ocr.predict(input=img_bgr)
            except TypeError:
                return self.ocr.predict(img_bgr)

        raise RuntimeError("Installed PaddleOCR object has no ocr() or predict() method.")

    @staticmethod
    def _box_to_xywh(box: Any) -> Box:
        """
        Convert PaddleOCR polygon/box to (x, y, w, h).
        """

        arr = np.asarray(box, dtype=np.float32)

        if arr.size == 0:
            return (0, 0, 0, 0)

        # Polygon points: [[x1, y1], [x2, y2], ...]
        if arr.ndim == 2 and arr.shape[1] >= 2:
            xs = arr[:, 0]
            ys = arr[:, 1]
            x1, y1 = float(np.min(xs)), float(np.min(ys))
            x2, y2 = float(np.max(xs)), float(np.max(ys))
            return (
                int(round(x1)),
                int(round(y1)),
                int(round(max(0.0, x2 - x1))),
                int(round(max(0.0, y2 - y1))),
            )

        flat = arr.reshape(-1)

        # Common rectangular formats: [x1, y1, x2, y2] or [x, y, w, h]
        if flat.size >= 4:
            x1, y1, a, b = [float(v) for v in flat[:4]]

            # Heuristic: if a/b look like bottom-right, convert x1y1x2y2.
            if a > x1 and b > y1:
                return (
                    int(round(x1)),
                    int(round(y1)),
                    int(round(a - x1)),
                    int(round(b - y1)),
                )

            return (
                int(round(x1)),
                int(round(y1)),
                int(round(max(0.0, a))),
                int(round(max(0.0, b))),
            )

        return (0, 0, 0, 0)

    @staticmethod
    def _looks_like_old_paddle_item(obj: Any) -> bool:
        """
        Check old PaddleOCR item format:
            [box, (text, score)]
        """

        if not isinstance(obj, (list, tuple)) or len(obj) < 2:
            return False

        rec = obj[1]

        if not isinstance(rec, (list, tuple)) or len(rec) < 2:
            return False

        return isinstance(rec[0], str)

    @staticmethod
    def _first_present(obj: Dict[str, Any], keys: List[str]) -> Any:
        """
        Return the first existing non-None value from a dict.

        Do not use `a or b` here because PaddleOCR may return numpy arrays.
        Numpy arrays cannot be evaluated directly as True/False.
        """

        for key in keys:
            if key in obj and obj[key] is not None:
                return obj[key]

        return None

    def _collect_paddle_items(self, obj: Any, items: List[Tuple[Box, str, float]]) -> None:
        """
        Recursively collect (box, text, confidence) from PaddleOCR outputs.
        Supports both old list output and newer dict/result-object output.
        """

        if obj is None:
            return

        # New result objects may expose .json, .res, or to_dict().
        if not isinstance(obj, (dict, list, tuple, str)):
            if hasattr(obj, "json"):
                try:
                    self._collect_paddle_items(getattr(obj, "json"), items)
                    return
                except Exception:
                    pass

            if hasattr(obj, "res"):
                try:
                    self._collect_paddle_items(getattr(obj, "res"), items)
                    return
                except Exception:
                    pass

            if hasattr(obj, "to_dict"):
                try:
                    self._collect_paddle_items(obj.to_dict(), items)
                    return
                except Exception:
                    pass

        # Newer dict-like result.
        if isinstance(obj, dict):
            # Some PaddleOCR 3.x results are {"res": {...}}.
            for key in ["res", "result", "data"]:
                if key in obj:
                    self._collect_paddle_items(obj[key], items)

            texts = self._first_present(
                obj,
                ["rec_texts", "texts", "text", "ocr_texts"],
            )

            scores = self._first_present(
                obj,
                ["rec_scores", "scores", "confs", "confidences"],
            )

            boxes = self._first_present(
                obj,
                ["rec_boxes", "dt_polys", "rec_polys", "boxes", "dt_boxes"],
            )

            if isinstance(texts, (list, tuple)):
                n = len(texts)

                for i in range(n):
                    text = _clean_token(texts[i])

                    if not text:
                        continue

                    score = 1.0

                    if isinstance(scores, (list, tuple)) and i < len(scores):
                        score = _normalize_conf(_safe_float(scores[i], 1.0))

                    if score < 0 or score < self.min_token_conf:
                        continue

                    box = (0, 0, 0, 0)

                    if isinstance(boxes, (list, tuple, np.ndarray)) and i < len(boxes):
                        box = self._box_to_xywh(boxes[i])

                    items.append((box, text, float(score)))

            return

        # Old PaddleOCR output or nested list output.
        if isinstance(obj, (list, tuple)):
            if self._looks_like_old_paddle_item(obj):
                box = self._box_to_xywh(obj[0])
                rec = obj[1]
                text = _clean_token(rec[0])
                score = _normalize_conf(_safe_float(rec[1], 1.0))

                if text and score >= 0 and score >= self.min_token_conf:
                    items.append((box, text, float(score)))

                return

            for child in obj:
                self._collect_paddle_items(child, items)

    def _parse_paddle_output(
        self,
        raw: Any,
        image_shape: Tuple[int, int, int],
        latency_ms: float,
    ) -> OCRResult:
        """
        Parse PaddleOCR raw output into OCRResult.
        """

        items: List[Tuple[Box, str, float]] = []
        self._collect_paddle_items(raw, items)

        # Sort top-to-bottom, then left-to-right.
        items.sort(key=lambda item: (item[0][1], item[0][0]))

        lines: List[OCRLine] = []
        boxes: List[Box] = []
        confidences: List[float] = []

        for idx, (box, text, conf) in enumerate(items):
            lines.append(
                OCRLine(
                    text=text,
                    conf=float(conf),
                    box=box,
                    block_num=0,
                    par_num=0,
                    line_num=idx,
                )
            )
            boxes.append(box)
            confidences.append(float(conf))

        full_text = "\n".join(line.text for line in lines).strip()
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        return OCRResult(
            text=full_text,
            lines=lines,
            boxes=boxes,
            confidences=confidences,
            avg_conf=avg_conf,
            n_boxes=len(boxes),
            latency_ms=float(latency_ms),
            backend=self.backend_name,
            lang=self.lang,
            psm=-1,
            oem=-1,
            image_shape=image_shape,
            error=None,
            raw={
                "raw_type": type(raw).__name__,
                "num_items": len(items),
                "device": self.device,
                "min_token_conf": self.min_token_conf,
            },
        )

# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------

def create_ocr_engine(
    backend: str = "tesseract",
    **kwargs: Any,
) -> BaseOCREngine:
    """
    Create OCR engine by backend name.

    Current supported backends:
    - tesseract
    - paddle

    Future backends can be added:
    - easyocr
    - vietocr
    """

    backend = backend.lower().strip()

    if backend == "tesseract":
        return TesseractOCREngine(**kwargs)

    if backend in {"paddle", "paddleocr"}:
        return PaddleOCREngine(**kwargs)

    raise ValueError(
        f"Unsupported OCR backend: {backend}. "
        f"Currently supported: ['tesseract', 'paddle']"
    )


def run_ocr(
    image: ImageInput,
    backend: str = "tesseract",
    **kwargs: Any,
) -> OCRResult:
    """
    Convenience function for quick one-off OCR.

    For batch processing, prefer:
        engine = create_ocr_engine(...)
        result = engine.recognize(image)
    """

    engine = create_ocr_engine(backend=backend, **kwargs)
    return engine.recognize(image)


# ---------------------------------------------------------------------
# Debug drawing
# ---------------------------------------------------------------------

def draw_ocr_boxes(
    image: ImageInput,
    result: OCRResult,
    min_conf: float = 0.0,
    show_text: bool = True,
) -> np.ndarray:
    """
    Draw OCR boxes on image for UI/debug.

    Returns BGR image.
    """

    img = read_bgr(image)

    for line in result.lines:
        if line.conf < min_conf:
            continue

        x, y, w, h = line.box

        cv.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 180, 0),
            2,
        )

        if show_text:
            label = f"{line.conf:.2f}: {line.text[:32]}"

            cv.putText(
                img,
                label,
                (x, max(0, y - 6)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 180, 0),
                1,
                cv.LINE_AA,
            )

    return img


__all__ = [
    "OCRLine",
    "OCRResult",
    "BaseOCREngine",
    "TesseractOCREngine",
    "PaddleOCREngine",
    "create_ocr_engine",
    "run_ocr",
    "draw_ocr_boxes",
]
