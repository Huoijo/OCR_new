# Product stage → `submission.csv` — Hướng dẫn bàn giao

Tài liệu này mô tả phần **product_name (Cell 1 → Cell 5) + tạo submission**, để bàn giao
cho người làm **OCR + preprocessing + engine**. Mục tiêu: bạn cắm output OCR của mình vào
là chạy ra `submission.csv` đúng format thi đấu.

> Quy ước: giải thích tiếng Việt, technical term giữ English.

---

## 0. Ranh giới trách nhiệm (ai làm gì)

```
┌─────────────────────────────┐        ┌──────────────────────────────────────┐
│  BẠN (OCR side)             │        │  PHẦN NÀY (product → submission)       │
│  preprocess/router.py       │ ─────► │  Cell 1  A_candidate.py                │
│  ocr/engine.py (PaddleOCR)  │ OCR    │  Cell 2  B_gazeetteer.py               │
│  ocr/quality.py             │ Select │  Cell 3  C_fuzzy_linker.py             │
│  → OCRSelection             │        │  Cell 4  D_confidence_gate.py + rules  │
└─────────────────────────────┘        │  Cell 5  E_predict.py → submission.csv │
                                        │  rules.py (registry luật)              │
                                        └──────────────────────────────────────┘
```

- **Phần OCR (của bạn) KHÔNG bị sửa.** Product stage chỉ *tiêu thụ* output của nó.
- **Điểm bàn giao = `OCRSelection`** (output của `run_ocr_with_quality_gate`). Xem §2.

---

## 1. Chạy nhanh (end-to-end)

```bash
# (Windows PowerShell) bật UTF-8 cho console 1 lần / phiên
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"

# Chạy thử 5 ảnh đầu  → outputs/submission.csv
python scripts/make_submission.py --limit 5

# Chạy đầy đủ public test (2006 ảnh; lần đầu ~18 phút vì OCR, sau đó cache)
python scripts/make_submission.py

# Phase 2 private (tự thêm cột brand_name)
python scripts/make_submission.py --split phase2
```

`scripts/make_submission.py` đã wire sẵn **OCR (router + engine + quality) → Cell 1-5 → submission**.
Bạn chỉ cần đảm bảo phần OCR import được (xem §6 cấu trúc).

### Các cờ (flags)

| Flag | Ý nghĩa |
|---|---|
| `--split test\|phase2` | chọn tập (public / private). Mặc định `test`. |
| `--limit N` | chỉ xử lý N ảnh đầu (smoke test). |
| `--rebuild-ocr` | bỏ OCR cache, OCR lại từ đầu. |
| `--rebuild-gazetteer` | build lại gazetteer từ train_labels.csv. |
| `--no-rules` | **ablation**: tắt tích hợp `rules.py` (đo đóng góp của luật). |
| `--no-clean-ocr` | tắt lọc ký tự lạ ở cột `ocr_text` (xem §5). |
| `--output PATH` | đổi đường dẫn file ra. |
| `--backend paddle\|tesseract` | backend OCR. Mặc định `paddle`. |

---

## 2. Hợp đồng dữ liệu OCR → product (QUAN TRỌNG NHẤT khi tích hợp)

Product stage nhận **một trong hai** dạng input. Cả hai đều do bridge
`src/ocr_cpu/ocr/A_candidate_sp.py` dựng từ `OCRSelection`.

### Cách A — truyền thẳng `OCRSelection` (đang dùng trong make_submission)

```python
from ocr_cpu.ocr.quality import run_ocr_with_quality_gate, OCRQualityConfig
from ocr_cpu.product.A_candidate import (
    selections_to_ocr_dataframe,        # -> ocr_df (cột ocr_text + variant)
    selections_to_candidate_dataframe,  # -> candidate_df (Cell 1)
)

selection = run_ocr_with_quality_gate(variants, decision, engine, OCRQualityConfig())
selections = [(image_id, selection), ...]     # list (image_id, OCRSelection)

ocr_df   = selections_to_ocr_dataframe(selections)
cand_df  = selections_to_candidate_dataframe(selections, filler_tokens=gz.filler_tokens)
```

### Cách B — truyền `ocr_df` (nếu bạn đã OCR sẵn và lưu ra bảng)

```python
from ocr_cpu.product.A_candidate import generate_candidate_dataframe
cand_df = generate_candidate_dataframe(ocr_df, filler_tokens=gz.filler_tokens)
```

### Schema `ocr_df` mà product stage cần (do `selection_to_product_input_fields` sinh)

| Cột | Kiểu | Bắt buộc | Ý nghĩa |
|---|---|---|---|
| `image_id` | str | ✅ | khóa ảnh |
| `ocr_text` | str | ✅ | text OCR (variant đã chọn). Dùng cho cột submission + rule context |
| `selected_text` | str | ✅ | = ocr_text của variant `selected` |
| `selected_conf` | float | ✅ | confidence trung bình |
| `selected_boxes` | int | ✅ | số box |
| `selected_lines_json` | str (JSON) | ✅ | **line records**, xem dưới |
| `raw_* / soft_* / hard_*` | … | tùy chọn | nếu có nhiều variant (tăng chất lượng candidate) |

**`*_lines_json`** = chuỗi JSON, list các dòng OCR:
```json
[{"text": "ĐỒ HỘP HẠ LONG", "conf": 0.91, "box": [x, y, w, h],
  "block_num": 0, "par_num": 0, "line_num": 0}]
```
> `box` là `[x, y, w, h]`. `line_num/par_num/block_num` để gom dòng. Thiếu lines_json thì
> candidate generation chỉ còn dựa text phẳng (vẫn chạy nhưng yếu hơn về position/ngram).

→ **Để tích hợp OCR của bạn:** chỉ cần sinh được `OCRSelection` (đã có sẵn) là cách A chạy luôn.
Không cần đổi gì ở product stage.

---

## 3. Luồng 5 Cell (Cell 1 → 5)

```python
from ocr_cpu.product.B_gazeetteer import build_product_gazetteer, ProductGazetteer
from ocr_cpu.product.C_fuzzy_linker import link_candidates_with_gazetteer
from ocr_cpu.product.D_confidence_gate import apply_confidence_gating, GateConfig
from ocr_cpu.product.E_predict import build_final_submission, FinalPredictorConfig

# Cell 2 (build/load 1 lần; cũng cấp filler_tokens cho Cell 1)
gz = ProductGazetteer.from_json("outputs/gazetteer/product_gazetteer.json")
gz.enrich_with_rule_aliases()                 # nạp alias từ rules.py (nếu load từ cache)

# image_text_map: BẮT BUỘC để rule rescue chạy cho ảnh OCR nhiễu (xem §4)
image_text_map = dict(zip(ocr_df["image_id"], ocr_df["ocr_text"]))

# Cell 3 → 5
linked     = link_candidates_with_gazetteer(cand_df, gz, image_text_map=image_text_map)
decisions  = apply_confidence_gating(linked, GateConfig(), image_text_map=image_text_map)
submission = build_final_submission(ocr_df, decisions, gazetteer=gz,
                                    config=FinalPredictorConfig(),
                                    sample_submission_path="the-2nd-ura-hackathon/sample_submission.csv")
```

| Cell | File | Vai trò |
|---|---|---|
| 1 | `A_candidate.py` | sinh candidate phrases từ OCR lines; lọc junk; tái dùng `normalize_*` |
| 2 | `B_gazeetteer.py` | gazetteer từ train_labels.csv (6 index folded), filler tokens, alias |
| 3 | `C_fuzzy_linker.py` | link candidate ↔ canonical (rapidfuzz, exact/folded), reject short |
| 4 | `D_confidence_gate.py` | gom theo image_id, quyết định fill/blank (**precision-first**), rule override |
| 5 | `E_predict.py` | dựng product_name cuối + lắp submission đúng schema |

---

## 4. `rules.py` — registry luật tập trung (đọc nếu cần chỉnh)

`src/ocr_cpu/product/rules.py` là **một chỗ duy nhất** chứa luật product_name (thay if/else
rải rác). Tích hợp vào A/C/D/E qua optional args, backward-compatible.

- **Resolver**: sắp theo `priority > score > len(canonical)` (KHÔNG chọn candidate dài nhất).
- **CP context-sensitive**: `CP` là product thật trong train (vd "CP vị hôi hôi...") nhưng cũng
  là viết tắt pháp lý ("Công ty CP") / OCR nhiễu ("Cập"→"Cp"). Luật chỉ giữ `CP` khi
  `is_cp_product_context()` đúng; ngược lại reject. **Không blacklist `CP` toàn cục.**
- **Rule override (Cell 4)**: luật mạnh (`priority >= 90`) thắng fuzzy → fix các case OCR nhiễu.
- **Rescue (Cell 4)**: ảnh OCR nhiễu tới mức **không sinh candidate nào** vẫn emit được nếu luật
  mạnh fire — **nhưng chỉ khi caller truyền `image_text_map` đủ tất cả image_id** (make_submission
  đã làm). Đây là lý do §3 phải có `image_text_map`.
- **Canonical đã đối chiếu `train_labels.csv`** (chọn dạng có thật trong train).

Thêm luật mới: thêm 1 dict `{name, priority, canonical, patterns}` vào nhóm tương ứng
(`PATE_RULES`, `NESTLE_NAN_RULES`, `MISC_EXACT_RULES`, …). Patterns match trên text đã chuẩn hóa
(`normalize_rule_text`: lowercase + bỏ dấu + chỉ a-z0-9 + gộp space). Luôn chọn `canonical` là
**chuỗi có thật trong train_labels.csv**.

---

## 5. Làm sạch cột `ocr_text` (CER) — whitelist

OCR đôi khi ảo ra ký tự lạ (CJK `兴`, box-drawing, emoji…) làm **tăng CER**. Hàm
`E_predict.clean_ocr_text_for_output` lọc bằng **whitelist (không phải blacklist)**:

- **NFC trước** (gộp dấu thanh decomposed; nếu strip thô sẽ phá tiếng Việt).
- Giữ: ASCII + Latin/Việt (`00C0-024F`, `1E00-1EFF`) + combining marks + dấu typographic + `₫`.
- Xóa **mọi thứ khác** → ký tự lạ *chưa từng thấy* (emoji mới, Cyrillic, Thái…) tự loại, không
  cần sửa code.
- Blank nếu sau lọc không còn từ ≥3 chữ cái (ảnh OCR rác thuần).

Bật mặc định trong make_submission; tắt bằng `--no-clean-ocr`.
⚠️ ~15/4892 dòng GT train có CJK **hợp lệ** → nên **đo CER trên train** trước khi chốt.

---

## 6. Cấu trúc thư mục & file quan trọng

```
src/ocr_cpu/
  preprocess/router.py        # [OCR side] phân loại + sinh variant
  ocr/engine.py               # [OCR side] PaddleOCR vi, CPU
  ocr/quality.py              # [OCR side] quality gate -> OCRSelection
  ocr/A_candidate_sp.py       # BRIDGE OCRSelection -> ocr_df / candidate row
  product/
    A_candidate.py  B_gazeetteer.py  C_fuzzy_linker.py
    D_confidence_gate.py  E_predict.py  rules.py        # ← phần này
scripts/
  make_submission.py          # ENTRY end-to-end
  time_one_image.py           # đo thời gian OCR vs product
  test_rules.py               # 11 unit test luật
  test_rules_e2e.py           # 5 test A→C→D→E thật
  test_gazetteer.py           # 13 test gazetteer
the-2nd-ura-hackathon/
  train_labels.csv            # nguồn build gazetteer (image_id, ocr_text, product_name)
  test.csv, sample_submission.csv, test_images/
  phase2_dataset/.../ (private_test.csv, sample_submission_private.csv, images/)
outputs/
  gazetteer/product_gazetteer.json   # cache gazetteer
  ocr_cache_test.csv                 # cache OCR (tạo sau lần chạy đầu)
  submission.csv                     # KẾT QUẢ
```

---

## 7. Output schema (submission.csv)

| Phase | Cột | Blank |
|---|---|---|
| **public** (`--split test`) | `image_id, ocr_text, product_name` | ô rỗng `""` (không NaN) |
| **phase2** (`--split phase2`) | `image_id, ocr_text, brand_name, product_name` | rỗng `""` |

- Số dòng = đúng `sample_submission` (theo thứ tự sample). Không thiếu/thừa/trùng image_id.
- `ocr_text` flatten về **1 dòng** (GT train là single-line).
- `product_name` **precision-first**: không chắc → để blank (blank là output hợp lệ; scoring
  `0.6·F1_product + 0.4·(1−CER)`).

---

## 8. Caching & thời gian (đo thật trên 1 CPU)

| Việc | Thời gian |
|---|---|
| OCR (router+engine+quality) | **~32 s/ảnh** (~98% tổng) |
| Product stage Cell 1→5 / ảnh | **~0.57 s** (Cell 3 link ~0.53s) |
| Engine init / gazetteer load | ~7 s / ~0.01 s (1 lần) |
| Full public (2006 ảnh) lần đầu | **~18 phút** (chủ yếu OCR) |

→ **OCR cache là thiết yếu**: lần đầu OCR hết và ghi `outputs/ocr_cache_test.csv`; các lần sau
chỉnh luật/gate chỉ tốn ~vài giây (không OCR lại). Dùng `--rebuild-ocr` khi đổi OCR/engine.

Đo lại 1 ảnh:
```bash
python scripts/time_one_image.py --image img_0019.jpg --split train --repeat 3
```

---

## 9. Kiểm thử (chạy trước khi tin kết quả)

```bash
python scripts/test_rules.py        # 11/11 luật (spec cases)
python scripts/test_rules_e2e.py    # 5/5  A→C→D→E thật
python scripts/test_gazetteer.py    # 13/13 gazetteer (build + folded lookup)
```

---

## 10. Troubleshooting (đã gặp thật)

- **`UnicodeEncodeError` khi in `đ`** (console Windows cp1252): set
  `$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"`. Không phải bug code.
- **`cv2.imread ... WebP ... has_animation`**: ảnh test là *animated webp* giả đuôi `.jpg`.
  Chỉ là **cảnh báo stderr**, engine có loader fallback nên vẫn OCR được (không chí mạng).
- **Ảnh test nằm sâu** `test_images/test_images/images/`: `make_submission` rglob tự tìm theo
  `image_id`, không cần chỉnh.
- **Tên file Cell 2 là `B_gazeetteer.py`** (2 chữ 'e') — import đúng tên này; `E_predict` có
  fallback `B_gazetteer` nhưng file thật là `B_gazeetteer`.

---

## 11. Ràng buộc bất biến (đừng phá khi chỉnh)

- KHÔNG sửa `preprocess/` và `ocr/engine.py` (đã chốt benchmark).
- KHÔNG đổi dataclass `ProductCandidate` — cell sau ghi vào `matched`/`debug_reason` hoặc thêm cột.
- Tái dùng `normalize_light`/`normalize_dedupe_key`/`tokenize_text` (A) và `fold_*` (B); không tự
  định nghĩa lại. `rules.py` tự chứa `normalize_rule_text` riêng để tránh circular import.
- `canonical` trong rule **phải** là chuỗi có thật trong `train_labels.csv`.
```
