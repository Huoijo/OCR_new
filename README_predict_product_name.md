# Predict `product_name` — Pipeline hướng dẫn chi tiết (Cell 1 → Cell 6)

> Tài liệu **hướng dẫn thiết kế** cho stage suy luận `product_name`, viết để **đồng bộ với
> source code hiện tại** (`src/ocr_cpu/product/A_candidate.py`,
> `src/ocr_cpu/ocr/A_candidate_sp.py`). Đây là **guide, chưa phải code**.
>
> **Ràng buộc đã chốt:** preprocessing (`preprocess/router.py`) và OCR engine
> (`ocr/engine.py`, PaddleOCR `vi` + `enable_mkldnn=False`) đã benchmark ở mức ổn —
> **KHÔNG chỉnh** hai phần này. Mọi nâng cấp nằm ở tầng `product/` (Cell 1→6).

---

## 0. Bức tranh tổng thể & "tính liền mạch"

Toàn bộ stage product là một chuỗi 6 cell, mỗi cell **nhận output cell trước, thêm
thông tin, đẩy sang cell sau** — không cell nào đọc lại CSV trung gian hay OCR lại ảnh:

```
OCRSelection  (engine + quality, ĐÃ XONG, không sửa)
   │  selections_to_candidate_dataframe(...)        ← cầu nối đã code
   ▼
Cell 1  Candidate Generation     → candidate DataFrame (1 row / candidate)   [ĐÃ CODE: A_candidate.py]
   ▼
Cell 2  Gazetteer Builder        → ProductGazetteer (memory brand/product)   [ĐÃ CODE: B_gazeetteer.py]
   ▼
Cell 3  Fuzzy Linker             → gắn match_score / matched_entry vào candidate   [GUIDE §3 — chưa code]
   ▼
Cell 4  Confidence Gating        → quyết định blank vs fill (precision-first), 1 quyết định / image   [ĐÃ CODE: D_confidence_gate.py]
   ▼
Cell 5  Normalization + Final Predictor → product_name cuối (Title Case, giữ dấu, nối " + ")   [GUIDE §5 — chưa code]
   ▼
Cell 6  Validation / Ablation    → F1_product + (1−CER), bật/tắt từng bước để đo đóng góp
```

### Hợp đồng dữ liệu xuyên suốt (đọc kỹ — đây là "sợi chỉ liền mạch")

Ba thứ phải **dùng chung một định nghĩa** ở mọi cell, nếu lệch là cả pipeline lệch:

1. **Hàm normalize** — mọi text (candidate lẫn gazetteer) đều đi qua **cùng** các hàm
   trong `product/A_candidate.py`:
   - `normalize_light(text)` — NFC, bỏ emoji/control, gộp space, thống nhất dash/quote.
   - `normalize_dedupe_key(text)` — `normalize_light` + lowercase + bỏ ký tự ngoài `\w`/chữ
     Việt → **key để so khớp**. **Candidate và gazetteer PHẢI cùng dùng key này.**
   - `tokenize_text(text)` — `TOKEN_PATTERN` (đã hỗ trợ chữ Việt có dấu) → token set.
2. **Khóa join** — `image_id` (đã có trong candidate row nhờ `selection_to_product_input_fields(..., image_id=...)`).
3. **Các field "chừa sẵn" trong `ProductCandidate`** để cell sau ghi vào, KHÔNG tạo schema mới:
   - `matched: bool` (mặc định `False`) → Cell 3 set `True` khi link được.
   - `debug_reason: Optional[str]` → mọi cell ghi lý do (giữ/loại/blank) để Cell 6 ablation.
   - `normalized_key`, `tokenized`, `source`, `ocr_conf`, `structure_score`,
     `variant_agreement_score`, `position_score` → feature có sẵn cho Cell 3/4 chấm điểm.

> Nguyên tắc: **không cell nào tự ý OCR lại, không tự định nghĩa lại normalize, không đổi
> schema `ProductCandidate`.** Chỉ thêm cột/đổ giá trị vào field đã chừa.

---

## 1. Cell 1 — Candidate Generation  *(đã code — đây là input của Cell 2)*

**File:** `src/ocr_cpu/product/A_candidate.py`
**Entry chính:** `selections_to_candidate_dataframe(selections, filler_tokens=None, ...)`
hoặc `generate_candidate_dataframe(ocr_df, ...)`.

### 1.1 Cell 1 sinh ra cái gì
Một **DataFrame, mỗi row là một "ứng viên" (candidate)** cho `product_name` của một ảnh.
Các cột quan trọng (chính là field của dataclass `ProductCandidate` + 3 cột phụ):

| Cột | Ý nghĩa | Ai dùng |
|---|---|---|
| `image_id` | khóa ảnh | join mọi cell |
| `variant` | `selected` / `raw` / `soft` / `hard` | Cell 4 (đồng thuận variant) |
| `source` | `line` / `whole_line` / `segment` / `ngram` | Cell 3/4 (ưu tiên span) |
| `clean_text` | text ứng viên đã `normalize_light` + strip filler | hiển thị / fuzzy |
| `tokenized` | list token | **Cell 2/3 token-set match** |
| `normalized_key` | **key chuẩn hóa để so khớp** | **Cell 3 link với gazetteer** |
| `ocr_conf` | confidence OCR của span | Cell 4 gating |
| `structure_score` | điểm "trông giống tên SP" | Cell 4 gating |
| `variant_agreement_score`, `variant_support_count` | mức đồng thuận giữa variant | Cell 4 |
| `source_score`, `source_priority` | trọng số theo nguồn | Cell 3/4 ranking |
| `position_score`, `bbox_coords` | vị trí span trên ảnh | Cell 4 (bonus tiêu đề/giữa ảnh) |
| `matched`, `debug_reason` | **chừa trống** cho Cell 3+ ghi vào | Cell 3/4/6 |

### 1.2 Đặc tính cần nhớ (đã quan sát trên `img_0006`)
- Với ảnh **1 sản phẩm, line không có separator, 1 variant** → candidate ≈ line OCR
  nguyên văn (vd `ĐỒ HỘP HẠ LONG`). Generator **chỉ "nở" ngram/segment khi line có dấu
  phân tách** (`+ | / : ;`). ⇒ **Brand thường là sub-span của candidate dài hơn.**
- ⇒ **Hệ quả ràng buộc cho Cell 2/3:** vì brand hay nằm *bên trong* candidate, gazetteer
  phải cho phép **partial / token-set match**, không chỉ exact full-string.

### 1.3 Điểm nối Cell 1 ↔ Cell 2 (quan trọng)
- `generate_candidate_dataframe(..., filler_tokens=...)` **nhận filler list ở Cell 1**, nhưng
  **filler list lại được Cell 2 dựng**. ⇒ Vòng lặp: **Cell 2 build filler → feed ngược vào
  tham số `filler_tokens` của Cell 1** (qua `load_filler_tokens_from_txt(path)` đã có sẵn).
  Lần chạy thật: build gazetteer + filler (Cell 2) **một lần**, rồi mới chạy Cell 1 cho toàn
  bộ ảnh với `filler_tokens` đó.
- Candidate `normalized_key` và gazetteer key phải **cùng** `normalize_dedupe_key` → Cell 3
  mới so khớp công bằng.

---

## 2. Cell 2 — Gazetteer Builder  *(trọng tâm tài liệu này)*

### 2.0 Gazetteer là gì & vì sao cần
"Gazetteer" = **bộ nhớ chuẩn hóa các tên product/brand đã biết** + biến thể OCR + filler +
prior tần suất. Cell 3 sẽ **link candidate (text OCR đọc được) → một entry chuẩn** trong
gazetteer rồi lấy đúng chuỗi `product_name` canonical để xuất.

Vì sao đây là cách đúng cho bài này:
- Dataset xoay quanh **một tập hữu hạn product/brand viral** (đồ hộp Hạ Long, Pate Cột Đèn
  Hải Phòng, Highlands Coffee…). Test gần như **trùng brand với train**.
- Scoring **precision-first** (`0.6·F1_product + 0.4·(1−CER)`, blank hợp lệ) ⇒ thà map về
  một tên chuẩn đã biết, còn không chắc thì blank — gazetteer cho phép kiểm soát điều đó.

### 2.1 Nguồn dữ liệu (chốt: dùng chính `train_labels.csv`)
**File:** `the-2nd-ura-hackathon/train_labels.csv` — 3 cột `image_id, ocr_text, product_name`.

Tách train thành **3 class** (đúng định nghĩa bài toán):
| Class | Điều kiện | Dùng để |
|---|---|---|
| **blank** | `ocr_text` rỗng | bỏ qua khi build gazetteer |
| **text-no-product** | `ocr_text` có, `product_name` rỗng | **mine filler/negative** (text KHÔNG phải product) |
| **text-with-product** | cả hai có | **nguồn gold dựng entry gazetteer** |

Nguồn phụ (tùy chọn, hand-curated, để tăng recall an toàn):
- `configs/product_aliases.yaml` *(tạo mới)* — seed brand → biến thể chính tả ("Hạ Long" ↔
  "Halong" ↔ "HA LONG"; "Pate" ↔ "Patê" ↔ "Paté"; "Highlands" ↔ "Highland").
- `configs/filler_tokens.txt` *(tạo mới)* — filler thủ công bổ sung cho phần auto-mine.
- `configs/product_rules.yaml` *(đã có)* — `fuzzy_threshold: 88`, `strong_threshold: 94`
  (ngưỡng dùng ở Cell 3; Cell 2 chỉ cần biết để chuẩn bị cấu trúc index phù hợp).

### 2.2 Schema một entry gazetteer (đề xuất — đồng bộ field Cell 1)
Mỗi entry là một "đối tượng có thể xuất ra làm product_name", gồm:

| Field | Ý nghĩa | Nguồn |
|---|---|---|
| `entry_id` | id ổn định (sha1 của `canonical_key`) | sinh khi build |
| `canonical_display` | **chuỗi xuất ra cuối cùng** (Title Case, giữ dấu, nối `" + "`) | product_name train (đã canonicalize) |
| `canonical_key` | `normalize_dedupe_key(canonical_display)` — **khóa exact có dấu** | dùng chung normalize Cell 1 |
| `folded_key` | `fold_key(canonical_display)` — **khóa exact đã fold dấu** | chống OCR rớt dấu |
| `token_keys` | `[normalize_dedupe_key(tok)]` — token có dấu | token match (có dấu) |
| `folded_token_keys` | `[fold_token_key(tok)]` — token đã fold dấu | token match (folded) |
| `brand_anchor` | brand gốc (1–2 token đầu / theo seed) | extract / seed |
| `item_count` | số item (đếm `" + "` + 1) | xử lý multi-item |
| `surface_variants` | các surface OCR từng thấy map về entry này | **mine từ ocr_text (2.5)** |
| `folded_surface_keys` | `fold_key(surface)` của từng surface | lookup folded surface |
| `frequency_prior` | số lần product_name = entry này trong train | đếm |
| `source_class` | `full` (cả chuỗi) / `atomic` (1 item tách từ `+`) / `mixed` | build |

> **Diacritic folding (đã hiện thực).** PaddleOCR thật **rớt dấu tiếng Việt** (đã chứng minh:
> `ĐỒ HỘP HẠ LONG` → OCR ra `DO HOP HA LONG`). Vì vậy mỗi entry giữ **song song** key/token
> **có dấu** (precision) và **đã fold dấu** (recall). `fold_text` / `fold_key` /
> `fold_token_key` được **export** từ `B_gazeetteer.py` để **Cell 3 tái dùng** — không tự
> định nghĩa fold riêng.

### 2.3 Bước build — chi tiết

**Step 2.3.1 — Load & phân loại.**
Đọc `train_labels.csv` bằng pandas; chuẩn hóa NaN→"" (đã có helper `_safe_str`). Chia 3
class như 2.1. **Vocab entry dựng từ MỌI row `product_name ≠ ""`** (kể cả row có `ocr_text`
rỗng — để biết hết product đã biết) ⇒ coverage đạt 100%. **Surface mining** (2.3.5) thì chỉ
dùng row **text-with-product** (cần ocr_text để căn).

**Step 2.3.2 — Canonicalize `product_name`.**
- Gom tất cả chuỗi `product_name` không rỗng.
- Tách multi-item theo `" + "` → giữ **cả chuỗi full lẫn từng atomic item** (entry riêng,
  đánh dấu `source_class`). Lý do: candidate có thể chỉ chứa 1 item.
- Với mỗi chuỗi: tính `canonical_key = normalize_dedupe_key(...)`,
  `folded_key = fold_key(...)`, `token_keys` (có dấu) và `folded_token_keys` (fold dấu).
- **Dedupe theo `canonical_key`.** Khi nhiều display khác nhau cùng key (vd
  `Đồ Hộp Hạ Long` vs `Đồ hộp Hạ Long`), chọn **canonical_display** theo quy tắc cố định
  (đề xuất: form **tần suất cao nhất**; hòa thì lấy Title-Case-chuẩn) và cộng dồn
  `frequency_prior`. ⚠️ Quy tắc chọn phải **deterministic** để Cell 6 ablation tái lập được.

**Step 2.3.3 — Brand anchor layer.**
- Trích `brand_anchor` cho mỗi entry: hoặc theo **seed brand list** (`product_aliases.yaml`),
  hoặc heuristic "cụm token đầu lặp lại nhiều entry". Lưu để Cell 3 ưu tiên match theo brand
  trước, rồi mới tới descriptor.
- ⚠️ **Ambiguity có thật:** surface "Hạ Long"/"Halong" xuất hiện ở **2 canonical khác nhau**
  (`Đồ Hộp Hạ Long` và `Halong Canfoco`). ⇒ `ambiguous_anchors` (đã hiện thực) gom theo
  **folded brand anchor** (`anchor:<...>`) **và** cặp **folded-token bigram chia sẻ ≥2 token**
  (`tok:<a>|<b>`), có **document-frequency filter** để không bùng nổ với token chung. Cell 4
  đọc theo prefix key để biết entry nào "ambiguous → cần evidence mạnh hơn hoặc blank".

**Step 2.3.4 — Index cho Cell 3 (cấu trúc tra cứu — đã hiện thực, có folded).**
Dựng **6 index** = 3 loại × {có dấu, fold dấu}, để retrieval bền với OCR rớt dấu:
- `exact_index: canonical_key → entry_id` *(có dấu, 1-1)* &
  `folded_exact_index: folded_key → List[entry_id]` *(folded **collide** nên là List)*.
- `token_inverted_index: token → List[entry_id]` *(có dấu)* &
  `folded_token_inverted_index: folded_token → List[entry_id]` *(fold dấu)*.
- `surface_index: surface_key → List[entry_id]` *(có dấu)* &
  `folded_surface_index: folded_surface_key → List[entry_id]` *(fold dấu)*.
- `entries: List[entry]` — để Cell 3 chấm `rapidfuzz` (`token_set_ratio` / `partial_ratio`)
  trên tập đã thu hẹp, gate bằng `88/94`.

**Step 2.3.5 — Mine surface variants từ `ocr_text` (bước tăng lực, data-driven).**
Với mỗi row `text-with-product`: lấy `gt ocr_text`, **căn (align) bằng fuzzy** để tìm span
trong ocr_text giống `product_name` nhất, ghi lại **surface OCR thực tế** → thêm vào
`surface_variants` của entry tương ứng. Việc này "dạy" gazetteer các lỗi chính tả OCR thật
(vd `Patê`/`PATE`, `HALONG`, mất dấu `DO HOP HA LONG`) ⇒ Cell 3 match đúng hơn nhiều so với
chỉ dựa chuỗi label sạch.
> Lưu ý dùng **`gt ocr_text` của train** ở đây là hợp lệ (đang build từ nhãn train), nhưng
> **không** được trộn nhãn test. Giữ build chỉ-từ-train để tránh leakage và để ablation sạch.

**Step 2.3.6 — Build filler / stopword list (đóng vòng về Cell 1).**
- Auto: gom token tần suất cao từ class **text-no-product** + token xuất hiện trong ocr_text
  nhưng **không bao giờ** thuộc product_name nào (channel/genre/corp): `review`, `mukbang`,
  `góc chia sẻ`, `tiktok`, `news`, `pov`, `công ty cổ phần`, `trả lời`, `kênh tin tức`…
- Hợp nhất với `configs/filler_tokens.txt` thủ công.
- **Xuất ra file** để Cell 1 nạp qua `load_filler_tokens_from_txt(path)` →
  `generate_candidate_dataframe(filler_tokens=...)`. Đây chính là mắt nối ngược 1.3.
  (helper sẵn: `rerun_candidates_with_gazetteer_fillers(ocr_df, gazetteer)`.)
- ⚠️ Precision của filler: chỉ strip token **chắc chắn** là nhiễu; đừng strip nhầm token có
  thể là brand/descriptor. **Guard đã folded-aware**: token bị bảo vệ nếu **dạng có dấu HOẶC
  dạng fold** trùng token brand (vd filler không được nuốt `pate`/`patê`, `ha`/`hạ`).

**Step 2.3.7 — Persist artifact (reproducible).**
- Trong notebook: giữ object in-memory.
- Cache tùy chọn: `outputs/gazetteer/product_gazetteer.json` (thư mục `outputs/` đã được
  `.gitignore`). Build **deterministic** (sort entry, seed cố định) để Cell 6 lặp lại được.

### 2.4 Output của Cell 2 (hợp đồng đẩy sang Cell 3) — `ProductGazetteer`
Object `ProductGazetteer` (build bằng `build_product_gazetteer(train_labels_path, ...)`) expose:
- 6 index ở 2.3.4 (`exact_index`, `folded_exact_index`, `token_inverted_index`,
  `folded_token_inverted_index`, `surface_index`, `folded_surface_index`) + `entries`,
  `entry_by_id`.
- `filler_tokens` (set) — đã feed ngược Cell 1.
- `ambiguous_anchors: Dict[str, Set[entry_id]]` — key prefix `anchor:` / `tok:`.
- `lookup_candidates(candidate_row, max_entries=80) → List[GazetteerEntry]` — **đã làm 6-way
  retrieval** (exact/surface/token × có dấu/folded) rồi **rank ưu tiên có dấu > folded**, trả
  về **List[GazetteerEntry]** (không phải id) đã thu hẹp cho Cell 3 fuzzy-score.
- `save_json` / `from_json` — cache `outputs/gazetteer/product_gazetteer.json` (giữ cả folded).
- `fold_text` / `fold_key` / `fold_token_key` (export) cho Cell 3 dùng lại.

### 2.5 Sanity check trước khi sang Cell 3 (đã có script tự test)
Chạy `python scripts/test_gazetteer.py` (13 check, in PASS/FAIL). Các check chính:
- **Coverage = 100%** (vocab từ mọi `product_name`); < 100% là lỗi normalize/dedupe hoặc vocab.
- **Folded lookup (fix cốt lõi):** candidate OCR rớt dấu `DO HOP HA LONG` → `lookup_candidates`
  trả `ĐỒ HỘP HẠ LONG` ở **#1** (folded_key cùng `"do hop ha long"`, tie-break theo
  `frequency_prior`). Candidate đủ dấu `ĐỒ HỘP HẠ LONG` vẫn ưu tiên entry có dấu (precision).
- **Filler probe:** `mukbang`/`review` ∈ `filler_tokens`; brand (`hạ`,`long`,`pate`,`highlands`)
  ∉ (kiểm bằng `filler_safety_check`, kỳ vọng `bad=0`).
- **Determinism:** build 2 lần → `entry_id`/thứ tự y hệt.

### 2.6 Bẫy cần tránh ở Cell 2
- ❌ Định nghĩa normalize/fold riêng cho gazetteer → lệch key với candidate. ✅ Import
  `normalize_dedupe_key`/`tokenize_text` từ `A_candidate.py`, dùng `fold_*` của `B_gazeetteer.py`.
- ❌ Để `folded_exact_index` là `Dict[str,str]` → mất entry vì folded **collide**. ✅ Phải là
  `Dict[str, List[entry_id]]`.
- ❌ Merge "Halong Canfoco" và "Đồ Hộp Hạ Long" thành một entry → sai chuỗi xuất ra. ✅ Giữ
  **mỗi canonical_display là một entry riêng**; ambiguity xử ở `ambiguous_anchors` + Cell 4.
- ❌ Strip filler quá tay (mất brand). ✅ Guard folded-aware, có `filler_safety_check`.
- ❌ Build phụ thuộc thứ tự dòng / không seed → ablation Cell 6 không tái lập. ✅ Deterministic.

---

## 3. Cell 3 — Fuzzy Linker *(chi tiết)*

### 3.0 Mục tiêu & vị trí
Cell 3 **link từng candidate** (row của Cell 1) tới **entry gazetteer tốt nhất** (Cell 2) và
ghi một **match record cho mỗi candidate**. **Chưa quyết định** product_name của ảnh — đó là
Cell 4. Cell 3 chỉ trả lời: *"candidate này khớp entry nào, điểm bao nhiêu, có chắc không,
có nhập nhằng không?"*

> Nguyên tắc bất biến: **không OCR lại, không đổi `ProductCandidate` dataclass, tái dùng
> `normalize_dedupe_key`/`tokenize_text` (A_candidate) + `fold_key`/`fold_token_key`
> (B_gazeetteer).** Mọi kết quả ghi vào **cột DataFrame mới** + `matched`/`debug_reason`.

### 3.1 Inputs
- `candidate_df` (Cell 1): từ `selections_to_candidate_dataframe(...)` hoặc
  `rerun_candidates_with_gazetteer_fillers(ocr_df, gazetteer)` — cột dùng ở Cell 3:
  `image_id, clean_text, normalized_key, tokenized, source, source_score, ocr_conf,
  structure_score, variant_agreement_score, position_score, token_count`.
- `gazetteer` (Cell 2): `ProductGazetteer` — dùng `lookup_candidates(...)`, các entry và
  `ambiguous_anchors`. **Không** tự quét `entries`.
- `thresholds`: đọc từ `configs/product_rules.yaml` (`fuzzy_threshold: 88`,
  `strong_threshold: 94`). ⚠️ Repo **chưa có loader YAML** → Cell 3 tự đọc (pyyaml có sẵn),
  nhận default `88/94` và cho override bằng tham số.

### 3.2 Output — cột MỚI ghi vào `candidate_df` (hợp đồng đẩy sang Cell 4)

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `matched` | bool | True nếu link vượt ngưỡng (set vào field có sẵn) |
| `matched_entry_id` | str | entry thắng (rỗng nếu không match) |
| `matched_display` | str | `canonical_display` của entry thắng (chuỗi xuất tiềm năng) |
| `match_score` | float 0–100 | điểm tốt nhất (đã lấy max 2 trục) |
| `match_type` | str | `exact_dia`/`exact_fold`/`surface_dia`/`surface_fold`/`fuzzy` |
| `match_used_fold` | bool | điểm thắng đến từ **trục folded** (⇒ OCR đã mất dấu) |
| `is_ambiguous` | bool | entry thắng thuộc nhóm `ambiguous_anchors` / có đối thủ sát điểm |
| `matched_frequency_prior` | int | prior của entry thắng (Cell 4 dùng để phân giải) |
| `match_debug` | str | lý do (mirror vào `debug_reason`) |

### 3.3 Thuật toán cho 1 candidate
1. **Guard rỗng/ngắn:** `clean_text` rỗng hoặc `token_count == 0` → `matched=False`,
   `match_debug="empty"`. (Phần lớn junk đã bị Cell 1 lọc, đây là phòng thủ.)
2. **Retrieval:** `entries = gazetteer.lookup_candidates(row, max_entries=80)`. Rỗng →
   `matched=False`, `match_debug="no_gazetteer_candidate"`. **Đây là default precision-first:
   không nằm trong vocab đã biết thì KHÔNG fill.**
3. **Chuẩn bị 2 chuỗi candidate:**
   - `cand_dia = normalized_key` (đã có; fallback `normalize_dedupe_key(clean_text)`).
   - `cand_fold = fold_key(clean_text)`.
4. **Chấm điểm từng entry trên 2 trục** (xem 3.4), lấy `best_dia` và `best_fold`.
5. **Chọn winner:** entry có `max(best_dia, best_fold)` cao nhất; **hòa thì theo thứ tự
   `lookup_candidates`** (đã ưu tiên có-dấu + `frequency_prior`).
6. **Phân loại `match_type`** theo cách khớp: exact key có dấu → `exact_dia`; exact
   `folded_key` → `exact_fold`; trùng surface → `surface_*`; còn lại `fuzzy`. Set
   `match_used_fold = (best_fold > best_dia)`.
7. **Gate** (xem 3.5) → `matched` True/False + `match_debug`.
8. **Ambiguity** (xem 3.6) → `is_ambiguous`.
9. **Ghi cột** ở 3.2.

### 3.4 Scoring (rapidfuzz — đã cài 3.14.5)
Brand thường là **sub-span** của candidate (line dài) ⇒ một metric không đủ. Cho mỗi
(candidate, target) tính:
```
s = max( fuzz.token_set_ratio(a, b),   # bỏ qua thứ tự + token thừa quanh brand
         fuzz.partial_ratio(a, b) )    # căn sub-span tốt nhất
```
- **Trục có dấu:** `a = cand_dia`; `b` ∈ {`entry.canonical_key`} ∪ {`normalize_dedupe_key(v)`
  cho `v` trong `entry.surface_variants`} → `best_dia`.
- **Trục folded:** `a = cand_fold`; `b` ∈ {`entry.folded_key`} ∪ `entry.folded_surface_keys`
  → `best_fold`.
- **Tận dụng exact index:** nếu `cand_dia == entry.canonical_key` → `best_dia = 100`
  (`exact_dia`); nếu `cand_fold == entry.folded_key` → `best_fold = 100` (`exact_fold`). Khỏi
  fuzzy cho mấy case này.
- **Guard candidate ngắn:** nếu `token_count <= 1` và `len(cand_dia) < 4` → bỏ qua
  `partial_ratio` (chỉ dùng `token_set_ratio`) để tránh `partial_ratio` thổi điểm ảo cho token
  cụt (`te`, `O`, `Y`…).

### 3.5 Gate & precision-first
- `score = max(best_dia, best_fold)` của winner.
- **Trục có dấu** (`match_used_fold == False`): `score >= 88` → match (≥94 = strong).
- **Trục folded** (`match_used_fold == True`): **nâng bar** — yêu cầu `score >= strong_threshold
  (94)`. Vì fold mất thông tin & dễ collision (`Hạ Long`↔`Ha Long`), match chỉ-folded phải
  chắc hơn mới được fill.
- Không đạt → `matched=False`, `match_debug=f"below_threshold:{score:.0f}"`.

### 3.6 Ambiguity (đẩy tín hiệu cho Cell 4, KHÔNG tự quyết)
Đánh `is_ambiguous = True` khi một trong các điều kiện:
- Winner thắng ở **trục folded** và `folded_exact_index[winner.folded_key]` có **≥2 entry**
  (vd nhiều entry cùng fold `"do hop ha long"`), hoặc
- Winner nằm trong nhóm `ambiguous_anchors` (`anchor:` hoặc `tok:`) cùng với một entry khác
  **cũng được retrieve**, hoặc
- Entry đứng #2 có `match_score` cách winner **≤ 3 điểm** và là entry khác canonical.

Cell 4 sẽ giải bằng `matched_frequency_prior` (vd `Đồ Hộp Hạ Long` 793 ≫ `Halong Canfoco`),
đồng thuận nhiều candidate/variant, hoặc **blank nếu vẫn không chắc**.

### 3.7 Config loader (việc nhỏ cần làm cho Cell 3)
Viết helper đọc `configs/product_rules.yaml` (pyyaml) → trả `(fuzzy_threshold, strong_threshold)`
với default `(88, 94)` nếu file thiếu/khoá thiếu. Đặt cạnh Cell 3 (vd `C_fuzzy_linker.py`), để
Cell 6 ablation truyền thẳng tham số (88 vs 94, fold-bar on/off) mà không cần sửa YAML.

### 3.8 Performance
`lookup_candidates` đã thu hẹp ≤80 entry/candidate ⇒ ≤80×(1–~40 surface) phép fuzzy — chấp
nhận được. Nếu chạy toàn test cần nhanh hơn: gom theo `normalized_key` **dedupe candidate
trùng** trước khi score (nhiều ảnh cùng text), hoặc dùng `rapidfuzz.process.cdist`. Không bắt
buộc ở bản đầu.

### 3.9 Sanity / test (mở rộng `scripts/test_gazetteer.py` hoặc thêm `test_fuzzy_linker.py`)
- `DO HOP HA LONG` (OCR rớt dấu) → `matched=True`, `matched_display` thuộc họ `Đồ Hộp Hạ Long`,
  `match_used_fold=True`, `match_score` cao.
- `ĐỒ HỘP HẠ LONG` → `matched=True`, `match_type=exact_dia`, `match_used_fold=False`.
- `MUKBANGReview`, `te`, `cangi` → `matched=False` (no gazetteer / dưới ngưỡng).
- Chạy trên `candidate_df` thật của vài ảnh (vd `img_0006`) và soi mắt `match_*`.

### 3.10 Bẫy cần tránh ở Cell 3
- ❌ Tự viết hàm fold/normalize. ✅ Dùng `fold_key`/`fold_token_key` + `normalize_dedupe_key`.
- ❌ Cho match-chỉ-folded cùng độ tin như có-dấu. ✅ Nâng bar lên `strong_threshold` cho folded.
- ❌ Quyết định blank/fill **theo ảnh** ở Cell 3. ✅ Cell 3 là **per-candidate**; gộp theo
  `image_id` là việc Cell 4.
- ❌ Fill khi `lookup_candidates` rỗng. ✅ Default **blank** (không có trong vocab → không đoán).
- ❌ Sửa dataclass `ProductCandidate`. ✅ Chỉ thêm cột DataFrame + dùng `matched`/`debug_reason`.

---

## 4. Cell 4 — Confidence Gating *(chi tiết)*

### 4.0 Mục tiêu & vị trí
Cell 4 **gom các candidate đã link (Cell 3) theo `image_id`** và ra **một quyết định cho mỗi
ảnh**: chọn (các) entry để xuất, hoặc **blank**. Đây là nơi đặt **precision-first** — vì scoring
`Score = 0.6·F1_product + 0.4·(1−CER)`, một product_name **sai** làm tụt F1 (false positive),
nên *khi không chắc → blank*.

> Ranh giới với Cell 5: **Cell 4 quyết định "ảnh này dùng entry nào / blank"** (logic chọn +
> gate). **Cell 5 mới dựng chuỗi `product_name` cuối** từ entry đã chọn (Title Case, giữ dấu,
> nối `" + "`, brand correction). Cell 4 KHÔNG tự ghép chuỗi xuất.

### 4.1 Inputs — cột từ `linked_candidate_df` (Cell 3)
Per-candidate, các cột Cell 4 dùng (đã có thật):
- **Match:** `matched`, `is_strong_match`, `matched_entry_id`, `matched_display`,
  `matched_source_class`, `matched_frequency_prior`, `matched_item_count`, `match_score`,
  `second_match_score`, `match_margin`, `match_type` (`exact`/`folded_*`/`fuzzy_*`).
- **Bằng chứng candidate (Cell 1):** `source` + `source_priority` (line>segment>ngram),
  `structure_score`, `ocr_conf`, `variant_agreement_score`, `variant_support_count`,
  `available_variant_count`, `position_score`, `line_index`.
- Tra cứu entry khi cần: `gazetteer.entry_by_id[...]` (lấy `item_count`, `source_class`,
  `brand_anchor`, `frequency_prior`) và `gazetteer.ambiguous_anchors`.

### 4.2 Output — 1 row / ảnh (đẩy sang Cell 5)
| Cột | Ý nghĩa |
|---|---|
| `image_id` | khóa ảnh |
| `decision` | `fill` / `blank` |
| `chosen_entry_ids` | list entry_id được chọn (1 phần tử = single; ≥2 = compose) |
| `compose_mode` | `single` / `multi` / `none` |
| `image_confidence` | điểm tổng hợp (0–100) để gate & ablation |
| `n_matched_candidates`, `n_distinct_entries` | thống kê bằng chứng |
| `is_ambiguous` | có nhập nhằng chưa giải được |
| `decision_reason` | vì sao fill/blank (mirror `debug_reason`) |

> `chosen_entry_ids` là **id**, không phải chuỗi — Cell 5 map sang `canonical_display`. Như vậy
> compose multi-item là việc nối `canonical_display` ở Cell 5, Cell 4 chỉ **chọn entry nào**.

### 4.3 Bước 1 — Gom bằng chứng theo entry (trong 1 ảnh)
Lọc candidate `matched == True` của ảnh, **gom theo `matched_entry_id`** (một entry có thể
được nhiều candidate/line/ngram trỏ tới). Mỗi entry-evidence tính:
- `best_score = max(match_score)`, `any_strong = any(is_strong_match)`.
- `support = số candidate` trỏ tới; `distinct_sources = #{source}`; `distinct_variants`.
- `best_structure`, `best_ocr_conf`, `best_position` (max), `min_margin` (cảnh báo nhập nhằng).
- entry-level: `frequency_prior`, `source_class`, `item_count` (từ gazetteer).
- **`entry_confidence`** (heuristic, deterministic, tunable Cell 6):
  ```
  conf = best_score
       + bonus_source(line/segment > ngram)        # vd +4 / +2 / 0
       + bonus_agreement(distinct_sources, support) # nhiều nguồn đồng thuận
       + bonus_structure*best_structure
       + small_bonus(log frequency_prior)           # prior nhẹ, tránh đè evidence
       − penalty_fold(match_type bắt đầu 'folded_') # fold lossy
       − penalty_low_margin(min_margin nhỏ)
  ```

### 4.4 Bước 2 — Chọn **full vs compose** (mấu chốt, bám `img_0011`)
Quan sát thật: với ảnh multi-item, chuỗi **full** (`Highlands Coffee Trà Sen Vàng + Bánh Mì
Que`) match **fuzzy 96**, còn **atomic** (`Trà Sen Vàng`/`Bánh Mì Que`) match **exact 100**.
⇒ **Đừng chỉ lấy max score** (sẽ ra 1 atomic, mất phần còn lại). Quy tắc:
1. **Ưu tiên `full` được chứng thực:** nếu có entry `source_class=full`, `item_count≥2`,
   `entry_confidence` đạt ngưỡng, **và** các item của nó được đối chứng bởi atomic-match cùng
   ảnh → **chọn full đó (`compose_mode=single`)**. Đây là case `img_0011` → ra đúng nhãn.
2. **Compose từ atomic:** nếu **không** có full đạt ngưỡng nhưng có **≥2 atomic phân biệt**
   (canonical khác nhau, không trùng brand) cùng đạt ngưỡng → **chọn các atomic đó**
   (`compose_mode=multi`), Cell 5 nối `" + "` theo thứ tự `line_index`/`position`.
3. **Single:** còn lại, nếu có **1 entry** (brand hoặc single-item) đạt ngưỡng mạnh → chọn nó.
4. **Chống double-count:** đã chọn full thì **không** thêm atomic con của nó; đã compose atomic
   thì không thêm brand-only trùng.

### 4.5 Bước 3 — Gate blank vs fill (precision-first)
- `image_confidence` = `entry_confidence` của lựa chọn (compose: min/trung bình các phần — vì
  yếu một phần là cả chuỗi sai).
- **Fill** chỉ khi: lựa chọn có **≥1 strong** (hoặc `image_confidence ≥ bar`), **mọi phần**
  (nếu compose) đều đạt ngưỡng, và **không** ambiguous-chưa-giải.
- **Blank** khi: không candidate matched; `image_confidence < bar`; chỉ có **ngram 1-token**
  yếu; ambiguous không giải được (4.6). *Blank là class hợp lệ — thà blank còn hơn sai F1.*

### 4.6 Bước 4 — Giải nhập nhằng (dùng tín hiệu đã có)
- Hai entry canonical khác nhau, điểm sát (`min_margin` nhỏ) và cùng nhóm
  `gazetteer.ambiguous_anchors` (`anchor:`/`tok:`) hoặc cùng `folded_key` collide
  (vd `Hạ Long` → `Đồ Hộp Hạ Long` vs `Halong Canfoco`):
  1. Phá hòa bằng **`frequency_prior`** (đồ hộp hạ long 793 ≫ canfoco) **nếu** chênh lệch lớn.
  2. Còn sát → ưu tiên entry được **nhiều source/variant đồng thuận** hơn.
  3. Vẫn không chắc → **blank** (`is_ambiguous=True`, `decision=blank`).
- Match **chỉ-folded** (`match_used_fold`/`match_type='folded_*'`) cần bằng chứng mạnh hơn mới
  fill (đồng bộ tinh thần Cell 3 §3.5).

### 4.7 Sanity check (bám ảnh thật)
- `img_0006` (`Đồ Hộp Hạ Long`): 1 entry chiếm ưu thế → `single`, fill.
- `img_0011` (`Highlands Coffee Trà Sen Vàng + Bánh Mì Que`): full được chứng thực bởi atomic →
  `single` chọn full (KHÔNG ra mỗi `Bánh Mì Que`).
- `img_0009` (`HÓNG SÀI GÒN`) / `img_0003` (`Hệ Lụy Đồ Hộp`): không entry nào đạt ngưỡng →
  **blank** (đúng nhãn — class text-no-product).
- Ảnh `ocr_text` rỗng: không candidate → blank.

### 4.8 Bẫy cần tránh ở Cell 4
- ❌ Quyết định theo từng candidate. ✅ **Gom theo `image_id`** rồi mới quyết.
- ❌ Lấy max `match_score` → ra 1 atomic, mất multi-item. ✅ Ưu tiên **full được chứng thực**,
  else compose ≥2 atomic.
- ❌ Nối **mọi** `matched_display` → trùng/nhiễu (vd `BÁNH MÌ HIGHLANDS`). ✅ Gate từng phần +
  dedup + chống double-count.
- ❌ Fill khi chỉ có ngram 1-token / chỉ-folded yếu. ✅ Blank precision-first.
- ❌ Ghép chuỗi xuất ở Cell 4. ✅ Cell 4 chọn **entry_id**; Cell 5 dựng chuỗi.
- ❌ Output `canonical_key`/text folded (mất dấu). ✅ Cell 5 dùng `canonical_display`.

---

## 5. Cell 5 — Normalization + Final Predictor *(chi tiết)*

### 5.0 Mục tiêu & vị trí
Cell 5 biến quyết định của Cell 4 thành **chuỗi `product_name` cuối** đúng format thi đấu
(**Title Case giữ dấu**, brand correction, nối multi-item `" + "`) và **lắp submission đúng
schema**. Đây là tầng cuối trước validator (Cell 6 / `validate_submission.py`).

> Cell 4 đã chọn **entry** (và có sẵn `product_name_candidate` ghép thô từ `canonical_display`).
> Cell 5 là **bộ chuẩn hóa chính thức**: không chọn lại entry, chỉ làm sạch/định dạng chuỗi +
> đóng gói submission. Tái dùng `normalize_light`/`tokenize_text` (A) và `fold_*` (B).

### 5.1 Inputs
- `decision_df` (Cell 4): `image_id`, `emit_product`, `compose_mode` (`single`/`multi`/`none`),
  `chosen_entry_ids`, `chosen_displays`, `product_name_candidate`, `gate_score`, `is_ambiguous`.
- **OCR text** cho cột `ocr_text`: lấy `selected_result.text` của pipeline (hoặc `selected_text`
  trong ocr_df). **KHÔNG** đọc lại OCR — dùng text đã có.
- `gazetteer` (tùy chọn): `entry_by_id[...]` để lấy `brand_anchor` (phase 2 `brand_name`),
  `source_class`, `item_count`.
- `configs/product_aliases.yaml` *(tạo mới, tùy chọn)*: **brand-correction map** + **acronym
  lexicon** (xem 5.3).

### 5.2 Output schema — **theo phase** (đã quét dataset)
| Phase | File mẫu | Cột bắt buộc |
|---|---|---|
| **Public** | `sample_submission.csv` | `image_id, ocr_text, product_name` |
| **Phase 2 private** | `phase2_dataset/.../sample_submission_private.csv` | `image_id, ocr_text, **brand_name**, product_name` |

- `test.csv` / `private_test.csv` chỉ có `image_id` ⇒ `ocr_text` do pipeline OCR sinh.
- **Blank = `""` (không NaN).** Row count = đúng test set; image_id không thiếu/thừa/trùng.
- ⚠️ Phase 2 cần thêm cột **`brand_name`** — đừng quên (5.5).

### 5.3 Chuẩn hóa một item (`normalize_product_display`)
Vào: một `canonical_display` (từ `chosen_displays`). Ra: chuỗi Title Case chuẩn.
1. `normalize_light` (NFC, bỏ emoji/control, gộp space, thống nhất dash/quote).
2. **Title Case giữ dấu Việt**: token hóa, mỗi token `first.upper() + rest.lower()` trên NFC
   (đ→Đ, ạ→Ạ đúng). *Không* dùng `str.title()` ngây thơ (vỡ ở dấu nháy/số).
3. **Acronym/brand-casing lexicon** (giữ nguyên, KHÔNG Title Case): `NAN`, `A2`, `HMO`,
   `OPTIPRO`?, `VTDC`, `POV`… → map `lowercased token → casing chuẩn`.
4. **Brand correction**: gộp biến thể chính tả về canonical (`Highland Coffee`/`Highlands
   Cofee` → `Highlands Coffee`; `Halong` → theo ngữ cảnh). Dùng `product_aliases.yaml`/dict;
   key bằng `normalize_dedupe_key`/`fold_key` để bắt cả bản rớt dấu.
5. Strip filler còn sót (an toàn — **đừng cắt brand**); trim ký tự rác đầu/cuối.

### 5.4 Compose multi-item & blank
- `compose_mode == 'multi'`: chuẩn hóa **từng** `chosen_displays` → **dedup sau chuẩn hóa** →
  nối `" + "` theo thứ tự Cell 4 đưa (đã sort `line_index`).
- `compose_mode == 'single'`: nếu display đã chứa `" + "` (full multi-item label) → **chuẩn
  hóa từng phần quanh `" + "`** rồi nối lại (đừng Title Case cả separator); ngược lại chuẩn hóa
  nguyên chuỗi.
- `emit_product == False` / `decision == blank` → `product_name = ""`.

### 5.5 `brand_name` (chỉ phase 2)
- Lấy `brand_anchor` của **entry chính** (`chosen_entry_ids[0]`) qua
  `gazetteer.entry_by_id[...]`, áp Title Case (5.3).
- Nếu compose nhiều item **cùng brand** → brand đó; **khác brand** → để brand chính hoặc `""`
  (precision-first, không đoán bừa).

### 5.6 Lắp submission (nơi `scripts/make_submission.py` gọi)
1. Khung từ `test.csv`/`sample_submission` (đủ & đúng thứ tự `image_id`).
2. LEFT JOIN `decision_df` → `product_name` (đã chuẩn hóa); điền `ocr_text` từ OCR; `fillna("")`.
3. Chọn đúng cột theo phase (thêm `brand_name` nếu phase 2); **không NaN**.
4. Pipeline submission đầy đủ: **OCR test → Cell 1→5 → ghi `submission.csv`**.

### 5.7 Sanity check (bám ảnh thật)
- `img_0006` → `Đồ Hộp Hạ Long` (Title Case từ canonical `ĐỒ HỘP HẠ LONG`).
- `img_0011` → `Highlands Coffee Trà Sen Vàng + Bánh Mì Que`.
- `img_0009` → `""` (blank).
- Acronym: `Sữa NAN` giữ `NAN` (không thành `Nan`); `A2` giữ nguyên.

### 5.8 Bẫy cần tránh ở Cell 5
- ❌ Title Case ngây thơ làm hỏng acronym (`NAN`→`Nan`, `HMO`→`Hmo`). ✅ acronym lexicon.
- ❌ Quên cột `brand_name` ở **phase 2**. ✅ schema theo phase (5.2).
- ❌ Xuất `NaN`/thiếu/thừa/trùng `image_id`. ✅ build từ `test.csv` + `fillna("")` +
  validate (Cell 6).
- ❌ Bỏ dấu tiếng Việt (CER tăng). ✅ **giữ dấu**.
- ❌ Cắt substring nguyên văn OCR làm product_name. ✅ dùng `canonical_display` (đã chuẩn).
- ❌ Title Case cả chuỗi `"A + B"` gộp luôn `+`. ✅ chuẩn hóa **từng phần** quanh `" + "`.

---

## 6. Định hướng Cell 6 *(tóm tắt — chi tiết hóa ở vòng tiếp)*

- **Cell 6 — Validation / Ablation + post-processing.** Tính `F1_product` + `1−CER` trên
  train; **bật/tắt từng bước** (filler on/off, ngram on/off, threshold 88 vs 94, surface-mine
  on/off, full-vs-compose on/off, context-lexicon on/off) để đo đóng góp. Cũng **validate
  schema** submission (`validate_submission.py`: đúng image_id, không NaN/thiếu/thừa/trùng,
  đúng cột theo phase, đếm blank). Đây là chỗ chuẩn hóa lại metric (hiện CER đang inline trong
  `scripts/audit_train_subset.py` — Cell 6 sẽ tách về `evaluation/`).

---

## 7. Checklist "đã đồng bộ với source code"
- [x] Cell 2 **import** `normalize_light` / `normalize_dedupe_key` / `tokenize_text` từ
      `product/A_candidate.py` (không viết lại).
- [x] Gazetteer key = `normalize_dedupe_key` (cùng candidate) + `fold_key` cho trục folded.
- [x] **Folded-index** (`folded_exact_index` kiểu `Dict[str,List]`, `folded_token_inverted_index`,
      `folded_surface_index`) — chống OCR rớt dấu; `lookup_candidates` 6-way, rank có dấu > folded.
- [x] Vocab entry từ **mọi** `product_name ≠ ""`; surface mining chỉ từ row có `ocr_text`.
- [x] Filler list xuất ra file → Cell 1 nạp bằng `load_filler_tokens_from_txt`
      (`rerun_candidates_with_gazetteer_fillers`); guard folded-aware.
- [x] Mỗi `canonical_display` (kể cả casing/`+`) → một entry; không merge; ambiguity ở
      `ambiguous_anchors` (`anchor:`/`tok:`) + Cell 4.
- [x] **Cell 3** (`C_fuzzy_linker.py`) tái dùng `fold_key`/`fold_token_key`; `lookup_candidates`
      6-way; drop cột trùng trước concat (`matched` của Cell 1); rapidfuzz có fallback stdlib.
- [x] **Cell 4** (`D_confidence_gate.py`) gom theo `image_id` (per-entry aggregation); ưu tiên
      **full được chứng thực** (coverage + ≥2 atomic high-trust) else compose ≥2 atomic
      (`compose_mode`/`chosen_entry_ids`); blank precision-first + negative-context; không đổi
      dataclass. *(Lưu ý: context-lexicon là phần thêm ngoài §4, cần Cell 6 validate.)*
- [ ] **Cell 5** map `chosen_displays` → Title Case giữ dấu + acronym lexicon + brand
      correction + nối `" + "`; **schema theo phase** (public 3 cột / phase 2 thêm `brand_name`);
      blank=`""` không NaN; build submission từ `test.csv`.
- [ ] Không sửa `preprocess/`, `ocr/engine.py`; không đổi dataclass `ProductCandidate`
      (cell sau ghi vào `matched`/`debug_reason` hoặc thêm cột DataFrame).
- [x] Build deterministic + cache `outputs/gazetteer/` cho ablation.
- [x] Sanity `scripts/test_gazetteer.py` pass (13/13) trước khi code Cell 3.
