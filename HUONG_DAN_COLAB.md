# Chạy trên Colab — bảng kiểm từng bước

Số ô (`ô #n`) đếm theo **ô code**, bỏ qua ô chữ — đúng như Colab đánh số khi chạy.
Thời gian **đã gồm** ~15 phút cài đặt + nạp model mỗi phiên GPU.

> ⚠ Các mốc thời gian là **ước lượng sau khi nâng trần sinh 4096 → 8192**. Phần đội thêm
> rơi gần hết vào ~5 % mẫu khó nhất; số còn lại kết thúc sớm nên không đổi.

Thang bậc: **1** prompt cơ bản → **2** prompt hoàn chỉnh → **4** self-eval → **3** SFT →
**5** ACE, cộng hai nhánh rẽ (**5c** ACE trên prompt cơ bản, đối chứng bullet ngẫu nhiên)
và **6** ma trận tổ hợp.

---

## BUỔI 1 — kết quả chính (~4,3 h)

### Bước 1 · `00_data_audit` · CPU · 2 ph

Mở → CPU → Chạy tất cả → cho phép gắn Drive.

| ⛔ Cổng | |
|---|---|
| §2 — `test … 497/497  100.0%` | không đạt → **dừng hẳn**, executor sai thì mọi số sau vô nghĩa |

§5 — chụp bảng đính chính EA, dùng cho báo cáo.

---

### Bước 2 · `01_baseline_basic` · A100 · ~30 ph

Mở → A100 → bật **Thực thi nền** → Chạy tất cả.

Colab sẽ hiện **RESTART SESSION** sau ô #1 → bấm, rồi bấm ô #2 → `Ctrl+F10` (chạy ô này
và các ô sau).

| ⛔ Cổng | Phải thấy |
|---|---|
| ô #3 | `[GÓI] vllm=0.23.0+cu129 \| transformers=4.57.6 \| trl=0.24.0` — **`vllm` phải có đuôi `+cu129`** |
| ô #3 | `[SELF-TEST] executor tái tạo exe_ans trên test: 497/497` |
| ô #4 | `[CFG] max_seq=25000 max_tokens=8192 temp=0.1 (cố định mọi GPU)` |
| ô #6 | `[PROMPT] ✅ mọi prompt đều lọt ngân sách, không cần cắt` |
| ô #6 | `thang lồng nhau: basic=2535 ký tự ⊂ no_fewshot=5408 ⊂ engineered=5924` |

Thấy `[PROMPT] ⚠ đã bật cắt ngữ cảnh` là đang không ở A100/L4 — đổi runtime rồi chạy lại.
**Kết quả có cắt ngữ cảnh không trộn chung bảng được với kết quả không cắt.**

Nấc 1 dùng `PROMPT_LEVEL = "basic"`: model **đã biết** có những phép toán nào và phải trả
lời theo định dạng nào, chỉ chưa được mách chọn phép nào cho loại câu hỏi nào và chưa thấy
ví dụ mẫu. Không sửa gì.

**Cuối §4** — dòng mới, đây là chỗ kiểm trần token:

```
   Bị cắt vì trần token, tách theo bước:
     step1        0.x%  (n/497 lượt)
```

| Quan sát | |
|---|---|
| `step1` < 1 % | ✅ trần 8192 đủ |
| `step1` > 1 % | ⛔ báo lại — trần còn thiếu, đừng chạy tiếp cả thang rồi mới phát hiện |

→ Chạy tiếp ngay, **không cần dừng**.

---

### Bước 3 · `02_prompt_engineering` · A100 · ~30 ph

Mở → A100 → Chạy tất cả. Không sửa gì — ô #6 ghim sẵn `PROMPT_LEVEL = "engineered"`.

Ô #7 in bảng tách đôi phần nấc 2 thêm vào so với nấc 1 (hướng dẫn từ khoá, rồi ví dụ mẫu)
— phải thấy cả hai dòng `✅ chèn thuần`. Không thấy nghĩa là prompt đã bị sửa ở phần dùng
chung, hiệu số nấc 1 → 2 khi đó lẫn cả chuyện đổi câu chữ.

| Quan sát ở §4 (`PA_loose` lặp lại ở bảng §5) | |
|---|---|
| `PA_loose` ≥ 55 % **và** `step1` < 1 % | ✅ |
| `PA_loose` < 53 % | ⛔ dừng, báo lại |
| `step1` > 1 % | ⛔ trần 8192 chưa đủ |

→ **DỪNG. Gửi khối meta.**

---

### Bước 4 · `04_self_evaluation` · A100 · ~45 ph

Mở → A100 → **ô #6 giữ `USE_SFT_ADAPTER = False`** → Chạy tất cả.

Phải in `[MODEL] dùng Qwen3-8B gốc (chưa SFT)`.

| ⛔ Cổng §7 | |
|---|---|
| `PA_loose` cao hơn nấc 2 ít nhất **2 điểm** | sàn nhiễu đo được là 1,6 điểm |

**Đọc kỹ dòng bị cắt ở §5** — nấc này sinh hai lượt mỗi mẫu:

```
     step1        0.x%
     step2        0.y%
```

`step2` cao hơn `step1` quá 2 điểm thì notebook tự cảnh báo: prompt bước 2 chứa nguyên lời
giải bước 1 nên dài hơn, bị cắt nhiều hơn nghĩa là self-eval **mất cơ hội sửa** chứ không
phải sửa sai — hiệu số đo được khi đó là **cận dưới**.

→ **DỪNG. Gửi khối meta.**

---

### Bước 5 · `05_ace` · A100 · ~150 ph

Trước khi chạy: nạp `OPENAI_API_KEY` vào **Colab Secrets** (🔑 bên trái), bật *Notebook
access*.

Mở → A100. Ba cờ đều đã đúng mặc định:

| Ô | Cờ | Giá trị |
|---|---|---|
| #6 | `USE_SFT_ADAPTER` | `False` |
| #8 | `ACE_TREN_PROMPT` | `"engineered"` |
| #15 | `RUN_RANDOM_CONTROL` | `True` |

Phải thấy `[ACE] ✅ gọi thử gpt-4o-mini OK`. Không thấy thì dừng — Reflector hỏng là cả pha
A vô ích.

**Pha A (§5, ~70 ph).** Checkpoint của lần chạy cũ phải tự bị loại:

```
[RESUME] ⚠ CẤU HÌNH ĐÃ ĐỔI → KHÔNG nối tiếp, học lại từ đầu.
```

Thấy `[RESUME] tiếp từ vòng 19` → **dừng ngay**, xoá tay `progress_ace_base.json` rồi chạy lại.

| ⛔ Cổng cuối §5 | |
|---|---|
| playbook ≥ 5 bullet, composite dương | dưới mức đó thì ACE chưa học được gì để đo |

→ **DỪNG. Gửi khối meta + `playbook_ace_base.txt`.**

---

### Bước 6 · `07_final_report` · CPU · 2 ph

| ⛔ Cổng ô #4 | |
|---|---|
| `✅ Mọi nấc cùng GPU, cùng trần token…` | |
| Cột **bước 1** mọi nấc < 1 % | |
| `⚠ KHÔNG KIỂM ĐƯỢC … thiếu '<nấc>_meta.json'` | ⛔ thiếu file meta — bảng công bằng không kết luận được gì |

→ Gửi `bang_ket_qua_*.csv` + `kiem_dinh_*.csv`.

---

## BUỔI 2 — SFT (~2,4 h)

Một notebook `03_sft_qwen3`, **ba phần, hai lần restart**. Mở → A100.

### Bước 7 · phần A — dựng dữ liệu · ~80 ph

Bấm ô chữ **⚠ RESTART RUNTIME TẠI ĐÂY** (cái thứ nhất) → *Thời gian chạy* → **Chạy trước**.
Tức là chạy ô #1 → #10.

Ô #7 để nguyên `SFT_TRAIN_SUBSET = 2000`, `ADD_GOLD_FALLBACK = False`.

Ô #9 giờ có reset riêng — lượt sinh này dựng **dữ liệu huấn luyện**, bị cắt ở đây là mất
đúng những mẫu khó nhất khỏi tập SFT. Xem dòng `step1`.

| ⛔ Cổng §5 | |
|---|---|
| mẫu SFT ≥ 800, phân bố có mẫu ≥ 3 bước | |

→ **Khởi động lại phiên.** Chỉ báo em nếu < 800.

### Bước 8 · phần B — huấn luyện · ~40 ph

Bấm ô #11, `Shift`+bấm ô #16 → **Chạy phần đã chọn**.

| ⛔ Cổng | Phải thấy |
|---|---|
| ô #12 | `[SFT] dữ liệu: …` — số mẫu khớp phần A |
| ô #13 | `MAX_SEQ_LENGTH = 12288` (dài hơn trần sinh 8192 + prompt) |
| ô #14 | `[DATA] train X → Y (lọc mẫu > 12288 token)` và `[DATA] mất Z% vì quá dài` |
| ô #16 | val loss không tăng ngược |

`Z` **trên 5 %** là notebook cảnh báo — báo em, đừng train tiếp. Mất nhiều nghĩa là đang
cắt mất chính những mẫu dài mà SFT cần học.

→ **Restart session lần nữa.**

### Bước 9 · phần C — chấm test · ~24 ph

Bấm ô chữ **⚠ RESTART RUNTIME TẠI ĐÂY** (lần 2) → bấm ô #17 → **Chạy ô này và các ô sau**.

> ⚠ Đừng chạy lại ô #11–#16 — ô #15 là ô huấn luyện, mất thêm 30 phút vô ích.

Ô #23 phải in `[LoRA] đã nạp adapter từ …`.

Ô #25 so với nấc 2. **Cả ba khả năng (hơn / ngang / kém) đều đáng báo cáo** — bảng cuối
notebook gợi ý câu chữ cho từng trường hợp.

→ **DỪNG. Gửi khối meta.**

---

## BUỔI 3 — nấc 5c: ACE trên prompt cơ bản · ~110 ph

Mở lại `05_ace` từ GitHub → A100 → **ô #8 sửa một dòng**:

```python
ACE_TREN_PROMPT = "basic"
```

Phải in:

```
[ACE] chồng lên prompt 'basic' | self-eval=False | so với nấc '01_basic'
```

Ghi sang `05c_ace_basic_base`, **không đè** nấc 5. Checkpoint và playbook của nấc này
mang tên riêng (`progress_ace_basic_base.json`, `playbook_ace_basic_base.txt`).

Câu hỏi nấc này trả lời: *ACE có tự khám phá lại được thứ mà người viết prompt đã viết tay
không?* Trên prompt hoàn chỉnh, ACE gần như hết đất diễn vì prompt đã chứa sẵn ánh xạ từ
khoá → phép toán. Trên prompt cơ bản thì khoảng trống để lấp rộng hơn hẳn.

| Δ so với nấc 1 | |
|---|---|
| > +15 điểm | ACE thay được phần lớn công viết prompt tay — luận điểm mạnh nhất |
| +5 … +15 | khám phá được một phần |
| < +2 | không tự tìm ra — **vẫn là kết quả đáng báo cáo** |

→ **DỪNG. Gửi khối meta + playbook.**

---

## BUỔI 4 — ma trận tổ hợp (~3,3 h)

### Bước 11 · `04` lần 2 · ~45 ph
Mở `04` → ô #6 → `USE_SFT_ADAPTER = True` → Chạy tất cả.
Phải in `[MODEL] dùng adapter đã SFT` và `nấc trước để so sánh: 03_sft`.

### Bước 12 · `05` lần 2 · ~120 ph
Mở `05` → ô #6 → `True`; ô #8 → `"engineered"`; ô #15 → `False`.

### Bước 13 · `06` rồi `07` · ~50 ph
`06` → A100 → Chạy tất cả (§5 phải hiện **6/8 ô đã có sẵn**, chỉ chạy 2 ô mới).
`07` → CPU → Chạy tất cả.

→ Gửi `ma_tran_to_hop_*.csv` + gói zip cuối.

---

## Gửi lại gì

**Bốn lần dừng bắt buộc:** sau bước 3, 4, 5, 9.

Cuối mỗi nấc notebook tự in khối meta giữa hai đường kẻ ngang — bôi đen copy nguyên khối.

Kèm theo, mỗi lần: dòng **`Bị cắt vì trần token, tách theo bước`**. Đó là thứ quyết định
đọc hiệu số giữa các nấc như thế nào.
