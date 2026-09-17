# Kế hoạch thử nghiệm — Colab Pro, GPU A100

Ngân sách ~100 compute unit (CU)/tháng. A100 ≈ **11.8 CU/giờ**.

Tài liệu này trả lời *vì sao chạy theo thứ tự này và tốn bao nhiêu*. Thao tác cụ thể
từng bước nằm ở [HUONG_DAN_COLAB.md](HUONG_DAN_COLAB.md).

## Đính chính so với lần ước lượng trước

Trước đó em khuyên tránh A100 vì "đắt gấp 2.5× mà không nhanh tương ứng". **Nói vậy là quá
mạnh.** Với đúng workload này:

| | L4 24GB | A100 40GB | Tỉ lệ |
|---|---|---|---|
| Giá | ~4.8 CU/giờ | ~11.8 CU/giờ | 2.4× đắt hơn |
| Băng thông bộ nhớ | 300 GB/s | 1.555 GB/s | 5.2× |
| FLOPS bf16 | 121 TF | 312 TF | 2.6× |
| KV cache khả dụng | ~15 GB | ~28 GB | 1.9× |
| **Nhanh hơn thực tế** | — | — | **~2.5×** |

Prompt của ViNumQA rất dài (3.6k–7.4k token) nên phần **prefill chiếm ưu thế**, mà prefill
thì A100 nhanh hơn đúng theo tỉ lệ FLOPS. Kết quả: **chi phí CU gần như hoà, nhưng thời gian
thực giảm ~2.7×** — 11.7 giờ xuống còn 4.4 giờ.

Với A100 thì chọn A100 là hợp lý. Chỉ cần nhớ A100 trên Colab Pro **không phải lúc nào cũng
có**; xem mục "Nếu không xin được A100" ở cuối.

---

## Tổng quan

### Chi phí cố định mỗi session — đừng bỏ sót

Mỗi lần mở một notebook GPU mới, trước khi tính toán gì đã mất:

| Việc | Thời gian | ~CU |
|---|---:|---:|
| `pip install` unsloth + vLLM | **12–20 phút** | ~3.5 |
| Nạp Qwen3-8B 4-bit vào vLLM | ~4 phút | ~0.8 |
| **Cộng mỗi session** | **~22 phút** | **~4.3** |

Restart runtime **không** phải cài lại (chỉ nạp lại model), nên `03` với hai lần restart chỉ
tốn một lần cài.

### Tổng — đã tính cả chi phí cố định

| Gói | Notebook | Session GPU | Chi phí cố định | Tính toán | Giờ A100 | ~CU |
|---|---|:--:|---:|---:|---:|---:|
| **A — kết quả chính** | 00, 01, 02, 04, 05, 07 | 4 | 1.5 h | 2.0 h | 3.4 | **~40** |
| **B — SFT** | 03 | 1 | 0.5 h | 1.2 h | 1.7 | **~20** |
| **C — ma trận tổ hợp** | 04b, 05b, 06 | 3 | 1.1 h | 1.7 h | 2.8 | **~32** |
| | **Tổng** | **8** | **3.1 h** | **4.9 h** | **7.9** | **~92 / 100** |

**Chi phí cố định chiếm ~36 CU, tức 40 % ngân sách** — gần bằng cả gói B. Ước tính trước đó
(~57 CU) chỉ đếm thời gian tính toán nên thấp hơn thực tế đáng kể.

Với 100 CU/tháng thì chạy trọn cả ba gói là **vừa khít, không còn dự phòng**. Hai lựa chọn:

| Cách | Bỏ gì | Còn lại | Mất gì |
|---|---|---:|---|
| **Khuyến nghị** | Chạy A + B trước, quyết định C sau khi xem kết quả | ~60 CU dùng, ~40 dự phòng | Chưa có ma trận tương tác — nhưng đã đủ một bài hoàn chỉnh |
| Chạy hết | — | ~92 CU dùng, ~8 dự phòng | Một lần chạy lại vì lỗi là hết ngân sách |

Đừng mở notebook GPU chỉ để "xem thử" — mỗi lần mở là 4.3 CU dù không chạy gì.

### Thứ tự chạy khác với số thứ tự notebook

```
00 → 01 → 02 → 04 → 05 → [KẾT QUẢ CHÍNH] → 03 → 04b → 05b → 06 → 07
```

**`03` (SFT) chạy SAU `05` (ACE)**, dù số nhỏ hơn. Ba lý do:

1. Câu hỏi chính của bài là **ACE có cải tiến không** — `05` trả lời, `03` thì không.
2. `03` đắt nhất (14 CU) mà lần chạy trước đã cho thấy fine-tuning thua inference-time.
3. `03` không phải điều kiện của `04`/`05` trên model gốc — chỉ cần cho gói C.

Nếu hết ngân sách giữa chừng, dừng sau `05` vẫn có một bài hoàn chỉnh.

---

## Gói A — kết quả chính (1.9 giờ, ~23 CU)

Chạy gọn trong **một buổi**.

| # | Notebook | Runtime | Thời gian | CU | Sinh ra |
|---|---|---|---:|---:|---|
| 1 | `00_data_audit` | **CPU** | 1 phút | 0 | Chốt thước đo + bảng đính chính EA |
| 2 | `01_baseline_basic` | A100 | 12 phút | 2.4 | Mốc dưới — prompt cơ bản |
| 3 | `02_prompt_engineering` | A100 | 12 phút | 2.4 | Δ hướng dẫn từ khoá + few-shot |
| 4 | `04_self_evaluation` | A100 | 18 phút | 3.5 | **Tái lập mốc tham chiếu** |
| 5 | `05_ace` pha A | A100 | 45 phút | 8.8 | Playbook |
| 6 | `05_ace` pha B | A100 | 18 phút | 3.5 | **Δ của ACE** |
| 7 | `05_ace` §8 đối chứng | A100 | 18 phút | 3.5 | Truy hồi có giá trị không |
| 8 | `07_final_report` | **CPU** | 1 phút | 0 | Bảng + biểu đồ |

Ở bước 4, để `USE_SFT_ADAPTER = False`. Ở bước 7, `RUN_RANDOM_CONTROL = True`.

### Bốn cổng kiểm tra

| Sau bước | Kiểm tra | Ngưỡng | Nếu trượt |
|---|---|---|---|
| 1 | `00` §2: executor tái tạo gold trên test | **100 %** | Dừng hẳn. Mọi số sau đó vô nghĩa. |
| 3 | `02` PA_loose | **~51 %** (tham chiếu: 51.34 %) | Lệch > 5 điểm ⇒ sai môi trường. Kiểm tra `transformers==4.56.2`, `temperature=0.1`, và dòng `[PROMPT]`. |
| 4 | `04` PA_loose | **~59.5 %** (tham chiếu: 59.56 %) | Đây là mốc tái lập chính. Sai ở đây thì đừng tin nấc 5. |
| 5 | `05` pha A: số bullet | **≥ 8**, dev composite dương | Rỗng ⇒ đọc `qg_reasons`. Nhiều `chua_so_lieu_cu_the` = Reflector chép đáp án; nhiều `verify_khong_sua_duoc` = bullet nghe hay nhưng vô dụng. |

Cổng thứ 3 quan trọng nhất: nó xác nhận hạ tầng mới tái lập được mốc tham chiếu. Chỉ tốn
37 phút để tới đó.

---

## Gói B — SFT (1.2 giờ, ~14 CU)

`03_sft_qwen3` có **ba phần, phải restart runtime hai lần**:

| Phần | Việc | Thời gian | Chế độ |
|---|---|---:|---|
| A (§1–§5) | Sinh lời giải trên 2000 mẫu train → lọc → ghi JSONL | 36 phút | vLLM |
| ⚠ | **Runtime → Restart session**, chạy tiếp từ §6 | | vLLM không nhả VRAM cho training |
| B (§6–§9) | Huấn luyện LoRA (batch 8 × accum 2 trên A100 40GB) | 28 phút | training |
| ⚠ | **Restart lần nữa**, chạy §1–§3 rồi §10 | | quay lại vLLM để chấm |
| C (§10) | Nạp adapter, chấm trên test | 9 phút | vLLM + `load_lora` |

**Cổng kiểm tra:**

| Kiểm tra | Ngưỡng | Nếu trượt |
|---|---|---|
| §5: số mẫu SFT giữ lại | **≥ 800** (lý tưởng 1000–1200) | < 200 notebook tự dừng. Tăng `SFT_TRAIN_SUBSET` hoặc `ACCEPT="ea"`. |
| §5: phân bố số phép toán | có mẫu ≥ 3 bước | Toàn bài 1 bước ⇒ model chỉ làm đúng bài dễ, SFT không dạy được gì mới. |
| §9: đường cong val loss | **không tăng ngược** | Tăng ngược = overfit như lần chạy trước. Giảm epoch, hoặc `ACCEPT="pa"` cho dữ liệu sạch hơn. |

Cả ba khả năng ở §10 (SFT hơn / ngang / kém nấc 2) đều là kết quả đáng báo cáo — bảng cuối
notebook `03` giải thích viết gì cho từng trường hợp.

---

## Gói C — ma trận tổ hợp (1.5 giờ, ~20 CU)

Trả lời câu hỏi đáng giá nhất: **SFT và ACE đều học từ train — bổ sung hay trùng nhau?**

| # | Việc | Cấu hình | Thời gian | CU |
|---|---|---|---:|---:|
| 1 | `04_self_evaluation` lần 2 | `USE_SFT_ADAPTER = True` | 18 phút | 3.5 |
| 2 | `05_ace` lần 2, pha A + B | `USE_SFT_ADAPTER = True` | 63 phút | 12.4 |
| 3 | `06_combination` | tự chạy 2 ô còn thiếu | 18 phút | 3.5 |

`06` **đọc lại 6 ô đã có từ đĩa**, chỉ sinh 2 ô mới — ACE **không kèm** self-eval.

**Bốn kết quả có thể ra, và viết gì cho từng cái:**

| Quan sát | Kết luận |
|---|---|
| Tương tác SFT×ACE **âm rõ** | Hai cái học cùng thứ ⇒ chọn một. ACE nếu muốn khỏi huấn luyện. |
| Tương tác SFT×ACE **≈ 0** | Trọng số và ngữ cảnh là hai kênh độc lập ⇒ cộng dồn được. Kết quả đáng giá. |
| **`E+A` ≥ `E+S`** | ACE **thay được** self-eval với nửa chi phí (1 lượt sinh thay vì 2). Luận điểm mạnh nhất cho bối cảnh tài nguyên hạn chế. |
| `F+S+A` tốt nhất, p < 0.05 so với `E+S` | Tổ hợp đầy đủ thắng cấu hình tham chiếu ⇒ con số chính của báo cáo. |

---

## Nếu thiếu ngân sách — cắt theo thứ tự

| Ưu tiên | Chạy | CU cộng dồn | Đủ để nói gì |
|:---:|---|---:|---|
| 1 | 00, 01, 02 | 4 | Prompt engineering đóng góp bao nhiêu |
| 2 | + 04, 07 | 7 | Tái lập được mốc tham chiếu trên hạ tầng mới |
| 3 | + 05 pha A + B | 19 | **ACE có cải tiến không** — câu hỏi chính |
| 4 | + 05 đối chứng ngẫu nhiên | 23 | Cải tiến do truy hồi hay chỉ do prompt dài |
| 5 | + 03 (SFT) | 37 | Đủ cả 5 nấc |
| 6 | + gói C | 57 | Tương tác giữa các kỹ thuật |

**Đừng cắt mục 3 và 4** — đó là phần mới của bài, và đối chứng ngẫu nhiên là câu người phản
biện chắc chắn sẽ hỏi.

---

## Riêng cho A100

| Việc | Ghi chú |
|---|---|
| Cấu hình tự nhận | `max_seq=15000` (đúng notebook gốc), `batch=500`, `max_num_seqs=64` (80GB: 128) |
| Ngữ cảnh | **Không bị cắt.** Phải thấy `[PROMPT] ✅ mọi prompt đều lọt ngân sách` |
| SFT | batch 8 × accum 2 (40GB) hoặc 16 × 1 (80GB) — **batch hiệu dụng vẫn 16** để lr 2e-4 còn hợp lệ |
| bf16 | A100 là CC 8.0 nên có bf16; `03` tự bật |
| Background execution | Bật trong Pro để đóng tab không chết session |

### Nếu không xin được A100

L4 chạy được y hệt, **cùng chi phí CU**, chỉ lâu hơn ~2.5× (11.7 giờ thay vì 4.4). Cấu hình
L4 đã được chỉnh lên `max_seq=13500` nên **cũng không cắt ngữ cảnh** → kết quả A100 và L4
so sánh trực tiếp được với nhau, trộn chung trong một bảng cũng hợp lệ.

**T4 thì khác**: 15 GB không đủ, notebook sẽ tự cắt ngữ cảnh và in cảnh báo. Kết quả từ T4
**không** trộn chung được. Nếu buộc phải dùng T4, chỉ dùng cho pha A của `05` (xây playbook),
đừng dùng cho các con số báo cáo.

---

## Rủi ro và cách xử lý

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| Session ngắt giữa pha A của `05` | | Checkpoint mỗi vòng. Chạy lại đúng cell là tiếp tục. |
| `03` quên restart runtime | OOM ở phần B | Restart rồi chạy từ §6. Dữ liệu SFT đã ghi ra Drive. |
| Playbook rỗng | `qg_reasons` toàn `chua_so_lieu_cu_the` | Reflector đang chép đáp án. Đổi `REFLECTOR_BACKEND="gemini"` — nhưng khi đó **phải báo cáo ở nhóm unconstrained**, vì thiết lập constrained của dự án cấm API ngoài. |
| OOM khi nạp model | | `GPU_MEM_UTIL = 0.80`, `MAX_NUM_SEQS = 32`. |
| Δ nhỏ, p ≥ 0.05 | | Với n = 497, chênh lệch < ~2 điểm EA thường không đạt. Là **giới hạn cỡ mẫu**, không phải kỹ thuật thất bại. Cách cứu rẻ nhất: chạy thêm trên `valid` (584 mẫu) rồi gộp → n = 1.081. |

---

## Đọc kết quả cho đúng

* Chỉ **`PA_loose`** so trực tiếp được với cột PA tham chiếu. EA tham chiếu tính bằng
  executor lỗi `table_*` nên thấp hơn thực tế — notebook `00` định lượng chênh lệch.
* Chạy nhiều cấu hình thì dễ có cái "đạt p < 0.05" do may mắn. Kết luận mạnh chỉ nên dựa vào
  **các so sánh đã định trước**: từng bước leo thang, `05` vs `04`, và truy hồi vs ngẫu nhiên.
* Báo cáo kèm **chi phí**: lượt sinh mỗi mẫu, có cần huấn luyện không, có cần API ngoài không.
  Dự án đặt trong bối cảnh tài nguyên hạn chế nên cột này quan trọng ngang EA.

## Bảng cần có khi viết bài

`07_final_report` xuất sẵn hai CSV. Bài cần ít nhất:

1. **Thang bậc** — 5 nấc × (EA, PA_strict, PA_loose, lượt sinh/mẫu).
2. **Kiểm định** — mỗi bước: Δ, KTC 95 %, p-value, số mẫu bất đồng.
3. **Đính chính EA** (notebook `00`) — 5 model tham chiếu, EA cũ vs EA chấm đúng.
4. **Ma trận tổ hợp + tương tác** (nếu chạy gói C).
5. **Playbook học được** — in nguyên văn vào phụ lục; đó là thứ trực quan nhất để người đọc
   thấy ACE học được gì.
