# Bàn giao — ViNumQA: đo tác động của từng kỹ thuật lên suy luận số học tài chính tiếng Việt

> **Tài liệu này để mở một phiên làm việc mới. Đọc hết §1–§4 và §9 trước khi sửa bất cứ thứ gì.**
> Mốc: 187 test · 8 notebook · 111 ô code · nhánh `main`.
> Kiểm nhanh mọi thứ trong tài liệu này còn đúng không: `python tools/kiem_tra.py`

---

## 1. Bài toán

Cho một câu hỏi số học trên báo cáo tài chính tiếng Việt, kèm văn bản trước/sau bảng và
một bảng số liệu, **model phải sinh ra một CHƯƠNG TRÌNH tính toán** (không tự tính nhẩm),
rồi chương trình đó được chạy bằng máy để ra đáp án.

### DSL

```
add(a,b)  subtract(a,b)  multiply(a,b)  divide(a,b)  exp(a,b)  greater(a,b)
table_max(nhãn,none)  table_min(nhãn,none)  table_sum(nhãn,none)  table_average(nhãn,none)
```

| Quy tắc | |
|---|---|
| Nối bước | `#0` = kết quả phép thứ nhất, `#1` phép thứ hai… |
| **Không lồng nhau** | `add(1,0.15), divide(5310,#0)` ✅ — `divide(5310, add(1,0.15))` ❌ |
| **`table_*` đọc theo NHÃN HÀNG** | tham số là chuỗi ở **ô đầu mỗi dòng**, không phải tên cột |
| Một đầu ra | phép cuối cùng là đáp án; không thừa phép |
| Định dạng | trả lời trong khối ` ```plaintext ` gồm đúng 2 dòng `program:` và `answer:` |

### Dữ liệu

| | train | valid | **test** |
|---|--:|--:|--:|
| số mẫu | 2 993 | 584 | **497** |

Phân bố tập test theo độ phức tạp của gold: **1 phép 319 · 2 phép 161 · 3 phép 10 ·
4 phép 4 · 5 phép 3**. Trong đó **61 mẫu dùng `table_*`** (59 nằm ở ô 1 phép).

### Chỉ số

| | nghĩa |
|---|---|
| **EA** | chạy chương trình ra **đúng số** |
| **PA_strict** | chương trình **khớp gold** sau chuẩn hoá (giao hoán `add`/`multiply`, `100.00`→`100`, `20%`→`0.2`) |
| PA_loose | bản so chuỗi **cũ**, giữ cho tương thích — trượt khi chỉ khác cách viết số |

> ⚠ `PA_loose` **không** phải bản nới lỏng của `PA_strict`. Hai normalizer khác nhau, nên
> `PA_loose < PA_strict` là bình thường. **Đọc `PA_strict` là con số thật.**

### Môi trường

`unsloth/Qwen3-8B` 4-bit, vLLM qua unsloth, Colab A100-40GB. Code + data clone thẳng từ
GitHub; kết quả ghi lên Drive.

---

## 2. Mục tiêu người dùng đặt ra

1. 🎯 **Hai ô ma trận tổ hợp (`06_comb_E_A`, `06_comb_F_A`) phải đạt > 70 % EA VÀ > 70 % PA.**
2. Kết quả phải **nhìn rõ tác động thực sự của từng phương pháp**. Nấc nào cho Δ nằm
   trong nhiễu thì **phải nói thẳng là nằm trong nhiễu**, không tô thành "cải tiến".
3. Tối ưu tốc độ inference, tận dụng tài nguyên.

### Ràng buộc bắt buộc giữ

- **Dự án chạy độc lập.** Không nhắc bài báo gốc / FJCAI / "đã công bố" ở bất kỳ đâu. Số
  liệu cũ vẫn dùng nhưng gọi là **"mốc tham chiếu"**. PDF bài báo gitignore, không được
  vào lịch sử git.
- Dữ liệu **không** riêng tư → code *và* `data/` đều nằm trong repo công khai.
- `OPENAI_API_KEY` (Reflector của ACE, `gpt-4o-mini`) nạp qua **Colab Secrets**, tuyệt
  đối không viết vào notebook.
- Nấc 3 dùng **rejection sampling**, không gọi API ngoài — giữ thiết lập *constrained*.

---

## 3. Trạng thái

### Đã chạy thật (A100-40GB, cấu hình CŨ trần 8192)

| nấc | EA | PA_strict | PA_loose | phút |
|---|--:|--:|--:|--:|
| `01_basic` — prompt cơ bản | 44,67 | 42,05 | 39,84 | 17,1 |
| `02_prompt_eng` — prompt hoàn chỉnh | **64,79** | **60,36** | 57,34 | 16,4 |
| `04_selfeval_base` — + self-eval | 64,99 | 60,36 | 57,75 | 42,4 |

| bước | Δ EA | KTC 95 % | p | kết luận |
|---|--:|---|--:|---|
| nấc 1 → 2 | **+0,2012** | [+0,151, +0,252] | 0,000 | ✅ có ý nghĩa |
| nấc 2 → 4 | +0,0020 | [−0,026, +0,030] | 1,000 | ⚠ **không tách được khỏi nhiễu** |

Phân rã lỗi ở nấc 2 (497 mẫu): `dung` 300 · **`sai` 136** · `khong_co_program` 29 ·
`dung_nhung_khac_program` 22 · `program_khong_chay_duoc` 10.

### Chưa chạy

`02c` · `03` (SFT) · `05` (ACE) · `05c` · `04b` · `05b` · **hai ô ma trận `06`**.

### ⚠ Cấu hình đã đổi sau các lần chạy trên → **phải chạy lại từ nấc 1**

```
TEMPERATURE 0.1   MAX_TOKENS 4096   MAX_SEQ_LENGTH 17000
REPETITION_PENALTY 1.0   RANDOM_SEED 42
A100-40GB: GPU_MEM_UTIL 0.85 · MAX_NUM_SEQS 48 · BATCH_SIZE 512
```

### Thang bậc hiện tại (12 nấc)

```
01_basic · 02_prompt_eng · 02c_prompt_v2 · 03_sft · 04_selfeval_base · 04_selfeval_sft
05_ace_base · 05_ace_sft · 05c_ace_basic_base · 05_ace_random_base
06_comb_E_A · 06_comb_F_A
```

### Thang prompt (lồng nhau — có test canh)

| mức | thêm gì so với mức trên | ký tự |
|---|---|--:|
| `plain` | *đã rời thang bậc* — prompt trần, thiếu cả quy tắc định dạng | 530 |
| `basic` | mở đầu + danh sách phép toán + yêu cầu định dạng | 2 535 |
| `no_fewshot` | + ánh xạ từ khoá → phép toán (14 mục) — *bậc giữa, không chạy* | 5 408 |
| `engineered` | + 2 ví dụ mẫu có khung | 5 924 |
| `engineered_v2` | nhánh rẽ: sửa 13 chỗ dạy ngược về `table_*` | 6 186 |

`basic ⊂ no_fewshot ⊂ engineered` là **chèn thuần** — mỗi bước chỉ THÊM, phần dùng chung
giống nhau **từng ký tự**. Sửa chữ ở phần chung là hỏng phép đo mà không ai thấy.

---

## 4. Những gì đã rút ra — phần đắt nhất, đọc kỹ

### 4.1 Đã xác minh bằng dữ liệu

**a) Prompt đang dạy NGƯỢC về `table_*`.**

```
61/61 mẫu test dùng table_* có nhãn khớp NHÃN HÀNG (ô đầu mỗi dòng)
 0/61 khớp tên cột
```

Prompt lại nói "cột" ở mọi chỗ, tệ nhất là *"table_ functions chỉ nhận đúng 1 cột, KHÔNG
THÊM HÀNG"* — lái thẳng khỏi đáp án đúng. Executor `table_row_values` đọc theo nhãn hàng.
**61 mẫu = 12,3 % tập test.** Đã làm thành nấc `02c`, **chưa đo**.

**b) Câu 1 phép đang KHÓ hơn câu 2 phép — vô lý, và là manh mối chính.**

| số phép | mẫu | EA nấc 2 |
|---|--:|--:|
| 1 phép | 319 | **63,0 %** |
| 2 phép | 161 | **71,4 %** |

Nếu ô 1 phép đạt bằng ô 2 phép → **EA = 70,2 %**, vừa đúng mốc. `table_*` chiếm 18 % ô
đó và là nghi can số một.

**c) Trần token 4096 đã bão hoà.** Nấc 2 chạy hai lần, cùng prompt cùng GPU, chỉ khác trần:

| | 4096 | 8192 |
|---|--:|--:|
| lượt bị cắt | 30 | 30 |
| mẫu mất trắng | 28 | 28 |
| EA | ,6479 | ,6479 |

Gấp đôi ngân sách cứu **đúng 0 mẫu**. Chúng cũng **không lặp** (trung vị lặp 0,0) →
`repetition_penalty` cũng không phải thuốc. **Đừng nâng trần nữa.**

**d) Self-eval hiện cho ĐÚNG 0.** Bất đồng **0 mẫu**, KTC `[+0,0000, +0,0000]`. Không
phải "hiệu ứng nhỏ" — là **không có hiệu ứng**. Nguyên nhân: prompt bước 2 gần như toàn
quy tắc **định dạng**, mà sau nấc 2 lỗi định dạng chỉ còn 10/497. Nó nhắm vào mục tiêu
đã không còn tồn tại.

**e) Chế độ suy nghĩ Qwen3 phải để template tự quyết** (`enable_thinking = None`). Ép
`False` từng làm mất ~10 điểm PA. Dấu hiệu: 497 mẫu xong trong ~1 phút thay vì ~16 phút.

**f) Sàn nhiễu ~1,6 điểm EA**, đo từ hai lần chạy input giống hệt nhau.

**g) Tương tác trong ma trận 2×2×2 gần như KHÔNG tách được khỏi nhiễu.** Thử với tương
tác thật −0,05 (5 điểm): KTC vẫn chứa 0. Đừng viết "cộng hưởng"/"trùng nhau" vào báo cáo
khi KTC chứa 0.

### 4.2 Ba lần đoán mù đều SAI — đừng lặp lại

| đoán | thực tế |
|---|---|
| "prompt trần hỏng vì thiếu quy tắc định dạng" | `basic` có đủ quy tắc mà vẫn 33 % program không chạy được |
| "164 mẫu không chạy được chủ yếu do nhãn bảng sai" | nhãn bảng **7 mẫu**; thủ phạm là `cu_phap_di_dang` **120 mẫu** |
| "nâng trần token sẽ cứu mẫu bị cắt" | cứu được **0** |

> **Bài học: không sửa prompt / ACE khi chưa có bảng phân loại lỗi.** Công cụ chẩn đoán
> đã có đủ (§5) — chạy `07` lấy số rồi mới sửa.

---

## 5. Công cụ chẩn đoán đã cài — dùng chúng

Mọi bảng dưới đây **tính lại được từ jsonl của nấc ĐÃ CHẠY XONG**, chạy `07` trên CPU
~2 phút, không tốn GPU.

| bảng | hàm | trả lời |
|---|---|---|
| `vi_sao_khong_co_program` | `phan_loai_khong_co_program` | bị cắt vì trần hay sai định dạng; kèm **mức lặp** |
| `vi_sao_khong_chay_duoc` | `phan_loai_khong_chay_duoc` | executor từ chối vì đâu — 6 nhóm, **kèm program hỏng thật** |
| `vi_sao_sai` | `phan_loai_sai` | sai kiểu gì — thiếu/thừa bước, nhầm phép, **dãy phép trùng khít chỉ sai toán hạng**, bỏ qua/lạm dụng `table_*` |
| `by_phep` | `nhom_phep` | EA/PA theo **loại** phép — `by_steps` gộp theo số phép nên `table_*` vô hình |
| `bi_cat_theo_buoc` | — | bước 1 vs bước 2; bước 2 cắt nhiều hơn = phương pháp bị **pha loãng**, Δ là cận dưới |
| bảng bước 2 | `so_sanh_hai_buoc` | bước 2 **đổi** bao nhiêu program: chép y nguyên / đổi giữ giá trị / đổi cả giá trị |
| tác động chính + tương tác | `ktc_hieu_ung`, `ktc_tuong_tac` | cho **cả EA lẫn PA**, kèm KTC bootstrap |
| — | `bo_sung_ly_do` | điền bù lý do cho nấc đã chạy, phân tích lại không cần GPU |

`07` còn có **bảng kiểm công bằng**: cảnh báo khi các nấc khác GPU / khác trần / khác chế
độ suy nghĩ, và khi **thiếu metadata**. `06` có **cổng 70/70** in thẳng còn thiếu bao
nhiêu mẫu.

---

## 5b. ACE hoạt động thế nào trong repo này

Cần biết trước khi "cải tiến ACE" — nếu không sẽ sửa mù.

**Vòng lặp:** Generator (Qwen3 sinh program) → **Reflector** (`gpt-4o-mini` qua API, đọc
ca sai rồi đề xuất một chiến lược) → **Curator** (nhận/loại bullet) → **Verify** (sinh lại
với bullet mới, giữ nếu sửa được) → nhập playbook.

| Thành phần | Tham số | Ghi chú |
|---|---|---|
| Playbook | trần **30 bullet**, tối đa **3 bullet/cụm** | cụm = 11 loại lỗi, xem `ace/clusters.py` |
| Cụm lỗi | `C1_ty_trong` `C2_tang_truong` `C3_chenh_lech` `C4_dao_nguoc_tang_truong` `C5_tong_nhieu_ky` `C6_quy_doi_don_vi` `C7_max_min_bang` `C8_trung_binh_bang` `C9_ty_le_don_gian` `C10_nhieu_buoc` `C11_khac` | |
| Truy hồi | Tier-1 + Tier-2, `k = TOP_K_TIER1 + TOP_K_TIER2` | có **đối chứng bullet ngẫu nhiên** |
| Quality gate | `dedup_thresh 0.98` · `min_ops 1` | xem bẫy bên dưới |
| Verify | sinh lại **2 lượt**, chỉ sinh lại ứng viên còn trượt | temperature 0.1 vẫn ngẫu nhiên |
| Cách ly | chỉ áp cho bullet bị **gate** loại (lỗi nội tại) | KHÔNG áp cho trượt verify (ngẫu nhiên) hay quota |
| Reflector | biết **luật prompt đã có** (đừng đề xuất lại) + thấy **một ca cùng cụm đã làm đúng** | truy hồi phản chứng |

### Bẫy ACE đã dẫm phải

| Bẫy | Chi tiết |
|---|---|
| **Ngưỡng tương đồng KHÔNG chuyển được giữa hai embedder** | MiniLM/bge có sàn ~0,1–0,3; multilingual-e5 có sàn ~0,70–0,90. Đổi embedder mà giữ ngưỡng cũ → 113/217 bullet bị loại oan vì "trùng ngữ nghĩa" với một playbook chỉ có 1 bullet. |
| `min_ops = 2` chặn oan | 64 % tập test là câu **1 phép**; yêu cầu bullet phải nhắc 2 phép là loại sạch lời khuyên cho phần lớn dữ liệu. Nay `min_ops = 1`. |
| Đối chứng ngẫu nhiên **thoái hoá** | playbook 3 bullet mà k = 7 → cả hai bên lấy TOÀN BỘ bullet, prompt giống hệt nhau. Notebook tự phát hiện và gọi đúng tên: đó là **phép đo nhiễu**, không phải tác dụng của truy hồi. |
| Dev set 120 mẫu quá ồn để chọn snapshot | composite dao động 0,6383–0,6717 trên **cùng một** playbook 3 bullet. Nay `DEV_SUBSET = 240`, `EVAL_EVERY_ROUNDS = 5` (cùng chi phí). |
| ACE chồng lên prompt hoàn chỉnh gần như hết đất | prompt engineered **đã chứa sẵn** ánh xạ từ khoá → phép toán, tức đúng loại tri thức ACE định khám phá. Vì vậy mới có **nấc 5c** (`ACE_TREN_PROMPT = "basic"`): ở đó khoảng trống là 156 mẫu chứ không phải 10. |

---

## 6. Cải tiến đã cài nhưng CHƯA ĐO

| # | Cải tiến | Kỳ vọng |
|---|---|---|
| 1 | **Nấc 2c** `engineered_v2` — sửa chỗ prompt dạy ngược về `table_*` | 61 mẫu (12,3 %) |
| 2 | **Vớt mẫu bị cắt** — sinh lại với suy nghĩ TẮT, chỉ thay khi ra được program | tối đa +5,8 điểm EA/nấc |
| 3 | **Self-eval biết kết quả thực thi** — bước 2 nay thấy `⚙ Chạy thật … thì ra: 15000.0` | biến nấc 4 từ Δ=0 thành đo được |
| 4 | **Tốc độ** — một lô 512, `enable_prefix_caching`, `max_num_seqs` 48 | log cũ: lô 400 chạy 1,88 s/mẫu, lô 97 còn lại 2,83 s/mẫu |

---

## 7. Kiểm tra phần tính ma trận — đã làm

**Toán thì ĐÚNG.** Dựng ma trận 2×2×2 có đáp án đặt tay (SFT +0,10 · SE +0,04 · ACE +0,06
· SFT×ACE −0,05), code lấy ra **chính xác** cả ba tác động chính lẫn tương tác −0,0500.
McNemar đối chiếu với công thức nhị thức hai phía: khớp tới 1e-12 trên 5 ca. Bootstrap
lấy mẫu lại **theo cặp** (đúng), và `compare_pair` ném lỗi nếu hai nấc lệch thứ tự mẫu.

**Ba thiếu sót đã sửa:**

1. Tác động chính và tương tác chỉ tính **EA** → nay tính **cả PA_strict**.
2. Ngưỡng cứng ±0,01 (1 điểm EA) để phán "cộng hưởng"/"trùng nhau" — nằm **trong** sàn
   nhiễu 1,6 điểm → nay dùng **KTC bootstrap**, KTC chứa 0 thì nói thẳng là chưa tách
   được khỏi nhiễu.
3. Không có cổng kiểm **70/70** → nay in thẳng còn thiếu bao nhiêu mẫu mỗi chỉ số.

---

## 8. Các bước tiếp theo

### Bước 0 — lấy số, KHÔNG tốn GPU (làm trước tiên)

`git pull` → chạy **`07`** trên CPU. Bốn bảng ở §5 hiện ra trên các nấc đã chạy.
**Dòng `table_*` trong `by_phep` là con số quan trọng nhất** — xác nhận hoặc bác bỏ §4.1a,
và quyết định có đáng đổ công vào nấc 2c không.

### Bước 1 — chạy lại thang bậc

Thứ tự: `01` → `02` → `02c` → `04` → `05` → `07`, rồi SFT (`03` A/B/C), rồi `5c`, `4b`,
`5b`, cuối cùng ma trận `06`. Thao tác chi tiết, số ô, cổng kiểm: **`HUONG_DAN_COLAB.md`**.

### Bước 2 — cải tiến CÓ CĂN CỨ

Chỉ làm **sau khi** có `vi_sao_sai` và `by_phep`:

| nếu chi phối là | thì sửa |
|---|---|
| `table_*` thấp | nấc 2c là đường tới 70 %; cân nhắc gộp vào prompt chính |
| `dung_so_buoc_sai_phep` | ánh xạ từ khoá → phép toán |
| `dung_phep_sai_so_lieu` | dạy **đọc bảng** — model hiểu đúng bài, lấy nhầm số |
| `thieu_buoc` | dạy lập kế hoạch nhiều bước; ACE cũng có đất ở đây |

### Ngân sách tới 70/70

Từ nấc 2 (64,79 EA / 60,36 PA) cần **+26 mẫu EA / +48 mẫu PA**:

| nguồn | mẫu | EA | PA |
|---|--:|:-:|:-:|
| `khong_co_program` (bị cắt) → vớt | 29 | ✓ | ✓ |
| `dung_nhung_khac_program` → chuẩn hoá dạng | 22 | — | ✓ |
| `program_khong_chay_duoc` | 10 | ✓ | ✓ |
| **`sai`** (suy luận) | **136** | ✓ | ✓ |
| `table_*` sửa bởi nấc 2c | 61 *(chồng lấn)* | ✓ | ✓ |

Cơ học thuần **chưa đủ** — phải chuyển hoá được một phần ô `sai`, cộng đóng góp của SFT
và ACE. **PA là chỉ số khó hơn** (+48 so với +26).

---

## 9. Lời nhắc để không đi lạc hướng

### Về phương pháp

1. **Không đoán khi có thể đo.** Ba lần đoán mù gần nhất đều sai. Công cụ chẩn đoán đã có
   đủ và chạy trên CPU trong 2 phút.
2. **Kết quả âm tính là kết quả.** Nấc 4 cho Δ=0 thì báo cáo đúng như vậy kèm lý do. Đừng
   tô nhiễu thành cải tiến — đó là yêu cầu số 2 của người dùng.
3. **KTC chứa 0 = không kết luận được gì.** Áp cho cả tác động chính lẫn tương tác.
4. **Thêm nấc mới thay vì sửa đè nấc cũ.** Nấc 2c không sửa đè nấc 2; nhờ vậy chỗ sửa
   thành một con số đo được thay vì lặng lẽ đổi rồi không biết nó đóng góp bao nhiêu.
5. **Thay đổi tham số ảnh hưởng kết quả thì phải chạy lại mọi nấc liên quan.** Chốt chống
   ghi đè canh `MAX_TOKENS` và `MAX_SEQ_LENGTH`; `GPU_MEM_UTIL`/`MAX_NUM_SEQS`/`BATCH_SIZE`
   chỉ đổi tốc độ (mỗi request có seed riêng) nên đổi thoải mái.

### Lệnh kiểm bắt buộc trước mỗi commit

```bash
python -m pytest tests/ -q -W error     # 187 test, CPU, ~2 giây
python tools/kiem_tra.py                # notebook + cấu hình + thang prompt + mã chết
python tools/kiem_tra.py --day-du       # + chạy trọn notebook 07 với nấc dựng sẵn
```

`tools/kiem_tra.py` trả mã thoát khác 0 nếu có lỗi. Nó kiểm những thứ `pytest` **không**
kiểm được: LADDER có giống hệt nhau ở cả 8 notebook không, trần token có lệch giữa các ô
cấu hình không, ngân sách ngữ cảnh còn dư bao nhiêu, thang prompt còn lồng nhau không, và
notebook có sót output không.

### Vòng làm việc với người dùng

1. Người dùng chạy notebook trên Colab, **dán khối meta** (notebook tự in giữa hai đường
   kẻ ngang) hoặc CSV của `07` vào hội thoại.
2. Đọc số → đánh giá → **chỉ sửa khi có bảng phân loại lỗi**, không đoán.
3. Sửa xong: chạy 3 lệnh trên, commit, push. Người dùng `git pull` trên Colab rồi chạy tiếp.

Người dùng viết tiếng Việt và mong trả lời tiếng Việt. Compute Colab có hạn — mỗi lượt
chạy GPU là tiền thật, nên **đừng đề nghị chạy lại khi chưa cần**.

### Sản phẩm cuối

Bảng thang bậc + kiểm định McNemar + ma trận tổ hợp 2×2×2 (tác động chính, tương tác,
cổng 70/70), đủ để viết thành báo cáo. `07` xuất `bang_ket_qua_*.csv`, `kiem_dinh_*.csv`,
`bao_cao_*.png`; `06` xuất `ma_tran_to_hop_*.csv`.

### Về quy trình

6. Mọi thay đổi phải qua **cả ba lệnh kiểm ở trên**. Đừng commit khi chưa xanh.
7. Notebook sửa bằng **script Python thao tác JSON**, không sửa tay — 10 ô bootstrap phải
   giống hệt nhau, có bộ kiểm canh LADDER đồng nhất ở cả 8 notebook.
8. **Không dùng `open(p, "w")` để ghi đè file nguồn.** Nó **cắt cụt file NGAY khi mở**,
   trước cả khi kiểm tham số — đã từng làm mất trắng `prompts.py`. Ghi ra file tạm rồi
   `os.replace`.
9. **Bash heredoc nuốt `\n`** trong chuỗi Python. Dùng công cụ Write cho mã nhạy escape,
   hoặc `chr(10)` / `chr(92)`.

### Bẫy môi trường

| Bẫy | Cách tránh |
|---|---|
| `vllm` từ PyPI là bản CUDA 13, Colab là CUDA 12.8 | Cài wheel `+cu129` từ GitHub releases; ô `#3` phải in đúng đuôi đó |
| `sentence-transformers` 6.x âm thầm nâng `transformers` lên 5.x, vỡ vLLM | Ghim `5.7.0`; self-test dừng nếu `transformers` không phải 4.x |
| `%%capture` giấu một lần cài hỏng suốt 24 phút | Không dùng `%%capture` ở ô cài đặt |
| vLLM `CUDA error: invalid argument` khi dựng engine | **Không** phải tải model hỏng. Restart session; notebook tự lùi về `enforce_eager` + util 0.80 |
| Checkpoint ACE nối tiếp nhầm cấu hình mới | Phải thấy `[RESUME] ⚠ CẤU HÌNH ĐÃ ĐỔI`; thấy `tiếp từ vòng N` thì xoá `progress_*.json` |
| trl đổi API giữa các bản | `03` dò chữ ký thật bằng `inspect`, chạy được cả `≤0.22` và `0.24` |

---

## 10. Bố cục repo

```
vinumqa/            lõi: dsl · data · prompts · pipeline · sft · stats · io_utils · ace/
notebooks/00–07     mỗi nấc một notebook; 00 và 07 chạy CPU
tests/              187 test, CPU, ~2 giây, KHÔNG cần GPU
data/               ViNumQA (24 MB, nằm luôn trong repo)
reference/          hồ sơ xuất xứ + 5 CSV mốc tham chiếu (KHÔNG tái tạo được nếu mất)
BAN_GIAO.md         tài liệu này
tools/kiem_tra.py   bộ kiểm toàn dự án (notebook + cấu hình + thang prompt + mã chết)
HUONG_DAN_COLAB.md  bảng kiểm thao tác từng bước, số ô đã đối chiếu thật
CAC_BUOC_THUC_HIEN.md  runbook đầy đủ kèm cổng kiểm
```

Phần cần GPU nhận `generate_fn` **tiêm từ ngoài vào**, nên toàn bộ package test được trên
CPU mà không cần model.

Repo: <https://github.com/ThanhDatVN/vinumqa-numerical-reasoning>
