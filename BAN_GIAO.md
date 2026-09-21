# Bàn giao — ViNumQA: đo tác động của từng kỹ thuật lên suy luận số học tài chính tiếng Việt

> **Tài liệu này để mở một phiên làm việc mới. Đọc hết §1–§4 và §9 trước khi sửa bất cứ thứ gì.**
> Mốc: 209 test · 9 notebook · 123 ô code · nhánh `main`.
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

`03` (SFT) · `05` (ACE) · `05c` · `05b` · **hai ô ma trận `06`**. Nấc 1–2 phải chạy lại vì prompt đã sửa theo dữ liệu.

### ⚠ Cấu hình đã đổi sau các lần chạy trên → **phải chạy lại từ nấc 1**

```
TEMPERATURE 0.1   MAX_TOKENS 4096   MAX_SEQ_LENGTH 17000
REPETITION_PENALTY 1.0   RANDOM_SEED 42
A100-40GB: GPU_MEM_UTIL 0.85 · MAX_NUM_SEQS 48 · BATCH_SIZE 512
```

### Thang bậc hiện tại (12 nấc)

```
01_basic · 02_prompt_eng · 03_sft · 04_selfeval_base · 04_selfeval_sft
05_ace_base · 05_ace_sft · 05c_ace_basic_base · 05_ace_random_base
06_comb_E_A · 06_comb_F_A · 08_tu_nhat_quan · 09_vidu_dong
06_comb_E_A_moi · 06_comb_F_A_moi
```
**15 nấc.** Chỉ còn MỘT prompt engineered — nấc 2c đã gộp vào nó, xem §4.1i.
Hai nấc `08`/`09` là ba phương pháp mới, xem §6c. Hai nấc `*_moi` là hai ô MỤC TIÊU
chạy cấu hình tốt nhất — chỉ chạy nếu §6c cho thấy phương pháp mới vượt sàn nhiễu.

**Lộ trình chốt: 10 bước, ~19 giờ GPU, KHÔNG công tắc nào.** Xem `HUONG_DAN_COLAB.md`.

`04` và `05` **tự chọn cấu hình chưa có kết quả** (2 và 3 cấu hình), in ra còn lại
những gì; bấm ô #6 rồi `Ctrl+F10` là chạy tiếp trong cùng phiên, model vẫn trên GPU.
`03` (SFT) chuyển lên **trước** hai notebook đó nên chúng làm trọn mọi cấu hình một lần.

Ma trận 2×2×2 luôn giữ cấu hình thang bậc (1 mẫu, temp 0.1) để tác động chính và tương
tác sạch; hai ô mục tiêu `*_moi` chạy cấu hình tốt nhất và ghi ra tên nấc riêng.

### Thang prompt (lồng nhau — có test canh)

| mức | thêm gì so với mức trên | ký tự |
|---|---|--:|
| `basic` | mở đầu + danh sách phép toán + yêu cầu định dạng | 2 745 |
| `no_fewshot` | + ánh xạ từ khoá → phép toán (14 mục) — *bậc giữa, không chạy* | 5 898 |
| `engineered` | + 2 ví dụ mẫu có khung | 6 414 |

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

**f) Sàn nhiễu KHÔNG phải một con số cố định.** Con số "~1,6 điểm" dùng trước đây là
ước lượng quá lạc quan. Phép đo trực tiếp, ngày 22/09/2026: cùng adapter, cùng dữ liệu
SFT, cùng seed, cùng commit của `vinumqa/` — **chỉ đổi card** (A100-80GB với
`MAX_NUM_SEQS=128` sang A100-40GB với `MAX_NUM_SEQS=48`) — nấc `03_sft` cho

| | 80 GB | 40 GB | chênh |
|---|---:|---:|---:|
| EA | 0,6761 | 0,7002 | **+2,21 điểm** |
| PA_strict | 0,6197 | 0,6419 | +2,22 điểm |

vLLM gộp lô khác nhau thì thứ tự cộng dồn trong kernel khác nhau, logits lệch ở chữ số
cuối, và ở `temperature=0.1` là đủ để lật token rồi kéo theo cả chuỗi suy luận. Đây là
MỘT lần bốc từ phân phối nhiễu, không phải bằng chứng "40 GB tốt hơn".

Ngưỡng đúng phụ thuộc số mẫu hai cấu hình BẤT ĐỒNG. Theo McNemar, dưới giả thuyết không
thì hiệu số có độ lệch chuẩn `√(b+c)` mẫu, nên ngưỡng 95 % là `1,96·√(b+c)/497`:

| bất đồng (b+c) | ngưỡng 95 % |
|--:|--:|
| 30 | 2,2 điểm |
| 60 | 3,1 điểm |
| 90 | 3,7 điểm |
| 180 | 5,3 điểm |

**Đừng dùng một con số cố định. Đọc KTC mà `stats.compare_pair` in ra** — nó đã tính
đúng việc này từ đầu. Không kết luận nào trong dự án bị lật khi áp ngưỡng mới: mọi phát
hiện "có ý nghĩa" đều đã vượt ngưỡng tương ứng của nó, mọi kết luận "chưa tách được khỏi
nhiễu" vẫn nằm dưới.

**g) Tương tác trong ma trận 2×2×2 gần như KHÔNG tách được khỏi nhiễu.** Thử với tương
tác thật −0,05 (5 điểm): KTC vẫn chứa 0. Đừng viết "cộng hưởng"/"trùng nhau" vào báo cáo
khi KTC chứa 0.

**h) Bản sao prompt trong repo ĐÃ TỪNG trôi khỏi bản tham chiếu.**

`_prompt_text.py` tự nhận là "chép NGUYÊN VĂN", nhưng đối chiếu từng ký tự với
`reference/original_notebooks/inference_with_difference_models.ipynb` thì lệch **một ký
tự** ở mục 6 của STEP_1: repo ghi `add(#1,d)`, bản gốc ghi `add(#0,d)`. Ai đó đã sửa
lặng lẽ cho đúng toán — nhưng chỉ sửa STEP_1, không sửa STEP_2, nên hai prompt dạy ngược
nhau mà không ai biết.

Nay đã trả STEP_1 về **đúng bản gốc**, và bản sửa nằm ở `engineered_v2`. Có
`TestKhopBanThamChieu` canh cả hai chuỗi khớp từng ký tự — trôi lần nữa là test đỏ.

> Bản gốc có **hai** chỗ nói sai, đều giữ nguyên ở nấc 2 và đều sửa ở nấc 2c:
> `table_*` mô tả là đọc theo CỘT (61 mẫu), và `add(#0,c), add(#0,d)` cộng dồn bỏ mất số
> hạng giữa (17 mẫu). Bốn escape `	ext{`/`rac{` trong STEP_2 bị Python biến thành
> TAB/form-feed — repo lưu đúng bản đã diễn giải, tức đúng thứ model thật sự nhận,
> **không phải lỗi**.

**i) Prompt gốc nói SAI ba chỗ — đã đối chiếu với chính gold, không phải ý kiến.**

Đếm trên toàn bộ 4 074 mẫu train+valid+test:

| bản gốc nói | gold nói | số đo |
|---|---|---|
| `table_*` nhận **tên cột** | nhận **nhãn hàng** (ô đầu mỗi dòng) | **618/618** khớp nhãn hàng, **0** khớp tên cột |
| `add(#0,c), add(#0,d)` | mỗi bước tham chiếu bước **ngay trước** | **2 053** lần ngay trước / **109** lùi xa hơn |
| "giảm ⇒ kết quả **luôn** âm" | phụ thuộc dạng câu | tỷ lệ giảm (có chia): 61/74 giữ âm (82 %) · mức giảm tuyệt đối: 45/74 lấy dương (61 %) |

Ba chỗ này nay sửa thẳng trong `PromptKit.theo_du_lieu`, áp cho **cả** bước 1 lẫn bước 2,
nên **không còn nấc 2c**. Bản gốc chưa sửa giữ nguyên ở `_prompt_text.py` để truy xuất xứ,
có `TestKhopBanThamChieu` canh khớp notebook tham chiếu từng ký tự.

Thêm mấy con số đã đếm, dùng khi cần cãi về prompt:

* phép toán trong gold: `divide` 38,3 % · `subtract` 31,2 % · `add` 15,4 % · `multiply` 5,1 % · `table_*` 9,9 % · `greater` 1 mẫu · `exp` **0 mẫu**
* mẫu tăng trưởng: 1 016/1 061 lần mẫu số của `divide` đúng là **giá trị CŨ** (95,8 %)
* `multiply(#n,100)` trong gold: 101/4 069 — nhiễu nhãn, prompt cấm là đúng
* `table_sum` chỉ 61/374 câu hỏi có chữ "tổng"; còn lại dùng `add` liên tiếp
* gold **không có** phép lồng nhau: 0/4 069
* tham số thứ hai của `table_*` luôn là `none`: 618/618
* phân bố số phép: 1 phép 58,6 % · 2 phép 34,1 % · ≥3 phép 7,3 %

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
| 1 | **Prompt sửa theo dữ liệu** — ba chỗ bản gốc nói sai, xem §4.1i | 61 + 17 mẫu |
| 2 | **Vớt mẫu bị cắt** — sinh lại với suy nghĩ TẮT, chỉ thay khi ra được program | tối đa +5,8 điểm EA/nấc |
| 3 | **Self-eval biết kết quả thực thi** — bước 2 nay thấy `⚙ Chạy thật … thì ra: 15000.0` | biến nấc 4 từ Δ=0 thành đo được |
| 4 | **Tốc độ** — một lô 512, `enable_prefix_caching`, `max_num_seqs` 48 | log cũ: lô 400 chạy 1,88 s/mẫu, lô 97 còn lại 2,83 s/mẫu |
| 5 | **Bước 2 sửa y hệt bước 1** — trước đây bước 2 vẫn dạy "cột" nên **dạy lại điều sai ngay sau khi bước 1 vừa làm đúng** | mọi ô có self-eval |
| 6 | **Cổng bước 2** (`cong_buoc2`) — chỉ nhận program bước 2 khi nó CHẠY ĐƯỢC. Trước đó `prog2 or prog1` cho bước 2 thắng vô điều kiện | tính lại được từ jsonl, **0 GPU** |
| 7 | **Reflector hết mâu thuẫn** — `luat_dang_ap_dung` nay theo đúng nấc đang chạy và đã lọc các dòng dạy ngược về `table_*` ra khỏi danh sách "đừng đề xuất lại" | mở khoá nấc 5c và luật nhãn-hàng |
| 8 | **Ràng buộc Reflector ≥1 phép** — trước đòi ≥2 trong khi cổng đã hạ `min_ops=1`; 64 % test là câu 1 phép | nhóm 319 mẫu mới có lời khuyên |
| 9 | **Ma trận dùng chung một prompt** — `06` trước hard-code riêng, nay mọi nấc cùng một nền | nối lại đường tới 70/70 |

### 6c. Ba phương pháp MỚI — notebook `08`, hai nấc

Thiết kế để **một lượt GPU rút ra cả họ kết quả**, vì ngân sách không đủ chạy lại.

| phương pháp | cơ chế | căn cứ |
|---|---|---|
| **Self-consistency** | sinh K=5 mẫu, bỏ phiếu theo giá trị **thực thi** | ô `sai` là ô lớn nhất (136/497). GSM8K: 56,5 → 74,4 khi N=40 |
| **Ví dụ động (kNN)** | thay 2 ví dụ cố định bằng 3 ví dụ truy hồi từ train | đo trên bộ này: láng giềng gần nhất cùng dãy phép với gold ở **46,5 %** câu, top-3 là **66,2 %**; cả train chỉ có **86 dãy phép** |
| **Lượt sửa** | program bị executor từ chối → sinh lại 1 lượt kèm đúng lý do lỗi | chỉ ~10–40 mẫu phải sinh lại |

Cả ba **chỉ hỏi executor**, không đụng đáp án vàng.

Mọi mẫu sinh ra đều được lưu (`cac_program`, `cac_gia_tri`, `cac_ea`, `cac_pa`) cùng
`program_truoc_sua`, nên từ **hai** lượt GPU, `07` §5d bóc ra được: đường cong
self-consistency k=1…5, đóng góp của ví dụ động ở từng k, tương tác giữa hai thứ, đóng
góp của lượt sửa, ảnh hưởng của nhiệt độ, và **trần best-of-K**.

⚠ Hai nấc này chạy `temperature=0.7` (self-consistency cần đa dạng; ở 0,1 thì K mẫu
giống hệt nhau và bỏ phiếu vô nghĩa). Nấc 2 ở 0,1 — chênh lệch so với nấc 2 vì thế gồm
cả phần nhiệt độ, `07` tách riêng.

⚠ Đặt `SO_MAU` nhỏ rồi muốn tăng thì **phải chạy lại cả nấc**.

### 6b. Đợt dọn mã — một luồng

Mã nguồn nay chỉ còn **một đường chạy** cho mỗi nấc. Đã xoá hẳn:

| bỏ | vì |
|---|---|
| mức prompt `plain` | 0 notebook chạy; BAN_GIAO cũ đã ghi "sàn hỏng, đừng dùng để đo" |
| dò/hỗ trợ Kaggle (77 chỗ) | dự án chạy Colab |
| `save_details_csv`, `LEGACY_COLS`, `save_metrics_csv`, `ensure_details_dir` | nơi tiêu thụ (`results/`, `app.py`) **không tồn tại trong repo** |
| backend Reflector `gemini` | không bao giờ được chọn |
| `compare_ladder`, `has_two_ops`, `QualityGate(enabled=False)`, `section_aware`, `bao_gia_tri_cho_buoc2` | 0 lần gọi / 0 lần bật |
| tham số `PromptKit(repo_dir=…)` | vết tích chữ ký cũ |
| `PROMPT_LEVEL = "selfeval"` rồi ngay dòng sau đổi lại | lặp ở 5 notebook |

**Một công tắc duy nhất**: `MUC_PROMPT` ở **ô #2** của mọi notebook. `04`, `05`, `06` đều
đọc nó; `02` ghi đè có chủ ý vì việc của nó là so hai nền. `07` có bảng kiểm báo động nếu
các nấc lẫn hai nền.

> `no_fewshot` được GIỮ dù không nấc nào chạy: nó là bậc giữa để bộ kiểm chứng minh thang
> prompt là phép **chèn thuần** từng bước. Bỏ nó là bỏ một cái chốt canh phép đo.

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
python -m pytest tests/ -q -W error     # 209 test, CPU, ~4 giây
python tools/kiem_tra.py                # 6 phép kiểm tĩnh, ~5 giây
python tools/kiem_tra.py --day-du       # + 3 phép kiểm CHẠY THẬT, ~60 giây
```

`tools/kiem_tra.py` trả mã thoát khác 0 nếu có lỗi. Sáu phép kiểm tĩnh bắt những thứ
`pytest` **không** bắt được: LADDER có giống hệt nhau ở cả 9 notebook không, trần token có
lệch giữa các ô cấu hình không, ngân sách ngữ cảnh còn dư bao nhiêu, thang prompt còn lồng
nhau không, notebook có sót output không, **có tên nào được gọi mà không ô nào định nghĩa
không**, và **HUONG_DAN_COLAB có còn khớp notebook không** (số ô code + vị trí từng cổng).

`--day-du` thêm ba phép kiểm **thực thi mã thật**, vì biên dịch sạch không có nghĩa là
chạy được:

| | chạy gì | đã từng bắt được |
|--:|---|---|
| 7 | trọn `07` với nấc dựng sẵn | `save_stage`/`load_stage` không ai gọi → `KeyError: 'csv'` lọt lên Colab |
| 8 | phần phân tích của `06` | ba lỗi `NameError`/`TypeError`, rồi `NICE['E+A*']` sẽ nổ **sau** 160 phút GPU |
| 9 | logic 5 notebook GPU (`01`,`02`,`04`,`05`,`08`) bằng model giả | 38 ô trước đó có độ phủ **bằng 0** |

Phép kiểm 9 dùng `tools/chay_thu_notebook.py`: bỏ đúng ba loại ô — cài gói, kiểm gói, nạp
model — rồi tiêm `model`/`tokenizer`/`generate`/`SamplingParams` giả, trong đó `generate`
trả về **gold program** của chính mẫu đang hỏi, nên mọi nhánh chấm điểm chạy trên dữ liệu
thật. Nó **không** thay cho GPU: nó chỉ chứng minh mã chạy tới cuối, không nói gì về chất
lượng model.

Hai công cụ chạy riêng khi cần:

```bash
python tools/chay_thu_notebook.py 08    # chạy logic MỘT notebook, in từng ô
python tools/do_do_phu.py               # đo % dòng mã phân tích thật sự được chạy (nay 80%)
python tools/kiem_thu_tu_ten.py         # tên dùng ở ô i có định nghĩa ở ô ≤ i không
```

**Luật đã rút ra:** thêm một nhánh mới mà không thêm nó vào bộ chạy giả thì nhánh đó
**chưa được kiểm** — dù `pytest` và `kiem_tra.py` đều xanh.

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
   giống hệt nhau, có bộ kiểm canh LADDER đồng nhất ở cả 9 notebook.
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
notebooks/00–08     mỗi nấc một notebook; 00, 06 (phần phân tích) và 07 chạy CPU
tests/              209 test, CPU, ~4 giây, KHÔNG cần GPU
data/               ViNumQA (24 MB, nằm luôn trong repo)
reference/          hồ sơ xuất xứ + 5 CSV mốc tham chiếu (KHÔNG tái tạo được nếu mất)
BAN_GIAO.md         tài liệu này
tools/kiem_tra.py         bộ kiểm toàn dự án — 6 phép tĩnh + 3 phép chạy thật
tools/chay_thu_notebook.py  chạy logic notebook GPU bằng model giả (phép kiểm 9)
tools/do_do_phu.py          đo độ phủ dòng của phần phân tích 06/07
tools/kiem_thu_tu_ten.py    tên dùng ở ô i phải định nghĩa ở ô ≤ i
HUONG_DAN_COLAB.md  runbook MỘT LUỒNG: 10 bước, ~19 giờ GPU, KHÔNG công tắc nào
```

Phần cần GPU nhận `generate_fn` **tiêm từ ngoài vào**, nên toàn bộ package test được trên
CPU mà không cần model.

Repo: <https://github.com/ThanhDatVN/vinumqa-numerical-reasoning>
