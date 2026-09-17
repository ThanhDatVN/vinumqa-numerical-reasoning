# Chạy trên Colab — một luồng, làm từ trên xuống

Mỗi bước là một notebook. Làm đúng thứ tự, không nhảy cóc. Ô nào có **⛔ Cổng** thì phải
thấy đúng dòng đó rồi mới đi tiếp.

* `ô #n` đếm theo **ô code**, bỏ qua ô chữ — đúng như Colab đánh số. Số ô trong tài liệu
  này đã đối chiếu với notebook thật.
* Runtime: **A100-40GB** cho mọi bước có GPU. `00` và `07` chạy **CPU**, miễn phí.
* Thời gian đã gồm ~15 phút cài đặt + nạp model mỗi phiên GPU.
* Sau mỗi bước: dán **khối meta** (notebook tự in giữa hai đường kẻ ngang) vào hội thoại.

| bước | notebook | runtime | ~ | ra nấc |
|--:|---|---|--:|---|
| 1 | `00_data_audit` | CPU | 2 ph | — |
| 2 | `01_baseline_basic` | A100 | 35 ph | `01_basic` |
| 3 | `02_prompt_engineering` | A100 | 35 ph | `02_prompt_eng` |
| 4 | **`08_phuong_phap_moi`** | A100 | 180 ph | `08_tu_nhat_quan`, `09_vidu_dong` |
| 5 | `07_final_report` | CPU | 2 ph | — |
| 6 | `04_self_evaluation` | A100 | 60 ph | `04_selfeval_base` |
| 7 | `05_ace` | A100 | 150 ph | `05_ace_base` + `05_ace_random_base` |
| 8 | `03_sft_qwen3` | A100 | 145 ph | `03_sft` |
| 9 | `04` + `05` lần 2 | A100 | 170 ph | `04_selfeval_sft`, `05_ace_sft` |
| 10 | `05_ace` nhánh 5c | A100 | 110 ph | `05c_ace_basic_base` |
| 11 | `06_combination` | A100 | 100 ph | `06_comb_E_A`, `06_comb_F_A` |
| 12 | `07_final_report` | CPU | 2 ph | — |

---

## Trước khi bắt đầu

```
Colab → Secrets (biểu tượng chìa khoá) → thêm OPENAI_API_KEY → bật "Notebook access"
```

Chỉ bước 7 và 10 (`05_ace`) cần nó. **Không bao giờ viết key vào notebook.**

Cấu hình cố định của cả thang bậc, giống nhau ở mọi notebook — **không sửa**:

```
TEMPERATURE 0.1   MAX_TOKENS 4096   MAX_SEQ_LENGTH 17000   REPETITION_PENALTY 1.0
RANDOM_SEED 42    GPU_MEM_UTIL 0.85  MAX_NUM_SEQS 48       BATCH_SIZE 512
```

> **Đừng nâng `MAX_TOKENS`.** Đã đo sạch: 4096 và 8192 cho cùng 30 lượt bị cắt, cùng 28
> mẫu mất trắng, cùng EA 0,6479. Gấp đôi ngân sách cứu **0 mẫu**, chỉ tốn thêm ~50 % thời
> gian. Mẫu bị cắt được chữa bằng **lượt vớt** (sinh lại với suy nghĩ tắt), không phải
> bằng trần. Đổi trần thì `save_stage` tự ghi sang tên nấc khác, không đè lên bản chuẩn.

### Chỉ còn MỘT prompt

Không còn nền prompt nào để chọn, không còn công tắc nào để đặt nhầm. Prompt hoàn chỉnh
là bản gốc đã sửa **ba chỗ nói sai so với chính dữ liệu gold** — mỗi chỗ neo vào một con
số đếm được trên toàn bộ 4 074 mẫu train+valid+test:

| bản gốc nói | gold nói | số đo |
|---|---|---|
| `table_*` nhận **tên cột** | nhận **nhãn hàng** (ô đầu mỗi dòng) | **618/618** khớp nhãn hàng, **0** khớp tên cột |
| `add(#0,c), add(#0,d)` | mỗi bước tham chiếu bước **ngay trước** | **2 053** lần ngay trước / **109** lùi xa hơn |
| "giảm ⇒ kết quả **luôn** âm" | phụ thuộc dạng câu | tỷ lệ giảm: 61/74 giữ âm · mức giảm tuyệt đối: 45/74 lấy dương |

Sửa ở **cả** bước 1 lẫn bước 2. Bản gốc chưa sửa vẫn nằm nguyên trong
`vinumqa/_prompt_text.py`, có test canh khớp notebook tham chiếu từng ký tự.

---

## Bước 1 · `00_data_audit` · CPU · 2 ph

Mở → CPU → Chạy tất cả → cho phép gắn Drive. **6 ô code.**

| ô | ⛔ Cổng — phải thấy |
|--:|---|
| #2 | `[MÔI TRƯỜNG] Colab`, và `train=2993 valid=584 test=497` |
| #3 | `✅ Executor tái tạo đúng 100% nhãn vàng trên test → thước đo dùng được.` |

Không đạt ô #3 → **dừng hẳn**. Executor sai thì mọi con số sau đều vô nghĩa.

Ô #4–#6 in phân bố dữ liệu, audit nhãn nhiễu, và chấm lại mốc tham chiếu — đọc để biết,
không có cổng nào.

---

## Bước 2 · `01_baseline_basic` · A100 · ~35 ph

Mở → A100 → bật **Thực thi nền** → Chạy tất cả. **10 ô code.**

Colab hiện **RESTART SESSION** sau ô #1 → bấm, rồi bấm ô #2 → `Ctrl+F10` (chạy ô này và
mọi ô sau).

| ô | ⛔ Cổng — phải thấy |
|--:|---|
| #3 | `[GÓI] vllm=…+cu129 \| transformers=4.… \| trl=0.…` — **`vllm` phải có đuôi `+cu129`** |
| #3 | `[SELF-TEST] executor tái tạo exe_ans trên test: 497/497` |
| #3 | `[SELF-TEST] ✅ executor / PA / EA đạt` |
| #4 | `[CFG] max_seq=17000 max_tokens=4096 temp=0.1` |
| #5 | `[MODEL] ✅ sẵn sàng sau … phút` rồi `[WARMUP] ✅` |
| #6 | `[PROMPT] mức = basic \| self-eval = False \| suy nghĩ = template tự quyết (Qwen3: BẬT)` |
| #6 | `thang lồng nhau: basic=… ⊂ no_fewshot=… ⊂ engineered=…` — ba số phải **tăng dần** |
| #6 | `[PROMPT] ✅ mọi prompt đều lọt ngân sách, không cần cắt` |
| #8 | `NẤC: 01_basic \| prompt=basic \| self-eval=False` |

**Dừng lại nếu thấy:**

- `[PROMPT] ⚠ suy nghĩ đang TẮT` → sai chế độ, PA sẽ hụt ~10 điểm. Dấu hiệu phụ: 497 mẫu
  xong trong ~1 phút thay vì ~16 phút.
- `[PROMPT] ⚠ đã bật cắt ngữ cảnh` → không phải A100. Kết quả có cắt ngữ cảnh **không
  trộn chung bảng** với kết quả không cắt.
- `[MODEL] ✅ nạp được ở chế độ an toàn` → vLLM phải lùi cấu hình. Chỉ đổi tốc độ, nhưng
  ghi lại để còn truy nếu con số trông lạ.

Nấc 1 dùng prompt `basic`: model đã biết có những phép nào và phải trả lời ra sao, chỉ
chưa được mách chọn phép nào cho loại câu hỏi nào và chưa thấy ví dụ mẫu. **Không sửa gì.**

Cuối ô #8, ngoài bảng kết quả còn hai thứ đáng ghi:

```
    01_basic/vớt: k/n mẫu bị cắt đã cứu được
    Bị cắt vì trần token, tách theo bước:
      step1        x.x%  (n/497 lượt)
```

Ô #9 in vài ca sai để xem prompt cơ bản hụt ở đâu. Ô #10 ghi nấc và in khối meta.

---

## Bước 3 · `02_prompt_engineering` · A100 · ~35 ph

Mở → A100 → Chạy tất cả. **11 ô code. Không sửa gì.**

| ô | ⛔ Cổng — phải thấy |
|--:|---|
| #3–#5 | như bước 2 |
| #6 | `[NẤC] 2 — prompt hoàn chỉnh \| system prompt … ký tự (nấc 1 dùng …)` |
| #7 | thang prompt là phép **chèn thuần** — mỗi mức chỉ THÊM khối, không sửa chữ phần chung |
| #8 | `NẤC: 02_prompt_eng \| prompt=engineered \| self-eval=False` |

Hiệu số nấc 1 → nấc 2 chính là đóng góp của prompt engineering: khối ánh xạ từ khoá →
phép toán (14 mục) cộng 2 ví dụ mẫu. Lần chạy trước cho **+20,1 điểm EA**
(KTC [+0,151, +0,252], p = 0,000) — hiệu ứng lớn nhất cả dự án.

Ô #9 so trực tiếp với nấc 1 bằng McNemar. Ô #10 xem ca nấc 2 sửa được mà nấc 1 thì không.

---

## Bước 4 · `08_phuong_phap_moi` · A100 · ~180 ph

Ba phương pháp mới, đo trọn trong **một** lượt. **14 ô code. Không sửa gì.**

| phương pháp | cơ chế |
|---|---|
| **Self-consistency** | sinh K = 5 mẫu, bỏ phiếu theo giá trị THỰC THI |
| **Ví dụ động (kNN)** | thay 2 ví dụ cố định bằng 3 ví dụ truy hồi từ train |
| **Lượt sửa** | program bị executor từ chối → sinh lại một lượt kèm lý do lỗi |

Cả ba chỉ hỏi executor, không đụng đáp án vàng.

| ô | ⛔ Cổng — phải thấy |
|--:|---|
| #7 | `[MỚI] K=5 mẫu \| temp=0.7 top_p=0.95 \| 3 ví dụ truy hồi` |
| #7 | `[MỚI] kho ví dụ: ~2888/2993 mẫu train (đã bỏ nhãn nhiễu)` |
| #7 | in thử một khối ví dụ truy hồi — đọc xem nó có **cùng loại phép** với gold không |
| #8 | `[CỔNG] ✅ nhận đúng 5 mẫu cho mỗi prompt` — **cổng kiểm sớm, ~10 giây** |
| #8 | `[CỔNG] k/5 chương trình khác nhau` — thấy `1/5` là bỏ phiếu vô nghĩa, **dừng lại** |
| #8 | `[CỔNG] ✅ còn dư … token` — prompt ví dụ động vẫn lọt ngân sách |
| #9 | `NẤC: 08_tu_nhat_quan \| K=5 mẫu \| ví dụ CỐ ĐỊNH \| sửa-khi-lỗi` |
| #10 | bảng **SELF-CONSISTENCY THEO k** — k=1…5 và trần best-of-5 |
| #12 | `NẤC: 09_vidu_dong \| K=5 mẫu \| 3 ví dụ TRUY HỒI` |
| #13 | bảng **VÍ DỤ ĐỘNG ĐÓNG GÓP BAO NHIÊU, Ở TỪNG MỨC k** |

**Vì sao chỉ hai nấc là đủ.** Mọi mẫu sinh ra đều được lưu (`cac_program`,
`cac_gia_tri`, `cac_ea`, `cac_pa`) cùng chương trình **trước** lượt sửa, nên từ hai lượt
GPU này `07` bóc ra được:

| phép so | đo cái gì |
|---|---|
| `08` k=1 → k=5 | self-consistency |
| `09` k=1 vs `08` k=1 | ví dụ động |
| `09` k=5 vs `08` k=5 | ví dụ động **dưới** self-consistency (tương tác) |
| có/không `program_truoc_sua` | lượt sửa |
| `08` k=1 vs nấc 2 | nhiệt độ (0,7 so với 0,1) |
| best-of-5 | **trần** của self-consistency |

⚠ Hai nấc này chạy ở `temperature = 0.7`, nấc 2 ở `0.1`. Chênh lệch so với nấc 2 vì thế
gồm **cả** phần nhiệt độ — `07` tách riêng, đừng gộp.

⚠ Đặt `SO_MAU` nhỏ rồi muốn tăng thì **phải chạy lại cả nấc**. Đường cong k = 1…K lấy
miễn phí từ K lớn, nên đừng tiết kiệm nhầm chỗ.

**Ô #8 là cổng chặn 80 phút.** Nó kiểm ba giả định chưa ai xác minh trên GPU: vLLM có
thật sự trả K mẫu không, K mẫu có khác nhau không, và prompt ví dụ động còn lọt ngân
sách token không. Sai bất kỳ cái nào thì nó `assert` ngay, đừng chạy tiếp.

Đọc gì ở bảng #10:

- đường cong đi ngang từ k=2 → self-consistency không có đất trên bộ này. Nói thẳng,
  đó vẫn là kết quả.
- **trần best-of-5 cao hơn hẳn k=5** → bỏ phiếu đang bỏ sót; chỗ đáng đầu tư tiếp là bộ
  chọn, không phải sinh thêm mẫu.

---

## Bước 5 · `07_final_report` lần 1 · CPU · 2 ph

Mở → **CPU** → Chạy tất cả. **13 ô code.** Không tốn GPU, chạy được bất cứ lúc nào.

| ô | Đọc gì |
|--:|---|
| #3 | nạp lại mọi nấc đã có — nấc nào thiếu sẽ hiện `⊘ … chưa có kết quả` |
| #4 | **KIỂM TRA CÔNG BẰNG** — mọi nấc cùng GPU, cùng trần, cùng chế độ suy nghĩ, cùng nền prompt |
| #6 | EA theo **số phép** và theo **nguồn dữ liệu** |
| #7 | **EA/PA theo LOẠI PHÉP TOÁN** — dòng `table_*` là con số quan trọng nhất |
| #7 | **VÌ SAO KHÔNG SINH ĐƯỢC PROGRAM** — bị cắt vs sai định dạng, kèm mức lặp |
| #8 | **PHƯƠNG PHÁP MỚI** — đường cong self-consistency, ví dụ động, lượt sửa, trần best-of-K |
| #9 | **CỔNG BƯỚC 2** — chỉ hiện khi đã có nấc self-eval |
| #10 | phân bố kết cục + vì sao executor từ chối |
| #12 | xuất `bang_ket_qua_*.csv`, `kiem_dinh_*.csv` |

| ô | ⛔ Cổng |
|--:|---|
| #4 | `✅ Mọi nấc cùng GPU, cùng trần token, cùng chế độ suy nghĩ, không cắt ngữ cảnh.` |
| #4 | **không** có `⚠ KHÔNG KIỂM ĐƯỢC … thiếu '<nấc>_meta.json'` |
| #4 | **không** có `⛔ Các nấc … KHÔNG cùng nền prompt` |

Dán CSV `bang_ket_qua_*.csv` vào hội thoại.

---

## Bước 6 · `04_self_evaluation` · A100 · ~60 ph

Mở → A100 → Chạy tất cả. **13 ô code. Không sửa gì** — ô #6 đã ghim `USE_SFT_ADAPTER = False`.

Hai lượt sinh mỗi mẫu nên lâu gấp đôi.

| ô | ⛔ Cổng — phải thấy |
|--:|---|
| #6 | `[MODEL] dùng model gốc` (chưa SFT) |
| #7 | `[PROMPT] ✅ mọi prompt đều lọt ngân sách` — bước 2 dài hơn nhiều, đây mới là chỗ dễ tràn |
| #8 | prompt bước 2 in ra có khối `⚙ Chạy thật chương trình trên bằng máy thì ra: …` |
| #9 | `NẤC: 04_selfeval_base \| 2 bước` |

Cuối ô #9, bảng bị cắt nay có **hai** dòng:

```
      step1        x.x%  (n/497 lượt)
      step2        y.y%  (n/497 lượt)
```

Thấy `⚠ bước 2 bị cắt nhiều hơn bước 1` → hiệu quả self-eval đang bị **pha loãng**, hiệu
số đo được chỉ là cận dưới. Ghi lại, đừng bỏ qua.

Nấc "self-eval **có cổng**" **không phải chạy lại** — `07` ô #8 tính thẳng nó từ jsonl.

---

## Bước 7 · `05_ace` · A100 · ~150 ph

Mở → A100 → Chạy tất cả. **16 ô code. Không sửa gì** — ô #6 `USE_SFT_ADAPTER = False`,
ô #8 `ACE_TREN_PROMPT = "engineered"`.

| ô | ⛔ Cổng — phải thấy |
|--:|---|
| #8 | `[ACE] ✅ gọi thử gpt-4o-mini OK → …` — **chưa qua thì đừng vào pha A** |
| #8 | `[ACE] chồng lên prompt 'engineered' \| self-eval=True` |
| #8 | `[ACE] cổng: dedup≥0.98 \| tối thiểu 1 phép DSL \| ≤3 bullet/cụm` |
| #8 | `[ACE] embedding=… \| top_k=3+4 \| trần=30 bullet \| reflector=openai` |
| #10 | pha A chạy; thấy `[RESUME] ⚠ CẤU HÌNH ĐÃ ĐỔI` → xoá `progress_*.json` rồi chạy lại |
| #12 | `NẤC: 05_ace_base` |

Ô #8 cũng in `[ACE] ⚠ Reflector gọi API ngoài` — đúng như thiết kế. Reflector dùng
`gpt-4o-mini` (~0,2 USD một lượt) nên nấc này thuộc nhóm **unconstrained**; báo cáo phải
ghi rõ, đừng để người đọc tưởng cùng thiết lập với nấc 1–4.

Ô #11 in playbook học được. Ô #13 so với nấc trước. Ô #14 chạy **đối chứng bullet ngẫu
nhiên** (`RUN_RANDOM_CONTROL = True`, ~20 phút) — giữ nguyên, đó là thước đo nhiễu.
Ô #15 chấm công từng bullet.

> Nếu playbook nhỏ hơn `k = 7` thì đối chứng **thoái hoá**: cả hai bên lấy toàn bộ bullet,
> prompt giống hệt nhau. Notebook tự gọi đúng tên — đó là **phép đo nhiễu**, không phải
> tác dụng của truy hồi.

---

## Bước 8 · `03_sft_qwen3` · A100 · ~145 ph · **3 phần, restart giữa chừng**

**24 ô code**, chia ba phần bởi hai ô chữ `⚠ RESTART RUNTIME TẠI ĐÂY`.

| phần | ô | việc | ~ |
|---|---|---|--:|
| A | #1–#9 | sinh lời giải trên train, lọc thành dữ liệu SFT | 80 ph |
| B | #10–#15 | nạp model chế độ huấn luyện, train LoRA, vẽ loss | 40 ph |
| C | #16–#24 | nạp lại model + adapter, chấm test | 25 ph |

Chạy hết phần A → **Runtime → Restart session** → chạy từ ô #10.
Chạy hết phần B → **Restart session** lần hai → chạy từ ô #16.

| ô | ⛔ Cổng — phải thấy |
|--:|---|
| #7 | `SFT_TRAIN_SUBSET = 2000`, `DROP_NOISY_GOLD = True`, `ADD_GOLD_FALLBACK = False` |
| #9 | thống kê dựng dữ liệu có `bo_nhan_nhieu` > 0 — mẫu gold `multiply(#n,100)` phải bị loại |
| #11 | đọc đúng file dữ liệu SFT vừa ghi (`latest.txt`) |
| #14 | val loss **không tăng** từ step 50 — tăng là overfit, dừng và báo lại |
| #21 | `ADAPTER_DIR` tồn tại, nạp được |
| #22 | `NẤC: 03_sft` |

> Đích huấn luyện là phần **sau** `</think>`. Nếu phần C cho PA tụt mạnh so với nấc 2,
> nghi can số một là SFT đang dạy model bỏ suy nghĩ — **báo lại, đừng tự chỉnh.**

Ô #23 so nấc 3 với nấc 2. Ô #24 ghi nấc.

---

## Bước 9 · `04` và `05` lần 2, trên model SFT · A100 · ~170 ph

Chạy lại hai notebook đã chạy ở bước 6 và 7, lần này bật adapter.

| notebook | sửa đúng một dòng | ra nấc |
|---|---|---|
| `04_self_evaluation` ô #6 | `USE_SFT_ADAPTER = True` | `04_selfeval_sft` |
| `05_ace` ô #6 | `USE_SFT_ADAPTER = True` | `05_ace_sft` |

| ⛔ Cổng | Phải thấy |
|---|---|
| ô #6 | `[MODEL] dùng adapter đã SFT: /content/drive/…/sft_adapter_qwen3` |
| ô #9 / #12 | tên nấc kết thúc bằng `_sft` |

Chưa chạy bước 8 thì ô #6 tự dừng với thông báo thiếu adapter.

---

## Bước 10 · `05_ace` nhánh 5c · A100 · ~110 ph

Vẫn `05_ace`, sửa đúng **một dòng** ở ô #8:

```python
ACE_TREN_PROMPT = "basic"
```

Nhớ đặt lại ô #6 về `USE_SFT_ADAPTER = False`.

| ô | ⛔ Cổng — phải thấy |
|--:|---|
| #6 | `[MODEL] dùng model gốc` |
| #8 | `[ACE] chồng lên prompt 'basic' \| self-eval=False` |
| #12 | `NẤC: 05c_ace_basic_base` |

Đây là phép đo có ý nghĩa nhất về ACE: **ACE có tự khám phá lại được thứ người viết prompt
đã viết tay không?** Trên nền `basic` khoảng trống là ~156 mẫu, chứ không phải ~10 như khi
chồng lên prompt hoàn chỉnh. Reflector nay được cho biết đúng prompt của nấc này, nên nó
không còn bị cấm đề xuất chính những quy tắc mà `basic` chưa có.

---

## Bước 11 · `06_combination` · A100 · ~100 ph

Mở → A100 → Chạy tất cả. **15 ô code. Không sửa gì.**

| ô | ⛔ Cổng — phải thấy |
|--:|---|
| #7 | bảng 8 ô, ô nào `đã có — đọc lại`, ô nào `CẦN CHẠY` |
| #8 | `[PLAYBOOK] model gốc → playbook_ace_base.txt (… bullet)` |
| #9 | chạy các ô còn thiếu — mỗi ô in `Ô E+A — prompt+ACE \| SFT=False self-eval=False ACE=True` |
| #10 | `MA TRẬN TỔ HỢP` rồi `CỔNG MỤC TIÊU — EA > 70 % VÀ PA_strict > 70 %` |
| #14 | ghi được `ma_tran_to_hop_*.csv` và `tuong_tac_*.json` |
| #15 | vẽ được `ma_tran_*.png` (cột **xám** = KTC chứa 0) |

Cổng 70/70 in thẳng còn thiếu bao nhiêu mẫu mỗi chỉ số. Chưa đạt thì xem `by_phep` và
`vi_sao_sai` ở `07`, **đừng đoán**.

Bốn ô cần SFT bị bỏ nếu chưa có adapter — notebook tự báo và tự thu ma trận lại còn 4 ô.

Ô #11 tính tác động chính + tương tác cho **cả EA lẫn PA**, kèm KTC bootstrap. Ô #12
kiểm định tổ hợp tốt nhất. Ô #13 tính chi phí mỗi điểm EA.

---

## Bước 12 · `07_final_report` lần cuối · CPU · 2 ph

Chạy lại như bước 5, lần này đã đủ 13 nấc.

| ô | ⛔ Cổng |
|--:|---|
| #4 | `✅ Mọi nấc cùng GPU, cùng trần token, cùng chế độ suy nghĩ, không cắt ngữ cảnh.` |
| #4 | **không** có `⚠ KHÔNG KIỂM ĐƯỢC … thiếu meta` |
| #12 | ghi `bang_ket_qua_*.csv`, `kiem_dinh_*.csv` |
| #13 | ghi `bao_cao_*.png` |

---

## Gửi lại gì

Sau mỗi bước, dán vào hội thoại **một** trong hai:

1. **khối meta** — notebook tự in giữa hai đường kẻ ngang ở ô cuối, hoặc
2. CSV của `07` (`bang_ket_qua_*.csv`, `kiem_dinh_*.csv`).

Đọc số xong mới sửa. **Không sửa prompt / ACE / tham số khi chưa có bảng phân loại lỗi** —
ba lần đoán mù gần nhất đều sai.

---

## Đọc kết quả cho đúng

| | |
|---|---|
| **EA** | chạy chương trình ra **đúng số** |
| **PA_strict** | chương trình **khớp gold** sau chuẩn hoá — **đây là con số thật** |
| PA_loose | bản so chuỗi cũ, chỉ để đối chiếu mốc tham chiếu. `PA_loose < PA_strict` là **bình thường**, hai normalizer khác nhau |

- **Sàn nhiễu ~1,6 điểm EA.** Chênh lệch dưới mức đó không phải phát hiện.
- **KTC chứa 0 = không kết luận được gì.** Không viết "cải tiến" hay "cộng hưởng".
- **Δ = 0 là kết quả.** Nấc nào cho Δ = 0 thì nói thẳng, kèm lý do.
- Cột **xám** trong biểu đồ ma trận nghĩa là KTC chứa 0 — đừng đọc nó như xanh hay đỏ.

---

## Bẫy môi trường

| Bẫy | Cách tránh |
|---|---|
| `vllm` từ PyPI là bản CUDA 13, Colab là CUDA 12.8 | ô #3 phải in đúng đuôi `+cu129` |
| `sentence-transformers` 6.x âm thầm nâng `transformers` lên 5.x, vỡ vLLM | đã ghim sẵn; self-test dừng nếu `transformers` không phải 4.x |
| `%%capture` giấu một lần cài hỏng | ô cài đặt của notebook GPU **không** dùng `%%capture`; chỉ hai notebook CPU dùng, cho `pandas/matplotlib` |
| vLLM `CUDA error: invalid argument` khi dựng engine | **không** phải model hỏng. Restart session; notebook tự lùi về `enforce_eager` + util 0,80 và in `[MODEL] ✅ nạp được ở chế độ an toàn` |
| Checkpoint ACE nối tiếp nhầm cấu hình | phải thấy `[RESUME] ⚠ CẤU HÌNH ĐÃ ĐỔI`; thấy `tiếp từ vòng N` mà cấu hình đã đổi thì xoá `progress_*.json` |
| Ép `enable_thinking=False` | mất ~10 điểm PA. Dấu hiệu: 497 mẫu xong trong ~1 phút thay vì ~16 phút |
| Drive hết chỗ giữa chừng | mỗi nấc ghi ~20 MB (jsonl + raw). Cả 11 nấc ~250 MB |

---

## Sửa code dưới máy

Trước mỗi lần commit, cả ba lệnh phải xanh:

```bash
python -m pytest tests/ -q -W error     # 209 test, CPU, ~4 giây
python tools/kiem_tra.py                # 6 phép kiểm nhanh
python tools/kiem_tra.py --day-du       # + chạy thật notebook 07 và phần phân tích của 06
```

`kiem_tra.py` bắt những thứ `pytest` không bắt được: LADDER lệch giữa các notebook, trần
token lệch giữa các ô cấu hình, thang prompt hết lồng nhau, notebook còn sót output, tên
được gọi mà không ô nào định nghĩa, và `save_stage`/`load_stage` có thật sự chạy không.

Xong thì `git push`, rồi trên Colab chạy lại **ô #2** của notebook đang mở — nó tự
`git pull`.
