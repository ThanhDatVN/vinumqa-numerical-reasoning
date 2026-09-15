# Các bước thực hiện thí nghiệm

Tài liệu này là **bảng thao tác** — vừa chạy vừa mở. Mỗi bước nói rõ *sửa gì → chạy gì →
phải thấy gì → nếu sai thì làm gì*.

Ba tài liệu chia việc như sau: [HUONG_DAN_COLAB.md](HUONG_DAN_COLAB.md) là cách đưa dự án
lên Colab; [KE_HOACH_THU_NGHIEM.md](KE_HOACH_THU_NGHIEM.md) là *vì sao* chạy theo thứ tự này
và tốn bao nhiêu; tài liệu này là *làm gì, theo thứ tự nào, bấm ở đâu*.

## Quy ước

- **"code cell #N"** = ô **code** thứ N tính từ trên xuống. **Ô markdown không đếm.**
  Cell #1 luôn là ô cài đặt (`%%capture`), cell #2 luôn là ô cấu hình.
- **"§N"** = mục trong notebook, thấy ở panel mục lục bên trái Colab.
- ⛔ = cổng kiểm tra. Trượt thì **dừng lại xử lý**, đừng chạy tiếp cho tốn compute unit.

---

## Toàn cảnh 13 bước

| # | Bước | Notebook | Runtime | Thời gian | CU | Sinh ra |
|:--:|---|---|---|--:|--:|---|
| | **BUỔI 1 — kết quả chính** | | | **1,9 h** | **~23** | |
| 1 | Audit & chốt thước đo | `00` | CPU | 1 ph | 0 | bảng đính chính EA |
| 2 | Nấc 1 — inference thường | `01` | A100 | 9 ph | 1,8 | `01_plain` |
| 3 | Nấc 2 — prompt engineering | `02` | A100 | 9 ph | 1,8 | `02_prompt_eng` |
| 4 | Nấc 4 — self-eval (gốc) | `04` | A100 | 18 ph | 3,5 | `04_selfeval_base` |
| 5 | Nấc 5 — ACE (gốc) | `05` | A100 | 81 ph | 15,8 | `05_ace_base` + playbook + đối chứng |
| 6 | Báo cáo giữa kỳ | `07` | CPU | 1 ph | 0 | bảng + biểu đồ |
| | **BUỔI 2 — SFT** | | | **1,2 h** | **~14** | |
| 7 | `03` phần A — dựng dữ liệu | `03` | A100 | 36 ph | 7,1 | `sft_data/*.jsonl` |
| 8 | `03` phần B — huấn luyện | `03` | A100 | 28 ph | 5,5 | `sft_adapter_qwen3/` |
| 9 | `03` phần C — chấm test | `03` | A100 | 9 ph | 1,8 | `03_sft` |
| | **BUỔI 3 — ma trận tổ hợp** | | | **1,5 h** | **~20** | |
| 10 | Self-eval trên model SFT | `04` | A100 | 18 ph | 3,5 | `04_selfeval_sft` |
| 11 | ACE trên model SFT | `05` | A100 | 63 ph | 12,4 | `05_ace_sft` |
| 12 | Ma trận 2×2×2 | `06` | A100 | 18 ph | 3,5 | `06_comb_E_A`, `06_comb_F_A` |
| 13 | Báo cáo cuối | `07` | CPU | 1 ph | 0 | CSV để dán vào bài |

**Tổng ~57 CU / 100 CU/tháng.** Hết ngân sách giữa chừng thì dừng sau **bước 6** vẫn có một
bài hoàn chỉnh — xem mục "Cắt theo thứ tự nào" trong [KE_HOACH_THU_NGHIEM.md](KE_HOACH_THU_NGHIEM.md).

---

## Bước 0 — chuẩn bị, làm một lần

```bash
python -m pytest tests/ -q                                    # phải xanh 114/114
python tools/set_repo.py https://github.com/<bạn>/<repo> --init
git push -u origin main
```

Sau đó mở notebook bằng huy hiệu **Mở trong Colab** trong [README.md](README.md).

Kiểm lại ba thứ trước khi tốn CU đầu tiên:

- [ ] `pytest` xanh 114/114
- [ ] Repo trên GitHub có thư mục `data/` với đủ `train.json`, `valid.json`, `test.json`
- [ ] Colab Pro còn ≥ 60 compute unit

---

# BUỔI 1 — kết quả chính

## Bước 1 — `00_data_audit` · CPU · 1 phút

Bước duy nhất **không tốn CU**, mà lại là bước quan trọng nhất: nó chứng minh thước đo đúng.
Chạy sai ở đây thì 12 bước sau đều vô nghĩa.

**Runtime:** Runtime → Change runtime type → **CPU** (không phải GPU — chọn GPU ở đây là đốt
CU vô ích).

**Sửa:** nếu chưa chạy `set_repo.py` thì sửa `GITHUB_REPO` ở **code cell #2**. Chỉ lần này.

**Chạy:** Runtime → Run all.

**Phải thấy:**

```
[CODE] đang clone https://github.com/… → /content/<repo>
[MÔI TRƯỜNG] Colab | vinumqa v2.0.0
[DỮ LIỆU] /content/<repo>/data
          train=2993 valid=584 test=497
  0/10 nấc đã có kết quả
```

**⛔ Cổng 1 — §2, quan trọng nhất cả dự án.** Bảng "executor tái tạo `exe_ans`":

| tập | ngưỡng |
|---|---|
| train | ≥ 99,8 % |
| valid | **100 %** |
| test | **100 %** |

Dưới ngưỡng → **dừng hẳn**, không chạy bước 2. Nguyên nhân gần như chắc chắn là `data/` bị
lệch phiên bản so với lúc xây `vinumqa/dsl.py`.

**Ghi lại từ §5:** bảng đính chính EA của 5 model tham chiếu (EA cũ vs EA chấm đúng). Đây là
một bảng độc lập cho báo cáo — nó định lượng lỗi `table_*` của executor cũ, và **không cần
GPU** để có.

---

## Bước 2 — `01_baseline_plain` · A100 · 9 phút · 1,8 CU

**Runtime:** Runtime → Change runtime type → **A100 GPU**. Bật **Background execution**
(Colab Pro) để đóng tab không chết session.

**Sửa:** không gì cả.

**Chạy:** Run all. Bốn mốc cần theo dõi:

| Cell | Việc | Bình thường mất |
|---|---|---|
| #1 | cài unsloth + vLLM | 5–10 phút |
| #2 | cấu hình, clone repo | 15 giây |
| #3 | self-test executor | tức thì |
| #4–#5 | nạp Qwen3-8B 4-bit | 3–4 phút |
| #8 | chạy 497 mẫu | 9 phút |

> Colab hiện nút **RESTART SESSION** sau cell #1 là chuyện bình thường. Bấm restart rồi
> **Run all lại** — cell #1 sẽ bỏ qua vì thư viện đã có sẵn.

**⛔ Cổng — cell #3 phải in:**

```
[SELF-TEST] executor tái tạo exe_ans trên test: 497/497
[SELF-TEST] ✅ executor / PA / EA đạt
```

Cell này cố tình đặt **trước** khi nạp model, để hỏng thì chưa tốn gì.

**⛔ Cổng — §3 phải in:**

```
[PROMPT] ✅ mọi prompt đều lọt ngân sách, không cần cắt
```

Nếu thấy `[PROMPT] ⚠ đã bật cắt ngữ cảnh` thì đang không ở A100/L4. Đổi runtime rồi chạy
lại — **kết quả có cắt ngữ cảnh không trộn chung bảng được với kết quả không cắt.**

**Cũng nên liếc §2:** `[GPU] NVIDIA A100-SXM4-40GB | 39.x GB | CC 8.0`.

**Ghi lại:** EA, PA_strict, PA_loose ở §4. Đây là **mốc dưới** của cả lộ trình, chưa có ngưỡng
nào phải đạt.

**Sinh ra:** `vinumqa_runs/stages/01_plain.{jsonl,csv,meta}`.

---

## Bước 3 — `02_prompt_engineering` · A100 · 9 phút · 1,8 CU

Giống hệt bước 2, chỉ khác `PROMPT_LEVEL = "engineered"` (đã đặt sẵn, không sửa gì).

**Chạy:** Run all.

**⛔ Cổng 2 — đây là cổng tái lập mốc tham chiếu, và chỉ mất 37 phút để tới.** §4 in PA_loose:

| Quan sát | Kết luận |
|---|---|
| PA_loose ≈ **51 %** (tham chiếu: 51,34 %) | ✅ môi trường khớp, chạy tiếp |
| Lệch > 5 điểm | ⛔ **Dừng.** Kiểm tra theo thứ tự: (1) `[PROMPT]` có báo cắt ngữ cảnh không; (2) cell #1 đã cài đúng `transformers==4.56.2` và `trl==0.22.2` chưa; (3) `TEMPERATURE` có đúng 0.1 không |

**Cũng xem §5:** Δ so với nấc 1 + kiểm định McNemar. Nếu Δ dương và p < 0,05 thì prompt
engineering có đóng góp đo được — đó là con số của nấc 2.

**Sinh ra:** `02_prompt_eng.*`.

---

## Bước 4 — `04_self_evaluation` lần 1 · A100 · 18 phút · 3,5 CU

Bỏ qua `03` lúc này là **cố ý** — `03` đắt nhất và không phải điều kiện của `04`/`05` trên
model gốc.

**Sửa — code cell #6**, xác nhận đúng dòng này (mặc định đã đúng):

```python
USE_SFT_ADAPTER = False
```

**Chạy:** Run all.

**Phải thấy ngay sau cell #6:**

```
[MODEL] dùng Qwen3-8B gốc (chưa SFT)
[MODEL] nấc trước để so sánh: 02_prompt_eng
```

Thấy `dùng adapter đã SFT` ở lần chạy này là **sai cờ** — dừng, sửa lại `False`, Run all lại.

**⛔ Cổng 3 — mốc tái lập chính.** §5 in PA_loose:

| Quan sát | Kết luận |
|---|---|
| PA_loose ≈ **59,5 %** (tham chiếu: 59,56 %) | ✅ hạ tầng mới tái lập được mốc tham chiếu |
| Lệch nhiều | ⛔ Sai ở đây thì **đừng tin nấc 5**. Cấu hình self-eval đang khác mốc tham chiếu. |

Notebook chạy **2 lượt sinh/mẫu** nên lâu gấp đôi bước 3 — đúng như thiết kế.

**Đáng đọc:** §6 in 5 ca bước 2 **sửa đúng** và các ca bước 2 **làm hỏng**. Đây là chất liệu
định tính tốt cho phần thảo luận.

**Sinh ra:** `04_selfeval_base.*`.

---

## Bước 5 — `05_ace` lần 1 · A100 · 81 phút · 15,8 CU

Bước dài nhất và là **câu hỏi chính của bài**. Gồm ba phần chạy liền trong một session.

**Sửa hai cờ:**

| Cell | Đặt | Ý nghĩa |
|---|---|---|
| #6 | `USE_SFT_ADAPTER = False` | ACE trên model gốc |
| #15 | `RUN_RANDOM_CONTROL = True` | **giữ nguyên True ở lần này** |

Đối chứng bullet ngẫu nhiên tốn thêm 18 phút nhưng là câu phản biện chắc chắn bị hỏi: *cải
tiến là do truy hồi đúng bullet, hay chỉ vì prompt dài thêm?* Đừng cắt ở lần chạy này.

**Chạy:** Run all, rồi theo dõi theo ba pha:

### Pha A — học playbook (§5, ~45 phút)

Phải thấy khi bắt đầu:

```
[PHA A] train_sub=600 dev_sub=120
[PHA A] nhãn nhiễu trong train_sub: NN (x.x%) — Reflector sẽ nhận tín hiệu sai ở các mẫu này
[ACE] embedding=… | top_k=3+4 | trần=30 bullet | reflector=slm
```

Nếu thấy `[ACE] ⚠ không có embedding đa ngữ` thì truy hồi yếu hẳn — cell #1 chưa cài xong
`sentence-transformers`. Restart, Run all lại.

Vòng lặp chạy 600 mẫu, mỗi 32 mẫu một vòng, cứ 3 vòng đánh giá trên 120 mẫu dev.
**Có checkpoint sau mỗi vòng** vào `vinumqa_runs/progress_ace_base.json` — mất session cũng
không mất công.

**⛔ Cổng 4 — cuối §5:**

| Kiểm tra | Ngưỡng | Trượt thì |
|---|---|---|
| Số bullet trong playbook | **≥ 8** | Đọc `qg_reasons`. Nhiều `chua_so_lieu_cu_the` = Reflector đang chép đáp án thay vì rút chiến lược. Nhiều `verify_khong_sua_duoc` = bullet nghe hay nhưng không sửa được lỗi thật. |
| `[PHA A] so với mốc rỗng: composite` | **dương** | Playbook không hơn prompt rỗng ⇒ pha B sẽ không cải thiện. Xem lại `qg_reasons` trước khi tốn thêm 18 phút. |

**In §5 ra và lưu lại playbook** — đây là thứ trực quan nhất cho phụ lục báo cáo: nó cho
người đọc thấy ACE *học được cái gì* bằng tiếng Việt, không phải chỉ một con số.

### Pha B — chấm test (§6, ~18 phút)

Ghi `05_ace_base`. §7 so với `04_selfeval_base` bằng McNemar.

| Quan sát | Nghĩa là |
|---|---|
| Δ EA dương, p < 0,05 | ACE có cải tiến đo được — **kết quả chính của bài** |
| Δ dương, p ≥ 0,05 | Với n = 497, chênh < ~2 điểm thường không đạt. Là **giới hạn cỡ mẫu**, không phải ACE thất bại. Ghi cả Δ lẫn khoảng tin cậy. |
| Δ âm | Cũng là kết quả đáng báo cáo. §9 chỉ ra bullet nào có tội. |

### Đối chứng ngẫu nhiên (§8, ~18 phút)

Ghi `05_ace_random_base`. Đọc như sau:

| Quan sát | Kết luận |
|---|---|
| ACE > ngẫu nhiên | ✅ Cải tiến đến từ **truy hồi đúng bullet**, không phải từ prompt dài thêm |
| ACE ≈ ngẫu nhiên | Cơ chế truy hồi chưa có tác dụng — tác dụng (nếu có) chỉ là prompt dài hơn |

**Sinh ra:** `05_ace_base.*`, `05_ace_random_base.*`, `playbook_ace_base.txt`,
`logs/ace_history_ace_base.jsonl`.

---

## Bước 6 — `07_final_report` · CPU · 1 phút

**Runtime: đổi về CPU.** Notebook này chỉ đọc file, nạp GPU là phí CU.

**Chạy:** Run all. Nó sẽ tự bỏ qua các nấc chưa có (`03`, `04_selfeval_sft`, …) và in
`⚠ chưa có 'xx'` — bình thường ở giai đoạn này.

Xem §3 (bảng thang bậc), §4 (kiểm định từng bước), §7 (đối chiếu mốc tham chiếu).

**Đến đây đã có một bài hoàn chỉnh**: tái lập được mốc tham chiếu + trả lời được câu hỏi ACE có cải
tiến không + có đối chứng. Hết ngân sách thì dừng ở đây cũng được.

---

# BUỔI 2 — SFT (bước 7–9)

`03_sft_qwen3` là **một notebook, ba phần, hai lần restart runtime**. Lý do: engine vLLM giữ
VRAM rất chặt, không nhả đủ cho huấn luyện. Dữ liệu ghi ra Drive sau mỗi phần nên restart
không mất gì.

Trong notebook có sẵn ô markdown **⚠ RESTART RUNTIME TẠI ĐÂY** làm mốc.

## Bước 7 — phần A: dựng dữ liệu SFT · 36 phút · 7,1 CU

**Chạy:** cell **#1 → #10**, rồi **DỪNG**.

Cách làm trong Colab: Run all rồi canh dừng ở #10; hoặc bấm chạy từng ô.

**Xem lại cell #7 nếu muốn chỉnh (thường không cần):**

```python
SFT_TRAIN_SUBSET = 2000      # None = dùng cả 2993 mẫu, lâu hơn ~50%
DROP_NOISY_GOLD  = True      # bỏ mẫu multiply(#n,100)
ACCEPT           = "pa_or_ea"
ADD_GOLD_FALLBACK = False    # bật = quay lại đúng lỗi overfit của SFT trên gold trần
```

`ADD_GOLD_FALLBACK = False` là điểm khác lần chạy trước quan trọng nhất ở nấc này: nấc 3 huấn luyện
trên **lời giải model tự làm đúng** (rejection sampling), không phải trên program trần.

**⛔ Cổng — §5:**

| Kiểm tra | Ngưỡng | Trượt thì |
|---|---|---|
| Số mẫu SFT giữ lại | **≥ 800** (lý tưởng 1000–1200) | < 200 thì notebook tự dừng. Tăng `SFT_TRAIN_SUBSET`, hoặc nới `ACCEPT = "ea"` |
| Phân bố số phép toán | có mẫu **≥ 3 bước** | Toàn bài 1 bước ⇒ model chỉ làm đúng bài dễ, SFT không dạy được gì mới |

**Rồi: Runtime → Restart session.** (Không phải "Disconnect and delete runtime" — cái đó xoá
luôn thư mục đã clone.)

**Sinh ra:** `vinumqa_runs/sft_data/qwen3_sft_<stamp>.jsonl` + `latest.txt`.

## Bước 8 — phần B: huấn luyện LoRA · 28 phút · 5,5 CU

**Chạy:** cell **#11 → #16**, rồi **DỪNG**.

Cell #11 là ô cấu hình lặp lại — **không phải điền lại `GITHUB_REPO`**, nó tự đọc bản ghi
nhớ trên Drive, và thư mục đã clone vẫn còn sau restart.

Cell #12 phải in `[SFT] dữ liệu: …/qwen3_sft_<stamp>.jsonl` và số mẫu khớp với phần A.

Trên A100 40GB: batch 8 × accum 2. **Batch hiệu dụng luôn là 16** ở mọi GPU, để `lr = 2e-4`
giữ nguyên ý nghĩa.

**⛔ Cổng — §9, đường cong loss (cell #16):**

| Quan sát | Kết luận |
|---|---|
| val loss giảm rồi phẳng | ✅ bình thường |
| val loss **tăng ngược** | Overfit — đúng cái lần chạy trước gặp. Giảm số epoch, hoặc siết `ACCEPT = "pa"` cho dữ liệu sạch hơn rồi chạy lại phần A |

**Rồi: Restart session lần nữa.**

**Sinh ra:** `vinumqa_runs/sft_adapter_qwen3/`.

## Bước 9 — phần C: chấm model đã SFT · 9 phút · 1,8 CU

**Chạy:** cell **#17 → #26**.

Cell #23 phải in `[LoRA] đã nạp adapter từ …/sft_adapter_qwen3`.

§10 so với nấc 2. **Cả ba khả năng đều là kết quả đáng báo cáo:**

| Quan sát | Viết gì |
|---|---|
| SFT > nấc 2 | Rejection sampling khắc phục được lỗi overfit của SFT trên gold trần |
| SFT ≈ nấc 2 | Fine-tune 8B với 1000 mẫu không thêm gì so với prompt tốt — củng cố luận điểm inference-time thắng training-time |
| SFT < nấc 2 | Vẫn overfit dù đã lọc; ghi rõ cấu hình để người sau không lặp lại |

Bảng cuối notebook `03` gợi ý câu chữ cho từng trường hợp.

**Sinh ra:** `03_sft.*`.

---

# BUỔI 3 — ma trận tổ hợp (bước 10–13)

Trả lời câu đáng giá nhất: **SFT và ACE đều học từ train — một cái vào trọng số, một cái vào
ngữ cảnh. Chúng bổ sung hay trùng nhau?**

## Bước 10 — `04_self_evaluation` lần 2 · 18 phút · 3,5 CU

Mở **lại chính notebook `04`**, đổi **một cờ** ở cell #6:

```python
USE_SFT_ADAPTER = True
```

Phải thấy:

```
[MODEL] dùng adapter đã SFT: /content/drive/MyDrive/vinumqa_runs/sft_adapter_qwen3
[MODEL] nấc trước để so sánh: 03_sft
```

Notebook tự đổi tên nấc thành `04_selfeval_sft` nên **không đè lên kết quả lần 1**.

Báo `Không thấy adapter tại …` ⇒ bước 8 chưa xong.

## Bước 11 — `05_ace` lần 2 · 63 phút · 12,4 CU

Hai cờ:

| Cell | Đặt | Vì sao |
|---|---|---|
| #6 | `USE_SFT_ADAPTER = True` | ACE trên model đã SFT |
| #15 | `RUN_RANDOM_CONTROL = **False**` | đối chứng đã có ở lần 1, chạy lại là phí 18 phút |

Pha A học **playbook mới** (`playbook_ace_sft.txt`) — đúng như vậy: model đã đổi thì lỗi nó
mắc cũng đổi, playbook phải học lại. Checkpoint riêng (`progress_ace_sft.json`), không đụng
lần 1.

**Sinh ra:** `05_ace_sft.*`, `playbook_ace_sft.txt`.

> Đối chiếu hai playbook `playbook_ace_base.txt` và `playbook_ace_sft.txt` là một phân tích
> hay cho bài: SFT đã "hấp thụ" sẵn những bullet nào?

## Bước 12 — `06_combination` · 18 phút · 3,5 CU

**Sửa:** không gì cả. Notebook tự dò xem ô nào đã có trên đĩa.

**Chạy:** Run all. §3 in bảng trạng thái:

```
ô        SFT  self-eval   ACE   nấc                 trạng thái
E          ·          ·     ·   02_prompt_eng       đã có — đọc lại
E+A        ·          ·     ✓   06_comb_E_A         CẦN CHẠY
E+S        ·          ✓     ·   04_selfeval_base    đã có — đọc lại
E+S+A      ·          ✓     ✓   05_ace_base         đã có — đọc lại
F          ✓          ·     ·   03_sft              đã có — đọc lại
F+A        ✓          ·     ✓   06_comb_F_A         CẦN CHẠY
F+S        ✓          ✓     ·   04_selfeval_sft     đã có — đọc lại
F+S+A      ✓          ✓     ✓   05_ace_sft          đã có — đọc lại

  6/8 ô đã có sẵn.
```

**6 ô đọc lại từ đĩa, chỉ 2 ô mới phải sinh** — đó là ACE **không kèm** self-eval, cấu hình
chưa nấc nào chạy. Vì thế bước này chỉ 18 phút chứ không phải chạy lại cả 8 ô.

Thấy `⚠ chưa có adapter` ⇒ nó tự bỏ 4 ô cần SFT và chạy ma trận 2×2 rút gọn. Vẫn dùng được,
nhưng không trả lời được câu hỏi tương tác SFT×ACE.

**Đọc kết quả — §7 (tác động chính & tương tác) và §8 (kiểm định):**

| Quan sát | Kết luận viết vào bài |
|---|---|
| Tương tác SFT×ACE **âm rõ** | Hai kỹ thuật học cùng một thứ ⇒ chọn một. Chọn ACE nếu muốn khỏi huấn luyện. |
| Tương tác SFT×ACE **≈ 0** | Trọng số và ngữ cảnh là hai kênh độc lập ⇒ cộng dồn được |
| **`E+A` ≥ `E+S`** | ACE **thay được** self-eval với **nửa chi phí** (1 lượt sinh thay vì 2). Luận điểm mạnh nhất cho bối cảnh tài nguyên hạn chế. |
| `F+S+A` tốt nhất và p < 0,05 so với `E+S` | Tổ hợp đầy đủ thắng cấu hình tham chiếu ⇒ con số chính của báo cáo |

§9 còn in **chi phí trên mỗi điểm EA** — cột này quan trọng ngang EA trong một bài về tài
nguyên hạn chế.

**Sinh ra:** `06_comb_E_A.*`, `06_comb_F_A.*`, `ma_tran_to_hop_<stamp>.csv`.

## Bước 13 — `07_final_report` lần cuối · CPU · 1 phút

**Runtime: CPU.** Run all. Giờ đủ 10 nấc, không còn dòng `⚠ chưa có`.

Xuất ra `bang_ket_qua_<stamp>.csv` — dán thẳng vào bài.

---

## Bảng ghi kết quả (điền tay khi chạy)

| Nấc | EA | PA_strict | PA_loose | Δ EA so với nấc trước | p | Phút |
|---|--:|--:|--:|--:|--:|--:|
| `01_plain` | | | | — | — | |
| `02_prompt_eng` | | | | | | |
| `03_sft` | | | | | | |
| `04_selfeval_base` | | | | | | |
| `04_selfeval_sft` | | | | | | |
| `05_ace_base` | | | | | | |
| `05_ace_sft` | | | | | | |
| `05_ace_random_base` | | | | | | |
| `06_comb_E_A` | | | | | | |
| `06_comb_F_A` | | | | | | |

Notebook `07` in lại bảng này, nhưng ghi tay lúc chạy giúp phát hiện ngay con số bất thường.

---

## Mất session giữa chừng thì sao

| Đang ở bước | Mất gì | Làm gì |
|---|---|---|
| 2, 3, 4, 10 | Cả lần chạy | Mở lại, Run all. Mất 9–18 phút. |
| 5 hoặc 11, **pha A** | Chỉ vòng đang dở | Mở lại, Run all — thấy `[RESUME] tiếp từ vòng N` là đúng |
| 5 hoặc 11, **pha B** | Cả pha B | Playbook đã lưu ra `.txt`; pha A không phải học lại |
| 7 (`03` phần A) | Cả phần A | Chạy lại #1 → #10 |
| 8 (`03` phần B) | Cả phần B | Dữ liệu SFT đã ở Drive; chạy lại #11 → #16 |
| 9 (`03` phần C) | Chỉ phần C | Adapter đã ở Drive; chạy lại #17 → #26 |

Notebook nào cũng in **bảng tiến độ** ở cell #2, nên mở ra là biết đang đứng ở đâu.

---

## Muốn thử nhanh trước khi chạy thật

Chèn một cell **ngay sau cell #2**:

```python
test_all = test_all[:40]
```

40 mẫu chạy ~1 phút thay vì 9, đủ để xác nhận toàn bộ đường ống thông. **Nhớ xoá cell đó
trước khi chạy thật** — và xoá luôn file nấc đã ghi bằng 40 mẫu, vì `save_stage` sẽ đè:

```python
import glob, os
for f in glob.glob(os.path.join(RESULT_DIR, "01_plain*")): os.remove(f)
```

---

## Ba lỗi đọc kết quả hay gặp

1. **So EA của mình với EA tham chiếu.** Không so được — EA tham chiếu tính bằng
   executor lỗi `table_*`. Chỉ **`PA_loose`** mới so trực tiếp được. Notebook `00` §5 định
   lượng chênh lệch này.
2. **Chạy nhiều cấu hình rồi nhặt cái p < 0,05.** Kết luận mạnh chỉ nên dựa vào các so sánh
   **đã định trước**: từng bước leo thang, `05` vs `04`, và truy hồi vs ngẫu nhiên.
3. **Báo cáo EA mà không báo cáo chi phí.** Mỗi nấc phải kèm: lượt sinh/mẫu, có cần huấn
   luyện không, có cần API ngoài không. Dự án đặt trong bối cảnh tài nguyên hạn chế nên cột đó
   quan trọng ngang EA.
