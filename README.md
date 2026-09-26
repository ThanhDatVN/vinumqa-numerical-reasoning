# Suy luận số học trên báo cáo tài chính tiếng Việt

Nghiên cứu thực nghiệm về các kỹ thuật nâng độ chính xác của mô hình ngôn ngữ 8B trên
bài toán hỏi–đáp số học có bảng biểu. Mô hình sinh ra một **chương trình DSL** thay vì
trả lời trực tiếp; chương trình được thực thi bằng máy để lấy đáp án.

**Mô hình:** Qwen3-8B (vLLM, 4-bit) · **Tập test:** 497 câu · **Phần cứng:** A100-40GB

---

## 1. Bài toán

Đầu vào là một câu hỏi tiếng Việt kèm trích đoạn báo cáo tài chính (văn bản trước, bảng
số liệu, văn bản sau). Đầu ra là một chương trình DSL phẳng:

```
program: subtract(19296, 18511), divide(#0, 18511)
answer: 0.0424
```

DSL gồm `add · subtract · multiply · divide · exp · greater` và bốn phép đọc bảng
`table_max · table_min · table_sum · table_average`. Các phép nối nhau qua tham chiếu
`#N`, không lồng nhau.

### Thước đo

| | ý nghĩa |
|---|---|
| **EA** (Executed Accuracy) | thực thi chương trình, so kết quả với đáp án vàng |
| **PA_strict** (Program Accuracy) | chuẩn hoá chương trình rồi so với chương trình vàng |
| **PA_loose** | bản chuẩn hoá lỏng hơn, dùng để đối chiếu với bảng tham chiếu |

EA là thước đo chính. PA chặt hơn vì đòi đúng cả cách giải, không chỉ đúng đáp án.

---

## 2. Kết quả

| nấc | cấu hình | EA | PA_strict | phút |
|---|---|---:|---:|---:|
| 1 | prompt cơ bản | 43,46 | 40,44 | 9,5 |
| 2 | prompt hoàn chỉnh | 68,01 | 63,98 | 9,7 |
| 3 | + SFT | 70,02 | 64,19 | 12,6 |
| 4 | + self-evaluation | 69,82 | 63,58 | 20,1 |
| 5 | + ACE | 69,82 | 64,39 | 21,9 |
| 8 | + self-consistency (K=5) | 73,24 | 67,81 | 32,6 |
| 9 | + ví dụ truy hồi kNN | 77,87 | 72,43 | 29,1 |
| **10** | **+ bộ chọn bằng mô hình** | **79,28** | **73,64** | **3,0** |

![Thang bậc](docs/hinh/01_thang_bac.png)

Cấu hình tốt nhất đạt **EA 79,28 · PA_strict 73,64**, tăng 35,82 điểm EA so với prompt
cơ bản và 11,27 điểm so với prompt hoàn chỉnh.

### Đóng góp của từng cơ chế

Mỗi cơ chế được đo bằng kiểm định McNemar ghép cặp trên cùng 497 mẫu. Cột *hỏng/sửa* là
số câu chỉ một bên làm đúng.

| cơ chế | Δ EA | hỏng | sửa | |
|---|---:|---:|---:|:-:|
| Prompt engineering | **+24,55** | 33 | 155 | ✅ |
| ACE trên prompt cơ bản | **+18,71** | 26 | 119 | ✅ |
| Self-consistency K=5 | **+5,23** | 12 | 38 | ✅ |
| Ví dụ truy hồi kNN | **+4,63** | 19 | 42 | ✅ |
| Supervised fine-tuning | +2,01 | 39 | 49 | — |
| Self-evaluation | +1,81 | 24 | 33 | — |
| Bộ chọn bằng mô hình | +1,41 | 10 | 17 | — |
| ACE trên prompt hoàn chỉnh | ±0,00 | 32 | 32 | — |

✅ = chênh lệch vượt ngưỡng ý nghĩa 95 % của phép đo. Dấu — nghĩa là ước lượng điểm
dương nhưng chưa đủ bằng chứng ở cỡ mẫu 497.

![Đóng góp từng kỹ thuật](docs/hinh/02_dong_gop.png)

Số liệu đầy đủ, gồm cả các nhánh đối chứng và toàn bộ 32 phép kiểm định:
[`docs/KET_QUA.md`](docs/KET_QUA.md).

---

## 3. Bốn nhận xét rút ra

### 3.1 Kỹ thuật ngữ cảnh vượt xa huấn luyện lại

Prompt engineering một mình đóng góp **+24,55 điểm** — lớn hơn tổng mọi cơ chế còn lại.
Trong khi đó SFT trên 1.251 mẫu tự sinh, tốn 100 phút GPU, cho +2,01 điểm và không vượt
ngưỡng ý nghĩa.

Nguyên nhân có tính cấu trúc: dữ liệu SFT chỉ gồm những câu mô hình **vốn đã làm đúng**
(1.251/2.000). Rejection sampling không dạy được gì về nhóm câu mô hình đang làm sai.

### 3.2 ACE khám phá lại được tri thức viết tay, nhưng chỉ khi prompt còn chỗ trống

ACE (Agentic Context Engineering) học một *playbook* các quy tắc từ tập train, rồi truy
hồi quy tắc liên quan vào prompt lúc suy luận. Dùng đối chứng bullet ngẫu nhiên lấy từ
**chính playbook đã học**, tách được phần đóng góp của *nội dung* khỏi phần của *cơ chế
truy hồi*:

| | trên prompt cơ bản | trên prompt hoàn chỉnh |
|---|---:|---:|
| nội dung playbook | **+16,50** ✅ | −0,40 |
| cơ chế truy hồi | +2,21 | +2,21 |

Đường học trên tập dev xác nhận: trên prompt cơ bản, điểm tổng hợp tăng đơn điệu qua các
vòng (0,458 → 0,538 → 0,562 → 0,569 → 0,608); trên prompt hoàn chỉnh nó dao động không
xu hướng (0,672 → 0,644 → 0,690 → 0,667 → 0,689).

![ACE: nội dung và truy hồi](docs/hinh/04_ace.png)

Nội dung ACE học được chính là ánh xạ từ khoá → phép toán mà prompt hoàn chỉnh đã chứa
sẵn, ví dụ: *"Khi hỏi tỷ lệ thay đổi giữa hai kỳ, dùng `subtract(giá_trị_mới,
giá_trị_cũ)`, `divide(#0, giá_trị_cũ)`"*. Khi tri thức đó đã có trong prompt, ACE không
còn chỗ để đóng góp.

**ACE hoạt động. Phần đóng góp của nó đã bị prompt engineering chiếm trước.**

### 3.3 Tính toán lúc suy luận hiệu quả nhưng bão hoà nhanh

Sinh K mẫu rồi bỏ phiếu theo giá trị thực thi cho +5,23 điểm; thay 2 ví dụ cố định bằng
ví dụ truy hồi kNN cho thêm +4,63 điểm. Nhưng đường cong theo K phẳng từ K=4:

| K | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| EA | 74,65 | 75,25 | 75,86 | **77,26** | 77,26 |

![Đường cong K](docs/hinh/03_duong_cong_k.png)

Tăng K tiếp không còn lợi. Đáng chú ý hơn: self-evaluation cho +1,81 điểm khi đứng một
mình, nhưng khi cộng lên cấu hình đã có bỏ phiếu thì cho **−1,41 điểm**. Bỏ phiếu 5 mẫu
đã làm sẵn việc mà self-evaluation định làm — bắt lỗi ở lần thử đầu — nên chồng thêm chỉ
đè lên những đáp án vốn đã đúng.

### 3.4 Nút thắt nằm ở khâu chọn, không nằm ở năng lực mô hình

Trần *best-of-5* của nấc 9 là **85,31** trong khi bỏ phiếu chỉ đạt 77,87. Phân tích 497
câu cho thấy:

| số giá trị phân biệt trong 5 mẫu | số câu |
|---|---:|
| 1 (hoặc 0) | 382 |
| ≥ 2 | **115** |

Ở nhóm 115 câu đó, bỏ phiếu đúng 48 câu (41,7 %) trong khi **88 câu (76,5 %) có đáp án
đúng nằm đâu đó trong 5 mẫu**. Toàn bộ 40 câu chênh lệch đều là trường hợp đáp án đúng
thuộc **thiểu số** (1/5 hoặc 2/5 mẫu) — phép đếm phiếu theo số đông về nguyên tắc không
thắng được.

![Bộ chọn](docs/hinh/06_bo_chon.png)

Bốn luật bỏ phiếu thay thế đã được thử trên cùng dữ liệu (bỏ phiếu theo cấu trúc chương
trình, hoà thì theo chương trình phổ biến, hoà thì chọn chương trình ngắn nhất) — không
luật nào vượt được luật hiện tại.

Nấc 10 thay phép đếm bằng một lượt để mô hình **so sánh và chấm** giữa các ứng viên phân
biệt, kèm giá trị mỗi ứng viên chạy ra. Kết quả trên nhóm 115 câu: 41,7 % → **47,8 %**,
lấp được 18 % khoảng cách tới trần, tốn 3 phút.

Hướng cải tiến rõ ràng nhất còn lại không phải là sinh tốt hơn, mà là **chọn tốt hơn**.

### 3.5 Một lỗi không cơ chế nào chạm tới

Phân loại lỗi cho thấy nhóm `đúng phép toán, sai số liệu` — mô hình hiểu đúng bài nhưng
lấy nhầm ô trong bảng — giữ nguyên tỷ trọng qua mọi nấc:

| nấc | 2 | 3 | 4 | 5 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| số câu | 67 | 52 | 61 | 65 | 58 | 60 | 56 |

![Cơ cấu lỗi](docs/hinh/05_phan_loai_loi.png)

Không kỹ thuật nào trong nghiên cứu này làm nó giảm đáng kể. Đây là bài toán **đọc bảng**
chứ không phải bài toán suy luận, và là hướng nghiên cứu tiếp theo.

---

## 4. Bố cục

```text
vinumqa/                    thư viện lõi
  dsl.py                    phân tích cú pháp + thực thi DSL, chấm EA/PA
  data.py                   nạp dữ liệu, lấy mẫu phân tầng, kiểm chất lượng nhãn
  prompts.py                thang prompt lồng nhau + prompt cho từng cơ chế
  pipeline.py               vòng suy luận: sinh, bỏ phiếu, sửa lỗi, chọn
  fewshot.py                kho ví dụ truy hồi theo BM25
  sft.py                    dựng dữ liệu SFT, cấu hình huấn luyện
  stats.py                  McNemar, khoảng tin cậy bootstrap
  io_utils.py               đọc/ghi kết quả từng nấc
  ace/                      Generator–Reflector–Curator, playbook, truy hồi

notebooks/                  00–10, mỗi nấc một notebook
data/                       ViNumQA: 2.993 train · 584 valid · 497 test
tests/                      245 test, chạy trên CPU, không cần mô hình
docs/
  PHUONG_PHAP.md            thiết kế thực nghiệm
  KET_QUA.md                bảng đầy đủ và kiểm định
  hinh/                     hình dựng lại được từ kết quả đã lưu
tools/
  kiem_tra.py               bộ kiểm toàn dự án — 8 phép
  chay_thu_notebook.py      chạy logic notebook GPU bằng mô hình giả
  ve_bieu_do.py             dựng hình cho tài liệu từ file nấc
  do_do_phu.py              đo độ phủ dòng của phần phân tích
  kiem_thu_tu_ten.py        tên dùng ở ô i phải định nghĩa ở ô ≤ i
```

Phần cần GPU nhận `generate_fn` tiêm từ ngoài vào, nên toàn bộ thư viện kiểm thử được
trên CPU không cần mô hình.

Kết quả chạy thực nghiệm ghi xuống `results/` (không đưa vào git): mỗi nấc một file
`.jsonl` chứa từng mẫu và một file `_meta.json` chứa chỉ số tổng hợp cùng cấu hình đã
dùng. `tools/ve_bieu_do.py` đọc thẳng các file này nên hình trong tài liệu luôn khớp số.

---

## 5. Tái lập

### Dựng lại kết quả mà không cần GPU

Kết quả từng mẫu của cả 13 nấc nằm sẵn trong `results/stages/` (6,5 MB). Mọi con số và
hình trong tài liệu tính lại được từ đó trên CPU:

```bash
pip install -r requirements.txt
python tools/ve_bieu_do.py          # dựng lại 6 hình của docs/hinh
```

Sáu hình dựng lại trùng từng byte với bản đang có trong repo.

Mỗi nấc gồm hai file: `<nấc>.jsonl` (từng mẫu — câu hỏi, chương trình sinh ra, giá trị
thực thi, EA/PA, phân loại lỗi, và với nấc K mẫu là cả năm ứng viên) và
`<nấc>_meta.json` (chỉ số tổng hợp, cấu hình sinh, phiên bản thư viện, commit, GPU).
Adapter SFT, checkpoint huấn luyện và log thô không đưa vào git vì dung lượng.

### Kiểm thử

```bash
python -m pytest tests/ -q          # 245 test, ~6 giây, không cần mô hình
python tools/kiem_tra.py --day-du   # 8 phép kiểm toàn dự án
```

### Chạy lại thực nghiệm

Mỗi notebook trong `notebooks/` là một nấc độc lập, chạy trên Google Colab với A100-40GB.
Ô đầu tiên tự lấy mã nguồn từ GitHub và gắn Google Drive; kết quả ghi xuống Drive dưới
dạng `<nấc>.jsonl` + `<nấc>_meta.json` nên nấc sau đọc lại được nấc trước.

Mỗi notebook ghi ra một hoặc hai nấc; số notebook và số nấc không trùng nhau ở một
chỗ — `08_phuong_phap_moi` sinh ra cả `08_tu_nhat_quan` lẫn `09_vidu_dong` vì hai nấc
đó chỉ khác nhau ở nguồn ví dụ và dùng chung một lần nạp mô hình.

| notebook | nấc ghi ra |
| --- | --- |
| `00_data_audit` | — (kiểm dữ liệu) |
| `01_baseline_basic` | `01_basic` |
| `02_prompt_engineering` | `02_prompt_eng` |
| `03_sft_qwen3` | `03_sft` |
| `04_self_evaluation` | `04_selfeval_base` |
| `05_ace` | `05_ace_base`, `05c_ace_basic_base` + hai nhánh đối chứng |
| `06_combination` | `06_comb_E_A`, `04_selfeval_base_moi` |
| `07_final_report` | — (tổng hợp) |
| `08_phuong_phap_moi` | `08_tu_nhat_quan`, `09_vidu_dong` |
| `10_bo_chon` | `10_bo_chon` |

Thứ tự phụ thuộc:

```
00 (kiểm dữ liệu, CPU)
01 → 02 → 03 (SFT)
       02 → 04 → 05 (ACE)
       02 → 08 → 09 → 10
       02 → 06 (ma trận tổ hợp)
07 (báo cáo, CPU) — đọc mọi nấc đã có
```

Tổng thời gian GPU cho toàn bộ lộ trình khoảng 15 giờ. Notebook `03` cần khởi động lại
runtime giữa ba pha (dựng dữ liệu → huấn luyện → chấm điểm); notebook `05` cần khoá
OpenAI API cho thành phần Reflector.

Chi tiết thiết kế: [`docs/PHUONG_PHAP.md`](docs/PHUONG_PHAP.md).

---

## 6. Giới hạn

**Cỡ mẫu.** Tập test có 497 câu. Với mức xáo trộn mà các cơ chế gây ra, ngưỡng ý nghĩa
95 % nằm trong khoảng 2–5 điểm EA tuỳ cặp so sánh. Các hiệu ứng 1–2 điểm (SFT,
self-evaluation, bộ chọn) có ước lượng điểm dương nhưng không thể chứng minh ở cỡ mẫu
này. Tập `valid` 584 câu chưa được dùng để đánh giá; gộp vào sẽ nâng n lên 1.081 và hạ
ngưỡng xuống khoảng 1,4–2,4 điểm.

**Tính tái lập số học.** Suy luận theo lô bằng vLLM không tất định tuyệt đối: thành phần
lô đổi thì thứ tự cộng dồn trong kernel đổi, logits lệch ở chữ số cuối, và ở
`temperature = 0.1` đủ để lật một token rồi kéo theo cả chuỗi suy luận. Mọi nấc trong
nghiên cứu này chạy trên cùng một loại card (A100-40GB) với cùng tham số lô để loại yếu
tố đó khỏi các phép so sánh.

**Nhãn vàng.** Tập train có 5 nhãn vàng cụt cú pháp và 100 nhãn dùng `multiply(#n, 100)`
để biểu diễn phần trăm. Cả hai nhóm đều bị lọc khỏi dữ liệu SFT và kho ví dụ few-shot
(xem `vinumqa/data.py::is_noisy_gold`).

**Thành phần ngoài.** Reflector của ACE dùng `gpt-4o-mini` qua API, đúng như thiết kế
gốc của phương pháp. Nấc 5 vì vậy không thuộc nhóm *constrained* như các nấc còn lại;
trường `reflector_backend` trong metadata giữ dấu vết này.
