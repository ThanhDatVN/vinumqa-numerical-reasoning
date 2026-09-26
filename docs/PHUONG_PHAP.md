# Thiết kế thực nghiệm

Tài liệu này mô tả cách từng cơ chế được cài đặt và đo. Kết quả số nằm ở
[`KET_QUA.md`](KET_QUA.md).

---

## 1. Nguyên tắc chung

**Mỗi nấc thêm đúng một thay đổi.** Nấc sau giữ nguyên mọi thứ của nấc trước và chỉ bật
thêm một cơ chế, để hiệu số quy được về cơ chế đó.

**Mọi nấc chấm trên cùng 497 mẫu test, cùng thứ tự.** Nhờ vậy mọi so sánh đều là kiểm
định ghép cặp (McNemar), mạnh hơn hẳn so sánh hai tỷ lệ độc lập.

**Thang bậc là cây, không phải dây.** Nấc 4 xây trên nấc 2 chứ không phải nấc 3; nấc 8
cũng vậy. Mỗi nấc được so với đúng nấc nó xây lên:

```text
01_basic
  └─ 02_prompt_eng
       ├─ 03_sft
       ├─ 04_selfeval_base ─ 05_ace_base
       ├─ 06_comb_E_A
       └─ 08_tu_nhat_quan ─ 09_vidu_dong ─ 10_bo_chon
```

**Cấu hình sinh cố định giữa các nấc thang bậc.** `temperature = 0.1`,
`max_tokens = 4096`, `max_seq_length = 17000`, seed cố định. Các nấc 8–10 cố ý đổi cấu
hình sinh (K mẫu, nhiệt độ cao hơn) — đó là nội dung của chính cơ chế, và được ghi rõ
trong metadata của nấc.

**Phần cứng đồng nhất.** Toàn bộ nấc chạy trên A100-40GB với cùng `max_num_seqs = 48`.
Notebook `07` có một bước kiểm tra công bằng, báo lỗi nếu phát hiện nấc nào lệch GPU,
trần token, hay chế độ suy nghĩ.

---

## 2. Thang prompt

Ba mức prompt **lồng nhau**: mỗi mức là phép chèn thuần vào mức trước, phần dùng chung
giống nhau từng ký tự.

| mức | nội dung thêm | dùng ở |
|---|---|---|
| `basic` | danh sách phép toán + yêu cầu định dạng | nấc 1, 5c |
| `no_fewshot` | + hướng dẫn ánh xạ từ khoá → phép toán | (trung gian) |
| `engineered` | + hai ví dụ mẫu | nấc 2 trở đi |

Tính lồng nhau được kiểm tự động (`tools/kiem_tra.py`, phép kiểm 3): sửa chữ ở phần dùng
chung sẽ làm hỏng phép đo nên bộ kiểm chặn lại.

Ba chỉnh sửa trong prompt `engineered` được rút ra từ **thống kê trên nhãn vàng tập
train**, không phải từ phỏng đoán:

- quy tắc dùng `table_*` (618/618 nhãn vàng dùng `none` làm tham số thứ hai)
- cách biểu diễn phần trăm (2.053 nhãn dùng tỷ lệ thập phân, 109 nhãn nhân 100)
- thứ tự toán hạng khi hỏi mức giảm (61/74 nhãn lấy giá trị mới trừ giá trị cũ)

---

## 3. Các cơ chế

### 3.1 Supervised fine-tuning (nấc 3)

Rejection sampling: chạy mô hình gốc trên 2.000 mẫu train, giữ lại những mẫu nó làm đúng
(PA hoặc EA), dùng chính lời giải của nó làm đích huấn luyện. LoRA r=16, alpha=32, batch
hiệu dụng 16, 3 epoch.

Đích huấn luyện **giữ nguyên khối suy luận** `<think>…</think>`. Đây là điểm mấu chốt:
bản cài đặt đầu tiên cắt bỏ khối này, và chat template của Qwen3 khi đó dựng lượt
assistant thành `<think></think>` rỗng — tức 1.251 ví dụ dạy mô hình bỏ qua suy luận. Hệ
quả đo được: 497/497 mẫu test sinh khối suy nghĩ dài 2 ký tự, thời gian suy luận từ 10,1
phút xuống 1,4 phút, EA tụt 11,27 điểm. Sau khi giữ lại khối suy luận, kết quả hồi phục
+11,07 điểm.

Lọc dữ liệu: bỏ mẫu có nhãn vàng nhiễu (`vinumqa/data.py::is_noisy_gold`), bỏ mẫu quá
dài. Còn 1.251/2.000 bản ghi.

### 3.2 Self-evaluation (nấc 4)

Hai lượt sinh cho mỗi câu. Lượt hai nhận lời giải của lượt một kèm **giá trị nó chạy
thật ra**, và được yêu cầu soát lại. Độ lớn con số là chỗ lộ lỗi rõ nhất nên nó được đưa
vào prompt.

Có một biến thể **có cổng**: chỉ nhận chương trình của bước hai khi nó thực thi được,
hoặc khi bước một vốn cũng không chạy được. Biến thể này tính lại được từ file jsonl
trên CPU, không cần chạy lại GPU.

Cách đo: ngoài phép so với nấc 2, nấc này còn có **đối chứng trong cùng lượt chạy** —
chấm điểm chương trình của bước 1 và chương trình cuối trên đúng một lần sinh. Đối chứng
này loại được nhiễu giải mã giữa hai lần chạy nên nhạy hơn hẳn.

### 3.3 ACE — Agentic Context Engineering (nấc 5, 5c)

Vòng lặp ba vai trên tập train:

1. **Generator** chạy mô hình trên một lô 32 mẫu
2. **Reflector** (`gpt-4o-mini`) xem các câu sai, đề xuất quy tắc rút ra được
3. **Curator** cho quy tắc qua cổng chất lượng rồi chèn vào playbook

Cổng chất lượng loại quy tắc trùng ngữ nghĩa (ngưỡng tương đồng 0,98), trùng cấu trúc,
dưới một phép toán, và quy tắc **không sửa được câu nó sinh ra để sửa** (kiểm chứng bằng
cách chạy lại thật). Playbook trần 30 bullet, tối đa 3 bullet mỗi cụm lỗi.

Lúc suy luận, `Retriever` chọn tối đa 7 bullet liên quan cho mỗi câu: 3 từ Tier-1 (bullet
đã chứng minh hữu ích) và 4 từ Tier-2.

Snapshot playbook được chọn theo điểm tổng hợp `0,6·EA + 0,4·PA` trên tập dev 240 mẫu,
đo mỗi 5 vòng.

**Đối chứng.** Một nhóm đối chứng dùng `RandomRetriever` — lấy bullet **ngẫu nhiên từ
chính playbook đã học**. Nhờ vậy tách được hai phần:

| phép so | đo cái gì |
|---|---|
| không playbook → bullet ngẫu nhiên | **nội dung** playbook |
| bullet ngẫu nhiên → bullet truy hồi | **cơ chế truy hồi** |

Đối chứng chỉ có nghĩa khi playbook lớn hơn k=7; cả hai lần chạy đều thoả (12 và 16
bullet) và notebook tự cảnh báo nếu không.

**Nấc 5c** lặp lại toàn bộ quy trình trên prompt `basic`. Prompt `engineered` đã chứa
sẵn ánh xạ từ khoá → phép toán, tức đúng loại tri thức ACE định rút ra; nấc 5c đo xem
ACE có tự khám phá lại được nó không khi prompt không có gì.

### 3.4 Self-consistency (nấc 8)

Sinh K=5 mẫu cho mỗi câu ở `temperature = 0.7`, thực thi cả năm, bỏ phiếu theo **giá trị
thực thi** chứ không theo văn bản chương trình. Hai chương trình khác chữ mà cùng kết
quả là cùng một lá phiếu.

Nhiệt độ phải cao hơn 0,1 thì năm mẫu mới khác nhau; ở `temperature = 0.1` chúng gần như
trùng nhau và bỏ phiếu vô nghĩa. Phần đóng góp riêng của việc đổi nhiệt độ được đo tách
ra (so k=1 ở hai nhiệt độ).

Cả năm mẫu được lưu vào `cac_program` / `cac_gia_tri`, nên đường cong theo K và mọi luật
bỏ phiếu thay thế dựng lại được trên CPU từ một lần chạy GPU.

**Lượt sửa khi lỗi.** Sau khi chốt chương trình, câu nào executor từ chối được sinh lại
một lượt kèm đúng thông báo lỗi. Chỉ câu hỏng mới đi qua bước này nên rất rẻ.

### 3.5 Ví dụ truy hồi kNN (nấc 9)

Thay hai ví dụ cố định trong prompt bằng 3 ví dụ truy hồi từ tập train theo BM25. Kho ví
dụ lọc bỏ mẫu có nhãn vàng nhiễu (2.888/2.993 mẫu được giữ), tất định, không cần embedding
ngoài. Với ví dụ dùng `table_*`, chỉ nhãn hàng được đưa vào để tiết kiệm ngân sách token.

### 3.6 Bộ chọn bằng mô hình (nấc 10)

Thay phép đếm phiếu bằng một lượt để mô hình tự chấm. Các mẫu được gom thành **ứng viên
phân biệt** theo giá trị thực thi, rồi đưa cả nhóm vào prompt kèm giá trị mỗi ứng viên
chạy ra; mô hình chọn một.

Ba quyết định thiết kế:

- **Chỉ chạy trên câu có ≥2 ứng viên.** 382/497 câu chỉ ra một giá trị duy nhất — ở đó
  không có gì để chọn. Nhờ vậy chỉ phải sinh 115 prompt.
- **Không đưa số phiếu vào prompt.** Đó đúng là tín hiệu mà bỏ phiếu đã dùng, và ở nhóm
  này nó sai 58 %. Thứ tự ứng viên theo lần xuất hiện đầu, không theo số phiếu.
- **Fail-closed mọi nhánh.** Không đọc được lựa chọn, chọn ngoài phạm vi, hay câu một
  ứng viên → giữ nguyên đáp án bỏ phiếu.

Nấc này không sinh lại lời giải nào: năm mẫu đã nằm trong file jsonl của nấc 9.

---

## 4. Ma trận tổ hợp (nấc 6)

Ma trận 2×2 đo **tác động chính và tương tác** của self-evaluation × ACE, giữ nguyên cấu
hình sinh của thang bậc (1 mẫu, `temperature = 0.1`, ví dụ cố định) để hiệu số không lẫn
phần đổi ngân sách tính toán.

| ô | self-eval | ACE | nấc |
|---|:-:|:-:|---|
| E | · | · | `02_prompt_eng` |
| E+A | · | ✓ | `06_comb_E_A` |
| E+S | ✓ | · | `04_selfeval_base` |
| E+S+A | ✓ | ✓ | `05_ace_base` |

Tác động chính của mỗi yếu tố là trung bình hai cặp chỉ khác đúng yếu tố đó, kèm khoảng
tin cậy bootstrap lấy mẫu lại theo mẫu.

Ngoài ma trận còn một **ô mục tiêu**: ô ma trận tốt nhất chạy lại ở cấu hình tốt nhất
(K=5, nhiệt độ 0,7, ví dụ truy hồi, sửa-khi-lỗi). Hai nhóm ghi ra tên nấc khác nhau nên
không đè lên nhau.

---

## 5. Thống kê

**McNemar ghép cặp** cho mọi so sánh giữa hai nấc. Báo cáo số câu chỉ một bên làm đúng
(`b` và `c`) bên cạnh hiệu số, vì chúng cho biết mức xáo trộn — hai cấu hình có thể lệch
nhau rất ít về tổng mà vẫn đổi đáp án ở hàng chục câu.

**Ngưỡng ý nghĩa 95 %** tính theo `1,96·√(b+c)/n`, không dùng một hằng số. Hai cấu hình
xáo trộn nhiều mẫu cần chênh lệch lớn hơn mới kết luận được.

**Khoảng tin cậy bootstrap** lấy mẫu lại theo mẫu (mọi ô chấm trên cùng 497 câu nên đây
là cách đúng). Kết luận "có tác dụng" chỉ đưa ra khi p < 0,05 **và** khoảng tin cậy không
chứa 0.

---

## 6. Chất lượng dữ liệu

Tập train có hai nhóm nhãn vàng không đáng tin, đều bị lọc khỏi dữ liệu SFT và kho ví dụ
few-shot:

| nhóm | train | valid | test |
|---|---:|---:|---:|
| `multiply(#n, 100)` biểu diễn phần trăm | 100 | 1 | 0 |
| nhãn không thực thi được | 5 | 0 | 0 |

Năm nhãn cụt cú pháp trong train có dạng `subtract(a, b), divide(#0, b` — thiếu dấu đóng
ngoặc. Vá dấu ngoặc không phải cách chữa: chương trình vá xong chạy ra giá trị khác với
`exe_ans` đã ghi, tức nhãn tự mâu thuẫn chứ không chỉ cụt chữ.

---

## 7. Bộ kiểm

`tools/kiem_tra.py` chạy tám phép kiểm, sáu phép tĩnh và ba phép thực thi mã thật:

| | kiểm gì |
|--:|---|
| 1 | notebook hợp lệ, biên dịch được, không sót output hay metadata Colab |
| 2 | thang bậc, trần token, ngân sách ngữ cảnh đồng nhất giữa các notebook |
| 3 | thang prompt là phép chèn thuần |
| 4 | khai báo không ai dùng, import thừa |
| 5 | tên gọi trong notebook mà không ô nào định nghĩa |
| 6 | chạy trọn notebook `07` với nấc dựng sẵn |
| 7 | chạy phần phân tích của notebook `06`, đọc ma trận thẳng từ notebook |
| 8 | chạy logic sáu notebook GPU bằng mô hình giả |

Phép kiểm 8 bỏ đúng ba loại ô — cài gói, kiểm gói, nạp mô hình — rồi tiêm
`model`/`tokenizer`/`generate` giả, trong đó `generate` trả về nhãn vàng của chính mẫu
đang hỏi. Mọi nhánh chấm điểm vì vậy chạy trên dữ liệu thật. Nó không thay cho việc chạy
GPU: chỉ chứng minh mã chạy tới cuối, không nói gì về chất lượng mô hình.
