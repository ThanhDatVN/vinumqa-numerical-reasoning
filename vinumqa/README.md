# Package `vinumqa`

Lõi dùng chung của lộ trình 5 nấc trong [`../notebooks/`](../notebooks/). Tách thành package
để bảy notebook không ai giữ một bản sao riêng của executor rồi lệch nhau.

## Bố cục

| Module | Nội dung |
|---|---|
| `dsl` | Executor DSL + chấm PA/EA — **phần quan trọng nhất về độ chính xác** |
| `data` | Nạp ViNumQA, tách nguồn FinQA-Vi / Vi Data, audit nhiễu nhãn |
| `prompts` | `PromptKit`, thang prompt lồng nhau + prompt self-eval |
| `_prompt_text` | Prompt tham chiếu, chép nguyên văn — **không sửa** |
| `pipeline` | `run_pipeline` dùng chung cho cả 5 nấc + `summarize` |
| `sft` | Dựng dữ liệu SFT bằng rejection sampling + cấu hình LoRA/TrainingArguments |
| `stats` | McNemar chính xác theo cặp + bootstrap CI |
| `io_utils` | Ghi artifact, chấm lại dự đoán đã lưu |
| `ace/` | `clusters`, `playbook` (Embedder/Retriever/QualityGate/Curator), `reflector`, `trainer` |

## Dùng nhanh (CPU, không cần model)

```python
import sys; sys.path.insert(0, "/đường/dẫn/tới/repo")
from vinumqa import dsl, data, pipeline, io_utils

dsl.execute_program("add(1, 0.15), divide(5310, #0)", [])   # → 4617.39...
dsl.check_pa("add(1, 2)", "add(2, 1)")                      # → (True, True)

train = data.load_split("data", "train")
data.print_audit(data.audit_gold(train, "train"))
```

Phần cần GPU (`run_pipeline`, `AceTrainer`) nhận `generate_fn` **tiêm từ ngoài vào**, nên
toàn bộ package chạy và test được trên CPU mà không cần model.

## Thang prompt lồng nhau

Ba mức dưới đây **cắt ra từ chính prompt hoàn chỉnh**, nên phần dùng chung giống nhau
từng ký tự và mỗi bước là một phép *chèn thuần*. Nhờ vậy hiệu số giữa hai nấc kề nhau đo
đúng phần vừa thêm, không lẫn chuyện viết lại câu chữ (có test canh điều này).

| Mức | Thêm gì so với mức trên | Ký tự | Nấc |
|---|---|--:|---|
| `basic` | mở đầu + danh sách phép toán + yêu cầu định dạng | 2 535 | 1 |
| `no_fewshot` | + ánh xạ từ khoá tiếng Việt → phép toán (14 mục) | 5 408 | *bậc giữa, không chạy* |
| `engineered` | + 2 ví dụ mẫu có khung | 5 924 | 2, 3, 4, 5 |
| `self_eval` | prompt bước 2: ngữ cảnh + lời giải bước 1, yêu cầu tự soát | — | 4, 5 |

`plain` (530 ký tự) vẫn còn trong mã nhưng **đã rời thang bậc**: nó thiếu cả quy tắc định
dạng nên 32 % mẫu sinh ra program không chạy được — một cái sàn hỏng, khiến hiệu số so với
nấc 2 chủ yếu là công dạy cú pháp DSL chứ không phải prompt engineering.

`engineered` và `self_eval` nằm trong `vinumqa/_prompt_text.py`, chép **nguyên văn** từ bản
đã dùng ở lần chạy tham chiếu (bản gốc còn lưu ở
`reference/original_notebooks/prompt_builder.py` để đối chiếu). Đừng sửa nội dung hai chuỗi
đó — sửa là mất hiệu lực so sánh với mốc tham chiếu.

## Khác biệt so với ACE gốc (FinQA, tiếng Anh)

Bản tham khảo: `reference/original_notebooks/02_ace_finqa_ENGLISH.ipynb`. Tám điểm đã chỉnh cho ViNumQA:

1. **Không có `const_*`** — ViNumQA dùng số literal thuần (0/4074 gold có `const_`).
   Executor trả `None` nếu gặp `const_5`, không âm thầm coi là 0.
2. **`table_*(nhãn, none)` đọc theo NHÃN HÀNG** (cột 0), không phải tên cột.
3. **Ô bảng tiếng Việt**: `(79)` → −79, `—`/`n/a` → bỏ qua, `26% ( 26 % )` → 0.26,
   `22.0x` → 22.0, `$ 1,234` → 1234.
4. **Toán hạng `20%`** = 0.2 (61 gold program trong train dùng dạng này).
5. **Embedding đa ngữ** (`multilingual-e5-base`) — bản gốc dùng `bge-base-en-v1.5`,
   xếp hạng bullet tiếng Việt gần như ngẫu nhiên.
6. **Reflector, playbook, quality gate bằng tiếng Việt**; backend mặc định là chính SLM.
7. **Cụm lỗi định nghĩa trong code**, không cần `phase0_clusters.json`.
8. **Vòng lặp ACE theo mini-batch** — generate/reflect/verify gom lô qua vLLM.

## Kiểm thử

```bash
python -m pytest tests/ -v      # 114 test, ~1 giây
```

Chạy trên **dữ liệu thật**. Các test quan trọng nhất:

* `TestAgainstRealData` — executor tái tạo `exe_ans` từ gold: train 99.83 %, valid 100 %,
  test 100 %; riêng `table_*` là 427/427, 94/94, 61/61.
* `TestGoldAudit` — nhiễu `multiply(#n,100)` chỉ ở train (100 mẫu), valid 1, test 0.
* `TestQualityGate` — mỗi luật loại bullet có test riêng.
* `test_sft.py::TestTrainingConfig` — batch hiệu dụng giữ nguyên 16 trên mọi GPU, nếu không
  learning rate 2e-4 lấy từ notebook cũ sẽ không còn hợp lệ.

## Lưu ý

Đừng chấm điểm bằng `reference/original_notebooks/pa_ea_calculator_BUGGY.py`: nó đọc `table_*` theo tên cột
DataFrame nên **mọi câu dùng `table_*` đều bị chấm sai** (0/427 trên train, 0/61 trên test).
Notebook `00_data_audit.ipynb` định lượng ảnh hưởng lên từng model tham chiếu.
