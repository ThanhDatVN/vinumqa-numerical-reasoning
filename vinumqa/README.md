# Package `vinumqa`

Lõi dùng chung của mười notebook trong [`../notebooks/`](../notebooks/). Tách thành
package để không notebook nào giữ một bản sao riêng của executor rồi lệch nhau — độ
chính xác báo cáo phụ thuộc trực tiếp vào việc mọi nấc chấm điểm bằng cùng một đoạn mã.

## Bố cục

| module | nội dung |
| --- | --- |
| `dsl` | phân tích cú pháp + thực thi DSL, chấm EA/PA — **phần quyết định con số** |
| `data` | nạp ViNumQA, lấy mẫu phân tầng, kiểm chất lượng nhãn vàng |
| `prompts` | `PromptKit`: thang prompt lồng nhau + prompt cho self-eval, sửa lỗi, chọn |
| `_prompt_text` | văn bản prompt gốc, chép nguyên văn — **không sửa** |
| `pipeline` | `run_pipeline` dùng chung cho mọi nấc: sinh, bỏ phiếu, sửa lỗi, chọn, chấm |
| `fewshot` | kho ví dụ truy hồi theo BM25, tự chứa, không cần embedding ngoài |
| `sft` | dựng dữ liệu SFT bằng rejection sampling + cấu hình LoRA/TrainingArguments |
| `stats` | McNemar chính xác theo cặp + khoảng tin cậy bootstrap |
| `io_utils` | đọc/ghi kết quả từng nấc |
| `ace/` | `clusters`, `playbook` (Embedder/Retriever/QualityGate/Curator), `reflector`, `trainer` |

## Nguyên tắc thiết kế

**Mọi phần cần GPU nhận `generate_fn` tiêm từ ngoài vào.** `run_pipeline` và `AceTrainer`
không biết mô hình được nạp bằng gì; notebook truyền vào một hàm nhận danh sách prompt
và trả danh sách lời giải. Nhờ vậy toàn bộ package kiểm thử được trên CPU bằng mô hình
giả, và `tools/chay_thu_notebook.py` chạy được logic của cả sáu notebook GPU.

**Executor fail-closed.** `execute_program` trả `None` khi gặp chương trình không hợp lệ,
`normalize_program_strict` trả `""`, `_day_phep` trả danh sách rỗng. Một nhãn vàng hỏng
được phép làm sai một con số, không được phép làm hỏng cả lượt chạy sau khi GPU đã tiêu.

**Mọi cơ chế chốt đáp án đều chỉ hỏi executor, không hỏi đáp án vàng.** Bỏ phiếu theo
giá trị thực thi, lượt sửa khi chương trình bị từ chối, cổng bước hai — cả ba đều dùng
được lúc suy luận thật.

**Lưu đủ để tính lại trên CPU.** Nấc dùng K mẫu ghi cả `cac_program` và `cac_gia_tri`,
nên đường cong theo K, các luật bỏ phiếu thay thế, và nấc bộ chọn đều dựng lại được từ
một lần chạy GPU duy nhất.

## Dùng nhanh (CPU, không cần mô hình)

```python
from vinumqa import data, dsl, pipeline

tap = data.load_all("data")
s = tap["test"][0]                      # nhãn vàng: add(30, 1) → 31.0

# thực thi một chương trình DSL trên bảng của mẫu
gt = dsl.execute_program(s["qa"]["program"], s["table"])        # 31.0

# chấm điểm: EA so giá trị thực thi, PA so chương trình sau chuẩn hoá
dsl.check_ea(gt, s["qa"]["exe_ans"])                            # True
dsl.check_pa("add(1, 30)", s["qa"]["program"])                  # (True, True) — add giao hoán
dsl.check_pa("subtract(30, 1)", s["qa"]["program"])             # (False, False)
```

Chạy `run_pipeline` với mô hình giả:

```python
def generate(prompts, sp=None, desc=None, batch_size=None):
    return ["```plaintext\nprogram: add(1, 2)\nanswer: 3\n```"] * len(prompts)

rows = pipeline.run_pipeline(tap["test"][:5], prompt_kit, generate)
pipeline.print_summary(pipeline.summarize(rows, "thử"))
```

## Kiểm thử

```bash
python -m pytest tests/ -q          # 245 test, ~6 giây
python tools/kiem_tra.py --day-du   # 8 phép kiểm toàn dự án
```
