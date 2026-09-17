# reference — tư liệu tham chiếu

Thư mục này **không phải code chạy**. Nó giữ lại những gì cần để đối chiếu với mốc tham
chiếu và để truy nguyên xuất xứ của các lựa chọn thiết kế.

Dự án chạy được mà không cần thư mục này, **trừ** `baseline_results/` — notebook `00` và
`07` đọc file ở đây để đối chiếu với con số tham chiếu.

## `baseline_results/`

Dự đoán của 5 model trên 497 mẫu test — mốc tham chiếu của cả lộ trình.
**Không tái tạo được** nếu mất, vì cần chạy lại cả 5 model.

| File | Model | PA tham chiếu | EA tham chiếu |
|---|---|---|---|
| `Qwen3-8B_program.csv` | Qwen3-8B + self-eval | 59.56 % | 64.19 % |
| `Phi4_program.csv` | Phi-4 + self-eval | 54.69 % | 59.31 % |
| `Phi4_finetuned_program.csv` | Phi-4 SFT + self-eval | 52.52 % | 56.54 % |
| `mistral-7b-instruct-v0.3_program.csv` | Mistral-7B | 38.63 % | 41.65 % |
| `Llama-3.1-8B_program.csv` | Llama-3.1-8B | 0.20 % | 0.40 % |

Schema 6 cột: `id, question, gold_program, gold_answer, program_step1, program_step2`.

> ⚠ Cột **EA tham chiếu** được tính bằng executor cũ, vốn đọc `table_*` theo tên cột nên bỏ sót
> mọi câu dùng `table_*`. Chấm lại bằng `vinumqa.dsl` cho Qwen3-8B ra **EA 65.19 %**.
> Notebook `00_data_audit.ipynb` định lượng chênh lệch này cho từng model.

## `original_notebooks/`

Notebook và mã của lần chạy trước, **đã xoá output** cho nhẹ. Giữ làm hồ sơ xuất xứ:

| File | Vai trò |
|---|---|
| `inference_with_difference_models.ipynb` | Nguồn của mọi tham số inference đang dùng: `load_in_4bit=True`, `fast_inference=True`, `temperature=0.1`, khối cài đặt unsloth/vLLM. `max_tokens` thì **không** giữ: 3000 → 8192 vì ở mức cũ 5–10 % mẫu bị cắt giữa lúc suy nghĩ |
| `finetune_phi4.ipynb` | Nguồn của cấu hình LoRA đang dùng: r=16, alpha=32, `paged_adamw_8bit`, cosine, early stopping |
| `prompt_builder.py` | Bản gốc của hai system prompt. `vinumqa/_prompt_text.py` chép nguyên văn từ đây — đã kiểm chứng khớp từng ký tự |
| `02_ace_finqa_ENGLISH.ipynb` | Bản ACE gốc cho FinQA (tiếng Anh). Tám điểm phải sửa để dùng cho ViNumQA được ghi trong `vinumqa/README.md` |
| `pa_ea_calculator_BUGGY.py` | Executor cũ. **Đừng dùng** — đọc `table_*` theo tên cột DataFrame nên chấm sai mọi câu dùng `table_*` (0/427 trên train). Không mã nào nạp file này; giữ làm hồ sơ cho phần đính chính EA ở notebook `00` |

## Những gì đã bỏ

Phần còn lại của lần chạy trước đã xoá vì pipeline mới không dùng tới, và tái tạo được:
bản port ACE đầu tiên (`ace_vinumqa/`, đã thay bằng `vinumqa/`), bộ test của nó, demo
Streamlit, notebook phân tích dữ liệu, tài liệu ACE gốc.

Riêng dữ liệu SFT do Gemini sửa (`data_finetune/`) bỏ hẳn có chủ ý: dùng nó là rơi sang
thiết lập *unconstrained*. Nấc 3 nay dùng rejection sampling nên không cần API ngoài. Nếu
cần lại, chúng nằm trong bản lưu của lần chạy trước.
