# Numerical Reasoning QA cho báo cáo tài chính tiếng Việt

Lộ trình thí nghiệm 5 nấc trên **ViNumQA** với **Qwen3-8B (4-bit, vLLM)**, chạy được trên
Google Colab. Mỗi nấc thêm đúng một kỹ thuật và đo phần đóng góp riêng của nó.

Tư liệu tham chiếu — dự đoán của 5 model trên cùng tập test, notebook gốc, prompt gốc —
nằm trong [`reference/`](reference/).

```
vinumqa/      lõi dùng chung: executor, prompt, pipeline, SFT, thống kê, ACE
notebooks/    00 → 07, mỗi nấc một notebook chạy độc lập được
tests/        114 test CPU trên dữ liệu thật (~2 giây)
data/         ViNumQA: 2.993 train / 584 valid / 497 test
reference/    tư liệu đối chiếu, không phải code chạy
tools/        set_repo.py — điền URL repo vào notebook, chạy một lần
```

---

## Lộ trình

| Nấc | Kỹ thuật | Notebook | GPU | Mở |
|:---:|---|---|:---:|:---:|
| — | Audit dữ liệu & chốt thước đo | [`00_data_audit`](notebooks/00_data_audit.ipynb) | ❌ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/vinumqa-numerical-reasoning/blob/main/notebooks/00_data_audit.ipynb) |
| 1 | Inference thông thường | [`01_baseline_plain`](notebooks/01_baseline_plain.ipynb) | ✅ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/vinumqa-numerical-reasoning/blob/main/notebooks/01_baseline_plain.ipynb) |
| 2 | + Prompt engineering | [`02_prompt_engineering`](notebooks/02_prompt_engineering.ipynb) | ✅ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/vinumqa-numerical-reasoning/blob/main/notebooks/02_prompt_engineering.ipynb) |
| 3 | + SFT trên Qwen3-8B | [`03_sft_qwen3`](notebooks/03_sft_qwen3.ipynb) | ✅ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/vinumqa-numerical-reasoning/blob/main/notebooks/03_sft_qwen3.ipynb) |
| 4 | + Self-evaluation 2 bước | [`04_self_evaluation`](notebooks/04_self_evaluation.ipynb) | ✅ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/vinumqa-numerical-reasoning/blob/main/notebooks/04_self_evaluation.ipynb) |
| 5 | + ACE (playbook + truy hồi) | [`05_ace`](notebooks/05_ace.ipynb) | ✅ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/vinumqa-numerical-reasoning/blob/main/notebooks/05_ace.ipynb) |
| — | **Ma trận tổ hợp** — bổ sung hay trùng nhau? | [`06_combination`](notebooks/06_combination.ipynb) | ✅ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/vinumqa-numerical-reasoning/blob/main/notebooks/06_combination.ipynb) |
| — | Tổng hợp & kiểm định | [`07_final_report`](notebooks/07_final_report.ipynb) | ❌ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/vinumqa-numerical-reasoning/blob/main/notebooks/07_final_report.ipynb) |

Mỗi notebook **chạy độc lập được**, tự đọc kết quả nấc trước từ `OUTPUT_DIR/stages/` và so
sánh bằng **McNemar theo cặp** (cùng 497 mẫu test, nên đây là dữ liệu cặp).

### Vì sao cần notebook 06

Năm nấc đầu là một **chuỗi** — mỗi nấc chồng lên nấc trước. Chuỗi không trả lời được câu hỏi
đáng giá nhất: **SFT và ACE đều học từ tập train** (một cái vào trọng số, một cái vào ngữ
cảnh) — chúng bổ sung hay đang học cùng một thứ? Tương tự, self-eval và ACE đều là cơ chế
sửa lỗi lúc suy luận nên có thể chồng lấn.

Notebook 06 chạy **ma trận 2×2×2** (SFT × self-eval × ACE) rồi tính **tác động chính** và
**tương tác** của từng kỹ thuật. Ô nào đã chạy ở nấc trước thì đọc lại từ đĩa — chỉ sinh hai
ô mới (`prompt+ACE` và `SFT+ACE`, tức ACE **không kèm** self-eval), nên tốn thêm ~2 giờ chứ
không phải chạy lại cả 8 ô.

| Ô | SFT | Self-eval | ACE |
|---|:---:|:---:|:---:|
| `E` | ❌ | ❌ | ❌ |
| `E+A` | ❌ | ❌ | ✅ |
| `E+S` | ❌ | ✅ | ❌ |
| `E+S+A` | ❌ | ✅ | ✅ |
| `F` | ✅ | ❌ | ❌ |
| `F+A` | ✅ | ❌ | ✅ |
| `F+S` | ✅ | ✅ | ❌ |
| `F+S+A` | ✅ | ✅ | ✅ |

Một kết quả có thể rất đáng giá: nếu **`E+A` ≥ `E+S`** thì ACE **thay được** self-eval với
nửa chi phí (một lượt sinh thay vì hai) — luận điểm mạnh cho bối cảnh tài nguyên hạn chế.

### Bốn kỹ thuật khác nhau ở đâu

| | Học từ train? | Đổi trọng số? | Cần API ngoài? | Lượt sinh/mẫu |
|---|:---:|:---:|:---:|:---:|
| Prompt engineering (nấc 2) | ❌ | ❌ | ❌ | 1 |
| SFT (nấc 3) | ✅ | ✅ | ❌ | 1 |
| Self-eval (nấc 4) | ❌ | ❌ | ❌ | 2 |
| ACE (nấc 5) | ✅ | ❌ | ❌ | 2 |

**Không nấc nào cần API ngoài**, nên toàn bộ lộ trình nằm trong thiết lập
*constrained-resource* của dự án.

---

## Bắt đầu

Code **và** dữ liệu nằm trong repo này; notebook tự `git clone` về máy ảo Colab. Chỉ **kết
quả** ghi lên Drive, nên mất session không mất kết quả, và sửa code chỉ cần `git push`.

**Một lần, ở máy** — tạo repo trống trên GitHub rồi:

```bash
python -m pytest tests/ -q                                       # 114 test, ~1,5 giây
python tools/set_repo.py https://github.com/ThanhDatVN/vinumqa-numerical-reasoning --init
git push -u origin main
```

`set_repo.py` điền URL repo vào cả 10 cell cấu hình của 8 notebook. Push xong thì **trên
Colab không phải sửa dòng nào** — chỉ mở notebook (huy hiệu ở bảng trên), chọn runtime,
**Run all**.

Không muốn dùng script cũng được: sửa tay **một dòng, một lần** ở cell đầu notebook `00`,
URL sẽ được ghi nhớ cho bảy notebook còn lại.

```python
GITHUB_REPO = "https://github.com/ThanhDatVN/vinumqa-numerical-reasoning"   # ← URL repo của bạn
OUTPUT_DIR  = "/content/drive/MyDrive/vinumqa_runs"               # không cần sửa
```

Chạy lần lượt 00 → 07. Cell đầu của notebook nào cũng in **bảng tiến độ** cho biết nấc nào
đã có kết quả:

```
  nấc                           n       EA  PA_strict        chạy lúc
  01_plain              ✓     497   0.xxxx     0.xxxx   20260915_1030
  02_prompt_eng         ⊘       —        —          —       chưa chạy
```

**Runtime:** GPU cho notebook 01–06; CPU cho notebook 00 và 07 (đỡ tốn compute unit).
**A100** nhanh hơn L4 ~2.5× với chi phí compute unit gần như hoà (prompt dài nên prefill
chiếm ưu thế) — có A100 thì dùng. L4 cho kết quả **so sánh trực tiếp được** với A100 vì cả
hai đều không phải cắt ngữ cảnh. Riêng T4 thì có, nên đừng trộn kết quả từ T4 vào bảng.

📖 **[HUONG_DAN_COLAB.md](HUONG_DAN_COLAB.md)** — đẩy lên GitHub, mở trên Colab, sửa gì,
notebook nào cần nhập gì.
✅ **[CAC_BUOC_THUC_HIEN.md](CAC_BUOC_THUC_HIEN.md)** — bảng thao tác 13 bước: sửa cell nào,
chạy cell nào, phải thấy gì, sai thì làm gì. **Mở cái này lúc ngồi chạy.**
📋 **[KE_HOACH_THU_NGHIEM.md](KE_HOACH_THU_NGHIEM.md)** — chia buổi, ngân sách compute unit,
cổng kiểm tra sau mỗi nấc, thứ tự cắt nếu thiếu ngân sách.

---

## Thư mục làm việc

Mọi notebook ghi vào cùng một chỗ theo cùng quy ước:

```
OUTPUT_DIR/
├── stages/       kết quả từng nấc — <nấc>.jsonl (đầy đủ)
│                                     <nấc>_program.csv (6 cột, mở Excel được)
│                                     <nấc>_meta.json (cấu hình + metrics)
├── logs/         output thô của model
└── artifacts/    playbook, LoRA adapter, biểu đồ
```

Ba hàm dùng chung, có sẵn ngay sau cell đầu: `save_stage()`, `load_stage()`, `stage_status()`.

---

## Package `vinumqa/`

| Module | Nội dung |
|---|---|
| `dsl` | Executor DSL + chấm PA/EA. **Phần quan trọng nhất về độ chính xác.** |
| `data` | Nạp ViNumQA, tách nguồn FinQA-Vi / Vi Data, audit nhiễu nhãn |
| `prompts` | Ba mức prompt: `plain` / `engineered` / `self_eval` |
| `pipeline` | `run_pipeline` dùng chung cho cả 5 nấc |
| `sft` | Dựng dữ liệu SFT bằng rejection sampling + cấu hình LoRA |
| `stats` | McNemar theo cặp + bootstrap CI |
| `io_utils` | Ghi artifact, chấm lại dự đoán đã lưu |
| `ace/` | Playbook, truy hồi Tier-1/2, quality gate, Reflector, vòng lặp ACE |

Phần cần GPU nhận `generate_fn` **tiêm từ ngoài**, nên toàn bộ package test được trên CPU.

Chi tiết: [`vinumqa/README.md`](vinumqa/README.md).

---

## Hai quyết định thiết kế đáng lưu ý

**1. Executor phải tái tạo được nhãn vàng.** `table_*(nhãn, none)` trong ViNumQA đọc theo
**nhãn hàng** (cột đầu), không phải tên cột — đã kiểm chứng 456/456 tham số trong train.
Executor cũ (`reference/original_notebooks/pa_ea_calculator_BUGGY.py`) đọc theo tên
cột nên chấm sai **mọi** câu dùng
`table_*` (0/427 trên train). Notebook 00 định lượng ảnh hưởng.

Tỉ lệ tái tạo `exe_ans` của executor hiện tại: train 99.83 %, valid 100 %, test 100 %.

**2. SFT không fine-tune thẳng trên gold.** Cách đó đã được thử và overfit ngay (val loss tăng từ
step 50). Nấc 3 dùng **rejection sampling**: chạy model, giữ mẫu nó làm đúng, dùng chính lời
giải đó — *đã có chuỗi suy luận* — làm đích huấn luyện. Không cần API ngoài, và mẫu nhãn
nhiễu phần lớn tự rơi ra.

---

## Dữ liệu

ViNumQA (VLSP 2025): 2.993 train / 584 valid / 497 test. Mỗi mẫu gồm `pre_text`, `table`,
`post_text`, và `qa` với `question`, `program`, `exe_ans`.

Hai nguồn, tách được từ `id`: **FinQA-Vi** (dịch từ FinQA) và **Vi Data** (báo cáo doanh
nghiệp Việt Nam 2020–2025). Vi Data dùng `table_*` dày hơn hẳn.

**Nhiễu nhãn đo được:** `multiply(#n,100)` xuất hiện 100 lần ở train, 1 ở valid, **0 ở test**.
Prompt của dự án cấm dạng này, nên trên các mẫu đó model làm đúng vẫn bị chấm sai. Nấc 3 và
nấc 5 đều lọc chúng bằng `data.is_noisy_gold()`.

---

## Môi trường

Giữ đúng của notebook gốc
(`reference/original_notebooks/inference_with_difference_models.ipynb`):
unsloth + vLLM, `transformers==4.56.2`, `trl==0.22.2`, `unsloth/Qwen3-8B` với
`load_in_4bit=True`, `fast_inference=True`, `temperature=0.1`, `max_tokens=3000`.

Khối cài đặt được chép nguyên vào mỗi notebook.

---

## Dự án này độc lập

Không phụ thuộc gì ngoài `data/` và `vinumqa/`. Prompt tham chiếu được chép **nguyên
văn** vào `vinumqa/_prompt_text.py` (đã kiểm chứng khớp từng ký tự với bản gốc, và
`table_to_str` cho cùng kết quả trên toàn bộ 497 mẫu test), nên không còn import chéo sang
thư mục nào khác.

`reference/` chỉ dùng cho một việc: notebook `00` và `07` đọc `baseline_results/` để đối
chiếu với mốc tham chiếu. Không có `reference/` thì hai notebook đó vẫn chạy, chỉ bỏ phần đối chiếu
(và 2 test tự skip).

## Đã kiểm chứng

| Hạng mục | Kết quả |
|---|---|
| Test CPU trên dữ liệu thật | 114/114 pass, ~2 giây |
| Executor tái tạo `exe_ans` từ gold | train 99.83 %, valid 100 %, test 100 % |
| Riêng câu dùng `table_*` | 427/427, 94/94, 61/61 |
| `PA_loose` tái lập mốc tham chiếu | Qwen3-8B 59.56 % ✓, Mistral-7B 38.63 % ✓ |
| 8 notebook | compile sạch, nbformat hợp lệ, outputs rỗng, chạy hết cell |

## Giấy phép

MIT — xem [LICENSE](LICENSE).
