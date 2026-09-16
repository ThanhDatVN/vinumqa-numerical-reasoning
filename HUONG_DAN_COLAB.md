# Hướng dẫn chạy trên Colab

**Câu trả lời ngắn:** làm **một lần** ở máy (push lên GitHub), sau đó mỗi notebook chỉ là
*mở → chọn runtime → Run all*. Không phải upload gì, không phải sửa đường dẫn.

Cách hoạt động: **code và dữ liệu nằm trong repo GitHub**, cell đầu mỗi notebook tự
`git clone` về máy ảo Colab. **Chỉ kết quả ghi lên Drive**, nên mất session cũng không mất
kết quả. Sửa code dưới máy chỉ cần `git push` — lần chạy sau Colab tự lấy bản mới.

---

## Phần A — làm một lần, ở máy

### 1. Tạo repo trống trên GitHub

github.com → **New repository** → đặt tên (ví dụ `vinumqa-ladder`) → **Create**.
Đừng tick "Add a README" — repo phải trống.

Public hay private đều được. Private thì Colab cần token (xem [Trục trặc](#trục-trặc)),
nên nếu không có gì phải giữ kín thì để **public** cho đỡ phiền.

### 2. Điền URL rồi push

```bash
cd <thư mục dự án>
python tools/set_repo.py https://github.com/ThanhDatVN/vinumqa-numerical-reasoning --init
git push -u origin main
```

`set_repo.py` điền URL vào **cả 10 cell cấu hình** của 8 notebook và các liên kết trong tài
liệu, rồi `git init` + commit đầu + gắn `origin`. Push xong là **Colab không cần sửa gì
nữa**.

> Bỏ `--init` nếu muốn tự chạy các lệnh git. Đổi tên repo sau này thì chạy lại script với
> URL mới.

Repo khoảng **25 MB** (`data/train.json` 18 MB) — thoải mái so với hạn mức GitHub.

### 3. Kiểm tra trước khi đốt compute unit

```bash
python -m pytest tests/ -q      # 114 test, ~1,5 giây, không cần GPU
```

---

## Phần B — trên Colab

### 1. Mở notebook

Vào repo trên GitHub, mở file `.ipynb`, bấm huy hiệu **Mở trong Colab** ở đầu notebook.

Hoặc trong Colab: **File → Open notebook → GitHub** → dán URL repo → chọn notebook.

> Mở kiểu này thì notebook luôn là bản mới nhất trên GitHub. Sửa gì trong Colab sẽ **không**
> lưu ngược lại repo (trừ khi bấm *File → Save a copy to GitHub*) — không sao, vì kết quả
> nằm ở Drive chứ không nằm trong notebook.

### 2. Chọn runtime

**Runtime → Change runtime type**:

| Notebook | Runtime | Vì sao |
|---|---|---|
| `00`, `07` | **CPU** | không nạp model — chọn CPU thì tốn 0 compute unit |
| `01`–`06` | **A100 GPU** | |

### 3. Run all

**Runtime → Run all**. Bật **Background execution** để đóng tab cũng không chết session.

Cell đầu sẽ in:

```
[CODE] đang clone https://github.com/ThanhDatVN/vinumqa-numerical-reasoning.git … (~25 MB, khoảng 15 giây)
[CODE] → /content/<tên-repo>
[MÔI TRƯỜNG] Colab | vinumqa v2.0.0
[REPO]  /content/<tên-repo>
[RA]    /content/drive/MyDrive/vinumqa_runs
[DỮ LIỆU] /content/<tên-repo>/data
          train=2993 valid=584 test=497
```

kèm **bảng tiến độ** cho biết nấc nào đã chạy.

---

## Nếu không dùng `set_repo.py`

Sửa **một dòng, một lần**, ở code cell #2 của notebook `00`:

```python
GITHUB_REPO = "https://github.com/ThanhDatVN/vinumqa-numerical-reasoning"   # ← URL repo của bạn
OUTPUT_DIR  = "/content/drive/MyDrive/vinumqa_runs"               # không cần sửa
```

Chạy xong cell này, URL được **ghi nhớ** vào `MyDrive/.vinumqa_paths.json`; bảy notebook còn
lại tự đọc, kể cả sau khi restart runtime.

---

## Cần nhập gì cho từng notebook

| Notebook | Cần sửa | Ở đâu |
|---|---|---|
| `00_data_audit` | `GITHUB_REPO`, nếu chưa chạy `set_repo.py` | code cell #2 |
| `01_baseline_plain` | — | chỉ Run all |
| `02_prompt_engineering` | — | chỉ Run all |
| `03_sft_qwen3` | **2 lần restart** (xem dưới) | |
| `04_self_evaluation` | `USE_SFT_ADAPTER` | code cell #6 |
| `05_ace` | `USE_SFT_ADAPTER`, `RUN_RANDOM_CONTROL` | cell #6, #15 |
| `06_combination` | — | chỉ Run all |
| `07_final_report` | — | chỉ Run all |

### `04` và `05` chạy hai lần

| Lần | Cờ ở code cell #6 | So với nấc | Khi nào |
|---|---|---|---|
| 1 | `USE_SFT_ADAPTER = False` | model gốc | gói A |
| 2 | `USE_SFT_ADAPTER = True` | model đã SFT | gói C, sau khi chạy `03` |

Notebook tự đặt tên nấc (`04_selfeval_base` / `04_selfeval_sft`) nên hai lần chạy **không đè
lên nhau**.

Ở `05`, code cell #15 có `RUN_RANDOM_CONTROL = True` — giữ nguyên ở lần 1 (đây là nhóm đối
chứng quan trọng), đặt `False` ở lần 2 cho đỡ tốn.

### `03` — ba phần, hai lần restart

| Phần | Chạy cell | Việc | Sau đó |
|---|---|---|---|
| **A** | #1 → #10 | Sinh lời giải trên train, ghi JSONL | **Runtime → Restart session** |
| **B** | #11 → #16 | Huấn luyện LoRA | **Restart session** lần nữa |
| **C** | #17 → #26 | Nạp adapter, chấm test | xong |

Cách làm: Run all → dừng ở cell #10 → Restart → chạy từ cell #11 → dừng ở #16 → Restart →
chạy từ #17. Trong notebook có sẵn hai ô markdown **⚠ RESTART RUNTIME TẠI ĐÂY** làm mốc.

Lý do phải restart: engine vLLM giữ VRAM rất chặt, không nhả đủ cho huấn luyện. Dữ liệu SFT
đã ghi ra Drive nên phần B đọc lại được, không mất gì.

> Sau restart, cell cấu hình chạy lại **không clone lại** — thư mục `/content/...` vẫn còn
> nguyên sau khi restart runtime (chỉ mất khi *Disconnect and delete runtime*).

---

## Thứ tự chạy

```
00 → 01 → 02 → 04 → 05 → [KẾT QUẢ CHÍNH] → 03 → 04(lần 2) → 05(lần 2) → 06 → 07
```

`03` chạy sau `05` dù số nhỏ hơn — lý do ở [KE_HOACH_THU_NGHIEM.md](KE_HOACH_THU_NGHIEM.md).

> Từng bước một — sửa cell nào, chạy cell nào, phải thấy gì, sai thì làm gì — xem
> [CAC_BUOC_THUC_HIEN.md](CAC_BUOC_THUC_HIEN.md). Đó là tài liệu mở ra lúc ngồi chạy.

Notebook nào cũng in bảng tiến độ ở cell đầu, nên mở ra là biết đang ở đâu:

```
  nấc                           n       EA  PA_strict        chạy lúc
  01_plain              ✓     497   0.xxxx     0.xxxx   20260915_1030
  02_prompt_eng         ⊘       —        —          —       chưa chạy
```

Chạy nhầm thứ tự cũng không sao: notebook cần kết quả nấc trước sẽ báo
`⚠ chưa có 'xx' — chạy notebook tương ứng trước` rồi bỏ qua phần kiểm định, phần còn lại
vẫn chạy.

---

## Kết quả nằm ở đâu

Tất cả trong `MyDrive/vinumqa_runs/` — **không nằm trong repo**, nên `git push` không bao
giờ động tới kết quả:

```
vinumqa_runs/
├── stages/       <nấc>.jsonl       ← notebook sau đọc file này
│                 <nấc>_program.csv ← 6 cột, mở bằng Excel/Sheets được
│                 <nấc>_meta.json   ← cấu hình + metrics
├── logs/         output thô của model (để soi khi cần)
├── artifacts/    playbook, biểu đồ
├── sft_adapter_qwen3/   LoRA adapter sau khi chạy 03
└── bang_ket_qua_*.csv   bảng cuối cùng từ notebook 07
```

Không cần tải gì về — `07` đọc lại tất cả và xuất bảng để dán vào bài.

---

## Sửa code sau khi đã chạy

```bash
# ở máy
git add -A && git commit -m "sửa gì đó" && git push
```

Lần chạy Colab sau, cell đầu tự `git clone`/`git pull` bản mới. Không phải upload lại gì.

Hai lưu ý:

- Muốn lấy bản mới **ngay trong session đang chạy**: `Runtime → Restart session` rồi chạy
  lại cell đầu (nó sẽ `git pull`).
- Nếu sửa chính **notebook**, phải **mở lại** notebook từ GitHub — bản đang mở trong Colab
  là một bản sao.

---

## Trục trặc

| Hiện tượng | Xử lý |
|---|---|
| `Chưa điền GITHUB_REPO` | Chạy `tools/set_repo.py`, hoặc sửa tay dòng `GITHUB_REPO` ở cell đầu notebook `00`. |
| `git clone thất bại … not found` | Sai URL, hoặc repo private. Kiểm tra URL; repo private thì dùng `https://<token>@github.com/<tài-khoản>/<repo>` với token tạo ở *GitHub → Settings → Developer settings → Personal access tokens* (quyền `repo`). |
| `Không thấy dữ liệu tại …/data` | Repo thiếu `data/`. Kiểm tra `.gitignore` không chặn `data/`, rồi `git add data/ && git push`. |
| `compute capability 6.0 không chạy được vLLM` | Đang ở T4/P100. Runtime → Change runtime type → A100. |
| `[PROMPT] ⚠ đã bật cắt ngữ cảnh` | Đang ở T4. Đổi sang A100 hoặc L4 — kết quả có cắt không trộn chung được với kết quả không cắt. |
| Cell cài đặt chạy 6–12 phút | Bình thường, nó cài unsloth + vLLM. Chỉ chạy một lần mỗi session. |
| Cài đặt chạy >25 phút rồi vẫn thiếu `vllm`/`unsloth` | Phiên bản ghim không còn wheel cho image Colab mới. Mở terminal, chạy `pip install -U unsloth vllm`, xem lỗi thật, rồi ghim lại bộ mới vào ô cài đặt. |
| Colab bảo cần restart sau khi cài | Restart rồi Run all lại — cell cài đặt sẽ bỏ qua vì đã có sẵn. |
| OOM khi nạp model | Ở cell cấu hình GPU: `GPU_MEM_UTIL = 0.80`, `MAX_NUM_SEQS = 32`. |
| `Following weights were not initialized from checkpoint` | Model tải dở trong cache. Mở terminal: `rm -rf ~/.cache/huggingface/hub/models--unsloth--Qwen3-8B*` rồi đặt `TAI_CHAM_CHO_CHAC = True` ở đầu ô nạp model (`export` trong terminal KHÔNG tới được kernel notebook), Restart session, chạy lại từ cell #2. Kiểm `df -h /` trước — cần ≥ 20 GB trống. |
| Mất session giữa pha A của `05` | Có checkpoint mỗi vòng — mở lại, Run all, nó tự tiếp từ vòng dở. |
| Muốn chạy thử nhanh trước | Thêm một cell ngay sau cell cấu hình: `test_all = test_all[:40]`. Nhớ bỏ đi khi chạy thật. |

---

## Vẫn muốn để tất cả trên Drive?

Vẫn được — không cần GitHub. Upload cả thư mục lên `MyDrive/`, rồi ở cell cấu hình điền:

```python
REPO_DIR = "/content/drive/MyDrive/<tên-thư-mục-vừa-upload>"
```

Điền `REPO_DIR` thì bước clone bị bỏ qua và `GITHUB_REPO` không còn tác dụng. Đổi lại: mỗi
lần sửa code là phải upload lại 25 MB.
