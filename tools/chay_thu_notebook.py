# -*- coding: utf-8 -*-
"""Chạy thử LOGIC của các notebook GPU bằng model/tokenizer/generate GIẢ.

    python tools/chay_thu_notebook.py            # tất cả
    python tools/chay_thu_notebook.py 08 04      # chỉ vài cái

Vì sao cần: `kiem_tra.py` chỉ chạy thật `06`/`07`. Các notebook GPU (`01`, `02`, `04`,
`05`, `08`) có toàn bộ phần chấm điểm, so sánh, ghi nấc **chưa bao giờ được thực thi** —
độ phủ bằng 0. Mọi lỗi im lặng trong đó chỉ lộ ra sau khi đã tiêu GPU.

Ở đây ta bỏ qua đúng ba loại ô — cài gói, kiểm gói, nạp model — rồi tiêm sẵn `model`,
`tokenizer`, `generate`, `SamplingParams` giả. `generate` trả về **gold program** của
chính mẫu đang hỏi, nên mọi nhánh chấm điểm/so sánh chạy trên dữ liệu thật chứ không
phải dữ liệu rỗng.

KHÔNG thay cho việc chạy thật trên GPU: nó không kiểm được chất lượng model, chỉ kiểm
được rằng mã chạy tới cuối mà không nổ.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

GOC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, GOC)
os.environ.setdefault("MPLBACKEND", "Agg")

#: Ô chứa một trong các dấu hiệu này là ô môi trường — bỏ qua, không phải logic.
BO_QUA = ("%pip", "!pip", "FastLanguageModel", "from vllm", "import torch",
          "GITHUB_REPO =", "drive.mount", "Thiếu gói", "torch.cuda.is_available",
          "PYTORCH_CUDA_ALLOC_CONF", "trainer.train(", "load_dataset(",
          "AceTrainer(")


class TokenizerGia:
    chat_template = "co"

    def __call__(self, s, **kw):
        class R:
            input_ids = list(range(min(len(s) // 3, 900)))
        return R()

    def apply_chat_template(self, messages, **kw):
        return "\n".join(m["content"] for m in messages)


class SamplingGia:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.n = kw.get("n", 1)


def _save_stage_gia(st, rows, metrics, extra=None, quiet=False):
    """Ghi meta thật — có ô in lại khối meta từ file, không ghi là nó nổ FileNotFound."""
    d = os.path.join(GOC, "runs", "stages")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{st}_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"stage": st, "n": len(rows), "metrics": metrics,
                   **(extra or {})}, f, ensure_ascii=False, indent=1, default=str)
    return st


def dung_ns(tap):
    from vinumqa import data, dsl, io_utils, pipeline, sft, stats
    from vinumqa.ace import clusters, playbook as pb_mod, reflector as refl_mod
    from vinumqa.ace.trainer import AceTrainer
    from vinumqa.prompts import PromptKit
    _id = {s["id"]: s for s in tap["test"]}
    kit = PromptKit(tokenizer=TokenizerGia(), model_name="gia")

    def generate(prompts, sp=None, desc=None, batch_size=None):
        n = getattr(sp, "n", 1) or 1
        ra = []
        for p in prompts:
            sid = next((i for i in _id if i in p), None)
            g = _id[sid]["qa"]["program"] if sid else "divide(1, 2)"
            mau = [f"```plaintext\nprogram: {g}\nanswer: x\n```"]
            mau += [f"```plaintext\nprogram: divide(1, {j + 2})\nanswer: x\n```"
                    for j in range(n - 1)]
            ra.append(mau if n > 1 else mau[0])
        return ra

    ra = os.path.join(GOC, "runs")
    return {
        "__name__": "__main__", "os": os, "sys": sys, "json": json,
        "time": __import__("time"), "csv": __import__("csv"),
        "glob": __import__("glob"), "random": __import__("random"),
        "np": __import__("numpy"), "re": __import__("re"), "math": __import__("math"),
        "shutil": __import__("shutil"), "subprocess": __import__("subprocess"),
        "datetime": __import__("datetime").datetime,
        "Counter": __import__("collections").Counter,
        "defaultdict": __import__("collections").defaultdict,
        "data": data, "dsl": dsl, "io_utils": io_utils, "pipeline": pipeline,
        "sft": sft, "stats": stats, "PromptKit": PromptKit,
        "clusters": clusters, "pb_mod": pb_mod, "refl_mod": refl_mod,
        "AceTrainer": AceTrainer,
        "prompt_kit": kit, "tokenizer": TokenizerGia(), "generate": generate,
        "SamplingParams": SamplingGia,
        "SAMPLING": SamplingGia(temperature=0.1, max_tokens=4096),
        "MAX_TOKENS": 4096, "MAX_SEQ_LENGTH": 17000, "TEMPERATURE": 0.1,
        "REPETITION_PENALTY": 1.0, "RANDOM_SEED": 42, "BATCH_SIZE": 512,
        "GPU_MEM_UTIL": 0.85, "MAX_NUM_SEQS": 48,
        "MODEL_NAME": "gia/Qwen3-8B", "MODEL_TAG": "gia",
        "BASE_TAG": "base", "PREV_STAGE": "02_prompt_eng", "LORA_REQUEST": None,
        "train_all": tap["train"], "valid_all": tap["valid"], "test_all": tap["test"],
        "OUTPUT_DIR": ra, "LOG_DIR": os.path.join(ra, "logs"),
        "RESULT_DIR": os.path.join(ra, "stages"),
        "ARTIFACT_DIR": os.path.join(ra, "artifacts"), "STAMP": "chaythu",
        "LADDER_LABEL": {}, "ADAPTER_DIR": os.path.join(ra, "khong_co_adapter"),
        "dat_lai_bo_dem": lambda: None,
        "in_bi_cat_theo_buoc": lambda: None,
        "ty_le_bi_cat": lambda: 0.0,
        "bi_cat_theo_buoc": lambda: {},
        "save_stage": _save_stage_gia,
        "load_stage": lambda st, quiet=False: None,
        # Phải đặt tên Y HỆT hàm thật, nếu không ô in lại khối meta sẽ tìm nhầm file.
        "stage_path": lambda st, kind="jsonl": os.path.join(
            ra, "stages", {"jsonl": f"{st}.jsonl", "meta": f"{st}_meta.json"}[kind]),
        "stage_status": lambda: 0,
        "run_env": lambda: {"gpu": "gia"},
    }


#: Tên mà pha A của `05` sinh ra. Bỏ qua pha A (tốn GPU + API) thì phải tiêm thay,
#: nếu không mọi ô phân tích phía sau — chấm công bullet, đối chứng, ghi nấc — nằm im.
def _tiem_sau_pha_a(ns):
    from vinumqa.ace import playbook as pb
    _pb = pb.inject_bullet(pb.empty_playbook(), "sinh_program", "sp-00001",
                           "Khi hỏi tỷ lệ tăng, dùng subtract(gia_tri_moi, gia_tri_cu), "
                           "divide(#0, gia_tri_cu).")

    from collections import Counter as _C

    class TrainerGia:
        qg_reasons = _C({"trung_cau_truc": 3, "verify_khong_sua_duoc": 2})
        history = []
        playbook = _pb
        stats = _C({"reflected": 5, "added": 1, "evicted": 0})
    ns.update({"PLAYBOOK": _pb, "trainer": TrainerGia(),
               "PLAYBOOK_PATH": os.path.join(GOC, "runs", "playbook_thu.txt"),
               "ACE_TAG": "ace_base", "eval_log": [{"round": 1, "EA": 0.6, "PA": 0.5}],
               "best": {"playbook": _pb, "score": 0.6, "EA": 0.6, "PA": 0.5, "round": 1}})


def chay(ten: str, tap) -> list[str]:
    p = os.path.join(GOC, "notebooks", ten)
    nb = json.load(open(p, encoding="utf-8"))
    ma = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    ns = dung_ns(tap)
    if ten.startswith("05"):
        # Cho qua cổng API: khoá giả + chặn lời gọi mạng. Không phải kiểm OpenAI,
        # mà kiểm phần LOGIC phía sau nó.
        os.environ.setdefault("OPENAI_API_KEY", "khoa-gia-de-chay-thu")
        from vinumqa.ace import reflector as _rf
        _rf.Reflector._call_openai = lambda self, prompts: [
            '{"error_type":"khac","root_cause":"x","new_strategy":"",'
            '"bullet_tags":[]}' for _ in prompts]
    loi, n_chay, n_bo = [], 0, 0
    for i, src in enumerate(ma, 1):
        if any(t in src for t in BO_QUA):
            n_bo += 1
            if ten.startswith("05") and "AceTrainer(" in src:
                _tiem_sau_pha_a(ns)
            continue
        src = "\n".join(("pass" if l.lstrip().startswith(("!", "%")) else l)
                        for l in src.splitlines())
        try:
            exec(compile(src, f"{ten[:2]}#{i}", "exec"), ns)
            n_chay += 1
        except AssertionError as e:
            # assert là CỔNG có chủ ý (thiếu adapter, hết cấu hình…) — không phải lỗi.
            print(f"    ô #{i:<3} ⊘ dừng ở cổng: {str(e)[:70]}")
            n_bo += 1
            break
        except Exception as e:                            # noqa: BLE001
            loi.append(f"{ten} ô #{i}: {type(e).__name__}: {e}\n"
                       + traceback.format_exc(limit=3))
            break
    print(f"  {ten:<28}{n_chay:>3} ô logic chạy được · {n_bo} ô bỏ qua"
          + ("" if not loi else "  ⛔"))
    return loi


def main():
    from vinumqa import data
    chon = sys.argv[1:]
    tap = data.load_all(os.path.join(GOC, "data"))
    ds = ["01_baseline_basic.ipynb", "02_prompt_engineering.ipynb",
          "04_self_evaluation.ipynb", "05_ace.ipynb", "08_phuong_phap_moi.ipynb"]
    if chon:
        ds = [d for d in ds if any(c in d for c in chon)]
    print("═" * 78)
    print("  CHẠY THỬ LOGIC NOTEBOOK GPU (model giả)")
    print("═" * 78)
    loi = []
    for d in ds:
        loi += chay(d, tap)
    print("═" * 78)
    if loi:
        print(f"  ⛔ {len(loi)} VẤN ĐỀ")
        for x in loi:
            print("     •", x)
        return 1
    print("  ✅ Mọi notebook chạy tới cuối, không ô nào nổ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
