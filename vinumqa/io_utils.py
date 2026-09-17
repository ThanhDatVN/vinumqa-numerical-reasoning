# -*- coding: utf-8 -*-
"""Đọc/ghi artifact của một nấc, và chấm lại các dự đoán đã lưu sẵn.

Hai việc:

* **Ghi** kết quả từng mẫu ra ``*.jsonl`` — đây là thứ ``07`` đọc lại để phân tích
  mà không cần GPU, nên nó là artifact quan trọng nhất của mỗi nấc.
* **Đọc lại** các dự đoán mốc tham chiếu trong ``reference/baseline_results/*_program.csv``
  rồi chấm bằng executor đã kiểm chứng — nhờ vậy so được với số cũ mà không chạy lại GPU.
"""
from __future__ import annotations

import csv
import json
import os

from .dsl import (check_ea, check_pa, classify_outcome, execute_program,
                  ly_do_khong_chay, n_ops)

__all__ = ["save_full_jsonl", "save_raw_jsonl", "load_predictions",
           "score_saved_predictions", "BASELINE_RESULTS"]

#: Bảng kết quả tham chiếu của 5 model (post Step-2), dùng để đối chiếu.
BASELINE_RESULTS = {
    "Llama-3.1-8B":             {"PA": 0.20,  "EA": 0.40},
    "mistral-7b-instruct-v0.3": {"PA": 38.63, "EA": 41.65},
    "Phi4":                     {"PA": 54.69, "EA": 59.31},
    "Qwen3-8B":                 {"PA": 59.56, "EA": 64.19},
    "Phi4_finetuned":           {"PA": 52.52, "EA": 56.54},
}


def save_full_jsonl(rows, path: str, drop=("_sample", "raw_step1", "raw_step2",
                                           "bullets_text")) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k not in drop},
                               ensure_ascii=False, default=str) + "\n")
    return path


def save_raw_jsonl(rows, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r.get("id", ""),
                                "raw_step1": r.get("raw_step1", ""),
                                "raw_step2": r.get("raw_step2", "")},
                               ensure_ascii=False) + "\n")
    return path


# ─────────────────── chấm lại dự đoán đã lưu ───────────────────

def load_predictions(csv_path: str) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_saved_predictions(csv_path: str, samples: list[dict], label: str = "",
                            prefer: str = "step2") -> list[dict]:
    """Chấm lại một file ``*_program.csv`` bằng executor đã kiểm chứng.

    Trả về list row **cùng schema với** :func:`ace_vinumqa.pipeline.run_pipeline`,
    nên dùng thẳng được với ``summarize`` và ``stats.compare_pair``.

    ``prefer='step2'`` theo đúng logic của repo: lấy program của step 2, thiếu thì lấy step 1.
    Thứ tự row bám theo ``samples`` để ghép cặp thống kê luôn hợp lệ.
    """
    by_id = {r["id"]: r for r in load_predictions(csv_path)}
    rows = []
    for s in samples:
        rec = by_id.get(s["id"], {})
        p1 = (rec.get("program_step1") or "").strip()
        p2 = (rec.get("program_step2") or "").strip()
        final_prog = (p2 or p1) if prefer == "step2" else (p1 or p2)

        gold_prog = s.get("qa", {}).get("program", "") or ""
        gold_ans = s.get("qa", {}).get("exe_ans")
        value = execute_program(final_prog, s.get("table") or []) if final_prog else None
        ea = check_ea(value, gold_ans) if gold_ans is not None else False
        ea_loose = check_ea(value, gold_ans, abs_tol=1e-3) if gold_ans is not None else False
        pa_strict, pa_loose = check_pa(final_prog, gold_prog) if gold_prog else (False, False)

        rows.append({
            "id": s["id"],
            "question": s["qa"]["question"],
            "gold_program": gold_prog,
            "gold_answer": gold_ans,
            "program_step1": p1,
            "program_step2": p2,
            "final_program": final_prog,
            "pred_value": value,
            "pred_answer_text": None,
            "ea": ea, "ea_tol1e-3": ea_loose,
            "pa_strict": pa_strict, "pa_loose": pa_loose,
            "n_ops_gold": n_ops(gold_prog),
            "outcome": classify_outcome(ea, pa_strict, final_prog, value),
            "ly_do_khong_chay": (ly_do_khong_chay(final_prog, s.get("table") or [])
                                 if final_prog and value is None else None),
            "used_bullets": [],
            "bullets_text": "",
            "raw_step1": "", "raw_step2": "",
            "_source": label or os.path.basename(csv_path),
        })
    missing = [s["id"] for s in samples if s["id"] not in by_id]
    if missing:
        print(f"[SCORE] ⚠ {len(missing)} mẫu không có trong {os.path.basename(csv_path)} "
              f"→ tính là sai. Ví dụ: {missing[:3]}")
    return rows
