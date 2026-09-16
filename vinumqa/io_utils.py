# -*- coding: utf-8 -*-
"""Đọc/ghi artifact và chấm lại các dự đoán đã lưu sẵn.

Hai việc chính:

* **Ghi** kết quả ra ``results/details/`` đúng schema 6 cột của repo, để
  ``results/output_analyst.ipynb`` và ``app.py`` đọc được ngay.
* **Đọc lại** các dự đoán tham chiếu trong ``results/details/*_program.csv`` rồi chấm
  bằng executor đã kiểm chứng — nhờ vậy các thí nghiệm mới tái dùng được kết quả cũ
  thay vì chạy lại GPU.
"""
from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict

from .dsl import (check_ea, check_pa, classify_outcome, execute_program,
                  ly_do_khong_chay, n_ops)

__all__ = ["LEGACY_COLS", "save_details_csv", "save_full_jsonl", "save_raw_jsonl",
           "load_predictions", "score_saved_predictions", "save_metrics_csv",
           "ensure_details_dir", "BASELINE_RESULTS"]

LEGACY_COLS = ["id", "question", "gold_program", "gold_answer",
               "program_step1", "program_step2"]

#: Bảng kết quả tham chiếu của 5 model (post Step-2), dùng để đối chiếu.
BASELINE_RESULTS = {
    "Llama-3.1-8B":             {"PA": 0.20,  "EA": 0.40},
    "mistral-7b-instruct-v0.3": {"PA": 38.63, "EA": 41.65},
    "Phi4":                     {"PA": 54.69, "EA": 59.31},
    "Qwen3-8B":                 {"PA": 59.56, "EA": 64.19},
    "Phi4_finetuned":           {"PA": 52.52, "EA": 56.54},
}


def ensure_details_dir(repo_dir: str, fallback_dir: str) -> str:
    """Trả về thư mục ghi details; lùi về ``fallback_dir`` nếu repo chỉ đọc."""
    details = os.path.join(repo_dir, "results", "details")
    try:
        os.makedirs(details, exist_ok=True)
        probe = os.path.join(details, ".write_test")
        open(probe, "w").close()
        os.remove(probe)
        return details
    except OSError:
        os.makedirs(fallback_dir, exist_ok=True)
        print(f"[SAVE] Không ghi được vào repo — lưu tại {fallback_dir}")
        return fallback_dir


def save_details_csv(rows, path: str) -> str:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEGACY_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in LEGACY_COLS})
    return path


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


def save_metrics_csv(metrics_by_config: dict, path: str, model_tag: str) -> str:
    cols = ["config", "model", "n", "EA", "PA_strict", "PA_loose", "EA_tol1e-3",
            "no_program", "exec_none", "bullets", "minutes"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for name, m in metrics_by_config.items():
            w.writerow({"config": name, "model": model_tag,
                        **{k: m.get(k, "") for k in cols[2:]}})
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
