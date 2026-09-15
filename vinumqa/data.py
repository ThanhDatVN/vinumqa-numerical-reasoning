# -*- coding: utf-8 -*-
"""Nạp ViNumQA, tách nguồn dữ liệu, và audit chất lượng nhãn vàng.

Lần chạy trước nêu 5–10 % gold program có vấn đề. Module này định lượng được
hai loại nhiễu **đo được bằng máy**:

1. ``multiply(#n, 100)`` để ra phần trăm — prompt của dự án CẤM dạng này, nên trên
   các mẫu đó model làm đúng vẫn bị chấm sai cả PA lẫn EA.
   Đo được: **101 mẫu ở train, 1 ở valid, 0 ở test** → nhiễu chỉ nằm ở train,
   đúng là tập mà ACE dùng để phản tỉnh.
2. Gold program không tự thực thi ra đúng ``exe_ans`` (thường do thiếu dấu ngoặc).
   Đo được: 6 mẫu train, 1 mẫu valid, 0 mẫu test.

``source_of`` tách FinQA-Vi và Vi Data từ ``id`` (FinQA-Vi giữ tên file PDF gốc).
"""
from __future__ import annotations

import json
import os
import random
import re
from collections import Counter, defaultdict

from .dsl import check_ea, execute_program, n_ops

__all__ = [
    "load_split", "load_all", "source_of", "has_gold",
    "MULTIPLY_100_RE", "uses_multiply_100", "gold_executes",
    "is_noisy_gold", "audit_gold", "stratified_sample", "split_by_source",
]

MULTIPLY_100_RE = re.compile(r"multiply\s*\(\s*(?:#\d+\s*,\s*100|100\s*,\s*#\d+)\s*\)", re.I)


def load_split(data_dir: str, name: str) -> list[dict]:
    path = os.path.join(data_dir, f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Kiểm tra lại REPO_DIR / DATA_DIR."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all(data_dir: str) -> dict[str, list[dict]]:
    return {name: load_split(data_dir, name) for name in ("train", "valid", "test")}


def has_gold(sample: dict) -> bool:
    return bool(str(sample.get("qa", {}).get("program", "")).strip())


def source_of(sample: dict) -> str:
    """``'FinQA-Vi'`` hoặc ``'ViData'``.

    FinQA-Vi giữ nguyên id của FinQA gốc (``HIG/2004/page_140.pdf-3``); Vi Data dùng
    id do nhóm xây dựng đặt (``masvn/2020/.../page_1_QA7``).
    """
    return "FinQA-Vi" if ".pdf-" in str(sample.get("id", "")) else "ViData"


def split_by_source(samples: list[dict]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for s in samples:
        out[source_of(s)].append(s)
    return dict(out)


# ─────────────────────────────── nhiễu nhãn ───────────────────────────────

def uses_multiply_100(sample: dict) -> bool:
    """Gold có ``multiply(#n, 100)`` — dạng mà prompt của dự án cấm."""
    return bool(MULTIPLY_100_RE.search(sample.get("qa", {}).get("program", "") or ""))


def gold_executes(sample: dict) -> bool:
    """Gold program tự thực thi ra đúng ``exe_ans``."""
    qa = sample.get("qa", {})
    value = execute_program(qa.get("program", ""), sample.get("table") or [])
    return check_ea(value, qa.get("exe_ans"))


def is_noisy_gold(sample: dict) -> bool:
    """Mẫu mà tín hiệu giám sát KHÔNG đáng tin để phản tỉnh."""
    return uses_multiply_100(sample) or not gold_executes(sample)


def audit_gold(samples: list[dict], label: str = "") -> dict:
    """Thống kê chất lượng nhãn vàng của một tập."""
    n = len(samples)
    mul100, no_exec, noisy = [], [], []
    for s in samples:
        bad_mul = uses_multiply_100(s)
        bad_exec = not gold_executes(s)
        if bad_mul:
            mul100.append(s["id"])
        if bad_exec:
            no_exec.append(s["id"])
        if bad_mul or bad_exec:
            noisy.append(s["id"])
    by_source = Counter(source_of(s) for s in samples)
    by_ops = Counter(min(n_ops(s["qa"].get("program", "")), 5) for s in samples)
    return {
        "label": label,
        "n": n,
        "by_source": dict(by_source),
        "by_n_ops": dict(sorted(by_ops.items())),
        "multiply_100": len(mul100),
        "multiply_100_ids": mul100,
        "gold_not_executable": len(no_exec),
        "gold_not_executable_ids": no_exec,
        "noisy": len(noisy),
        "noisy_ids": noisy,
        "noisy_pct": round(len(noisy) / n * 100, 2) if n else 0.0,
        "executable_pct": round((n - len(no_exec)) / n * 100, 2) if n else 0.0,
        "table_ops": sum(1 for s in samples if "table_" in (s["qa"].get("program") or "")),
    }


def print_audit(report: dict) -> None:
    r = report
    print(f"[AUDIT] {r['label']:<8} n={r['n']:<5} nguồn={r['by_source']}")
    print(f"         gold thực thi đúng exe_ans : {r['executable_pct']:.2f}% "
          f"({r['gold_not_executable']} mẫu sai)")
    print(f"         gold dùng multiply(#n,100) : {r['multiply_100']} mẫu")
    print(f"         → tổng mẫu nhiễu           : {r['noisy']} ({r['noisy_pct']:.2f}%)")
    print(f"         program có table_*         : {r['table_ops']}")
    print(f"         phân bố số phép toán       : {r['by_n_ops']}")


# ─────────────────────────────── lấy mẫu ───────────────────────────────

def stratified_sample(samples: list[dict], total: int | None, seed: int = 42,
                      weights: dict[int, float] | None = None) -> list[dict]:
    """Lấy mẫu phân tầng theo số phép toán của gold (1, 2, 3, 4, 5+).

    Nâng trọng số cho bài nhiều bước: hiếm trong dữ liệu nhưng là chỗ model sai nhiều nhất,
    nên cũng là chỗ ACE có nhiều thứ để học nhất.
    """
    if total is None or total >= len(samples):
        return list(samples)
    weights = weights or {1: 0.38, 2: 0.32, 3: 0.15, 4: 0.08, 5: 0.07}

    buckets = defaultdict(list)
    for s in samples:
        buckets[min(n_ops(s["qa"].get("program", "")), 5)].append(s)
    rng = random.Random(seed)
    for v in buckets.values():
        rng.shuffle(v)

    picked, taken = [], {}
    for k in sorted(buckets):
        want = int(round(total * weights.get(k, 0.05)))
        take = min(want, len(buckets[k]))
        taken[k] = take
        picked += buckets[k][:take]
    for k in sorted(buckets, key=lambda k: -len(buckets[k])):       # bù cho đủ số lượng
        while len(picked) < total and taken[k] < len(buckets[k]):
            picked.append(buckets[k][taken[k]])
            taken[k] += 1
    rng.shuffle(picked)
    return picked[:total]
