# -*- coding: utf-8 -*-
"""Kiểm định ý nghĩa thống kê cho so sánh theo cặp.

Với n = 497 mẫu test, chênh lệch vài điểm EA rất dễ là nhiễu. Hai cấu hình chạy trên
**cùng một tập mẫu** nên đây là dữ liệu *cặp*, và kiểm định đúng là **McNemar**: nó chỉ
nhìn các mẫu mà hai cấu hình bất đồng, đếm ``b`` (chỉ A đúng) và ``c`` (chỉ B đúng), rồi
hỏi xác suất thấy chênh lệch lệch đến mức này nếu thật ra hai cấu hình tương đương.

Kèm khoảng tin cậy bootstrap **lấy mẫu lại theo cặp** (không phải hai mẫu độc lập),
để con số Δ có thanh sai số.
"""
from __future__ import annotations

from math import comb

import numpy as np

__all__ = ["mcnemar_exact", "bootstrap_delta_ci", "compare_pair", "interpret"]


def mcnemar_exact(flags_a, flags_b) -> dict:
    """McNemar chính xác (binomial hai phía) trên hai vector nhị phân theo cặp."""
    b = sum(1 for x, y in zip(flags_a, flags_b) if x and not y)
    c = sum(1 for x, y in zip(flags_a, flags_b) if y and not x)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p_value": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"b": b, "c": c, "n_discordant": n, "p_value": p}


def bootstrap_delta_ci(flags_a, flags_b, n_boot=10000, seed=42, alpha=0.05):
    """KTC cho ``Δ = mean(B) − mean(A)``, lấy mẫu lại THEO CẶP."""
    rng = np.random.default_rng(seed)
    a = np.asarray(flags_a, dtype=float)
    b = np.asarray(flags_b, dtype=float)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    deltas = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    lo = float(np.percentile(deltas, 100 * alpha / 2))
    hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    return lo, hi


def interpret(delta, p_value, lo, hi) -> str:
    if p_value < 0.05 and delta > 0:
        return "✅ cải tiến CÓ ý nghĩa thống kê (p < 0.05)"
    if p_value < 0.05 and delta < 0:
        return "❌ TỤT có ý nghĩa thống kê (p < 0.05)"
    if lo <= 0 <= hi:
        return "⚠ chưa phân biệt được với nhiễu (KTC chứa 0)"
    return "⚠ chưa đạt mức ý nghĩa 0.05"


def compare_pair(rows_base, rows_variant, key="ea", label="",
                 name_base="base", name_variant="variant", verbose=True) -> dict:
    """So sánh hai lần chạy trên cùng tập mẫu. Trả dict kết quả, in ra nếu ``verbose``."""
    ids_b = [r["id"] for r in rows_base]
    ids_v = [r["id"] for r in rows_variant]
    if ids_b != ids_v:
        raise ValueError("Hai cấu hình không cùng thứ tự mẫu — không ghép cặp được.")

    fb = [bool(r[key]) for r in rows_base]
    fv = [bool(r[key]) for r in rows_variant]
    mc = mcnemar_exact(fb, fv)
    lo, hi = bootstrap_delta_ci(fb, fv)
    delta = sum(fv) / len(fv) - sum(fb) / len(fb)
    verdict = interpret(delta, mc["p_value"], lo, hi)

    out = {"label": label, "key": key, "n": len(fb),
           "base": name_base, "variant": name_variant,
           "base_correct": sum(fb), "variant_correct": sum(fv),
           "delta": round(delta, 4), "ci95": (round(lo, 4), round(hi, 4)),
           "p_value": mc["p_value"], "b": mc["b"], "c": mc["c"],
           "n_discordant": mc["n_discordant"], "verdict": verdict}

    if verbose:
        print(f"\n  {label or f'{name_variant} so với {name_base}'}  [{key}]")
        print(f"    {name_base:<16} {sum(fb):>4}/{len(fb)}   →   "
              f"{name_variant:<16} {sum(fv):>4}/{len(fv)}")
        print(f"    Δ = {delta:+.4f}   KTC 95% bootstrap = [{lo:+.4f}, {hi:+.4f}]")
        print(f"    Bất đồng: {mc['n_discordant']} mẫu "
              f"({mc['c']} mẫu chỉ {name_variant} đúng, {mc['b']} mẫu chỉ {name_base} đúng)")
        print(f"    McNemar p = {mc['p_value']:.4f}  → {verdict}")
    return out
