# -*- coding: utf-8 -*-
"""Pipeline chấm điểm dùng chung cho cả 5 nấc của lộ trình thí nghiệm.

Một hàm :func:`run_pipeline` duy nhất phục vụ mọi cấu hình — đổi nấc bằng tham số
chứ không bằng code khác nhau, nhờ vậy các nấc luôn so sánh được với nhau::

    nấc 1  run_pipeline(..., prompt_level="plain",      use_selfeval=False)
    nấc 2  run_pipeline(..., prompt_level="engineered", use_selfeval=False)
    nấc 3  giống nấc 2 nhưng model đã SFT
    nấc 4  run_pipeline(..., prompt_level="engineered", use_selfeval=True)
    nấc 5  thêm retriever=<Retriever> và playbook

Lớp sinh văn bản được **tiêm vào** (``generate_fn``) nên module chạy và test được
trên CPU mà không cần model.

Khi ``use_selfeval=True``, chương trình cuối lấy theo đúng logic của notebook gốc:
**program của bước 2 → thiếu thì lấy của bước 1.**
"""
from __future__ import annotations

import gc
import re
import statistics
from collections import Counter, defaultdict

from .dsl import (check_ea, check_pa, classify_outcome, execute_program,
                  ly_do_khong_chay,
                  extract_program_answer, n_ops)
from .prompts import strip_assistant

__all__ = ["run_pipeline", "summarize", "print_summary", "compare_ladder",
           "phan_loai_khong_co_program"]


def run_pipeline(samples, prompt_kit, generate_fn, *,
                 prompt_level="engineered", use_selfeval=False,
                 playbook="", retriever=None,
                 sp_step1=None, sp_step2=None,
                 desc="infer", record_usage=False, keep_raw=True):
    """Chạy một cấu hình trên danh sách mẫu. Trả list dict kết quả từng mẫu."""
    if not samples:
        return []

    if retriever is not None and playbook and playbook.strip():
        retrieved = [retriever.retrieve(s["qa"]["question"], playbook, record=record_usage)
                     for s in samples]
    else:
        retrieved = [("", [])] * len(samples)
    bullets_texts = [r[0] for r in retrieved]
    used_ids_list = [r[1] for r in retrieved]

    raw1 = generate_fn(
        [prompt_kit.step1(s, b, level=prompt_level)
         for s, b in zip(samples, bullets_texts)],
        sp_step1, desc=f"{desc}/step1")
    raw1 = [strip_assistant(r) for r in raw1]

    if use_selfeval:
        raw2 = generate_fn(
            [prompt_kit.step2(s, r, b)
             for s, r, b in zip(samples, raw1, bullets_texts)],
            sp_step2, desc=f"{desc}/step2")
        raw2 = [strip_assistant(r) for r in raw2]
    else:
        raw2 = [""] * len(samples)

    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:                                    # noqa: BLE001
        pass

    rows = []
    for s, r1, r2, bt, ids in zip(samples, raw1, raw2, bullets_texts, used_ids_list):
        prog1, ans1 = extract_program_answer(r1)
        prog2, ans2 = extract_program_answer(r2) if use_selfeval else (None, None)

        final_prog = prog2 or prog1
        final_ans_text = ans2 if prog2 else ans1
        value = execute_program(final_prog, s.get("table") or []) if final_prog else None

        # Vì sao executor từ chối — tính NGAY ở đây vì chỉ chỗ này còn giữ `s["table"]`,
        # và lưu vào row để file jsonl sau này phân tích lại được mà không cần bảng.
        ly_do = (ly_do_khong_chay(final_prog, s.get("table") or [])
                 if final_prog and value is None else None)

        gold_prog = s.get("qa", {}).get("program", "") or ""
        gold_ans = s.get("qa", {}).get("exe_ans")
        ea = check_ea(value, gold_ans) if gold_ans is not None else False
        ea_loose = check_ea(value, gold_ans, abs_tol=1e-3) if gold_ans is not None else False
        pa_strict, pa_loose = check_pa(final_prog, gold_prog) if gold_prog else (False, False)

        rows.append({
            "id": s.get("id", ""),
            "question": s["qa"]["question"],
            "gold_program": gold_prog,
            "gold_answer": gold_ans,
            "program_step1": prog1 or "",
            "program_step2": prog2 or "",
            "final_program": final_prog or "",
            "pred_value": value,
            "pred_answer_text": final_ans_text,
            "ea": ea, "ea_tol1e-3": ea_loose,
            "pa_strict": pa_strict, "pa_loose": pa_loose,
            "n_ops_gold": n_ops(gold_prog),
            "outcome": classify_outcome(ea, pa_strict, final_prog, value),
            "ly_do_khong_chay": ly_do,
            "used_bullets": ids,
            "bullets_text": bt,
            "raw_step1": r1 if keep_raw else "",
            "raw_step2": r2 if keep_raw else "",
        })
    return rows


def ty_le_lap(text: str, n: int = 60) -> float:
    """Mức LẶP của đoạn cuối một lượt sinh. 0 = không lặp, gần 1 = quay vòng một câu.

    Dùng để phân biệt hai lý do rất khác nhau khi model chạm trần token:
    suy luận dài thật (lặp thấp — nâng trần sẽ cứu được) và quay vòng vô hạn
    (lặp cao — nâng trần chỉ tốn thêm thời gian, thuốc nằm ở repetition_penalty).
    """
    t = re.sub(r"\s+", " ", text or "").strip()[-4000:]
    if len(t) < n * 3:
        return 0.0
    buoc = max(1, n // 3)
    grams = [t[i:i + n] for i in range(0, len(t) - n + 1, buoc)]
    return round(1 - len(set(grams)) / len(grams), 3) if grams else 0.0


def phan_loai_khong_co_program(rows) -> dict | None:
    """Tách "không sinh được program" thành BỊ CẮT vs SAI ĐỊNH DẠNG.

    Hai thứ này cần cách chữa khác hẳn nhau, mà ``no_program`` gộp chung nên không
    biết đường nào mà lần:

    * **bị cắt giữa suy nghĩ** — output có ``<think>`` mở mà thiếu ``</think>``: model
      đang suy luận thì chạm ``max_tokens``. Chữa bằng cách nâng trần.
    * **sai định dạng** — nghĩ xong rồi nhưng không nhả ra khối ``program:`` đọc được.
      Nâng trần KHÔNG chữa được; đây là chuyện của prompt.

    Trả ``None`` nếu rows không giữ output thô (``keep_raw=False``).
    """
    if not any(r.get("raw_step1") or r.get("raw_step2") for r in rows):
        return None
    bi_cat = sai_dinh_dang = 0
    lap = []
    for r in rows:
        if r.get("final_program"):
            continue
        raws = [r.get("raw_step1") or "", r.get("raw_step2") or ""]
        cat = [t for t in raws if "<think>" in t and "</think>" not in t]
        if cat:
            bi_cat += 1
            lap.append(max(ty_le_lap(t) for t in cat))
        else:
            sai_dinh_dang += 1
    n = len(rows) or 1
    out = {"bi_cat_giua_suy_nghi": bi_cat, "sai_dinh_dang": sai_dinh_dang,
           "ty_le_bi_cat": round(bi_cat / n, 4),
           "ty_le_sai_dinh_dang": round(sai_dinh_dang / n, 4)}
    if lap:
        # Trung vị mức lặp của CHÍNH những lượt bị cắt: cao thì nâng trần là vô ích.
        out["lap_trung_vi"] = round(statistics.median(lap), 3)
        out["so_ca_lap_nang"] = sum(1 for x in lap if x >= 0.5)
    return out


def bo_sung_ly_do(rows, samples) -> int:
    """Điền ``ly_do_khong_chay`` cho row đọc từ file chạy TRƯỚC khi có trường này.

    Phải có ``samples`` vì lý do phụ thuộc vào BẢNG của mẫu, mà jsonl không lưu bảng.
    Nhờ hàm này, những nấc đã chạy xong vẫn phân tích lại được, không phải tốn GPU.

    Trả về số row vừa điền.
    """
    bang = {s.get("id"): (s.get("table") or []) for s in samples}
    n = 0
    for r in rows:
        if r.get("ly_do_khong_chay") or not r.get("final_program"):
            continue
        if r.get("pred_value") is not None:
            continue
        r["ly_do_khong_chay"] = ly_do_khong_chay(r["final_program"],
                                                 bang.get(r.get("id"), []))
        n += 1
    return n


def phan_loai_khong_chay_duoc(rows) -> dict | None:
    """Tách "program không chạy được" theo LÝ DO executor từ chối.

    Ở nấc prompt cơ bản đây là ô LỚN NHẤT của bảng kết cục (33 %) — lớn hơn cả ô "sai".
    Gộp chung thì không biết chữa đường nào, vì mỗi lý do cần một cách sửa khác hẳn:

    * ``nhan_bang_khong_khop`` — gọi ``table_*`` với nhãn không có hàng nào khớp. Chữa
      bằng hướng dẫn đọc bảng, không phải bằng quy tắc định dạng.
    * ``phep_long_nhau`` — ``divide(5310, add(1, 0.15))``. Prompt có hẳn một dòng cấm;
      con số này cho biết dòng đó có tác dụng hay không.
    * ``tham_chieu_sai`` — ``#N`` trỏ tới phép chưa có. Lỗi lập kế hoạch nhiều bước.
    * ``tham_so_khong_phai_so`` — nhét chữ vào chỗ cần số.

    Trả ``None`` nếu không mẫu nào thuộc diện này (khỏi in một khối rỗng).
    """
    dem = Counter(r["ly_do_khong_chay"] for r in rows
                  if r.get("ly_do_khong_chay"))
    if not dem:
        return None
    n = len(rows) or 1
    return {"tong": sum(dem.values()),
            "theo_ly_do": dict(dem.most_common()),
            "ty_le": {k: round(v / n, 4) for k, v in dem.most_common()}}


def summarize(rows, label="") -> dict:
    n = len(rows) or 1
    by_steps = defaultdict(lambda: [0, 0, 0])            # n_ops → [tổng, ea đúng, pa đúng]
    for r in rows:
        b = by_steps[r["n_ops_gold"]]
        b[0] += 1
        b[1] += r["ea"]
        b[2] += r["pa_strict"]
    return {
        "label": label,
        "n": len(rows),
        "EA": round(sum(r["ea"] for r in rows) / n, 4),
        "PA_strict": round(sum(r["pa_strict"] for r in rows) / n, 4),
        "PA_loose": round(sum(r["pa_loose"] for r in rows) / n, 4),
        "EA_tol1e-3": round(sum(r["ea_tol1e-3"] for r in rows) / n, 4),
        "no_program": round(sum(not r["final_program"] for r in rows) / n, 4),
        "exec_none": round(sum(r["pred_value"] is None for r in rows) / n, 4),
        "outcome": dict(Counter(r["outcome"] for r in rows)),
        "by_steps": {str(k): v for k, v in sorted(by_steps.items())},
        # vì sao "không sinh được program": bị cắt hay sai định dạng
        "vi_sao_khong_co_program": phan_loai_khong_co_program(rows),
        # vì sao "program không chạy được": nhãn bảng, lồng nhau, #N sai…
        "vi_sao_khong_chay_duoc": phan_loai_khong_chay_duoc(rows),
    }


def print_summary(m) -> None:
    print(f"\n  ── KẾT QUẢ [{m['label']}] — {m['n']} mẫu ──")
    print(f"  EA          : {m['EA']:.4f}")
    print(f"  PA (strict) : {m['PA_strict']:.4f}")
    print(f"  PA (loose)  : {m['PA_loose']:.4f}   ← so sánh được với bảng tham chiếu  ")
    _vs = m.get("vi_sao_khong_co_program")
    print(f"  Không sinh được program : {m['no_program']:.4f}"
          + (f"   (bị cắt {_vs['bi_cat_giua_suy_nghi']} | "
             f"sai định dạng {_vs['sai_dinh_dang']})" if _vs else ""))
    if _vs and _vs.get("lap_trung_vi") is not None:
        _l = _vs["lap_trung_vi"]
        print(f"      lặp của lượt bị cắt (trung vị) : {_l:.0%}"
              f"   {_vs['so_ca_lap_nang']}/{_vs['bi_cat_giua_suy_nghi']} ca lặp nặng")
        print("      → " + ("quay vòng, NÂNG TRẦN VÔ ÍCH — thuốc ở repetition_penalty"
                            if _l >= 0.5 else
                            "suy luận dài thật, nâng trần có thể cứu thêm"))
    print(f"  Program không chạy được : {m['exec_none']:.4f}")
    _kc = m.get("vi_sao_khong_chay_duoc")
    if _kc:
        for _k, _v in _kc["theo_ly_do"].items():
            print(f"      {_k:<32}{_v:>5} ({_v/m['n']:.1%})")
    print(f"\n  {'số phép':<9}{'mẫu':>6}{'EA':>9}{'PA':>9}")
    for k, (tot, ea, pa) in m["by_steps"].items():
        print(f"  {k:<9}{tot:>6}{ea/tot:>9.1%}{pa/tot:>9.1%}")
    print("\n  Phân bố kết cục:")
    for k, v in sorted(m["outcome"].items(), key=lambda x: -x[1]):
        print(f"    {k:<28}{v:>5} ({v/m['n']:.1%})")


def compare_ladder(metrics_by_stage: dict, order: list[str], key="EA") -> None:
    """In bảng thang bậc: mỗi nấc và phần tăng thêm so với nấc ngay trước."""
    print(f"\n{'═'*84}\n  THANG BẬC — {key}\n{'═'*84}")
    print(f"{'nấc':<28}{key:>10}{'Δ so với nấc trước':>22}{'Δ so với nấc 1':>18}")
    base = None
    prev = None
    for name in order:
        if name not in metrics_by_stage:
            continue
        v = metrics_by_stage[name][key]
        if base is None:
            base, prev = v, v
            print(f"{name:<28}{v:>10.4f}{'(mốc)':>22}{'—':>18}")
            continue
        print(f"{name:<28}{v:>10.4f}{v - prev:>+22.4f}{v - base:>+18.4f}")
        prev = v
