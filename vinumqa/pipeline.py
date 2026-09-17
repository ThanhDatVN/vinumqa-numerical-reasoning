# -*- coding: utf-8 -*-
"""Pipeline chấm điểm dùng chung cho cả 5 nấc của lộ trình thí nghiệm.

Một hàm :func:`run_pipeline` duy nhất phục vụ mọi cấu hình — đổi nấc bằng tham số
chứ không bằng code khác nhau, nhờ vậy các nấc luôn so sánh được với nhau::

    nấc 1   run_pipeline(..., prompt_level="basic",      use_selfeval=False)
    nấc 2   run_pipeline(..., prompt_level="engineered", use_selfeval=False)
    nấc 3   giống nấc 2 nhưng model đã SFT
    nấc 4   run_pipeline(..., prompt_level="engineered", use_selfeval=True)
    nấc 4b  như nấc 4 nhưng thêm cong_buoc2=True
    nấc 5   thêm retriever=<Retriever> và playbook

Lớp sinh văn bản được **tiêm vào** (``generate_fn``) nên module chạy và test được
trên CPU mà không cần model.

Khi ``use_selfeval=True``, chương trình cuối mặc định lấy **program của bước 2 → thiếu
thì lấy của bước 1**. Bật ``cong_buoc2=True`` thì bước 2 còn phải CHẠY ĐƯỢC mới được nhận.
"""
from __future__ import annotations

import gc
import re
import statistics
from collections import Counter, defaultdict

from .dsl import (EA_DECIMAL_PLACES, check_ea, check_pa, classify_outcome,
                  execute_program, ly_do_khong_chay, normalize_program_strict,
                  split_dsl_items, extract_program_answer, n_ops)
from .prompts import strip_assistant

__all__ = ["run_pipeline", "summarize", "print_summary",
           "phan_loai_khong_co_program", "phan_loai_khong_chay_duoc",
           "bo_sung_ly_do", "ap_cong_buoc2", "ty_le_lap", "so_sanh_hai_buoc",
           "phan_loai_sai", "nhom_phep", "bo_phieu", "tu_nhat_quan", "tran_best_of_k"]


def _chuan_hoa_lo(outs) -> list[list[str]]:
    """Mỗi phần tử đầu ra → ``list[str]``.

    ``generate_fn`` trả chuỗi khi sinh 1 mẫu, trả list khi sinh nhiều mẫu
    (``SamplingParams(n=K)``). Chuẩn hoá ở đây nên phần còn lại của pipeline chỉ phải
    biết một dạng duy nhất, và nấc 1 mẫu vẫn chạy y như cũ.
    """
    ra = []
    for o in outs:
        if isinstance(o, (list, tuple)):
            ra.append([strip_assistant(x) for x in o])
        else:
            ra.append([strip_assistant(o)])
    return ra


def run_pipeline(samples, prompt_kit, generate_fn, *,
                 prompt_level="engineered", use_selfeval=False,
                 playbook="", retriever=None,
                 kho_vi_du=None, n_vi_du=3,
                 sp_step1=None, sp_step2=None,
                 desc="infer", record_usage=False, keep_raw=True,
                 vot_mau_bi_cat=True, cong_buoc2=False, sua_khi_loi=False):
    """Chạy một cấu hình trên danh sách mẫu. Trả list dict kết quả từng mẫu.

    ``cong_buoc2`` — CỔNG cho self-eval (nấc 4b). Mặc định ``False`` = giữ đúng hành vi
    cũ (bước 2 luôn thắng). Bật lên thì chỉ nhận program của bước 2 khi nó **thực thi
    được**, hoặc khi bước 1 vốn cũng không chạy được.

    ``kho_vi_du`` — :class:`vinumqa.fewshot.KhoViDu`. Truyền vào thì 2 ví dụ cố định
    trong prompt bị thay bằng ``n_vi_du`` ví dụ TRUY HỒI từ train.

    ``sua_khi_loi`` — sau khi chốt program, mẫu nào executor từ chối thì sinh lại MỘT
    lượt kèm đúng thông báo lỗi. Chỉ những mẫu hỏng mới phải sinh lại nên rất rẻ.
    Chương trình trước khi sửa vẫn lưu ở ``program_truoc_sua``, nên "nếu không sửa thì
    sao" tính lại được trên CPU.

    Sinh **nhiều mẫu** (self-consistency): truyền ``sp_step1`` có ``n=K``. Mọi mẫu đều
    được lưu ở ``cac_program``/``cac_gia_tri``, và đáp án chốt bằng :func:`bo_phieu`.
    Nhờ lưu đủ, đường cong theo k dựng lại được trên CPU bằng :func:`tu_nhat_quan` —
    một lượt chạy GPU cho cả họ kết quả.

    Cả ba cơ chế chỉ hỏi **executor**, không hề đụng tới đáp án vàng.
    """
    if not samples:
        return []

    if retriever is not None and playbook and playbook.strip():
        retrieved = [retriever.retrieve(s["qa"]["question"], playbook, record=record_usage)
                     for s in samples]
    else:
        retrieved = [("", [])] * len(samples)
    bullets_texts = [r[0] for r in retrieved]
    used_ids_list = [r[1] for r in retrieved]

    # Ví dụ mẫu TRUY HỒI — tất định, chạy CPU, không tốn lượt sinh nào.
    if kho_vi_du is not None:
        vi_du_list = [kho_vi_du.van_ban_vi_du(s["qa"]["question"], n_vi_du,
                                              tru_id=s.get("id")) or None
                      for s in samples]
    else:
        vi_du_list = [None] * len(samples)

    def _p1(i):
        return prompt_kit.step1(samples[i], bullets_texts[i], level=prompt_level,
                                vi_du_dong=vi_du_list[i])

    raw1 = _chuan_hoa_lo(generate_fn([_p1(i) for i in range(len(samples))],
                                     sp_step1, desc=f"{desc}/step1"))

    # ── Vớt mẫu bị cắt giữa lúc suy nghĩ ──
    # Nâng trần không cứu được (đo sạch: 4096 và 8192 cùng mất 28 mẫu ở nấc 2), nên vớt
    # bằng một lượt sinh lại với suy nghĩ TẮT — không có đoạn <think> dài để mà bị cắt.
    n_vot = 0
    if vot_mau_bi_cat:
        # Chỉ vớt khi KHÔNG mẫu nào trong lô rút ra được program — còn một mẫu dùng được
        # thì self-consistency đã có cái để bỏ phiếu, không cần sinh thêm.
        _can = [i for i, lo in enumerate(raw1)
                if all(extract_program_answer(r)[0] is None for r in lo)
                and any("<think>" in r and "</think>" not in r for r in lo)]
        if _can:
            _cu = getattr(prompt_kit, "enable_thinking", None)
            prompt_kit.enable_thinking = False
            try:
                _lai = _chuan_hoa_lo(generate_fn([_p1(i) for i in _can],
                                                 sp_step1, desc=f"{desc}/vot-bi-cat"))
            finally:
                prompt_kit.enable_thinking = _cu
            for i, lo in zip(_can, _lai):
                # Chỉ thay khi lượt vớt THẬT SỰ ra được program, không thì giữ nguyên
                # bản cũ để con số "bị cắt" vẫn phản ánh đúng chuyện đã xảy ra.
                if any(extract_program_answer(r)[0] is not None for r in lo):
                    raw1[i] = lo
                    n_vot += 1
            print(f"    {desc}/vớt: {n_vot}/{len(_can)} mẫu bị cắt đã cứu được")

    # ── Chốt bước 1: thực thi từng mẫu rồi BỎ PHIẾU theo giá trị chạy được ──
    cac_prog, cac_val, chon1 = [], [], []
    for s, lo in zip(samples, raw1):
        bang = s.get("table") or []
        ps = [extract_program_answer(r)[0] or "" for r in lo]
        vs = [execute_program(p, bang) if p else None for p in ps]
        cac_prog.append(ps)
        cac_val.append(vs)
        chon1.append(bo_phieu(ps, vs))

    def _prog1(i):
        j = chon1[i]
        return cac_prog[i][j] if j >= 0 else ""

    # ── Lượt SỬA, chạy TRƯỚC bước 2 ──
    # Thứ tự quan trọng: nếu sửa sau bước 2 thì bước 2 đi soát một chương trình mà
    # pipeline sau đó vứt đi, còn `program_step1` lưu lại là bản khác hẳn. Sửa trước
    # thì bước 2 nhận đúng chương trình đã chạy được, và `gia_tri_buoc1` cũng là giá
    # trị thật của nó.
    sua_moi: dict[int, str] = {}
    if sua_khi_loi:
        _hong = []
        for i, s in enumerate(samples):
            p = _prog1(i)
            if not p:
                continue
            bang = s.get("table") or []
            if execute_program(p, bang) is None:
                _hong.append((i, p, ly_do_khong_chay(p, bang)))
        if _hong:
            _lai = _chuan_hoa_lo(generate_fn(
                [prompt_kit.step_sua(samples[i], p, ld, bullets_texts[i])
                 for i, p, ld in _hong], sp_step1, desc=f"{desc}/sua"))
            n_sua = 0
            for (i, _p, _ld), lo in zip(_hong, _lai):
                bang = samples[i].get("table") or []
                for r in lo:
                    pm, _ = extract_program_answer(r)
                    if pm and execute_program(pm, bang) is not None:
                        sua_moi[i] = pm
                        n_sua += 1
                        break
            print(f"    {desc}/sửa: {n_sua}/{len(_hong)} program hỏng đã sửa chạy được")

    def _prog_chot(i):
        """Chương trình bước 1 SAU cả bỏ phiếu lẫn lượt sửa."""
        return sua_moi.get(i, _prog1(i))

    # Lời giải thô của mẫu thắng phiếu; mẫu đã sửa thì ghép chương trình mới vào để
    # bước 2 không soát nhầm bản cũ.
    _raw_thang = []
    for i in range(len(samples)):
        r = raw1[i][chon1[i]] if chon1[i] >= 0 else raw1[i][0]
        if i in sua_moi:
            r = (f"{r}\n\n[Đã sửa cho chạy được]\n```plaintext\n"
                 f"program: {sua_moi[i]}\nanswer: \n```")
        _raw_thang.append(r)

    if use_selfeval:
        # Bước 2 soát chương trình ĐÃ THẮNG PHIẾU (và đã sửa nếu có), kèm giá trị nó
        # chạy thật ra — độ lớn con số là chỗ lộ lỗi rõ nhất.
        _gt = [execute_program(_prog_chot(i), samples[i].get("table") or [])
               if _prog_chot(i) else None for i in range(len(samples))]
        _prompts2 = [prompt_kit.step2(s, r, b, gia_tri_buoc1=g)
                     for s, r, b, g in zip(samples, _raw_thang, bullets_texts, _gt)]
        raw2 = [lo[0] for lo in
                _chuan_hoa_lo(generate_fn(_prompts2, sp_step2, desc=f"{desc}/step2"))]
    else:
        raw2 = [""] * len(samples)

    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:                                    # noqa: BLE001
        pass

    rows, n_cong_chan = [], 0
    for idx, (s, r1, r2, bt, ids) in enumerate(
            zip(samples, _raw_thang, raw2, bullets_texts, used_ids_list)):
        bang = s.get("table") or []
        prog_truoc_sua = _prog1(idx)
        _j = chon1[idx]
        ans1 = extract_program_answer(r1)[1]
        prog1 = sua_moi.get(idx, prog_truoc_sua)
        prog2, ans2 = extract_program_answer(r2) if use_selfeval else (None, None)

        val1 = execute_program(prog1, bang) if prog1 else None
        val2 = execute_program(prog2, bang) if prog2 else None

        # CỔNG BƯỚC 2. Không có cổng thì `prog2 or prog1` cho bước 2 thắng vô điều kiện,
        # kể cả khi bước 2 sinh ra một program KHÔNG CHẠY ĐƯỢC còn bước 1 thì chạy được —
        # tức là bước "tự soát" làm hỏng một mẫu vốn đã đúng. Cổng chỉ hỏi executor.
        if not prog2:
            lay_buoc2 = False
        elif not cong_buoc2:
            lay_buoc2 = True
        else:
            lay_buoc2 = (val2 is not None) or (val1 is None)

        final_prog = prog2 if lay_buoc2 else prog1
        final_ans_text = ans2 if lay_buoc2 else ans1
        value = val2 if lay_buoc2 else val1
        if cong_buoc2 and prog2 and not lay_buoc2:
            n_cong_chan += 1

        # Vì sao executor từ chối — tính NGAY ở đây vì chỉ chỗ này còn giữ `s["table"]`,
        # và lưu vào row để file jsonl sau này phân tích lại được mà không cần bảng.
        ly_do = (ly_do_khong_chay(final_prog, bang)
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
            "lay_buoc2": bool(lay_buoc2),
            # ── dữ liệu thô để tính lại trên CPU, không phải chạy GPU lần nữa ──
            "cac_program": cac_prog[idx],          # K chương trình đã sinh
            "cac_gia_tri": cac_val[idx],           # giá trị thực thi của từng cái
            "cac_ea": [check_ea(v, gold_ans) if gold_ans is not None else False
                       for v in cac_val[idx]],
            "cac_pa": [check_pa(p, gold_prog)[0] if gold_prog else False
                       for p in cac_prog[idx]],
            "k_da_sinh": len(cac_prog[idx]),
            "so_phieu": sum(1 for v in cac_val[idx]
                            if v is not None and _j >= 0
                            and _khoa_gia_tri(v) == _khoa_gia_tri(cac_val[idx][_j])),
            "program_truoc_sua": prog_truoc_sua,
            "da_sua": idx in sua_moi,
            "vi_du_dong": bool(vi_du_list[idx]),
        })
    if cong_buoc2 and n_cong_chan:
        print(f"    {desc}/cổng bước 2: giữ lại bước 1 ở {n_cong_chan} mẫu "
              f"(bước 2 sinh program không chạy được)")
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


def _khoa_gia_tri(v):
    """Khoá gộp phiếu. Làm tròn đúng số chữ số mà EA dùng, để hai lời giải chỉ khác
    sai số dấu phẩy động không bị đếm thành hai đáp án khác nhau."""
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip().casefold()
    try:
        return round(float(v), EA_DECIMAL_PLACES)
    except (TypeError, ValueError):
        return None


def bo_phieu(progs, vals) -> int:
    """Chọn chỉ số mẫu thắng trong self-consistency. Trả -1 nếu không có gì để chọn.

    Luật — **chỉ dùng executor, không hề đụng đáp án vàng**:

    1. Bỏ các mẫu không chạy được. Đáp án nào nhiều phiếu nhất thì thắng.
    2. Trong các mẫu cùng cho đáp án thắng, lấy **program xuất hiện nhiều nhất** (sau
       chuẩn hoá) — chọn cách viết phổ biến nhất giúp PA, không chỉ EA.
    3. Hoà ở bất kỳ bước nào → lấy mẫu có chỉ số NHỎ NHẤT. Tất định tuyệt đối.
    4. Không mẫu nào chạy được → lấy mẫu đầu tiên có program, để còn chấm được lý do.
    """
    n = min(len(progs), len(vals))
    chay = [i for i in range(n) if vals[i] is not None]
    if not chay:
        co = [i for i in range(n) if (progs[i] or "").strip()]
        return co[0] if co else -1

    phieu = Counter(_khoa_gia_tri(vals[i]) for i in chay)
    # sorted theo (-số phiếu, chỉ số nhỏ nhất của khoá đó) → tất định
    dau_tien = {}
    for i in chay:
        dau_tien.setdefault(_khoa_gia_tri(vals[i]), i)
    thang = min(phieu, key=lambda k: (-phieu[k], dau_tien[k]))

    ung = [i for i in chay if _khoa_gia_tri(vals[i]) == thang]
    dang = Counter(normalize_program_strict(progs[i]) or f"__raw{i}" for i in ung)
    dau_dang = {}
    for i in ung:
        dau_dang.setdefault(normalize_program_strict(progs[i]) or f"__raw{i}", i)
    dang_thang = min(dang, key=lambda k: (-dang[k], dau_dang[k]))
    return dau_dang[dang_thang]


def tu_nhat_quan(rows, samples, k: int) -> list[dict]:
    """Chấm lại một nấc ĐÃ CHẠY ở mức ``k`` mẫu — **không tốn GPU**.

    Nấc chạy với ``n_mau=K`` lưu cả ``cac_program`` lẫn ``cac_gia_tri``, nên đường cong
    self-consistency theo k = 1, 2, … K dựng được hết trên CPU từ MỘT lượt chạy GPU.
    ``k=1`` chính là "không self-consistency" — mốc để so.

    Trả list row MỚI; ``rows`` gốc không bị đụng.
    """
    if k < 1:
        raise ValueError("k phải ≥ 1")
    bang = {s.get("id"): (s.get("table") or []) for s in samples}
    ra = []
    for r in rows:
        progs = list(r.get("cac_program") or [])
        vals = list(r.get("cac_gia_tri") or [])
        if not progs:                       # nấc chạy 1 mẫu — giữ nguyên
            ra.append(dict(r))
            continue
        progs, vals = progs[:k], vals[:k]
        i = bo_phieu(progs, vals)

        moi = dict(r)
        moi["final_program"] = (progs[i] if i >= 0 else "") or ""
        moi["pred_value"] = vals[i] if i >= 0 else None
        moi["k_da_dung"] = len(progs)
        moi["so_phieu"] = sum(1 for v in vals
                              if v is not None
                              and _khoa_gia_tri(v) == _khoa_gia_tri(moi["pred_value"]))

        gold_prog = r.get("gold_program") or ""
        gold_ans = r.get("gold_answer")
        val = moi["pred_value"]
        moi["ea"] = check_ea(val, gold_ans) if gold_ans is not None else False
        moi["ea_tol1e-3"] = (check_ea(val, gold_ans, abs_tol=1e-3)
                             if gold_ans is not None else False)
        ps, pl = check_pa(moi["final_program"], gold_prog) if gold_prog else (False, False)
        moi["pa_strict"], moi["pa_loose"] = ps, pl
        moi["outcome"] = classify_outcome(moi["ea"], ps, moi["final_program"], val)
        moi["ly_do_khong_chay"] = (
            ly_do_khong_chay(moi["final_program"], bang.get(r.get("id"), []))
            if moi["final_program"] and val is None else None)
        ra.append(moi)
    return ra


def tran_best_of_k(rows, k: int | None = None) -> dict:
    """TRẦN của self-consistency: nếu luôn chọn được mẫu đúng nhất trong k mẫu thì EA/PA
    lên tới đâu. Không phải kết quả đạt được — là **cận trên** để biết còn bao nhiêu đất.

    Chỉ dùng được khi nấc đã lưu ``cac_ea``/``cac_pa`` (chấm sẵn từng mẫu lúc chạy).
    """
    n = len(rows) or 1
    ea = pa = 0
    for r in rows:
        e = (r.get("cac_ea") or [])[:k] if k else (r.get("cac_ea") or [])
        p = (r.get("cac_pa") or [])[:k] if k else (r.get("cac_pa") or [])
        ea += any(e)
        pa += any(p)
    return {"EA_tran": round(ea / n, 4), "PA_tran": round(pa / n, 4), "n": len(rows)}


def ap_cong_buoc2(rows, samples) -> list[dict]:
    """Áp CỔNG bước 2 lên một nấc ĐÃ CHẠY XONG, chấm lại — **không tốn GPU**.

    Nấc self-eval lưu cả ``program_step1`` lẫn ``program_step2``, mà cổng chỉ cần
    executor và bảng của mẫu. Vì vậy "nấc 4 có cổng" không phải chạy lại model: tính
    thẳng từ file jsonl của nấc 4.

    Trả về list row MỚI (không sửa ``rows`` gốc), cùng schema, để đưa thẳng vào
    :func:`summarize` và :func:`vinumqa.stats.compare_pair`.
    """
    bang = {s.get("id"): (s.get("table") or []) for s in samples}
    ra = []
    for r in rows:
        t = bang.get(r.get("id"), [])
        p1 = (r.get("program_step1") or "").strip()
        p2 = (r.get("program_step2") or "").strip()
        v1 = execute_program(p1, t) if p1 else None
        v2 = execute_program(p2, t) if p2 else None
        lay2 = bool(p2) and ((v2 is not None) or (v1 is None))

        moi = dict(r)
        moi["final_program"] = (p2 if lay2 else p1) or ""
        moi["pred_value"] = v2 if lay2 else v1
        moi["lay_buoc2"] = lay2

        gold_prog = r.get("gold_program") or ""
        gold_ans = r.get("gold_answer")
        val = moi["pred_value"]
        moi["ea"] = check_ea(val, gold_ans) if gold_ans is not None else False
        moi["ea_tol1e-3"] = (check_ea(val, gold_ans, abs_tol=1e-3)
                             if gold_ans is not None else False)
        ps, pl = check_pa(moi["final_program"], gold_prog) if gold_prog else (False, False)
        moi["pa_strict"], moi["pa_loose"] = ps, pl
        moi["outcome"] = classify_outcome(moi["ea"], ps, moi["final_program"], val)
        moi["ly_do_khong_chay"] = (ly_do_khong_chay(moi["final_program"], t)
                                   if moi["final_program"] and val is None else None)
        ra.append(moi)
    return ra


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


def _day_phep(prog: str) -> list[str]:
    """Dãy TÊN phép toán của một program, bỏ qua toán hạng."""
    ra = []
    for lenh in split_dsl_items((prog or "").strip()):
        m = re.match(r"\s*([a-z_]+)\s*\(", lenh, re.I)
        if m:
            ra.append(m.group(1).casefold())
    return ra


def nhom_phep(gold_program: str) -> str:
    """Xếp một program gold vào một nhóm phép toán, để đo EA/PA theo LOẠI câu hỏi.

    ``table_*`` tách riêng vì nó là loại câu duy nhất phải ĐỌC NHÃN trong bảng chứ không
    chỉ tính toán — hỏng ở đây có nguyên nhân khác hẳn, và cách chữa cũng khác hẳn.
    """
    day = _day_phep(gold_program)
    if not day:
        return "?"
    if any(x.startswith("table_") for x in day):
        return "table_*"
    return day[0] if len(day) == 1 else f"nhiều ({len(day)})"


def phan_loai_sai(rows) -> dict | None:
    """Tách ô "sai" theo KIỂU sai, so dãy phép toán của model với của gold.

    * ``thieu_buoc`` / ``thua_buoc`` — số phép ít hơn / nhiều hơn gold. Lỗi lập kế hoạch.
    * ``dung_so_buoc_sai_phep`` — đúng số phép nhưng chọn nhầm loại (``subtract`` chỗ
      đáng ``divide``). Đây là thứ ánh xạ từ khoá → phép toán phải chữa.
    * ``dung_phep_sai_so_lieu`` — dãy phép TRÙNG KHÍT gold, chỉ khác toán hạng: model
      hiểu đúng bài, nhưng lấy nhầm số khỏi bảng. Chữa bằng cách dạy đọc bảng, không
      phải bằng cách dạy chọn phép.
    * ``bo_qua_table`` / ``lam_dung_table`` — lệch nhau ở việc có dùng ``table_*``.

    Trả ``None`` nếu không có mẫu nào thuộc diện này.
    """
    dem, vi_du = Counter(), defaultdict(list)
    for r in rows:
        if r.get("outcome") != "sai":
            continue
        pm, pg = _day_phep(r.get("final_program")), _day_phep(r.get("gold_program"))
        tm = any(x.startswith("table_") for x in pm)
        tg = any(x.startswith("table_") for x in pg)
        if tg and not tm:
            k = "bo_qua_table"
        elif tm and not tg:
            k = "lam_dung_table"
        elif len(pm) < len(pg):
            k = "thieu_buoc"
        elif len(pm) > len(pg):
            k = "thua_buoc"
        elif pm != pg:
            k = "dung_so_buoc_sai_phep"
        else:
            k = "dung_phep_sai_so_lieu"
        dem[k] += 1
        if len(vi_du[k]) < 3:
            vi_du[k].append({"hoi": (r.get("question") or "")[:90],
                             "model": (r.get("final_program") or "")[:90],
                             "gold": (r.get("gold_program") or "")[:90]})
    if not dem:
        return None
    n = len(rows) or 1
    return {"tong": sum(dem.values()),
            "theo_kieu": dict(dem.most_common()),
            "ty_le": {k: round(v / n, 4) for k, v in dem.most_common()},
            "vi_du": {k: vi_du[k] for k, _ in dem.most_common()}}


def so_sanh_hai_buoc(rows, samples) -> dict | None:
    """Bước 2 (self-eval / ACE) thực sự đổi được bao nhiêu program so với bước 1.

    Nhìn EA trước/sau KHÔNG phân biệt được hai chuyện: bước 2 chép lại y nguyên bước 1
    (không có gì để sửa), hay nó sửa nhiều mà các thay đổi triệt tiêu nhau. Phải đếm
    thẳng số program bị đổi, và trong đó bao nhiêu cái đổi cả GIÁ TRỊ thực thi.

    Trả ``None`` nếu nấc này không có bước 2.
    """
    if not any((r.get("program_step2") or "").strip() for r in rows):
        return None
    bang = {s.get("id"): (s.get("table") or []) for s in samples}
    chuan = lambda p: re.sub(r"\s+", "", (p or "")).lower()      # noqa: E731
    d = Counter()
    for r in rows:
        p1, p2 = r.get("program_step1") or "", r.get("program_step2") or ""
        if not p1 and not p2:
            d["ca_hai_deu_trong"] += 1
        elif not p1:
            d["cuu_mau_buoc1_bo_trong"] += 1
        elif not p2:
            d["buoc2_bo_trong_giu_buoc1"] += 1
        elif chuan(p1) == chuan(p2):
            d["chep_lai_y_nguyen"] += 1
        else:
            t = bang.get(r.get("id"), [])
            v1 = execute_program(p1, t)
            v2 = execute_program(p2, t)
            d["doi_va_doi_ca_gia_tri" if repr(v1) != repr(v2)
              else "doi_nhung_gia_tri_giu_nguyen"] += 1
    n = len(rows) or 1
    # "Đổi" chỉ tính khi bước 2 THỰC SỰ đưa ra một program khác. Bước 2 bỏ trống rồi
    # lùi về dùng lại bước 1 KHÔNG phải là đổi — gộp vào là thổi phồng con số.
    doi = (d["doi_nhung_gia_tri_giu_nguyen"] + d["doi_va_doi_ca_gia_tri"]
           + d["cuu_mau_buoc1_bo_trong"])
    return {"theo_nhom": dict(d.most_common()),
            "so_program_bi_doi": doi,
            "ty_le_bi_doi": round(doi / n, 4)}


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
    dem, vi_du = Counter(), defaultdict(list)
    for r in rows:
        ly = r.get("ly_do_khong_chay")
        if not ly:
            continue
        dem[ly] += 1
        # Giữ vài program THẬT cho mỗi lý do. Con số "120 mẫu cú pháp dị dạng" không nói
        # được phải sửa gì; nhìn ba cái program hỏng là biết ngay.
        if len(vi_du[ly]) < 3:
            vi_du[ly].append((r.get("final_program") or "")[:160])
    if not dem:
        return None
    n = len(rows) or 1
    return {"tong": sum(dem.values()),
            "theo_ly_do": dict(dem.most_common()),
            "ty_le": {k: round(v / n, 4) for k, v in dem.most_common()},
            "vi_du": {k: vi_du[k] for k, _ in dem.most_common()}}


def summarize(rows, label="") -> dict:
    n = len(rows) or 1
    by_steps = defaultdict(lambda: [0, 0, 0])            # n_ops → [tổng, ea đúng, pa đúng]
    by_phep = defaultdict(lambda: [0, 0, 0])             # loại phép → [tổng, ea, pa]
    for r in rows:
        b = by_steps[r["n_ops_gold"]]
        b[0] += 1
        b[1] += r["ea"]
        b[2] += r["pa_strict"]
        q = by_phep[nhom_phep(r.get("gold_program"))]
        q[0] += 1
        q[1] += r["ea"]
        q[2] += r["pa_strict"]
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
        # tách theo LOẠI phép — by_steps gộp theo số phép nên table_* vô hình
        "by_phep": {k: v for k, v in sorted(by_phep.items(), key=lambda x: -x[1][0])},
        # vì sao "không sinh được program": bị cắt hay sai định dạng
        "vi_sao_khong_co_program": phan_loai_khong_co_program(rows),
        # vì sao "program không chạy được": nhãn bảng, lồng nhau, #N sai…
        "vi_sao_khong_chay_duoc": phan_loai_khong_chay_duoc(rows),
        # "sai" sai KIỂU gì: nhầm phép, nhầm số, thiếu bước…
        "vi_sao_sai": phan_loai_sai(rows),
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
        # Ba lý do lớn nhất, kèm một program thật — để biết phải sửa CÁI GÌ.
        for _k, _ in list(_kc["theo_ly_do"].items())[:3]:
            for _p in (_kc.get("vi_du", {}).get(_k) or [])[:1]:
                print(f"         {_k} ← {_p}")
    _vs2 = m.get("vi_sao_sai")
    if _vs2:
        print(f"  Sai (suy luận)          : {_vs2['tong']/m['n']:.4f}")
        for _k, _v in _vs2["theo_kieu"].items():
            print(f"      {_k:<32}{_v:>5} ({_v/m['n']:.1%})")
        for _k, _ in list(_vs2["theo_kieu"].items())[:2]:
            for _e in (_vs2.get("vi_du", {}).get(_k) or [])[:1]:
                print(f"         {_k}: model={_e['model']}")
                print(f"         {'':<{len(_k)}}  gold ={_e['gold']}")
    print(f"\n  {'số phép':<9}{'mẫu':>6}{'EA':>9}{'PA':>9}")
    for k, (tot, ea, pa) in m["by_steps"].items():
        print(f"  {k:<9}{tot:>6}{ea/tot:>9.1%}{pa/tot:>9.1%}")
    if m.get("by_phep"):
        print(f"\n  {'loại phép (gold)':<18}{'mẫu':>6}{'EA':>9}{'PA':>9}")
        for k, (tot, ea, pa) in m["by_phep"].items():
            _co = "   ← đọc nhãn bảng" if k == "table_*" else ""
            print(f"  {k:<18}{tot:>6}{ea/tot:>9.1%}{pa/tot:>9.1%}{_co}")
    print("\n  Phân bố kết cục:")
    for k, v in sorted(m["outcome"].items(), key=lambda x: -x[1]):
        print(f"    {k:<28}{v:>5} ({v/m['n']:.1%})")
