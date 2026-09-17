# -*- coding: utf-8 -*-
"""Bộ kiểm toàn dự án — chạy trước mỗi lần commit và trước mỗi lần chạy lại thang bậc.

    python tools/kiem_tra.py           # 4 phép kiểm nhanh, ~5 giây
    python tools/kiem_tra.py --day-du  # + chạy notebook 07 end-to-end với nấc dựng sẵn

Trả mã thoát khác 0 nếu có lỗi, để cắm vào CI hoặc pre-commit được.

Vì sao cần: biên dịch sạch KHÔNG có nghĩa là chạy được. Lỗi tên biến, khoá thiếu, định
dạng chuỗi chỉ lộ ra khi thực thi; còn LADDER lệch giữa các notebook hay trần token lệch
giữa các ô cấu hình thì chỉ lộ ra khi đối chiếu chéo. Đây là mẻ lưới cho cả hai loại.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import glob
import io
import json
import os
import re
import sys

GOC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NBDIR = os.path.join(GOC, "notebooks")
sys.path.insert(0, GOC)

# Cấu hình CHUẨN của cả thang bậc. Đổi ở đây thì phải đổi trong mọi ô cấu hình.
MAX_TOKENS_CHUAN, MAX_SEQ_CHUAN = 4096, 17000
MAX_SEQ_HUAN_LUYEN = 8192          # nb03 dùng riêng cho huấn luyện, khác mục đích
PHAN_PROMPT_CO_DINH = 7464         # phần prompt bước 2 KHÔNG phụ thuộc trần sinh (đo thật)

loi: list[str] = []


def _o_code(path):
    nb = json.load(io.open(path, encoding="utf-8"))
    return nb, [(k, c, "".join(c["source"]))
                for k, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]


def _sang_python(src: str) -> str:
    """Đổi dòng `!pip`/`%magic` thành `pass` — XOÁ hẳn sẽ để lại thân `if` rỗng và báo
    lỗi cú pháp oan ở các ô cài đặt."""
    ra = []
    for l in src.splitlines():
        t = l.lstrip()
        ra.append(l[:len(l) - len(t)] + "pass" if t.startswith(("!", "%")) else l)
    return "\n".join(ra)


# ══════════════════════ 1. notebook biên dịch được, không sót output ══════════════════
def kiem_notebook():
    import nbformat
    print("═" * 78)
    print("  1. NOTEBOOK — nbformat hợp lệ, biên dịch được, không sót output")
    n_nb = n_o = 0
    for p in sorted(glob.glob(os.path.join(NBDIR, "*.ipynb"))):
        n_nb += 1
        ten = os.path.basename(p)
        nb = nbformat.read(p, as_version=4)
        nbformat.validate(nb)
        for k, c in enumerate(nb.cells):
            if c.cell_type != "code":
                continue
            n_o += 1
            try:
                compile(_sang_python("".join(c.source)), f"{ten}#{k}", "exec")
            except SyntaxError as e:
                loi.append(f"{ten}#{k}: {e.msg} (dòng {e.lineno})")
            if c.get("outputs") or c.get("execution_count"):
                loi.append(f"{ten}#{k}: còn sót output — notebook phải được xoá output")
    print(f"   {n_nb} notebook · {n_o} ô code")


# ══════════════════════ 2. cấu hình đồng nhất giữa các notebook ═══════════════════════
def kiem_cau_hinh():
    print("═" * 78)
    print("  2. CẤU HÌNH — LADDER, trần token, ngân sách ngữ cảnh, bộ đếm")
    paths = sorted(glob.glob(os.path.join(NBDIR, "*.ipynb")))

    # LADDER phải giống hệt ở mọi ô bootstrap
    lad = {}
    for p in paths:
        for k, _c, s in _o_code(p)[1]:
            if "LADDER = [" in s:
                i = s.index("LADDER = [")
                lad.setdefault(s[i:s.index("]", i) + 1], []).append(f"{os.path.basename(p)}#{k}")
    if len(lad) != 1:
        loi.append(f"LADDER có {len(lad)} biến thể — mọi ô bootstrap phải giống hệt nhau")
    else:
        nacs = re.findall(r'\("([^"]+)",', list(lad)[0])
        print(f"   LADDER: 1 biến thể, {len(nacs)} nấc, {sum(len(v) for v in lad.values())} ô")

    # trần token
    val: dict[str, set] = {}
    for p in paths:
        for _k, _c, s in _o_code(p)[1]:
            for pat, nhan in ((r"TEMPERATURE, MAX_TOKENS = [\d.]+, (\d+)", "MAX_TOKENS"),
                              (r"^MAX_SEQ_LENGTH = (\d+)", "MAX_SEQ_LENGTH"),
                              (r"MAX_TOKENS_CHUAN, MAX_SEQ_CHUAN = (\d+), (\d+)", "CHUAN")):
                for m in re.finditer(pat, s, re.M):
                    val.setdefault(nhan, set()).add(m.group(0).strip())
    if val.get("MAX_TOKENS") != {f"TEMPERATURE, MAX_TOKENS = 0.1, {MAX_TOKENS_CHUAN}"}:
        loi.append(f"MAX_TOKENS không đồng nhất hoặc khác chuẩn: {val.get('MAX_TOKENS')}")
    ms = {int(re.search(r"(\d+)$", v).group(1)) for v in val.get("MAX_SEQ_LENGTH", set())}
    if ms - {MAX_SEQ_CHUAN, MAX_SEQ_HUAN_LUYEN}:
        loi.append(f"MAX_SEQ_LENGTH lạ: {ms}")
    if val.get("CHUAN") != {f"MAX_TOKENS_CHUAN, MAX_SEQ_CHUAN = {MAX_TOKENS_CHUAN}, {MAX_SEQ_CHUAN}"}:
        loi.append("MAX_TOKENS_CHUAN/MAX_SEQ_CHUAN lệch mức chuẩn — mọi lần chạy sẽ bị "
                   "gắn hậu tố oan vào tên nấc")
    print(f"   trần token: max_tokens={MAX_TOKENS_CHUAN} max_seq={MAX_SEQ_CHUAN} — đồng nhất")

    # ngân sách ngữ cảnh: prompt bước 2 phải lọt max_seq
    b2 = PHAN_PROMPT_CO_DINH + MAX_TOKENS_CHUAN
    du = (MAX_SEQ_CHUAN - MAX_TOKENS_CHUAN) - b2
    print(f"   ngân sách ngữ cảnh: prompt bước 2 xấu nhất {b2} / "
          f"{MAX_SEQ_CHUAN - MAX_TOKENS_CHUAN} → dư {du} token")
    if du <= 500:
        loi.append(f"ngân sách ngữ cảnh chỉ dư {du} token — sẽ bị cắt ngữ cảnh")

    # bộ đếm bị cắt: notebook nào có generate() thì phải định nghĩa + reset + in
    for p in paths:
        src = "\n".join(s for _k, _c, s in _o_code(p)[1])
        if "def generate(" not in src:
            continue
        thieu = [t for t, co in (("định nghĩa", "def bi_cat_theo_buoc" in src),
                                 ("reset", "dat_lai_bo_dem()" in src),
                                 ("in ra", "in_bi_cat_theo_buoc()" in src)) if not co]
        if thieu:
            loi.append(f"{os.path.basename(p)}: bộ đếm bị cắt thiếu {thieu}")
    print("   bộ đếm bị cắt: đủ ở mọi notebook có GPU")


# ══════════════════════ 3. thang prompt phải LỒNG NHAU ════════════════════════════════
def kiem_thang_prompt():
    print("═" * 78)
    print("  3. THANG PROMPT — mỗi bước phải là phép CHÈN THUẦN")
    from vinumqa.prompts import PromptKit
    kit = PromptKit()
    muc = [(m, getattr(kit, f"{m.upper()}_SYSTEM_PROMPT"))
           for m in ("basic", "no_fewshot", "engineered")]
    for n, t in muc:
        print(f"   {n:<14}{len(t):>6} ký tự")
    for (na, a), (nb, b) in zip(muc, muc[1:]):
        ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
               if o[0] != "equal"]
        if not (len(ops) == 1 and ops[0][0] == "insert"):
            loi.append(f"thang prompt {na}→{nb} KHÔNG lồng nhau ({[o[0] for o in ops]}) — "
                       "sửa chữ ở phần dùng chung là hỏng phép đo")
    print("   basic ⊂ no_fewshot ⊂ engineered: chèn thuần")


# ══════════════════════ 4. mã chết, import thừa ═══════════════════════════════════════
def kiem_ma_chet():
    print("═" * 78)
    print("  4. MÃ NGUỒN — khai báo không ai dùng, import thừa")
    van_ban = {}
    for p in (glob.glob(os.path.join(GOC, "vinumqa/**/*.py"), recursive=True)
              + glob.glob(os.path.join(GOC, "tests/*.py"))
              + glob.glob(os.path.join(GOC, "tools/*.py"))):
        van_ban[os.path.relpath(p, GOC)] = io.open(p, encoding="utf-8").read()
    for p in glob.glob(os.path.join(NBDIR, "*.ipynb")):
        van_ban[os.path.relpath(p, GOC)] = "\n".join(s for _k, _c, s in _o_code(p)[1])

    khai_bao = {}
    for f, s in van_ban.items():
        if not f.startswith("vinumqa"):
            continue
        for node in ast.parse(s).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                khai_bao[node.name] = f
    n_chet = 0
    for ten, f in sorted(khai_bao.items()):
        if ten.startswith("_"):
            continue
        dung = sum(len(re.findall(r"\b" + re.escape(ten) + r"\b", s)) - (g == f)
                   for g, s in van_ban.items())
        if dung == 0:
            loi.append(f"mã chết: {ten} ({f}) không ai gọi")
            n_chet += 1

    n_thua = 0
    for f, s in sorted(van_ban.items()):
        if not f.startswith("vinumqa"):
            continue
        than = re.sub(r"^\s*(from|import)\s.*$", "", s, flags=re.M)
        for node in ast.walk(ast.parse(s)):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for a in node.names:
                ten = (a.asname or a.name).split(".")[0]
                if ten in ("*", "annotations"):
                    continue
                if not re.search(r"\b" + re.escape(ten) + r"\b", than):
                    loi.append(f"import thừa: {ten} ({f}:{node.lineno})")
                    n_thua += 1
    print(f"   {len(khai_bao)} khai báo công khai · {n_chet} mã chết · {n_thua} import thừa")


# ══════════════════════ 5. (tuỳ chọn) notebook 07 chạy end-to-end ═════════════════════
def kiem_bao_cao_day_du():
    """Dựng vài nấc GIẢ rồi chạy trọn notebook 07.

    Không có dữ liệu thì phần lớn ô của 07 không chạy tới, nên phải bịa mới kiểm được.
    Ghi vào ``runs/stages`` (đã gitignore) rồi dọn sạch sau khi chạy.
    """
    print("═" * 78)
    print("  5. NOTEBOOK 07 — chạy trọn mọi ô với nấc dựng sẵn")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import random
    import traceback
    from vinumqa import data, dsl, pipeline

    ra = os.path.join(GOC, "runs")
    os.makedirs(os.path.join(ra, "stages"), exist_ok=True)
    test = data.load_all(os.path.join(GOC, "data"))["test"]
    rng = random.Random(7)
    HONG = ["table_max(khong_co_nhan, none)", "divide(5310, add(1, 0.15))",
            "add(#7, 1)", "add(abc, 2)"]

    def nac(ty_le, co_buoc2):
        rows = []
        for i, s in enumerate(test):
            gold = s["qa"].get("program") or ""
            prog = gold if rng.random() < ty_le else HONG[i % len(HONG)]
            if rng.random() < 0.024:
                prog = ""
            val = dsl.execute_program(prog, s.get("table") or []) if prog else None
            ea = dsl.check_ea(val, s["qa"].get("exe_ans")) if prog else False
            pa_s, pa_l = dsl.check_pa(prog, gold) if prog and gold else (False, False)
            rows.append({
                "id": s["id"], "question": s["qa"]["question"], "gold_program": gold,
                "gold_answer": s["qa"].get("exe_ans"), "program_step1": prog,
                "program_step2": (prog if co_buoc2 and i % 3 else ""),
                "final_program": prog, "pred_value": val, "pred_answer_text": None,
                "ea": ea, "ea_tol1e-3": ea, "pa_strict": pa_s, "pa_loose": pa_l,
                "n_ops_gold": dsl.n_ops(gold),
                "outcome": dsl.classify_outcome(ea, pa_s, prog, val),
                "used_bullets": [], "bullets_text": "",
                "raw_step1": "<think>xong</think>\n```plaintext\nok\n```", "raw_step2": ""})
        return rows

    NAC = [("01_basic", "Nấc 1 — prompt cơ bản (danh sách phép toán + yêu cầu)", 0.30, False),
           ("02_prompt_eng", "Nấc 2 — prompt hoàn chỉnh (+ hướng dẫn từ khoá + few-shot)",
            0.62, False),
           ("04_selfeval_base", "Nấc 4 — + self-eval (model gốc)", 0.64, True)]
    for stage, nhan, tl, b2 in NAC:
        rows = nac(tl, b2)
        with io.open(os.path.join(ra, "stages", stage + ".jsonl"), "w",
                     encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items()
                                    if k not in ("raw_step1", "raw_step2")},
                                   ensure_ascii=False, default=str) + "\n")
        m = pipeline.summarize(rows, stage)
        m["minutes"] = 12.0
        buoc = {"step1": {"n": 497, "bi_cat": 12, "ty_le": 0.0241}}
        if b2:
            buoc["step2"] = {"n": 497, "bi_cat": 20, "ty_le": 0.0402}
        with io.open(os.path.join(ra, "stages", stage + "_meta.json"), "w",
                     encoding="utf-8", newline="\n") as f:
            json.dump({"stage": stage, "label": nhan, "stamp": "kiemtra", "n": len(rows),
                       "metrics": m, "max_tokens": MAX_TOKENS_CHUAN,
                       "max_seq_length": MAX_SEQ_CHUAN, "ctx_truncated": False,
                       "enable_thinking": None, "ty_le_bi_cat_token": 0.0241,
                       "bi_cat_theo_buoc": buoc, "nap_an_toan": False,
                       "config": {"MAX_TOKENS": MAX_TOKENS_CHUAN,
                                  "MAX_SEQ_LENGTH": MAX_SEQ_CHUAN,
                                  "TEMPERATURE": 0.1, "MODEL_TAG": "Qwen3-8B"},
                       "env": {"gpu": "NVIDIA A100-SXM4-40GB", "commit": "kiemtra"}},
                      f, ensure_ascii=False, indent=1)

    ns = {"__name__": "__main__"}
    n_o = 0
    for k, _c, src in _o_code(os.path.join(NBDIR, "07_final_report.ipynb"))[1]:
        n_o += 1
        code = _sang_python(src.replace('REPO_DIR = ""', f'REPO_DIR = r"{GOC}"'))
        try:
            exec(compile(code, f"07#{k}", "exec"), ns)
        except Exception:                                # noqa: BLE001
            loi.append(f"07#{k} không chạy được:\n{traceback.format_exc(limit=4)}")
    for f in glob.glob(os.path.join(ra, "stages", "*")):
        if "noisy_train_ids" not in f:
            os.remove(f)
    for f in glob.glob(os.path.join(ra, "bao_cao_*.png")) \
            + glob.glob(os.path.join(ra, "bang_ket_qua_*.csv")) \
            + glob.glob(os.path.join(ra, "kiem_dinh_*.csv")):
        os.remove(f)
    print(f"   {n_o} ô đã chạy, đã dọn sạch nấc dựng sẵn")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day-du", action="store_true",
                    help="chạy thêm notebook 07 end-to-end (chậm hơn, cần matplotlib)")
    args = ap.parse_args()

    kiem_notebook()
    kiem_cau_hinh()
    kiem_thang_prompt()
    kiem_ma_chet()
    if args.day_du:
        kiem_bao_cao_day_du()

    print("═" * 78)
    if loi:
        print(f"  ⛔ {len(loi)} VẤN ĐỀ")
        for x in loi:
            print("     •", x)
    else:
        print("  ✅ Không có lỗi")
    print("\n  Nhớ chạy kèm:  python -m pytest tests/ -q -W error")
    return 1 if loi else 0


if __name__ == "__main__":
    raise SystemExit(main())
