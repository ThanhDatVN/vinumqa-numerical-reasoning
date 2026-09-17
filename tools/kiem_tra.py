# -*- coding: utf-8 -*-
"""Bộ kiểm toàn dự án — chạy trước mỗi lần commit và trước mỗi lần chạy lại thang bậc.

    python tools/kiem_tra.py           # 6 phép kiểm nhanh, ~5 giây
    python tools/kiem_tra.py --day-du  # + chạy notebook 07 end-to-end với nấc dựng sẵn

Trả mã thoát khác 0 nếu có lỗi, để cắm vào CI hoặc pre-commit được.

Vì sao cần: biên dịch sạch KHÔNG có nghĩa là chạy được. Lỗi tên biến, khoá thiếu, định
dạng chuỗi chỉ lộ ra khi thực thi; còn LADDER lệch giữa các notebook hay trần token lệch
giữa các ô cấu hình thì chỉ lộ ra khi đối chiếu chéo. Đây là mẻ lưới cho cả hai loại.
"""
from __future__ import annotations

import argparse
import ast
import builtins
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


# ══════════════════════ 5. tên dùng mà chưa bao giờ được định nghĩa ═══════════════════
_TEN_CO_SAN = set(dir(builtins)) | {
    "get_ipython", "display", "In", "Out", "_ih", "_oh", "__name__", "__file__"}


class _ThuTen(ast.NodeVisitor):
    """Gom tên được GÁN và tên được DÙNG trong một ô notebook."""

    def __init__(self):
        self.gan, self.dung = set(), []

    def visit_Name(self, n):
        (self.gan.add(n.id) if isinstance(n.ctx, ast.Store)
         else self.dung.append((n.id, n.lineno)))

    def visit_FunctionDef(self, n):
        self.gan.add(n.name)
        a = n.args
        for x in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]:
            if x is not None:
                self.gan.add(x.arg)
        self.generic_visit(n)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, n):
        a = n.args
        for x in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]:
            if x is not None:
                self.gan.add(x.arg)
        self.generic_visit(n)

    def visit_ClassDef(self, n):
        self.gan.add(n.name)
        self.generic_visit(n)

    def visit_Import(self, n):
        for a in n.names:
            self.gan.add((a.asname or a.name).split(".")[0])

    def visit_ImportFrom(self, n):
        for a in n.names:
            self.gan.add(a.asname or a.name)

    def visit_ExceptHandler(self, n):
        if n.name:
            self.gan.add(n.name)
        self.generic_visit(n)

    def visit_Global(self, n):
        self.gan.update(n.names)


def kiem_ten_notebook():
    """Bắt tên được gọi mà KHÔNG ô nào định nghĩa.

    Vì sao cần riêng phép kiểm này: `kiem_notebook` chỉ BIÊN DỊCH, nên `NameError` lọt
    qua sạch. Đã có tiền lệ đắt: notebook 06 gọi `ea(...)` ở hai ô mà hàm đó chưa bao giờ
    tồn tại — lỗi nằm NGAY SAU toàn bộ phần chạy GPU, làm hỏng ô ghi CSV và ô vẽ biểu đồ.
    """
    print("═" * 78)
    print("  5. NOTEBOOK — tên gọi mà không ô nào định nghĩa")
    tong = 0
    for p in sorted(glob.glob(os.path.join(NBDIR, "*.ipynb"))):
        _nb, o = _o_code(p)
        gan, dung = set(), []
        for k, _c, s in o:
            try:
                cay = ast.parse(_sang_python(s))
            except SyntaxError as e:
                loi.append(f"{os.path.basename(p)} ô {k}: không parse được ({e.msg})")
                continue
            t = _ThuTen()
            t.visit(cay)
            gan |= t.gan
            dung += [(ten, k) for ten, _ln in t.dung]
        # Tên nằm trong `"x" in globals()` là có chủ ý, đừng báo động.
        van_ban = "\n".join(s for _k, _c, s in o)
        thieu = {}
        for ten, k in dung:
            if ten in gan or ten in _TEN_CO_SAN:
                continue
            if f'"{ten}" in globals()' in van_ban or f"'{ten}' in globals()" in van_ban:
                continue
            thieu.setdefault(ten, set()).add(k)
        for ten, o_list in sorted(thieu.items()):
            loi.append(f"{os.path.basename(p)}: gọi `{ten}` ở ô {sorted(o_list)} "
                       f"mà không ô nào định nghĩa")
            tong += 1
    print(f"   {len(glob.glob(os.path.join(NBDIR, '*.ipynb')))} notebook · "
          f"{tong} tên chưa định nghĩa")


# ══════════════════════ 6. runbook khớp notebook ══════════════════════════════════════
#: (notebook, ô # theo ô CODE, mảnh chuỗi phải nằm trong ô đó). Đây là các mốc mà
#: HUONG_DAN_COLAB.md dẫn người chạy bám theo; lệch một ô là người chạy soi nhầm chỗ.
NEO_RUNBOOK = [
    ("00_data_audit", 3, "Executor tái tạo đúng 100%"),
    ("01_baseline_basic", 3, "[GÓI] "),
    ("01_baseline_basic", 3, "executor tái tạo exe_ans trên test"),
    ("01_baseline_basic", 4, "[CFG] max_seq="),
    ("01_baseline_basic", 5, "[WARMUP] ✅"),
    ("01_baseline_basic", 6, "thang lồng nhau"),
    ("01_baseline_basic", 6, "mọi prompt đều lọt ngân sách"),
    ("01_baseline_basic", 6, 'PROMPT_LEVEL = "basic"'),
    ("01_baseline_basic", 8, "NẤC: {STAGE}"),
    ("01_baseline_basic", 10, "save_stage"),
    ("02_prompt_engineering", 6, "[NẤC] 2 — prompt hoàn chỉnh"),
    ("02_prompt_engineering", 7, "chèn thuần"),
    ("02_prompt_engineering", 8, 'STAGE = "02_prompt_eng"'),
    ("02_prompt_engineering", 11, "save_stage"),
    ("03_sft_qwen3", 7, "SFT_TRAIN_SUBSET"),
    ("03_sft_qwen3", 9, "build_sft_records"),
    ("03_sft_qwen3", 11, "latest.txt"),
    ("03_sft_qwen3", 21, "ADAPTER_DIR"),
    ("03_sft_qwen3", 22, 'STAGE = "03_sft"'),
    ("03_sft_qwen3", 24, "save_stage"),
    ("04_self_evaluation", 6, "USE_SFT_ADAPTER"),
    ("04_self_evaluation", 7, "mọi prompt đều lọt ngân sách"),
    ("04_self_evaluation", 9, "04_selfeval_"),
    ("04_self_evaluation", 13, "save_stage"),
    ("05_ace", 6, "USE_SFT_ADAPTER"),
    ("05_ace", 8, "ACE_TREN_PROMPT"),
    ("05_ace", 8, "gọi thử"),
    ("05_ace", 10, "AceTrainer"),
    ("05_ace", 12, "NẤC: {STAGE}"),
    ("05_ace", 14, "RUN_RANDOM_CONTROL"),
    ("05_ace", 16, "save_stage"),
    ("06_combination", 7, "MATRIX = ["),
    ("06_combination", 8, "[PLAYBOOK]"),
    ("06_combination", 10, "CHAY_O_TOT_NHAT"),
    ("06_combination", 11, "CỔNG MỤC TIÊU"),
    ("06_combination", 12, "TÁC ĐỘNG CHÍNH"),
    ("06_combination", 15, "ma_tran_to_hop_"),
    ("06_combination", 16, "plt.savefig"),
    ("07_final_report", 3, "chưa có kết quả"),
    ("07_final_report", 4, "KIỂM TRA CÔNG BẰNG"),
    ("07_final_report", 6, "ĐỘ PHỨC TẠP"),
    ("07_final_report", 7, "EA THEO LOẠI PHÉP TOÁN"),
    ("07_final_report", 8, "SELF-CONSISTENCY THEO k"),
    ("07_final_report", 9, "ap_cong_buoc2"),
    ("07_final_report", 10, "PHÂN BỐ KẾT CỤC"),
    ("07_final_report", 12, "bang_ket_qua_"),
    ("07_final_report", 13, "bao_cao_"),
    ("08_phuong_phap_moi", 7, "SO_MAU"),
    ("08_phuong_phap_moi", 7, "KhoViDu"),
    ("08_phuong_phap_moi", 8, "CỔNG KIỂM SỚM"),
    ("08_phuong_phap_moi", 9, '08_tu_nhat_quan'),
    ("08_phuong_phap_moi", 10, "SELF-CONSISTENCY THEO k"),
    ("08_phuong_phap_moi", 12, '09_vidu_dong'),
    ("08_phuong_phap_moi", 13, "VÍ DỤ ĐỘNG ĐÓNG GÓP"),
]


def kiem_runbook():
    """HUONG_DAN_COLAB.md phải khớp notebook: số ô code và vị trí từng cổng kiểm.

    Vì sao cần: runbook dẫn người chạy tới ĐÚNG MỘT Ô rồi bảo phải thấy dòng gì. Thêm
    hay bớt một ô là mọi số sau đó lệch, và người chạy đi soi nhầm chỗ trên một phiên
    GPU đang tính tiền. Không ai phát hiện được điều đó bằng cách đọc.
    """
    print("═" * 78)
    print("  6. RUNBOOK — HUONG_DAN_COLAB khớp notebook")
    doc = os.path.join(GOC, "HUONG_DAN_COLAB.md")
    if not os.path.exists(doc):
        loi.append("thiếu HUONG_DAN_COLAB.md")
        return
    van = io.open(doc, encoding="utf-8").read()

    # (a) mỗi "## Bước k · `<notebook>`" kèm "**N ô code" phải đúng số ô thật
    n_dem = 0
    for m in re.finditer(r"## Bước \d+ · `([0-9]{2}_[a-z_0-9]+)`(.{0,600}?)\*\*(\d+) ô code",
                         van, re.S):
        nb_ten, _giua, n_noi = m.group(1), m.group(2), int(m.group(3))
        p = os.path.join(NBDIR, nb_ten + ".ipynb")
        if not os.path.exists(p):
            loi.append(f"runbook trỏ tới notebook không có: {nb_ten}")
            continue
        that = len(_o_code(p)[1])
        n_dem += 1
        if that != n_noi:
            loi.append(f"runbook nói {nb_ten} có {n_noi} ô code, thật ra {that}")

    # (b) từng cổng kiểm phải nằm đúng ô mà runbook dẫn tới
    n_neo = 0
    for nb_ten, so_o, xau in NEO_RUNBOOK:
        ma = _o_code(os.path.join(NBDIR, nb_ten + ".ipynb"))[1]
        if so_o > len(ma):
            loi.append(f"{nb_ten}: runbook dẫn tới ô #{so_o} nhưng chỉ có {len(ma)} ô")
            continue
        if xau not in ma[so_o - 1][2]:
            o_that = [i for i, (_k, _c, s) in enumerate(ma, 1) if xau in s]
            loi.append(f"{nb_ten} ô #{so_o} không chứa {xau!r}"
                       + (f" — thật ra ở ô {o_that}" if o_that else " — không ô nào có"))
        else:
            n_neo += 1
    print(f"   {n_dem} bước có số ô · {n_neo}/{len(NEO_RUNBOOK)} cổng kiểm đúng ô")


# ══════════════════ 7–8. (tuỳ chọn) notebook 07 và 06 chạy thật ══════════════════
def kiem_bao_cao_day_du():
    """Dựng vài nấc GIẢ rồi chạy trọn notebook 07.

    Không có dữ liệu thì phần lớn ô của 07 không chạy tới, nên phải bịa mới kiểm được.
    Ghi vào ``runs/stages`` (đã gitignore) rồi dọn sạch sau khi chạy.
    """
    print("═" * 78)
    print("  7. NOTEBOOK 07 — chạy trọn mọi ô với nấc dựng sẵn")
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

    def nac_k_mau(ty_le, K=5, co_sua=True, co_vidu=False):
        """Nấc kiểu MỚI: K mẫu + lượt sửa, đủ trường để §5d của 07 chạy THẬT.

        Không có nấc dạng này thì khối "phương pháp mới" chỉ in "⊘ chưa có", và mọi lỗi
        trong đó nằm im tới khi chạy Colab xong 3 tiếng GPU mới lộ ra.
        """
        from vinumqa import pipeline as _pl
        rows = []
        for i, s in enumerate(test):
            gold = s["qa"].get("program") or ""
            bang = s.get("table") or []
            progs = [gold if rng.random() < ty_le else HONG[(i + j) % len(HONG)]
                     for j in range(K)]
            vals = [dsl.execute_program(p, bang) if p else None for p in progs]
            j = _pl.bo_phieu(progs, vals)
            prog = progs[j] if j >= 0 else ""
            val = vals[j] if j >= 0 else None
            ea = dsl.check_ea(val, s["qa"].get("exe_ans")) if prog else False
            pa_s, pa_l = dsl.check_pa(prog, gold) if prog and gold else (False, False)
            rows.append({
                "id": s["id"], "question": s["qa"]["question"], "gold_program": gold,
                "gold_answer": s["qa"].get("exe_ans"), "program_step1": prog,
                "program_step2": "", "final_program": prog, "pred_value": val,
                "pred_answer_text": None,
                "ea": ea, "ea_tol1e-3": ea, "pa_strict": pa_s, "pa_loose": pa_l,
                "n_ops_gold": dsl.n_ops(gold),
                "outcome": dsl.classify_outcome(ea, pa_s, prog, val),
                "used_bullets": [], "bullets_text": "",
                "cac_program": progs, "cac_gia_tri": vals,
                "cac_ea": [dsl.check_ea(v, s["qa"].get("exe_ans")) for v in vals],
                "cac_pa": [dsl.check_pa(p, gold)[0] if gold else False for p in progs],
                "k_da_sinh": K,
                "so_phieu": sum(1 for v in vals
                                if v is not None and j >= 0 and v == vals[j]),
                "program_truoc_sua": prog,
                "da_sua": bool(co_sua and val is None and prog and i % 4 == 0),
                "vi_du_dong": co_vidu,
                "raw_step1": "", "raw_step2": ""})
        return rows

    NAC_MOI = {"08_tu_nhat_quan": (0.55, False), "09_vidu_dong": (0.60, True)}
    NAC = [("01_basic", "Nấc 1 — prompt cơ bản (danh sách phép toán + yêu cầu)", 0.30, False),
           ("02_prompt_eng", "Nấc 2 — prompt hoàn chỉnh (+ hướng dẫn từ khoá + few-shot)",
            0.62, False),
           ("04_selfeval_base", "Nấc 4 — + self-eval (model gốc)", 0.64, True),
           ("08_tu_nhat_quan", "Mới — self-consistency K mẫu (ví dụ cố định)", 0.55, False),
           ("09_vidu_dong", "Mới — self-consistency + ví dụ truy hồi", 0.60, True)]
    for stage, nhan, tl, b2 in NAC:
        # Truyền theo TÊN: `nac_k_mau(*NAC_MOI[stage])` đẩy nhầm cờ vào tham số K.
        rows = (nac_k_mau(NAC_MOI[stage][0], co_vidu=NAC_MOI[stage][1])
                if stage in NAC_MOI else nac(tl, b2))
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
                       **({"so_mau": 5, "temperature_moi": 0.7, "vi_du_dong": True,
                           "sua_khi_loi": True} if stage in NAC_MOI else {}),
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
    # ── GỌI THẬT save_stage / stage_path / load_stage ──
    # Ô bootstrap chỉ được ĐỊNH NGHĨA khi chạy 07, chưa bao giờ được GỌI. Đúng chỗ đó đã
    # để lọt `KeyError: 'csv'` ra tận Colab và làm hỏng hai lượt chạy GPU: notebook ghi
    # xong jsonl rồi mới nổ ở dòng in đường dẫn. Biên dịch sạch không bắt được khoá dict.
    try:
        _rows_thu = [{"id": "x", "question": "q", "gold_program": "add(1,2)",
                      "gold_answer": "3", "program_step1": "add(1,2)",
                      "program_step2": "", "final_program": "add(1,2)",
                      "pred_value": 3.0, "ea": True, "ea_tol1e-3": True,
                      "pa_strict": True, "pa_loose": True, "n_ops_gold": 1,
                      "outcome": "dung", "used_bullets": [], "bullets_text": "",
                      "raw_step1": "", "raw_step2": ""}]
        for _kind in ("jsonl", "meta"):
            ns["stage_path"]("_kiemtra", _kind)
        ns["save_stage"]("_kiemtra", _rows_thu, {"EA": 1.0, "PA_strict": 1.0},
                         extra={"prompt_level": "engineered"}, quiet=True)
        _lai = ns["load_stage"]("_kiemtra", quiet=True)
        if not _lai or _lai[0]["id"] != "x":
            loi.append("save_stage ghi rồi load_stage đọc lại KHÔNG ra đúng dữ liệu")
    except Exception:                                    # noqa: BLE001
        loi.append(f"ô bootstrap: save_stage/load_stage lỗi khi gọi thật:\n"
                   f"{traceback.format_exc(limit=4)}")

    for f in glob.glob(os.path.join(ra, "stages", "*")):
        if "noisy_train_ids" not in f:
            os.remove(f)
    for f in glob.glob(os.path.join(ra, "logs", "_kiemtra_raw_*.jsonl")):
        os.remove(f)
    for f in glob.glob(os.path.join(ra, "bao_cao_*.png")) \
            + glob.glob(os.path.join(ra, "bang_ket_qua_*.csv")) \
            + glob.glob(os.path.join(ra, "kiem_dinh_*.csv")):
        os.remove(f)
    print(f"   {n_o} ô đã chạy · save_stage/load_stage gọi thật · đã dọn nấc dựng sẵn")


def kiem_ma_tran_06():
    """Chạy các ô PHÂN TÍCH của notebook 06 với ma trận dựng sẵn.

    Phần đầu của 06 cần vLLM nên không chạy nổi ở đây; nhưng mọi lỗi từng lọt lưới đều
    nằm ở phần ĐUÔI — sau khi GPU đã chạy xong. Ba lỗi thật đã gặp: gọi `ea()` chưa
    định nghĩa, `json.dump` với khoá tuple, và vẽ biểu đồ trên tuple. Cả ba chỉ lộ ra
    khi thực thi, và lộ ra đúng lúc đắt nhất.
    """
    print("═" * 78)
    print("  8. NOTEBOOK 06 — chạy phần phân tích với ma trận dựng sẵn")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import random
    import traceback
    from vinumqa import data, dsl, pipeline, stats

    test = data.load_all(os.path.join(GOC, "data"))["test"]
    rng = random.Random(11)
    ra = os.path.join(GOC, "runs")
    os.makedirs(ra, exist_ok=True)

    def o_ma_tran(ty_le):
        rows = []
        for s in test:
            gold = s["qa"].get("program") or ""
            prog = gold if rng.random() < ty_le else "add(abc, 2)"
            val = dsl.execute_program(prog, s.get("table") or []) if prog else None
            ea = dsl.check_ea(val, s["qa"].get("exe_ans")) if prog else False
            pa_s, pa_l = dsl.check_pa(prog, gold) if prog and gold else (False, False)
            rows.append({"id": s["id"], "question": s["qa"]["question"],
                         "gold_program": gold, "gold_answer": s["qa"].get("exe_ans"),
                         "program_step1": prog, "program_step2": "",
                         "final_program": prog, "pred_value": val,
                         "ea": ea, "ea_tol1e-3": ea, "pa_strict": pa_s, "pa_loose": pa_l,
                         "n_ops_gold": dsl.n_ops(gold),
                         "outcome": dsl.classify_outcome(ea, pa_s, prog, val),
                         "used_bullets": [], "bullets_text": ""})
        return rows

    MATRIX = [("E", False, False, False, "02_prompt_eng"),
              ("E+A", False, False, True, "06_comb_E_A"),
              ("E+S", False, True, False, "04_selfeval_base"),
              ("E+S+A", False, True, True, "05_ace_base"),
              ("F", True, False, False, "03_sft"),
              ("F+A", True, False, True, "06_comb_F_A"),
              ("F+S", True, True, False, "04_selfeval_sft"),
              ("F+S+A", True, True, True, "05_ace_sft")]
    NICE = {"E": "prompt", "E+A": "prompt+ACE", "E+S": "prompt+selfeval",
            "E+S+A": "prompt+selfeval+ACE", "F": "SFT", "F+A": "SFT+ACE",
            "F+S": "SFT+selfeval", "F+S+A": "SFT+selfeval+ACE"}
    RESULTS = {}
    for i, (c, *_r) in enumerate(MATRIX):
        rows = o_ma_tran(0.60 + 0.02 * i)
        m = pipeline.summarize(rows, NICE[c])
        m["minutes"] = 20.0
        RESULTS[c] = (rows, m, _r[-1])

    ns = {"__name__": "__main__", "MATRIX": MATRIX, "NICE": NICE, "RESULTS": RESULTS,
          "test_all": test, "stats": stats, "pipeline": pipeline, "dsl": dsl,
          "OUTPUT_DIR": ra, "STAMP": "kiemtra", "MUC_PROMPT": "engineered",
          "os": os, "json": json, "csv": __import__("csv")}
    # Chỉ các ô PHÂN TÍCH: từ ô dựng bảng ma trận trở đi, bỏ ô cần model/playbook.
    _nb, o = _o_code(os.path.join(NBDIR, "06_combination.ipynb"))
    n_o = 0
    for k, _c, src in o:
        if k < 16:                       # ô 0–15 cần vLLM, Drive, playbook
            continue
        n_o += 1
        try:
            exec(compile(_sang_python(src), f"06#{k}", "exec"), ns)
        except Exception:                                # noqa: BLE001
            loi.append(f"06#{k} không chạy được:\n{traceback.format_exc(limit=4)}")
    for f in (glob.glob(os.path.join(ra, "ma_tran*"))
              + glob.glob(os.path.join(ra, "tuong_tac_*.json"))):
        os.remove(f)
    print(f"   {n_o} ô phân tích đã chạy, đã dọn file tạm")


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
    kiem_ten_notebook()
    kiem_runbook()
    if args.day_du:
        kiem_bao_cao_day_du()
        kiem_ma_tran_06()

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
