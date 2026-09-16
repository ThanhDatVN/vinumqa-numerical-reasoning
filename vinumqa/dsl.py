# -*- coding: utf-8 -*-
"""Bộ thực thi DSL của ViNumQA và các hàm chấm PA / EA.

Đây là phần quan trọng nhất về độ chính xác của toàn bộ dự án: nếu executor sai thì
mọi con số PA/EA đều vô nghĩa, và Reflector của ACE sẽ học từ tín hiệu nhiễu.

Khác biệt so với executor FinQA gốc trong ``ace/02_ace_finqa.ipynb`` — đã kiểm chứng
trên toàn bộ 4.074 mẫu ViNumQA:

* **Không có hằng số ``const_*``** — ViNumQA dùng số literal thuần (0/4074 gold có ``const_``).
* **``table_*(nhãn, none)`` đọc theo NHÃN HÀNG** ở cột đầu tiên, không phải tên cột
  (456/456 tham số ``table_*`` trong train khớp cột 0, 0 khớp header).
* **Ô bảng tiếng Việt**: ``(79)`` → −79, ``—``/``n/a`` → bỏ qua,
  ``26% ( 26 % )`` → 0.26, ``22.0x`` → 22.0, ``$ 1,234`` → 1234.
* **Toán hạng ``20%``** → 0.2 (61 gold program trong train dùng dạng này).
* **Fail-closed**: program dị dạng, chia 0, tham chiếu ``#N`` tiến/ngoài phạm vi → trả ``None``.

Tỉ lệ tái tạo ``exe_ans`` từ gold program: train 99.80 %, valid 99.83 %, test 100 %.
"""
from __future__ import annotations

import math
import re

__all__ = [
    "execute_program", "extract_program_answer",
    "normalize_program_strict", "normalize_program_loose",
    "check_pa", "check_ea", "n_ops", "first_op", "classify_outcome",
    "OP_NAMES", "OP_DETECT_RE", "OP_ARGS_RE", "EA_DECIMAL_PLACES",
]

EA_DECIMAL_PLACES = 5

_BS = chr(92)
_DASH_CELLS = {"", "-", "--", "---", chr(8212), chr(8211), "n/a", "na", "n.a", "none", "nm", "..."}
_NUMERIC_GOLD_RE = re.compile(r"^-?[0-9.]+(?:[eE][-+]?[0-9]+)?$")

OP_NAMES = ("add", "subtract", "multiply", "divide", "exp", "greater",
            "table_max", "table_min", "table_sum", "table_average")
OP_DETECT_RE = re.compile(r"\b(" + "|".join(OP_NAMES) + r")\s*\(", re.I)
OP_ARGS_RE = re.compile(r"(" + "|".join(OP_NAMES) + r")\s*\(([^)]*)\)", re.I)

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


# ─────────────────────────────── phân tích cú pháp ───────────────────────────────

def split_dsl_items(text: str) -> list[str]:
    """Tách danh sách ngăn bởi dấu phẩy, tôn trọng ngoặc và dấu nháy."""
    items, start, depth, quote = [], 0, 0, None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote and (i == 0 or text[i - 1] != _BS):
                quote = None
        elif ch in {"'", '"'}:
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("thừa dấu đóng ngoặc")
        elif ch == "," and depth == 0:
            items.append(text[start:i].strip())
            start = i + 1
    if quote or depth:
        raise ValueError("thiếu dấu đóng ngoặc/nháy")
    items.append(text[start:].strip())
    return items


def parse_numeric_literal(token) -> float:
    """``'$ 1,234'`` → 1234.0 | ``'(79)'`` → −79.0 | ``'20%'`` → 0.2 | ``'22.0x'`` → 22.0"""
    t = str(token).strip().replace(chr(8722), "-").replace(chr(8211), "-")
    negative = t.startswith("(") and t.endswith(")")
    if negative:
        t = t[1:-1].strip()
    t = re.sub(r"^[$€£¥]\s*", "", t).replace(",", "").replace(" ", "")
    t = re.sub(r"x$", "", t, flags=re.I)
    is_percent = t.endswith("%")
    if is_percent:
        t = t[:-1].strip()
    value = float(t)
    if not math.isfinite(value):
        raise ValueError("số không hữu hạn")
    if negative:
        value = -abs(value)
    return value / 100.0 if is_percent else value


def cell_to_number(cell):
    """Trả về float của một ô bảng, hoặc ``None`` nếu ô đó không phải số."""
    t = str(cell).strip()
    if t.casefold() in _DASH_CELLS:
        return None
    t = re.sub(r"\([^()]*%[^()]*\)", "", t).strip()          # '26% ( 26 % )' → '26%'
    m = re.match(r"^[\s$€£¥]*\(?\s*[-+]?[0-9][0-9,.]*\s*\)?\s*[%x]?", t, re.I)
    if not m:
        return None
    tok = re.sub(r"[.,]+$", "", m.group(0).strip())
    if tok.startswith("(") and not tok.endswith(")"):
        tok += ")"
    try:
        return parse_numeric_literal(tok)
    except (TypeError, ValueError):
        return None


def _norm_label(s) -> str:
    return " ".join(str(s).casefold().split())


def table_row_values(table, label) -> list[float]:
    """Lấy dãy số của HÀNG có nhãn ``label`` ở cột đầu tiên.

    Khớp chính xác trước; nếu không có thì khớp chứa-nhau (cho phép model viết tắt nhãn).

    Khi nhiều hàng cùng khớp: chỉ chấp nhận nếu **mọi hàng khớp đều cho cùng một dãy số**
    — bảng trong ViNumQA thỉnh thoảng lặp lại nguyên một hàng (vd ``PNC/2013/page_158.pdf-1``
    có hàng "tổng tdr" xuất hiện hai lần với số liệu y hệt), và khi đó không hề mơ hồ.
    Nếu các hàng khớp cho số liệu KHÁC nhau thì fail-closed, không đoán bừa.
    """
    if not isinstance(table, list) or len(table) < 2:
        raise ValueError("không có bảng")
    want = _norm_label(label)
    rows = [r for r in table[1:] if isinstance(r, list) and r]

    matches = [r for r in rows if _norm_label(r[0]) == want]
    if not matches:
        matches = [r for r in rows
                   if want and (want in _norm_label(r[0]) or _norm_label(r[0]) in want)]
    if not matches:
        raise ValueError("không có hàng nào khớp nhãn")

    value_sets = []
    for row in matches:
        vals = [v for v in (cell_to_number(c) for c in row[1:]) if v is not None]
        if vals:
            value_sets.append(vals)
    if not value_sets:
        raise ValueError("hàng không có giá trị số")
    if any(vs != value_sets[0] for vs in value_sets[1:]):
        raise ValueError("nhãn hàng khớp nhiều hàng có số liệu khác nhau")
    return value_sets[0]


_TABLE_FUNCS = {
    "table_max": max,
    "table_min": min,
    "table_sum": sum,
    "table_average": lambda v: sum(v) / len(v),
}


# ─────────────────────────────── thực thi ───────────────────────────────

def execute_program(program, table):
    """Thực thi program DSL. Trả ``None`` nếu bất hợp lệ (fail-closed)."""
    if not isinstance(program, str) or not program.strip():
        return None
    results = []

    def resolve(token):
        token = token.strip()
        if re.fullmatch(r"#\d+", token):
            idx = int(token[1:])
            if idx >= len(results) or isinstance(results[idx], str):
                raise ValueError("tham chiếu #N không hợp lệ")
            return float(results[idx])
        return parse_numeric_literal(token)

    try:
        for command in split_dsl_items(program.strip()):
            m = re.fullmatch(r"([a-z_]+)\s*\((.*)\)", command, re.I | re.S)
            if not m:
                raise ValueError("phép toán dị dạng")
            op, args = m.group(1).casefold(), split_dsl_items(m.group(2))
            if len(args) != 2:
                raise ValueError("phép toán cần đúng 2 tham số")
            left, right = args
            if op.startswith("table_"):
                if right.strip().casefold() != "none":
                    raise ValueError("table_* cần tham số thứ hai là none")
                if op not in _TABLE_FUNCS:
                    raise ValueError("table_* không hỗ trợ")
                vals = table_row_values(table, left.strip().strip('"').strip("'"))
                out = float(_TABLE_FUNCS[op](vals))
            else:
                a, b = resolve(left), resolve(right)
                if op == "add":
                    out = a + b
                elif op == "subtract":
                    out = a - b
                elif op == "multiply":
                    out = a * b
                elif op == "divide":
                    if b == 0:
                        raise ValueError("chia cho 0")
                    out = a / b
                elif op == "exp":
                    out = a ** b
                elif op == "greater":
                    out = "yes" if a > b else "no"
                else:
                    raise ValueError("phép toán không hỗ trợ")
            if isinstance(out, float) and not math.isfinite(out):
                raise ValueError("kết quả không hữu hạn")
            results.append(out)
    except (ArithmeticError, OverflowError, TypeError, ValueError, IndexError):
        return None
    return results[-1] if results else None


# ─────────────────────────────── trích xuất ───────────────────────────────

def extract_program_answer(text):
    """Trả ``(program, answer_text)``. Ưu tiên khối ```plaintext cuối cùng."""
    if not text:
        return None, None
    clean = _THINK_RE.sub("", text)
    # Qwen3 ở chế độ suy nghĩ không phải lúc nào cũng nhả đủ cặp thẻ:
    #   • chỉ có </think>  → template đã mở sẵn <think>, phần sinh ra bắt đầu từ suy luận
    #   • chỉ có <think>   → bị cắt vì chạm max_tokens, không có câu trả lời chốt
    # Trường hợp đầu phải bỏ phần suy luận đi; trường hợp sau đúng là "không sinh được".
    if "</think>" in clean:
        clean = clean.rsplit("</think>", 1)[-1]
    if "assistant" in clean[:200].lower():
        clean = clean.split("assistant", 1)[-1]
    clean = clean.strip()

    prog = ans = None
    blocks = re.findall(r"```(?:plaintext|text|python)?\s*\n?(.*?)\n?```", clean, re.S | re.I)
    for block in reversed(blocks):                    # khối cuối là câu trả lời chốt
        b = block.strip()
        m = re.search(r"program\s*:\s*(.+?)(?:\n|$)", b, re.I)
        if m:
            prog = m.group(1).strip()
        elif OP_DETECT_RE.search(b):
            for line in b.split("\n"):
                if OP_DETECT_RE.search(line):
                    prog = line.strip()
                    break
        m = re.search(r"(?:answer|output)\s*:\s*(.+?)(?:\n|$)", b, re.I)
        if m:
            ans = m.group(1).strip()
        if prog:
            break

    if prog is None:
        m = re.search(r"program\s*:\s*(.+?)(?:\n|```|$)", clean, re.I)
        if m:
            prog = m.group(1).strip()
    if ans is None:
        m = re.search(r"(?:answer|output)\s*:\s*(.+?)(?:\n|```|$)", clean, re.I)
        if m:
            ans = m.group(1).strip()

    if prog:
        prog = re.sub(r"^[`\s]+|[`\s}\]]+$", "", prog).strip()
        prog = re.sub(r"\s*(?:answer|output)\s*:.*$", "", prog, flags=re.I).strip()
        if not OP_DETECT_RE.search(prog):
            prog = None
    if ans:
        ans = re.sub(r"[`*]+", "", ans).strip().rstrip(".")
    return (prog or None), (ans or None)


# ─────────────────────────────── chuẩn hoá & chấm điểm ───────────────────────────────

def _canonical_operand(token: str, step_index: int) -> str:
    token = token.strip()
    if re.fullmatch(r"#\d+", token):
        ref = int(token[1:])
        if ref >= step_index:
            raise ValueError("tham chiếu tiến hoặc ngoài phạm vi")
        return f"#{ref}"
    value = parse_numeric_literal(token)
    return str(int(value)) if value == int(value) else f"{value:.10f}".rstrip("0").rstrip(".")


def normalize_program_strict(program) -> str:
    """Canonical hoá giữ nguyên ngữ nghĩa; trả ``''`` nếu program không hợp lệ.

    Giao hoán ``add``/``multiply``, chuẩn hoá ``100.00`` → ``100`` và ``20%`` → ``0.2``,
    kiểm tra ``#N`` hợp lệ.
    """
    if not isinstance(program, str) or not program.strip():
        return ""
    canonical = []
    try:
        for i, command in enumerate(split_dsl_items(program.strip())):
            m = re.fullmatch(r"([a-z_]+)\s*\((.*)\)", command, re.I | re.S)
            if not m:
                raise ValueError("dị dạng")
            op, args = m.group(1).casefold(), split_dsl_items(m.group(2))
            if len(args) != 2:
                raise ValueError("số tham số")
            if op.startswith("table_"):
                if op not in _TABLE_FUNCS:
                    raise ValueError("table_* lạ")
                left, right = _norm_label(args[0].strip().strip('"').strip("'")), "none"
            else:
                if op not in {"add", "subtract", "multiply", "divide", "exp", "greater"}:
                    raise ValueError("phép toán lạ")
                left, right = _canonical_operand(args[0], i), _canonical_operand(args[1], i)
                if op in {"add", "multiply"}:
                    left, right = sorted((left, right))
            canonical.append(f"{op}({left},{right})")
    except (TypeError, ValueError):
        return ""
    return ",".join(canonical)


def normalize_program_loose(prog) -> str:
    """Bản sao công thức trong ``calculator/pa_ea_calculator.py``.

    Giữ lại để ``PA_loose`` so sánh trực tiếp được với bảng kết quả tham chiếu
    (đã kiểm chứng: tái lập đúng 59.56 % cho Qwen3-8B, 38.63 % cho Mistral-7B).
    """
    if not prog:
        return ""
    prog = re.sub(r"\s+", "", str(prog).lower())
    prog = re.sub(r"add\(([^,]+),([^)]+)\)",
                  lambda m: "add(" + ",".join(sorted([m.group(1), m.group(2)])) + ")", prog)
    prog = re.sub(r"multiply\(([^,]+),([^)]+)\)",
                  lambda m: "multiply(" + ",".join(sorted([m.group(1), m.group(2)])) + ")", prog)
    prog = re.sub(r"table_(max|min|average|sum)\(([^,]+),none\)", r"table_\1(\2,none)", prog)
    refs = []

    def renumber(m):
        ref = m.group(0)
        if ref not in refs:
            refs.append(ref)
        return f"#{refs.index(ref)}"

    return re.sub(r"#\d+", renumber, prog)


def check_pa(pred_prog, gold_prog) -> tuple[bool, bool]:
    """Trả ``(pa_strict, pa_loose)``."""
    ps, gs = normalize_program_strict(pred_prog), normalize_program_strict(gold_prog)
    strict = bool(ps) and ps == gs
    pl, gl = normalize_program_loose(pred_prog), normalize_program_loose(gold_prog)
    loose = bool(pl) and pl == gl
    return strict, loose


def check_ea(pred_value, gold_answer, decimals=EA_DECIMAL_PLACES, abs_tol=None) -> bool:
    """So kết quả thực thi với ``exe_ans`` (là chuỗi, đã làm tròn 5 chữ số trong ViNumQA)."""
    if pred_value is None or gold_answer is None:
        return False
    gold_text = str(gold_answer).strip()
    if isinstance(pred_value, str) or not _NUMERIC_GOLD_RE.match(gold_text):
        return str(pred_value).strip().casefold() == gold_text.casefold()
    try:
        p, g = float(pred_value), float(gold_text)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(p) and math.isfinite(g)):
        return False
    if abs_tol is not None:
        return abs(p - g) <= abs_tol
    if round(p, decimals) == round(g, decimals):
        return True
    if abs(p - g) <= 10 ** (-decimals):
        return True
    return g != 0 and abs(p - g) / abs(g) <= 1e-6


def n_ops(prog) -> int:
    return len(OP_DETECT_RE.findall(prog or ""))


def first_op(prog) -> str:
    m = OP_DETECT_RE.search((prog or "").lower())
    return m.group(1).lower() if m else ""


def classify_outcome(ea, pa, pred_prog, pred_value) -> str:
    if not pred_prog:
        return "khong_co_program"
    if pred_value is None:
        return "program_khong_chay_duoc"
    if ea and pa:
        return "dung"
    if ea and not pa:
        return "dung_nhung_khac_program"          # lucky guess
    if pa and not ea:
        return "program_dung_ket_qua_lech"
    return "sai"
