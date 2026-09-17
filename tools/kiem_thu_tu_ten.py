# -*- coding: utf-8 -*-
"""Tên dùng ở ô i phải được định nghĩa ở ô ≤ i. Phép kiểm 5 chỉ hỏi 'có ở đâu đó
trong notebook không' — dùng TRƯỚC khi định nghĩa vẫn lọt."""
import ast
import builtins
import glob
import json
import os
import sys

CO_SAN = set(dir(builtins)) | {
    "get_ipython", "display", "In", "Out", "_ih", "_oh", "__name__", "__file__"}


class Thu(ast.NodeVisitor):
    def __init__(self):
        self.gan, self.dung = set(), []

    def visit_Name(self, n):
        (self.gan.add(n.id) if isinstance(n.ctx, ast.Store)
         else self.dung.append(n.id))

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


def sang_py(src):
    ra = []
    for l in src.splitlines():
        t = l.lstrip()
        ra.append(l[:len(l) - len(t)] + "pass" if t.startswith(("!", "%")) else l)
    return "\n".join(ra)


tong = 0
for p in sorted(glob.glob("notebooks/*.ipynb")):
    nb = json.load(open(p, encoding="utf-8"))
    o = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
    van = "\n".join(o)
    co = set(CO_SAN)
    bao = []
    for i, src in enumerate(o, 1):
        try:
            cay = ast.parse(sang_py(src))
        except SyntaxError:
            continue
        t = Thu()
        t.visit(cay)
        # Tên gán TRONG CÙNG ô cũng tính là đã có — trong một ô, thứ tự dòng do Python
        # lo, ta chỉ hỏi "ô này có tự định nghĩa nó không".
        co_ca_o = co | t.gan
        for ten in t.dung:
            if ten in co_ca_o:
                continue
            if f'"{ten}" in globals()' in van or f"'{ten}' in globals()" in van:
                continue
            bao.append((i, ten))
        co |= t.gan
    # gom theo tên, chỉ báo lần ĐẦU dùng
    thay = {}
    for i, ten in bao:
        thay.setdefault(ten, i)
    if thay:
        print(f"\n### {os.path.basename(p)}")
        for ten, i in sorted(thay.items(), key=lambda x: x[1]):
            print(f"    ô #{i:<3} dùng `{ten}` TRƯỚC khi ô nào định nghĩa")
            tong += 1

print(f"\n{'✅ không có tên nào dùng trước khi định nghĩa' if not tong else f'⛔ {tong} chỗ'}")
sys.exit(1 if tong else 0)
