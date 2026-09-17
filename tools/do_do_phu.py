# -*- coding: utf-8 -*-
"""Đo ĐỘ PHỦ: bộ kiểm --day-du thực sự chạy qua bao nhiêu dòng của 06 và 07.

Dòng nào không chạy = nhánh chưa được kiểm = chỗ lỗi còn nấp.
"""
import io
import json
import os
import sys
import trace as _trace

GOC = os.path.abspath(".")
sys.path.insert(0, GOC)
sys.argv = ["kiem_tra", "--day-du"]

# nạp kiem_tra như một module, rồi chạy main() dưới tracer
import importlib.util
spec = importlib.util.spec_from_file_location("kt", os.path.join(GOC, "tools", "kiem_tra.py"))
kt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kt)

dem = {}


def tracer(frame, event, arg):
    if event == "line":
        fn = frame.f_code.co_filename
        if fn.startswith("06#") or fn.startswith("07#"):
            dem.setdefault(fn, set()).add(frame.f_lineno)
    return tracer


_out = io.StringIO()
_cu = sys.stdout
sys.stdout = _out
sys.settrace(tracer)
try:
    kt.kiem_bao_cao_day_du()
    kt.kiem_ma_tran_06()
finally:
    sys.settrace(None)
    sys.stdout = _cu

# so với số dòng THỰC (bỏ dòng trống và comment) của từng ô
def dong_thuc(src):
    ra = set()
    for i, l in enumerate(src.splitlines(), 1):
        t = l.strip()
        if t and not t.startswith("#"):
            ra.add(i)
    return ra


print(f"{'ô':<10}{'dòng thực':>11}{'đã chạy':>10}{'phủ':>8}   chưa chạy")
print("-" * 78)
tong_t = tong_c = 0
for ten_nb, so in (("06_combination", "06"), ("07_final_report", "07")):
    nb = json.load(open(f"notebooks/{ten_nb}.ipynb", encoding="utf-8"))
    for k, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        key = f"{so}#{k}"
        if key not in dem:
            continue
        src = "".join(c["source"])
        thuc = dong_thuc(src)
        chay = dem[key] & thuc
        thieu = sorted(thuc - chay)
        tong_t += len(thuc)
        tong_c += len(chay)
        pct = len(chay) / len(thuc) * 100 if thuc else 100
        note = ""
        if thieu:
            note = f"{len(thieu)} dòng, vd dòng {thieu[:5]}"
        print(f"{key:<10}{len(thuc):>11}{len(chay):>10}{pct:>7.0f}%   {note}")
print("-" * 78)
print(f"{'TỔNG':<10}{tong_t:>11}{tong_c:>10}{tong_c/tong_t*100:>7.0f}%")
print()
print("Ô không xuất hiện = ô cần GPU, bộ kiểm cố ý bỏ qua.")
