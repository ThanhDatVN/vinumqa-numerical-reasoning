# -*- coding: utf-8 -*-
"""Dựng bộ hình cho tài liệu từ kết quả đã lưu.

    python tools/ve_bieu_do.py [--nguon results/stages] [--ra docs/hinh]

Đọc thẳng file nấc (`*.jsonl` + `*_meta.json`) và tính lại mọi con số, nên hình luôn
khớp số trong tài liệu. Không cần GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GOC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, GOC)

# Bảng màu dịu, phân biệt được cả khi in đen trắng nhờ khác độ đậm.
XANH, XAM, DO, CAM, TIM = "#2f6f9f", "#b0b7bd", "#a63d40", "#d9a441", "#6b5b95"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.autolayout": False,
})


def nap(nguon):
    nac, meta = {}, {}
    for f in sorted(os.listdir(nguon)):
        if not f.endswith(".jsonl"):
            continue
        t = f[:-6]
        nac[t] = [json.loads(l) for l in open(os.path.join(nguon, f), encoding="utf-8")
                  if l.strip()]
        mp = os.path.join(nguon, f"{t}_meta.json")
        if os.path.exists(mp):
            meta[t] = json.load(open(mp, encoding="utf-8"))
    return nac, meta


def mcnemar(x, y, key="ea"):
    """Trả (Δ điểm, hỏng, sửa, ngưỡng 95 %). Ngưỡng phụ thuộc số mẫu bất đồng."""
    n = len(x)
    b = sum(1 for p, q in zip(x, y) if p[key] and not q[key])
    c = sum(1 for p, q in zip(x, y) if q[key] and not p[key])
    ng = 1.96 * math.sqrt(b + c) / n * 100 if b + c else 0.0
    return (c - b) / n * 100, b, c, ng


def _ghi(fig, ra, ten):
    p = os.path.join(ra, ten)
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p}")


# ═══════════════════════════════ 1. thang bậc ═══════════════════════════════
def hinh_thang_bac(nac, meta, ra):
    thu_tu = ["01_basic", "02_prompt_eng", "03_sft", "04_selfeval_base", "05_ace_base",
              "06_comb_E_A", "08_tu_nhat_quan", "09_vidu_dong", "10_bo_chon"]
    nhan = ["prompt\ncơ bản", "+ prompt\nengineering", "+ SFT", "+ self-eval",
            "+ ACE", "ACE không\nself-eval", "+ self-\nconsistency",
            "+ ví dụ\ntruy hồi", "+ bộ\nchọn"]
    co = [s for s in thu_tu if s in nac]
    nhan = [n for s, n in zip(thu_tu, nhan) if s in nac]
    ea = [meta[s]["metrics"]["EA"] * 100 for s in co]
    pa = [meta[s]["metrics"]["PA_strict"] * 100 for s in co]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = range(len(co))
    w = 0.38
    mau = [XANH if s != co[-1] else "#1b4f72" for s in co]
    ax.bar([i - w / 2 for i in x], ea, w, color=mau, label="EA")
    ax.bar([i + w / 2 for i in x], pa, w, color=XAM, label="PA_strict")
    for i, (e, p) in enumerate(zip(ea, pa)):
        ax.text(i - w / 2, e + 1.1, f"{e:.1f}", ha="center", fontsize=7.5)
        ax.text(i + w / 2, p + 1.1, f"{p:.1f}", ha="center", fontsize=7.5, color="#555")
    ax.axhline(70, color=DO, lw=0.9, ls="--", zorder=0)
    ax.text(len(co) - 0.42, 71.2, "mục tiêu 70 %", color=DO, fontsize=7.5, ha="right")
    ax.set_xticks(list(x))
    ax.set_xticklabels(nhan, fontsize=7.5)
    ax.set_ylim(0, 92)
    ax.set_ylabel("%")
    ax.set_title(f"Thang bậc — Qwen3-8B, ViNumQA test ({len(nac[co[0]])} mẫu)")
    ax.legend(loc="upper left", fontsize=8)
    _ghi(fig, ra, "01_thang_bac.png")


# ═══════════════════════ 2. đóng góp từng kỹ thuật ═══════════════════════
def hinh_dong_gop(nac, ra):
    CAP = [("Prompt engineering", "01_basic", "02_prompt_eng"),
           ("ACE — nội dung playbook\n(trên prompt cơ bản)", "01_basic",
            "05c_ace_basic_random_base"),
           ("Self-consistency K=5", "02_prompt_eng", "08_tu_nhat_quan"),
           ("Ví dụ truy hồi kNN", "08_tu_nhat_quan", "09_vidu_dong"),
           ("ACE — cơ chế truy hồi", "05_ace_random_base", "05_ace_base"),
           ("SFT trên dữ liệu tự sinh", "02_prompt_eng", "03_sft"),
           ("Self-eval", "02_prompt_eng", "04_selfeval_base"),
           ("Bộ chọn", "09_vidu_dong", "10_bo_chon"),
           ("ACE — nội dung playbook\n(trên prompt đầy đủ)", "04_selfeval_base",
            "05_ace_random_base"),
           ("Self-eval trên cấu\nhình tốt nhất", "09_vidu_dong", "04_selfeval_base_moi")]
    d = [(t, *mcnemar(nac[a], nac[b])) for t, a, b in CAP if a in nac and b in nac]
    d.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    y = range(len(d))
    for i, (t, dd, _b, _c, ng) in enumerate(d):
        vuot = abs(dd) > ng
        ax.barh(i, dd, 0.62, color=(XANH if dd > 0 else DO) if vuot else XAM,
                zorder=3)
        ax.plot([ng, ng], [i - 0.34, i + 0.34], color="#333", lw=1.1, zorder=4)
        ax.plot([-ng, -ng], [i - 0.34, i + 0.34], color="#333", lw=1.1, zorder=4)
        # Đặt nhãn ra NGOÀI cả cột lẫn vạch ngưỡng, nếu không hai thứ đè nhau ở
        # những cặp có hiệu ứng nhỏ hơn ngưỡng — đúng nhóm cần đọc kỹ nhất.
        mep = max(abs(dd), ng) + 0.5
        ax.text(mep if dd >= 0 else -mep, i, f"{dd:+.2f}",
                va="center", ha="left" if dd >= 0 else "right", fontsize=8,
                fontweight="bold" if vuot else "normal")
    ax.axvline(0, color="#333", lw=0.9)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in d], fontsize=8)
    ax.set_xlabel("Δ EA (điểm phần trăm)")
    ax.set_title("Đóng góp của từng kỹ thuật, so với nấc nó xây lên")
    ax.set_xlim(-6, 29)
    ax.text(0.985, 0.045,
            "vạch dọc = ngưỡng ý nghĩa 95 % của riêng cặp đó\n"
            "cột đậm = vượt ngưỡng · cột xám = chưa tách được khỏi biến thiên đo",
            transform=ax.transAxes, ha="right", fontsize=7.3, color="#444")
    _ghi(fig, ra, "02_dong_gop.png")


# ═══════════════ 3. đường cong số mẫu và trần best-of-K ═══════════════
def hinh_duong_cong_k(nac, ra):
    from vinumqa import data, pipeline
    if "09_vidu_dong" not in nac:
        return
    test = data.load_all(os.path.join(GOC, "data"))["test"]
    r9 = nac["09_vidu_dong"]
    ks = list(range(1, (r9[0].get("k_da_sinh") or 5) + 1))
    ea = [pipeline.summarize(pipeline.tu_nhat_quan(r9, test, k), "")["EA"] * 100 for k in ks]
    tran = pipeline.tran_best_of_k(r9)["EA_tran"] * 100

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(ks, ea, "o-", color=XANH, lw=2, ms=6, label="bỏ phiếu theo giá trị", zorder=3)
    ax.axhline(tran, color=XAM, lw=1.6, ls="--", label=f"trần best-of-{ks[-1]} ({tran:.2f})")
    if "10_bo_chon" in nac:
        bc = sum(r["ea"] for r in nac["10_bo_chon"]) / len(nac["10_bo_chon"]) * 100
        ax.plot([ks[-1]], [bc], "D", color=DO, ms=7, zorder=4, label=f"bộ chọn ({bc:.2f})")
    ax.fill_between([ks[0], ks[-1]], ea[-1], tran, color=CAM, alpha=0.13, zorder=0)
    ax.annotate(f"{tran - ea[-1]:.2f} điểm\nchưa khai thác",
                xy=(ks[-1] - 0.9, (ea[-1] + tran) / 2), fontsize=8, color="#8a6d2f",
                ha="center", va="center")
    for k, v in zip(ks, ea):
        ax.text(k, v - 1.25, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(ks)
    ax.set_xlabel("số mẫu k")
    ax.set_ylabel("EA (%)")
    ax.set_ylim(min(ea) - 3, tran + 2.2)
    ax.set_title("Tăng số mẫu bão hoà; khoảng trống còn lại là bài toán chọn")
    ax.legend(loc="lower right", fontsize=8)
    _ghi(fig, ra, "03_duong_cong_k.png")


# ═══════════════════ 4. ACE: nội dung vs cơ chế truy hồi ═══════════════════
def hinh_ace(nac, meta, ra):
    bo = [("prompt cơ bản", "01_basic", "05c_ace_basic_random_base", "05c_ace_basic_base"),
          ("prompt đầy đủ", "04_selfeval_base", "05_ace_random_base", "05_ace_base")]
    bo = [b for b in bo if all(x in nac for x in b[1:])]
    if not bo:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))

    nh, nd, tr = [], [], []
    for ten, moc, ran, ace in bo:
        nh.append(ten)
        nd.append(mcnemar(nac[moc], nac[ran])[0])
        tr.append(mcnemar(nac[ran], nac[ace])[0])
    x = range(len(nh))
    w = 0.34
    ax1.bar([i - w / 2 for i in x], nd, w, color=XANH, label="nội dung playbook")
    ax1.bar([i + w / 2 for i in x], tr, w, color=XAM, label="cơ chế truy hồi")
    for i, (a, b) in enumerate(zip(nd, tr)):
        ax1.text(i - w / 2, a + (0.5 if a >= 0 else -1.4), f"{a:+.2f}",
                 ha="center", fontsize=8, fontweight="bold")
        ax1.text(i + w / 2, b + 0.5, f"{b:+.2f}", ha="center", fontsize=8)
    ax1.axhline(0, color="#333", lw=0.9)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(nh)
    ax1.set_ylabel("Δ EA (điểm)")
    ax1.set_title("ACE: nội dung học được và cơ chế truy hồi")
    ax1.legend(fontsize=8, loc="upper right")

    for ten, f, mau in (("prompt cơ bản", "05c_ace_basic_base", XANH),
                        ("prompt đầy đủ", "05_ace_base", XAM)):
        if f not in meta:
            continue
        el = meta[f].get("eval_log") or []
        v = [e.get("composite", 0) for e in el]
        ax2.plot(range(len(v)), v, "o-", color=mau, lw=1.9, ms=5, label=ten)
    ax2.set_xlabel("mốc đo trên tập dev (mỗi 5 vòng)")
    ax2.set_ylabel("điểm tổng hợp 0,6·EA + 0,4·PA")
    ax2.set_title("Đường học: tăng đơn điệu hay dao động")
    ax2.legend(fontsize=8, loc="lower right")
    _ghi(fig, ra, "04_ace.png")


# ══════════════════════════ 5. phân loại lỗi ══════════════════════════
def hinh_loi(meta, ra):
    thu_tu = [s for s in ("02_prompt_eng", "03_sft", "04_selfeval_base", "05_ace_base",
                          "08_tu_nhat_quan", "09_vidu_dong", "10_bo_chon") if s in meta]
    nhan = {"dung_phep_sai_so_lieu": "đúng phép, sai số liệu", "thua_buoc": "thừa bước",
            "thieu_buoc": "thiếu bước", "dung_so_buoc_sai_phep": "đúng số bước, sai phép",
            "lam_dung_table": "lạm dụng table_*", "bo_qua_table": "bỏ qua table_*"}
    mau = {"dung_phep_sai_so_lieu": DO, "thua_buoc": CAM, "thieu_buoc": TIM,
           "dung_so_buoc_sai_phep": XANH, "lam_dung_table": "#7fa8c9",
           "bo_qua_table": XAM}
    n = meta[thu_tu[0]]["metrics"]["n"]

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    duoi = [0.0] * len(thu_tu)
    for k in nhan:
        v = [(meta[s]["metrics"].get("vi_sao_sai") or {}).get("theo_kieu", {}).get(k, 0)
             / n * 100 for s in thu_tu]
        ax.bar(range(len(thu_tu)), v, 0.62, bottom=duoi, color=mau[k], label=nhan[k])
        duoi = [a + b for a, b in zip(duoi, v)]
    for i, t in enumerate(duoi):
        ax.text(i, t + 0.5, f"{t:.1f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(thu_tu)))
    ax.set_xticklabels([s.split("_")[0] for s in thu_tu])
    ax.set_ylabel("% tổng số mẫu")
    ax.set_title("Cơ cấu lỗi qua thang bậc — lỗi đọc số liệu không giảm")
    ax.legend(fontsize=7.6, ncol=2, loc="upper right")
    ax.set_ylim(0, max(duoi) * 1.32)
    _ghi(fig, ra, "05_phan_loai_loi.png")


# ═════════════════════════════ 6. bộ chọn ═════════════════════════════
def hinh_bo_chon(nac, ra):
    from vinumqa import pipeline
    if "09_vidu_dong" not in nac or "10_bo_chon" not in nac:
        return
    r9, r10 = nac["09_vidu_dong"], nac["10_bo_chon"]
    uv = [pipeline.nhom_ung_vien(r) for r in r9]
    can = [i for i, u in enumerate(uv) if len(u) >= 2]
    mot = [i for i in range(len(r9)) if i not in set(can)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0),
                                   gridspec_kw={"width_ratios": [1, 1.25]})
    from collections import Counter
    c = Counter(len(u) for u in uv)
    ks = sorted(c)
    ax1.bar([str(k) for k in ks], [c[k] for k in ks],
            color=[XAM if k < 2 else XANH for k in ks], width=0.62)
    for i, k in enumerate(ks):
        ax1.text(i, c[k] + 6, str(c[k]), ha="center", fontsize=8, fontweight="bold")
    ax1.set_xlabel("số giá trị phân biệt trong 5 mẫu")
    ax1.set_ylabel("số câu")
    ax1.set_title("Ba phần tư số câu không có gì để chọn")
    ax1.set_ylim(0, max(c.values()) * 1.18)

    dat = [sum(r9[i]["ea"] for i in can) / len(can) * 100,
           sum(r10[i]["ea"] for i in can) / len(can) * 100,
           sum(1 for i in can if any(r9[i]["cac_ea"])) / len(can) * 100]
    ax2.bar(range(3), dat, 0.55, color=[XAM, XANH, "#dfe4e8"])
    for i, v in enumerate(dat):
        ax2.text(i, v + 1.4, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["bỏ phiếu\ntheo số đông", "bộ chọn\n(model tự chấm)",
                         "trần\n(có mẫu đúng)"], fontsize=8.5)
    ax2.set_ylabel("EA (%)")
    ax2.set_ylim(0, 92)
    ax2.set_title(f"Trên {len(can)} câu có nhiều hơn một đáp án")
    lap = (dat[1] - dat[0]) / (dat[2] - dat[0]) * 100
    ax2.text(0.5, 0.06, f"lấp {lap:.0f} % khoảng cách tới trần",
             transform=ax2.transAxes, ha="center", fontsize=8.5, color="#444")
    _ghi(fig, ra, "06_bo_chon.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nguon", default=os.path.join(GOC, "results", "stages"))
    ap.add_argument("--ra", default=os.path.join(GOC, "docs", "hinh"))
    a = ap.parse_args()
    if not os.path.isdir(a.nguon):
        print(f"Không thấy {a.nguon} — cần thư mục nấc đã chạy.")
        return 1
    os.makedirs(a.ra, exist_ok=True)
    nac, meta = nap(a.nguon)
    print(f"  {len(nac)} nấc đọc từ {a.nguon}")
    hinh_thang_bac(nac, meta, a.ra)
    hinh_dong_gop(nac, a.ra)
    hinh_duong_cong_k(nac, a.ra)
    hinh_ace(nac, meta, a.ra)
    hinh_loi(meta, a.ra)
    hinh_bo_chon(nac, a.ra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
