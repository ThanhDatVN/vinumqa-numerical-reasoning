# -*- coding: utf-8 -*-
"""Truy hồi ví dụ mẫu ĐỘNG từ tập train, thay cho 2 ví dụ cố định trong prompt.

Vì sao đáng làm — đo trên chính bộ này, chỉ bằng BM25 thô trên câu hỏi:

* láng giềng train gần nhất có **cùng dãy phép** với gold: 231/497 = **46,5 %**
* ít nhất 1 trong top-3 láng giềng cùng dãy phép: 329/497 = **66,2 %**
* cả 2 993 mẫu train chỉ có **86 dãy phép** khác nhau; 10 dãy phổ biến nhất chiếm 85,2 %

Không gian chương trình hẹp và lặp lại — đúng điều kiện để truy hồi ăn tiền. Đây cũng
là hướng duy nhất nhắm thẳng vào **PA**: ví dụ truy hồi dạy đúng *văn phong gold*, thứ
PA đo, chứ không chỉ dạy ra đúng số.

Hai quyết định thiết kế, cả hai đều có lý do:

1. **Không kèm bảng của mẫu train.** Bảng dài, mà ngân sách ngữ cảnh chỉ dư 1 344 token.
   Thay vào đó, với ví dụ dùng ``table_*`` thì kèm **danh sách nhãn hàng** — đúng phần
   tín hiệu cần thiết, tốn vài chục token.
2. **Lọc nhãn nhiễu khỏi kho ví dụ.** Mẫu gold dùng ``multiply(#n,100)`` hoặc gold không
   tự thực thi ra ``exe_ans`` thì không được làm mẫu cho model bắt chước.

Chấm điểm bằng BM25 tự cài (không phụ thuộc ``rank_bm25``) nên hoàn toàn tất định và
chạy được trên CPU trong test.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .data import is_noisy_gold
from .dsl import OP_DETECT_RE

__all__ = ["KhoViDu", "tach_tu"]

_TU_RE = re.compile(r"[0-9a-zA-ZÀ-ỹ]+", re.UNICODE)

#: Từ xuất hiện ở quá nhiều câu hỏi thì không phân biệt được gì, chỉ tốn thời gian.
NGUONG_PHO_BIEN = 0.15


def tach_tu(s: str) -> list[str]:
    """Tách từ thô cho tiếng Việt: giữ token ≥2 ký tự, hạ chữ thường."""
    return [w for w in _TU_RE.findall((s or "").lower()) if len(w) > 1]


class KhoViDu:
    """Kho ví dụ mẫu lấy từ train, truy hồi theo BM25 trên câu hỏi.

    Parameters
    ----------
    train : danh sách mẫu train (có ``qa.program`` và ``qa.exe_ans``).
    bo_nhan_nhieu : bỏ mẫu có nhãn vàng nhiễu khỏi kho. Mặc định bật.
    k1, b : tham số BM25 chuẩn.
    """

    def __init__(self, train, *, bo_nhan_nhieu: bool = True, k1: float = 1.5,
                 b: float = 0.75):
        self.k1, self.b = k1, b
        self.mau = [s for s in train
                    if (s.get("qa") or {}).get("program", "").strip()
                    and not (bo_nhan_nhieu and is_noisy_gold(s))]
        self.tu = [tach_tu(s["qa"]["question"]) for s in self.mau]
        self.do_dai = [len(t) for t in self.tu]
        self.tb_dai = (sum(self.do_dai) / len(self.do_dai)) if self.do_dai else 1.0

        n = len(self.mau) or 1
        df: Counter = Counter()
        for t in self.tu:
            df.update(set(t))
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        # chỉ mục ngược, bỏ từ quá phổ biến
        self.nguoc: dict[str, list[int]] = defaultdict(list)
        tran = max(1, int(n * NGUONG_PHO_BIEN))
        for i, t in enumerate(self.tu):
            for w in set(t):
                if df[w] <= tran:
                    self.nguoc[w].append(i)
        self.tf = [Counter(t) for t in self.tu]

    def __len__(self) -> int:
        return len(self.mau)

    # ── truy hồi ──
    def xep_hang(self, cau_hoi: str, k: int = 3, tru_id: str | None = None) -> list[int]:
        """Chỉ số của ``k`` mẫu train gần nhất. Tất định: hoà điểm thì lấy chỉ số nhỏ hơn."""
        q = tach_tu(cau_hoi)
        if not q or not self.mau:
            return []
        diem: dict[int, float] = defaultdict(float)
        for w in set(q):
            if w not in self.nguoc:
                continue
            idf = self.idf.get(w, 0.0)
            for i in self.nguoc[w]:
                f = self.tf[i][w]
                mau_so = f + self.k1 * (1 - self.b + self.b * self.do_dai[i] / self.tb_dai)
                diem[i] += idf * (f * (self.k1 + 1)) / (mau_so or 1.0)
        if tru_id is not None:
            diem = {i: d for i, d in diem.items() if self.mau[i].get("id") != tru_id}
        return [i for i, _ in sorted(diem.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]

    # ── dựng văn bản ví dụ ──
    @staticmethod
    def _nhan_hang(mau, toi_da: int = 8) -> str:
        """Nhãn hàng của bảng — chỉ kèm khi ví dụ dùng ``table_*``."""
        t = mau.get("table") or []
        if not isinstance(t, list) or len(t) < 2:
            return ""
        nhan = [str(r[0]).strip() for r in t[1:]
                if isinstance(r, list) and r and str(r[0]).strip()]
        if not nhan:
            return ""
        cat = nhan[:toi_da]
        duoi = f" … (+{len(nhan) - len(cat)} hàng nữa)" if len(nhan) > len(cat) else ""
        return "Nhãn hàng của bảng: " + " | ".join(cat) + duoi

    def van_ban_vi_du(self, cau_hoi: str, k: int = 3, tru_id: str | None = None) -> str:
        """Khối ``=== VÍ DỤ ===`` dựng từ ``k`` mẫu train gần nhất. ``''`` nếu không có."""
        idxs = self.xep_hang(cau_hoi, k, tru_id)
        if not idxs:
            return ""
        khoi = []
        for thu_tu, i in enumerate(idxs, 1):
            s = self.mau[i]
            qa = s["qa"]
            prog = " ".join(str(qa["program"]).split())
            dong = [f"Ví dụ {thu_tu}:", f"Câu hỏi: {qa['question']}"]
            if OP_DETECT_RE.search(prog) and "table_" in prog.lower():
                nh = self._nhan_hang(s)
                if nh:
                    dong.append(nh)
            dong += ["Output:", f"program: {prog}", f"answer: {qa.get('exe_ans', '')}"]
            khoi.append("\n".join(dong))
        return "\n".join(khoi)
