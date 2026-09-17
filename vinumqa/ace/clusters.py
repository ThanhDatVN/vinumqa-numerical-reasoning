# -*- coding: utf-8 -*-
"""Cụm lỗi tiếng Việt cho ViNumQA.

Bản ACE gốc nạp ``phase0_clusters.json`` (12–16 cụm tiếng Anh, dựng sẵn cho FinQA).
Ở đây cụm được **định nghĩa trực tiếp trong code** từ chính bảng "hướng dẫn chọn phép
toán theo từ khoá" trong ``prompts/prompt_builder.py``, nên không cần file ngoài và
luôn khớp với prompt mà model thực sự nhìn thấy.

Cụm dùng để: (1) nói cho Reflector biết bài toán thuộc họ nào và mẫu chuẩn là gì,
(2) áp hạn ngạch ``MAX_BULLETS_PER_CLUSTER`` để playbook không bị một họ lỗi chiếm hết chỗ.

Phân bố đo được trên train (2.993 mẫu): C2 16.1 %, C5 11.3 %, C1 11.2 %, C9 9.5 %,
C7 7.8 %, C3 5.1 %, C8 4.9 %, C10 4.5 %, C4 1.1 %, C6 0.5 %, còn lại C11 28.4 %.
"""
from __future__ import annotations

import re
from collections import Counter

from ..dsl import first_op, n_ops

__all__ = ["VI_CLUSTERS", "CLUSTER_BY_ID", "CLUSTER_PRIORITY",
           "cluster_id_for_sample", "cluster_distribution", "print_cluster_distribution"]

VI_CLUSTERS = [
    {"id": "C1_ty_trong", "n_steps": 1, "first_op": "divide",
     "kw": r"tỷ trọng|tỉ trọng|chiếm bao nhiêu|chiếm tỷ lệ|phần trăm của|bao nhiêu phần trăm|tỷ lệ .{0,20}(trên|so với|trong)",
     "lesson": "Tỷ trọng / phần trăm của A so với tổng B — một phép chia duy nhất, kết quả là số thập phân.",
     "pattern": "divide(phan, tong)",
     "wrong": ["nhân thêm 100 để ra phần trăm", "chia ngược tổng cho phần"]},

    {"id": "C2_tang_truong", "n_steps": 2, "first_op": "subtract",
     "kw": r"tăng trưởng|tỷ lệ tăng|tỉ lệ tăng|tăng bao nhiêu|giảm bao nhiêu|tỷ lệ giảm|yoy|so với (năm|quý|cùng kỳ)",
     "lesson": "Tốc độ tăng/giảm giữa hai kỳ: subtract(mới, cũ) rồi chia cho giá trị cũ, "
               "GIỮ dấu âm nếu giảm. Chỉ áp cho câu hỏi TỶ LỆ (có phép chia) — đo trên "
               "gold: 49/58 mẫu dạng này giữ dấu âm.",
     "pattern": "subtract(gia_tri_moi, gia_tri_cu), divide(#0, gia_tri_cu)",
     # KHÔNG liệt "đảo thứ tự subtract" là sai: với câu hỏi MỨC giảm tuyệt đối (không
     # chia), gold chia gần đôi — 16 mẫu đảo thứ tự cho ra số dương, 14 mẫu giữ dấu âm.
     # Ở đó không có quy tắc đáng tin, nên đừng dạy Reflector một quy tắc không có thật.
     "wrong": ["chia cho giá trị mới thay vì giá trị cũ", "nhân 100 để ra phần trăm"]},

    {"id": "C3_chenh_lech", "n_steps": 1, "first_op": "subtract",
     "kw": r"chênh lệch|tăng thêm|nhiều hơn|ít hơn|cao hơn|thấp hơn|thay đổi .{0,20}(là|bao nhiêu)",
     "lesson": "Chênh lệch tuyệt đối — chỉ trừ, không chia.",
     "pattern": "subtract(gia_tri_moi, gia_tri_cu)",
     "wrong": ["chia thêm cho giá trị cũ dù đề chỉ hỏi chênh lệch"]},

    {"id": "C4_dao_nguoc_tang_truong", "n_steps": 2, "first_op": "add",
     "kw": r"đầu năm|đầu kỳ|đầu quý|ban đầu|gốc ban đầu|trước khi tăng|năm trước đó|cùng kỳ năm trước",
     "lesson": "Biết giá trị hiện tại và % tăng, cần tìm giá trị gốc: dựng hệ số 1+r rồi chia ngược.",
     "pattern": "add(1, ty_le_tang), divide(gia_tri_hien_tai, #0)",
     "wrong": ["nhân trực tiếp giá trị hiện tại với tỷ lệ", "trừ thẳng phần trăm"]},

    {"id": "C5_tong_nhieu_ky", "n_steps": 2, "first_op": "add",
     "kw": r"tổng|cộng dồn|cộng lại|trong vòng \d+ (năm|quý|tháng)|gộp",
     "lesson": "Cộng nhiều giá trị rời rạc bằng chuỗi add liên tiếp, nối bằng #0, #1 — không dùng table_sum.",
     "pattern": "add(a, b), add(#0, c)",
     "wrong": ["dùng table_sum cho 2-3 ô rời rạc"]},

    {"id": "C6_quy_doi_don_vi", "n_steps": 2, "first_op": "multiply",
     "kw": r"quy đổi|cùng đơn vị|nghìn tỷ|triệu đồng|(tỷ|tỉ) đồng|triệu usd|nghìn đồng",
     "lesson": "Hai số khác đơn vị: phải quy đổi bằng một phép nhân/chia trong program trước khi tính.",
     "pattern": "multiply(gia_tri, 1000), divide(gia_tri_khac, #0)",
     "wrong": ["tự nhẩm 1 tỷ = 1000 triệu rồi viết thẳng số đã đổi"]},

    {"id": "C7_max_min_bang", "n_steps": 1, "first_op": "table_max",
     "kw": r"lớn nhất|cao nhất|nhỏ nhất|thấp nhất|đỉnh|đáy",
     "lesson": "Giá trị lớn nhất/nhỏ nhất của một chỉ tiêu trong bảng — dùng table_max/table_min với nhãn hàng.",
     "pattern": "table_max(ten_chi_tieu, none)",
     "wrong": ["truyền tên năm thay vì nhãn chỉ tiêu", "liệt kê tay rồi so sánh"]},

    {"id": "C8_trung_binh_bang", "n_steps": 1, "first_op": "table_average",
     "kw": r"trung bình|bình quân",
     "lesson": "Trung bình cả một chỉ tiêu trong bảng — table_average; nếu chỉ vài ô rời rạc thì add rồi divide.",
     "pattern": "table_average(ten_chi_tieu, none)",
     "wrong": ["cộng tay rồi chia sai số lượng phần tử"]},

    {"id": "C9_ty_le_don_gian", "n_steps": 1, "first_op": "divide",
     "kw": r"tỷ lệ|tỉ lệ|gấp bao nhiêu lần|bao nhiêu lần|hệ số|biên",
     "lesson": "Tỷ số giữa hai đại lượng — một phép chia, tử số là đại lượng được hỏi.",
     "pattern": "divide(tu_so, mau_so)",
     "wrong": ["đảo tử/mẫu"]},

    {"id": "C10_nhieu_buoc", "n_steps": 0, "first_op": "*", "kw": r".*",
     "lesson": "Bài nhiều bước: tách thành các phép tuần tự, nối bằng #0, #1, chỉ để lại một đầu ra cuối cùng.",
     "pattern": "op1(a, b), op2(#0, c), op3(#1, d)",
     "wrong": ["lồng phép toán vào nhau", "để thừa phép không dẫn tới đầu ra cuối"]},

    {"id": "C11_khac", "n_steps": 0, "first_op": "*", "kw": r".*",
     "lesson": "Không thuộc họ nào rõ rệt — bám sát quy tắc chung trong system prompt.",
     "pattern": "(không có mẫu chuẩn)",
     "wrong": []},
]

CLUSTER_BY_ID = {c["id"]: c for c in VI_CLUSTERS}
CLUSTER_PRIORITY = [c["id"] for c in VI_CLUSTERS]
_CLUSTER_RE = {c["id"]: re.compile(c["kw"], re.I) for c in VI_CLUSTERS}

W_NSTEPS, W_FIRSTOP, W_REGEX, CLUSTER_THRESHOLD = 2, 1, 2, 3
_CATCH_ALL = ("C10_nhieu_buoc", "C11_khac")


def cluster_id_for_sample(sample: dict) -> str:
    """Chấm điểm từng cụm: khớp số bước (×2) + khớp phép đầu (×1) + khớp regex (×2).

    Bắt buộc phải khớp regex thì mới được tính, nên một cụm không bao giờ "nuốt" mẫu
    chỉ vì trùng số bước. Không cụm nào đạt ngưỡng → rơi về catch-all.
    """
    question = (sample.get("qa", {}).get("question", "") or "").lower()
    gold = sample.get("qa", {}).get("program", "") or ""
    steps, op = n_ops(gold), first_op(gold)

    scores = {}
    for idx, cid in enumerate(CLUSTER_PRIORITY):
        if cid in _CATCH_ALL:
            continue
        c = CLUSTER_BY_ID[cid]
        n_ok = (c["n_steps"] == 0) or (c["n_steps"] == steps)
        op_ok = (c["first_op"] == "*") or (c["first_op"] == op) or \
                (c["first_op"].startswith("table_") and op.startswith("table_"))
        re_ok = bool(_CLUSTER_RE[cid].search(question))
        score = W_NSTEPS * n_ok + W_FIRSTOP * op_ok + W_REGEX * re_ok
        if re_ok and score >= CLUSTER_THRESHOLD:
            scores[cid] = (score, -idx)
    if scores:
        return max(scores.items(), key=lambda kv: kv[1])[0]
    return "C10_nhieu_buoc" if steps >= 3 else "C11_khac"


def cluster_distribution(samples: list[dict]) -> Counter:
    return Counter(cluster_id_for_sample(s) for s in samples)


def print_cluster_distribution(samples: list[dict], label: str = "") -> Counter:
    counts = cluster_distribution(samples)
    total = sum(counts.values()) or 1
    print(f"\n[CLUSTER] Phân bố trên {label} ({total} mẫu):")
    for cid in CLUSTER_PRIORITY:
        c = counts.get(cid, 0)
        pct = c / total * 100
        print(f"  {cid:<26} {c:>5} ({pct:>5.1f}%) {chr(9608) * int(pct / 2)}")
    return counts
