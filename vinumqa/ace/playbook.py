# -*- coding: utf-8 -*-
"""Playbook ACE: lưu trữ, truy hồi, quality gate, curator — bản tiếng Việt cho ViNumQA.

Playbook là một tài liệu markdown, mỗi bullet có ID ổn định và bộ đếm ``helpful/harmful``::

    ## Chiến lược số học
    - [cl-00001] Khi hỏi tỷ lệ tăng trưởng, dùng subtract(gia_tri_moi, gia_tri_cu), divide(#0, gia_tri_cu). (h=12, harm=1)

Toàn bộ thao tác ghi đều đi qua :func:`render_playbook` nên bố cục không bao giờ trôi.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict

import numpy as np

from ..dsl import OP_ARGS_RE, OP_DETECT_RE, OP_NAMES, execute_program

__all__ = [
    "SECTION_SLUGS", "SECTION_TITLES", "SECTION_FALLBACK_ORDER", "ERROR_TO_SECTION",
    "all_bullets", "render_playbook", "empty_playbook", "inject_bullet",
    "update_bullet_counts", "section_counts", "next_id_map",
    "Embedder", "Retriever", "QualityGate", "Curator", "vi_tokens",
]

# ─────────────────────────────── định dạng playbook ───────────────────────────────

SECTION_SLUGS = {
    "chien_luoc_so_hoc": "cl",
    "doc_bang":          "db",
    "sinh_program":      "sp",
    "loi_thuong_gap":    "lt",
    "meo_ngu_canh":      "mn",
}
SECTION_TITLES = {
    "chien_luoc_so_hoc": "## Chiến lược số học",
    "doc_bang":          "## Đọc bảng",
    "sinh_program":      "## Sinh program",
    "loi_thuong_gap":    "## Lỗi thường gặp",
    "meo_ngu_canh":      "## Mẹo ngữ cảnh",
}
TITLE_TO_SECTION = {v.lower(): k for k, v in SECTION_TITLES.items()}
SECTION_FALLBACK_ORDER = ["chien_luoc_so_hoc", "doc_bang", "sinh_program",
                          "loi_thuong_gap", "meo_ngu_canh"]

ERROR_TO_SECTION = {
    "sai_don_vi":         "chien_luoc_so_hoc",
    "sai_dau":            "chien_luoc_so_hoc",
    "sai_bac_do_lon":     "chien_luoc_so_hoc",
    "lay_sai_so_lieu":    "doc_bang",
    "sai_ham_bang":       "doc_bang",
    "sai_phep_toan":      "sinh_program",
    "thieu_buoc":         "sinh_program",
    "thua_buoc":          "sinh_program",
    "sai_dinh_dang":      "sinh_program",
    "khong_co_program":   "sinh_program",
    "so_lieu_trong_text": "meo_ngu_canh",
    "suy_luan_sai":       "loi_thuong_gap",
    "khac":               "loi_thuong_gap",
}

SECTION_QUOTA_PCT = 0.50            # không mục nào chiếm quá 50 % playbook
SECTION_QUOTA_MIN_BULLETS = 8

_BULLET_RE = re.compile(r"^\s*-\s*\[([^\]]+)\]\s*(.+?)\s*(?:\(h=(\d+),\s*harm=(\d+)\))?\s*$")


def _parse_bullet_line(line: str):
    m = _BULLET_RE.match(line)
    if not m:
        return None, None, 0, 0
    return m.group(1), m.group(2).strip(), int(m.group(3) or 0), int(m.group(4) or 0)


def all_bullets(pb: str) -> list[dict]:
    bullets, section = [], None
    for line in (pb or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("##"):
            section = TITLE_TO_SECTION.get(stripped.lower(), "loi_thuong_gap")
        elif stripped.startswith("-"):
            bid, content, h, harm = _parse_bullet_line(line)
            if bid:
                bullets.append({"id": bid, "content": content,
                                "section": section or "loi_thuong_gap",
                                "helpful": h, "harmful": harm})
    return bullets


def render_playbook(bullets: list[dict]) -> str:
    """Dựng lại playbook từ danh sách bullet — bố cục luôn ổn định."""
    by_sec = defaultdict(list)
    for b in bullets:
        by_sec[b.get("section") or "loi_thuong_gap"].append(b)
    blocks = []
    for sec in SECTION_FALLBACK_ORDER:
        lines = [SECTION_TITLES[sec]]
        for b in by_sec.get(sec, []):
            suffix = (f" (h={b['helpful']}, harm={b['harmful']})"
                      if (b.get("helpful") or b.get("harmful")) else "")
            lines.append(f"- [{b['id']}] {b['content']}{suffix}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def empty_playbook() -> str:
    return render_playbook([])


def section_counts(pb: str) -> dict[str, int]:
    counts = {s: 0 for s in SECTION_SLUGS}
    for b in all_bullets(pb):
        counts[b["section"]] = counts.get(b["section"], 0) + 1
    return counts


def next_id_map(pb: str) -> dict[str, int]:
    out = {}
    for b in all_bullets(pb):
        parts = b["id"].split("-")
        if len(parts) == 2 and parts[1].isdigit():
            out[parts[0]] = max(out.get(parts[0], 0), int(parts[1]))
    return out


def inject_bullet(pb: str, section: str, bid: str, content: str) -> str:
    bullets = all_bullets(pb)
    bullets.append({"id": bid, "content": content.strip(), "section": section,
                    "helpful": 0, "harmful": 0})
    return render_playbook(bullets)


def update_bullet_counts(pb: str, tags) -> str:
    """``tags = [{'id': 'cl-00001', 'tag': 'helpful'|'harmful'|'neutral'}, ...]``"""
    tag_map = {t["id"]: t["tag"] for t in (tags or [])
               if isinstance(t, dict) and "id" in t and "tag" in t}
    if not tag_map:
        return pb
    bullets = all_bullets(pb)
    for b in bullets:
        tag = tag_map.get(b["id"])
        if tag == "helpful":
            b["helpful"] += 1
        elif tag == "harmful":
            b["harmful"] += 1
    return render_playbook(bullets)


# ─────────────────────────────── tokenise tiếng Việt ───────────────────────────────

VI_STOPWORDS = {
    "là", "của", "và", "có", "được", "cho", "trong", "với", "khi", "nếu", "thì", "các",
    "những", "một", "này", "đó", "ở", "từ", "đến", "bao", "nhiêu", "gì", "nào", "bằng",
    "hãy", "dùng", "sử", "dụng", "cần", "phải", "không", "để", "ra", "vào", "theo", "về",
    "so", "như", "vì", "mà", "sẽ", "đã", "còn", "hơn", "rất", "bạn", "tôi",
}


def vi_tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", (text or "").lower(), re.UNICODE)
            if w not in VI_STOPWORDS and len(w) > 1]


# ─────────────────────────────── embedding ───────────────────────────────

class Embedder:
    """Bọc sentence-transformers, ưu tiên model ĐA NGỮ.

    Bản ACE gốc dùng ``bge-base-en-v1.5`` — chỉ hiểu tiếng Anh nên xếp hạng bullet
    tiếng Việt gần như ngẫu nhiên. Ở đây mặc định là ``multilingual-e5-base``.
    Nếu không nạp được model nào, ``available`` = False và truy hồi lùi về BM25 + từ khoá.
    """

    CANDIDATES = ("intfloat/multilingual-e5-base", "BAAI/bge-m3", "all-MiniLM-L6-v2")

    def __init__(self, candidates=None, verbose=True):
        self.model, self.name = None, None
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            if verbose:
                print("[EMBED] ⚠ chưa cài sentence-transformers → chỉ dùng BM25 + từ khoá")
            return
        for name in (candidates or self.CANDIDATES):
            try:
                self.model, self.name = SentenceTransformer(name), name
                if verbose:
                    print(f"[EMBED] ✅ {name}")
                break
            except Exception as e:                       # noqa: BLE001
                if verbose:
                    print(f"[EMBED] ⚠ {name} lỗi: {type(e).__name__}")
        if self.model is None and verbose:
            print("[EMBED] ⚠ Không nạp được embedding nào → chỉ dùng BM25 + từ khoá")

    @property
    def available(self) -> bool:
        return self.model is not None

    @property
    def query_prefix(self) -> str:
        return "query: " if (self.name and "e5" in self.name) else ""

    @property
    def passage_prefix(self) -> str:
        return "passage: " if (self.name and "e5" in self.name) else ""

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True,
                                 show_progress_bar=False, batch_size=32)


# ─────────────────────────────── truy hồi ───────────────────────────────

class Retriever:
    """Truy hồi Tier-1 / Tier-2.

    ``score = w_sem·semantic + w_bm25·BM25 + w_value·(h/(h+harm+1)) + w_freq·(1/(1+lượt dùng))``

    Hai hạng mục cuối là cơ chế chống "một bullet chiếm sóng mọi câu hỏi" của ACE.
    **Tier-1** = bullet đã chứng minh có ích (đủ lượt dùng và lift PA dương) → luôn vào prompt.
    **Tier-2** = phần còn lại, chọn top-k động, có ưu tiên trải đều các mục.
    """

    def __init__(self, embedder: Embedder | None = None, k_tier1=3, k_tier2=4,
                 tier1_max=5, tier1_min_uses=12, tier1_min_lift=0.01,
                 max_bullets=30, section_aware_min=10,
                 w_sem=0.35, w_bm25=0.25, w_value=0.20, w_freq=0.20):
        self.embedder = embedder
        self.k_tier1, self.k_tier2 = k_tier1, k_tier2
        self.tier1_max = tier1_max
        self.tier1_min_uses, self.tier1_min_lift = tier1_min_uses, tier1_min_lift
        self.max_bullets = max_bullets
        self.section_aware_min = section_aware_min
        self.w_sem, self.w_bm25 = w_sem, w_bm25
        self.w_value, self.w_freq = w_value, w_freq

        self.tier1_ids: set[str] = set()
        self.usage: dict[str, dict] = defaultdict(lambda: {"used": 0, "pa_ok": 0, "ea_ok": 0})
        self.retrieval_count: Counter = Counter()
        self._emb_cache: dict[str, tuple[str, object]] = {}

        try:
            from rank_bm25 import BM25Okapi
            self._BM25 = BM25Okapi
        except ImportError:
            self._BM25 = None

    # ── điểm ──
    def _bullet_embeddings(self, bullets):
        if not (self.embedder and self.embedder.available):
            return None
        embs, todo, todo_idx = [None] * len(bullets), [], []
        for i, b in enumerate(bullets):
            key = hashlib.md5(b["content"].encode("utf-8")).hexdigest()[:12]
            cached = self._emb_cache.get(b["id"])
            if cached and cached[0] == key:
                embs[i] = cached[1]
            else:
                todo.append(self.embedder.passage_prefix + b["content"])
                todo_idx.append((i, key))
        if todo:
            for (i, key), e in zip(todo_idx, self.embedder.encode(todo)):
                embs[i] = e
                self._emb_cache[bullets[i]["id"]] = (key, e)
        return np.array(embs)

    def _bm25_scores(self, question, bullets):
        if self._BM25 is None or not bullets:
            return [0.5] * len(bullets)
        corpus = [vi_tokens(b["content"]) for b in bullets]
        if not any(corpus):
            return [0.5] * len(bullets)
        try:
            raw = self._BM25(corpus).get_scores(vi_tokens(question))
        except Exception:                                # noqa: BLE001
            return [0.5] * len(bullets)
        lo, hi = float(np.min(raw)), float(np.max(raw))
        return ([(s - lo) / (hi - lo) for s in raw] if hi - lo > 1e-8
                else [0.5] * len(bullets))

    def score(self, question, bullets):
        if not bullets:
            return []
        if self.embedder and self.embedder.available:
            q_emb = self.embedder.encode([self.embedder.query_prefix + question])[0]
            sem_raw = (self._bullet_embeddings(bullets) @ q_emb).tolist()
        else:
            qw = set(vi_tokens(question))
            sem_raw = [len(qw & set(vi_tokens(b["content"]))) for b in bullets]
        lo, hi = min(sem_raw), max(sem_raw)
        sem = ([(s - lo) / (hi - lo) for s in sem_raw] if hi - lo > 1e-8
               else [0.5] * len(bullets))
        bm = self._bm25_scores(question, bullets)

        out = []
        for i, b in enumerate(bullets):
            denom = b["helpful"] + b["harmful"]
            value = b["helpful"] / (denom + 1) if denom else 0.5
            freq = 1.0 / (self.retrieval_count.get(b["id"], 0) + 1)
            out.append((i, self.w_sem * sem[i] + self.w_bm25 * bm[i]
                        + self.w_value * value + self.w_freq * freq))
        return out

    def _section_aware_topk(self, question, bullets, k):
        scored = sorted(self.score(question, bullets), key=lambda x: -x[1])
        if len(bullets) < self.section_aware_min:
            return [i for i, _ in scored[:k]]
        by_sec = defaultdict(list)
        for i, _ in scored:
            by_sec[bullets[i]["section"]].append(i)
        picked, seen = [], set()
        for sec in SECTION_FALLBACK_ORDER:                   # mỗi mục một suất trước
            for i in by_sec.get(sec, [])[:1]:
                if i not in seen and len(picked) < k:
                    picked.append(i); seen.add(i)
        for i, _ in scored:                                  # còn chỗ thì lấp theo điểm
            if len(picked) >= k:
                break
            if i not in seen:
                picked.append(i); seen.add(i)
        return picked[:k]

    # ── API chính ──
    def retrieve(self, question, playbook, k_tier1=None, k_tier2=None, record=False):
        """Trả ``(đoạn text bullet, danh sách id đã dùng)``."""
        k1 = self.k_tier1 if k_tier1 is None else k_tier1
        k2 = self.k_tier2 if k_tier2 is None else k_tier2
        bullets = all_bullets(playbook)
        if not bullets or (k1 + k2) <= 0:
            return "", []

        t1 = [b for b in bullets if b["id"] in self.tier1_ids]
        t2 = [b for b in bullets if b["id"] not in self.tier1_ids]

        if not t1:                                           # chưa có Tier-1 → top-k phẳng
            chosen = [bullets[i]
                      for i in self._section_aware_topk(question, bullets, k1 + k2)]
        else:
            s1 = sorted(self.score(question, t1), key=lambda x: -x[1])
            chosen = [t1[i] for i, _ in s1[:k1]]
            if t2 and k2 > 0:
                chosen += [t2[i] for i in self._section_aware_topk(question, t2, k2)]

        if record:
            for b in chosen:
                self.retrieval_count[b["id"]] += 1
        return "\n".join(f"- {b['content']}" for b in chosen), [b["id"] for b in chosen]

    def record_outcome(self, used_ids, pa_ok: bool, ea_ok: bool):
        for bid in used_ids:
            st = self.usage[bid]
            st["used"] += 1
            st["pa_ok"] += int(pa_ok)
            st["ea_ok"] += int(ea_ok)

    def maybe_promote_tier1(self, playbook, baseline_pa: float):
        """Thăng Tier-1 cho bullet đủ lượt dùng và có lift PA dương, ưu tiên lift cao nhất."""
        if len(self.tier1_ids) >= self.tier1_max:
            return []
        candidates = []
        for b in all_bullets(playbook):
            if b["id"] in self.tier1_ids:
                continue
            st = self.usage[b["id"]]
            if st["used"] < self.tier1_min_uses:
                continue
            lift = st["pa_ok"] / st["used"] - baseline_pa
            if lift >= self.tier1_min_lift:
                candidates.append((b["id"], round(lift, 4)))
        promoted = []
        for bid, lift in sorted(candidates, key=lambda x: -x[1]):
            if len(self.tier1_ids) >= self.tier1_max:
                break
            self.tier1_ids.add(bid)
            promoted.append((bid, lift))
        return promoted

    def enforce_budget(self, playbook):
        """Vượt trần → loại bullet Tier-2 giá trị thấp nhất."""
        bullets = all_bullets(playbook)
        if len(bullets) <= self.max_bullets:
            return playbook, []
        t2 = [b for b in bullets if b["id"] not in self.tier1_ids]

        def value(b):
            st = self.usage[b["id"]]
            lift = (st["pa_ok"] / st["used"]) if st["used"] else 0.0
            return (b["helpful"] - 2 * b["harmful"]) + lift + 0.001 * st["used"]

        drop = {b["id"] for b in sorted(t2, key=value)[:len(bullets) - self.max_bullets]}
        return render_playbook([b for b in bullets if b["id"] not in drop]), sorted(drop)


# ─────────────────────────────── quality gate ───────────────────────────────

ALLOWED_LITERALS = {"0", "1", "2", "3", "4", "5", "10", "12", "100", "1000",
                    "10000", "1000000", "0.5", "1.0"}

VI_DOMAIN_KEYWORDS = [
    "tỷ lệ", "tỉ lệ", "tăng", "giảm", "doanh thu", "chi phí", "lợi nhuận", "chênh lệch",
    "chia", "trừ", "cộng", "nhân", "phép", "tính", "bảng", "cột", "hàng", "ô", "năm", "kỳ",
    "program", "bước", "công thức", "mẫu", "đơn vị", "quy đổi", "triệu", "tỷ", "nghìn",
    "lớn nhất", "nhỏ nhất", "trung bình", "tổng", "phần trăm", "tỷ trọng", "giá trị",
    "tử số", "mẫu số", "divide", "subtract", "multiply", "add", "table",
]

FORBIDDEN_PHRASES = [
    "đáp án là", "kết quả là", "câu trả lời là", "con số cụ thể", "nhớ giá trị",
    "the answer is", "use the correct answer",
]

VI_NAMED_OPERANDS = {
    "gia_tri_moi", "gia_tri_cu", "gia_tri_hien_tai", "gia_tri_goc", "gia_tri_dau",
    "gia_tri_cuoi", "gia_tri_nam_nay", "gia_tri_nam_truoc", "tong", "phan", "tu_so",
    "mau_so", "ty_le_tang", "ty_le", "he_so", "so_luong", "don_gia", "ten_chi_tieu",
    "gia_tri", "gia_tri_khac", "a", "b", "c", "d", "x", "y", "v1", "v2", "w1", "w2",
    "tong_trong_so", "a1", "a2", "b1", "b2",
}

_NUM_TOKEN_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_NESTED_RE = re.compile(r"\(\s*[^()]*\b(?:" + "|".join(OP_NAMES) + r")\s*\(", re.I)
_CHAIN_RE = re.compile(
    r"((?:" + "|".join(OP_NAMES) + r")\s*\([^)]*\)"
    r"(?:\s*,\s*(?:" + "|".join(OP_NAMES) + r")\s*\([^)]*\))*)", re.I)

# Bộ số giả để chạy thử công thức trong bullet — bắt công thức sai trước khi vào playbook.
SYNTH_CATEGORIES = [
    {"name": "tang_truong",
     "kw": ["tăng trưởng", "tỷ lệ tăng", "tỉ lệ tăng", "tăng/giảm", "tỷ lệ giảm", "so với kỳ"],
     "values": [{"gia_tri_moi": 120, "gia_tri_cu": 100, "gia_tri_nam_nay": 120,
                 "gia_tri_nam_truoc": 100, "a": 120, "b": 100, "expected": 0.20},
                {"gia_tri_moi": 80, "gia_tri_cu": 100, "gia_tri_nam_nay": 80,
                 "gia_tri_nam_truoc": 100, "a": 80, "b": 100, "expected": -0.20}],
     "range": (-1.0, 2.0), "tol": 0.05},
    {"name": "ty_trong",
     "kw": ["tỷ trọng", "tỉ trọng", "phần trăm của", "chiếm"],
     "values": [{"phan": 30, "tong": 100, "tu_so": 30, "mau_so": 100, "expected": 0.30},
                {"phan": 50, "tong": 200, "tu_so": 50, "mau_so": 200, "expected": 0.25}],
     "range": (0.0, 1.5), "tol": 0.05},
    {"name": "dao_nguoc",
     "kw": ["đầu năm", "năm trước", "giá trị gốc", "trước khi tăng"],
     "values": [{"ty_le_tang": 0.2, "gia_tri_hien_tai": 120, "gia_tri_nam_nay": 120,
                 "expected": 100.0},
                {"ty_le_tang": 0.5, "gia_tri_hien_tai": 150, "gia_tri_nam_nay": 150,
                 "expected": 100.0}],
     "range": (1, 10000), "tol": 0.05},
    {"name": "trung_binh",
     "kw": ["trung bình", "bình quân"],
     "values": [{"a": 10, "b": 20, "c": 30, "v1": 10, "v2": 20, "expected": 15.0},
                {"a": 100, "b": 200, "c": 300, "v1": 100, "v2": 200, "expected": 150.0}],
     "range": (1, 1000), "tol": 0.05},
]


class QualityGate:
    """Cổng kiểm duyệt giữ playbook khỏi biến thành "bộ nhớ đáp án".

    Một bullet bị loại nếu: có số liệu cụ thể (ngoài 1/2/3/10/100/1000/1000000) hoặc có năm;
    dưới 2 phép toán DSL; chứa ``multiply(#n, 100)``; phép toán lồng nhau hoặc ``#N`` tham chiếu
    tiến; chạy thử công thức trên bộ số giả ra sai; hoặc trùng cấu trúc / từ vựng / ngữ nghĩa
    với bullet đã có.
    """

    def __init__(self, embedder: Embedder | None = None, retriever: Retriever | None = None,
                 min_len=30, max_len=220, dedup_thresh=0.98, overlap_thresh=0.80,
                 min_ops=1):
        """``dedup_thresh`` phải khớp với embedder đang dùng — ngưỡng KHÔNG chuyển được
        giữa các model.

        Bản ACE gốc dùng ``all-MiniLM-L6-v2`` / ``bge-base-en``, cặp câu không liên quan
        cho cosine ~0.1–0.3 nên 0.85 là "trùng" thật. Ta dùng ``multilingual-e5-base``,
        vốn nén mọi cặp vào dải ~0.70–0.90 — 0.85 ở đó gần như là SÀN, không phải trùng.
        Giữ 0.85 khiến 113/217 đề xuất bị loại oan dù playbook chỉ có 1 bullet.
        Log của lượt sau (đã ghi kèm giá trị thật) cho thấy phân bố dồn ở 0.94–0.97,
        nên sàn của e5 là ~0.95 → đặt 0.98 mới chỉ chặn trùng lặp thật.

        ``min_ops`` = số phép DSL tối thiểu. Bản gốc ép 2. Nhưng 64 % tập test ViNumQA
        là câu MỘT phép (319/497), nên ép 2 là chặn hẳn mọi lời khuyên cho nhóm lớn nhất.
        Để 1: bullet vẫn phải có phép DSL thật, vẫn qua kiểm chứng + verify.
        """
        self.embedder = embedder
        self.retriever = retriever
        self.min_len, self.max_len = min_len, max_len
        self.min_ops = min_ops
        self.dedup_thresh, self.overlap_thresh = dedup_thresh, overlap_thresh

    # ── kiểm tra từng luật ──
    @staticmethod
    def has_forbidden_numbers(text: str) -> bool:
        body = re.sub(r"#\d+", "", text)
        return any(tok.replace(",", ".") not in ALLOWED_LITERALS
                   for tok in _NUM_TOKEN_RE.findall(body))

    def has_enough_ops(self, text: str) -> bool:
        return len(OP_DETECT_RE.findall(text)) >= self.min_ops

    @staticmethod
    def is_nested(text: str) -> bool:
        """Bắt phép toán lồng trong tham số, vd ``divide(30, add(1, 0.3))``."""
        return bool(_NESTED_RE.search(text))

    @staticmethod
    def check_refs(text: str) -> tuple[bool, str]:
        for i, (_op, args) in enumerate(OP_ARGS_RE.findall(text.lower())):
            for ref in re.findall(r"#(\d+)", args):
                if int(ref) >= i:
                    return False, f"#{ref}_o_buoc_{i}_khong_hop_le"
        return True, "ok"

    @staticmethod
    def validate_synthetically(content: str) -> tuple[bool, str]:
        m = _CHAIN_RE.search(content)
        if not m:
            return True, "khong_co_chuoi_dsl"
        chain = re.sub(r"\s+", "", m.group(1))
        low = content.lower()
        cat = next((c for c in SYNTH_CATEGORIES if any(k in low for k in c["kw"])), None)
        if cat is None:
            return True, "khong_thuoc_ho_nao"

        tested = passed = 0
        for vs in cat["values"]:
            sub = chain
            for name, val in sorted(vs.items(), key=lambda x: -len(x[0])):
                if name == "expected":
                    continue
                sub = re.sub(rf"\b{re.escape(name)}\b", str(val), sub, flags=re.I)
            leftover = [t for t in re.findall(r"[a-zA-Z_]{2,}", sub)
                        if t.lower() not in set(OP_NAMES) | {"none"}]
            if leftover:
                continue
            tested += 1
            result = execute_program(sub, [])
            if result is None or isinstance(result, str):
                return False, "chay_thu_that_bai"
            lo, hi = cat["range"]
            if not (lo <= float(result) <= hi):
                return False, f"ngoai_khoang_{cat['name']}"
            exp = vs["expected"]
            if abs(float(result) - exp) / (abs(exp) if exp else 1.0) < cat["tol"]:
                passed += 1
        if tested == 0:
            return True, f"khong_co_bo_so_thu_{cat['name']}"
        if passed < tested:
            return False, f"sai_cong_thuc_{cat['name']}_{passed}/{tested}"
        return True, f"dat_{cat['name']}_{passed}/{tested}"

    # ── chống trùng ──
    @staticmethod
    def _classify_operand(arg: str) -> str:
        a = arg.strip().lower()
        if a.startswith("#"):
            return "REF"
        if a in VI_NAMED_OPERANDS:
            return f"TEN({a})"
        try:
            float(a.replace(",", ""))
            return "SO"
        except ValueError:
            return f"KHAC({a[:15]})"

    @classmethod
    def _structural_signature(cls, content: str):
        m = re.match(r"\s*(?:khi|nếu|với|trong|đối với)\s+([^,.;:]{6,80})", content, re.I)
        trigger = {w for w in vi_tokens(m.group(1)) if len(w) > 2} if m else set()
        ops = []
        for mm in OP_ARGS_RE.finditer(content):
            kinds = tuple(cls._classify_operand(a) for a in mm.group(2).split(","))
            ops.append(f"{mm.group(1).lower()}({','.join(kinds)})")
        return trigger, "->".join(ops)

    def is_structural_duplicate(self, new_content, bullets, trigger_thr=0.75):
        new_trig, new_ops = self._structural_signature(new_content)
        if not new_ops:
            return False, None
        for b in bullets:
            b_trig, b_ops = self._structural_signature(b["content"])
            if b_ops != new_ops:
                continue
            if not new_trig or not b_trig:
                return True, f"{b['id']}@cung_cau_truc"
            denom = min(len(new_trig), len(b_trig))
            sim = len(new_trig & b_trig) / denom if denom else 0.0
            if sim >= trigger_thr:
                return True, f"{b['id']}@trung_dieu_kien={sim:.2f}"
        return False, None

    @staticmethod
    def jaccard(a, b) -> float:
        wa, wb = set(vi_tokens(a)), set(vi_tokens(b))
        return len(wa & wb) / len(wa | wb) if (wa | wb) else 0.0

    def is_duplicate_lexical(self, new_content, bullets):
        for b in bullets:
            if self.jaccard(new_content, b["content"]) >= self.overlap_thresh:
                return True, "trung_tu_vung"
        if self.embedder and self.embedder.available and bullets and self.retriever:
            emb = self.embedder.encode([self.embedder.passage_prefix + new_content])[0]
            mat = self.retriever._bullet_embeddings(bullets)
            if mat is not None:
                sim = float((mat @ emb).max())
                if sim >= self.dedup_thresh:
                    # kèm giá trị để lần sau đọc log là biết phân bố, khỏi đoán ngưỡng
                    return True, f"trung_ngu_nghia@{sim:.2f}"
        return False, "ok"

    # ── API ──
    def __call__(self, content: str, playbook: str) -> tuple[str, str]:
        """Trả ``('add'|'reject', lý do)``."""
        c = (content or "").strip()
        if len(c) < self.min_len:
            return "reject", "qua_ngan"
        if len(c) > self.max_len:
            return "reject", "qua_dai"
        low = c.lower()
        if not any(k in low for k in VI_DOMAIN_KEYWORDS):
            return "reject", "qua_chung_chung"
        for p in FORBIDDEN_PHRASES:
            if p in low:
                return "reject", f"cum_tu_cam:{p[:20]}"
        if _YEAR_RE.search(c):
            return "reject", "chua_nam_cu_the"
        if self.has_forbidden_numbers(c):
            return "reject", "chua_so_lieu_cu_the"
        if not self.has_enough_ops(c):
            return "reject", f"duoi_{self.min_ops}_phep_toan"
        if self.is_nested(c):
            return "reject", "phep_toan_long_nhau"
        if re.search(r"multiply\s*\(\s*#\d+\s*,\s*100\s*\)", low) or \
           re.search(r"multiply\s*\(\s*100\s*,\s*#\d+\s*\)", low):
            return "reject", "multiply_100_tinh_phan_tram"
        ok, reason = self.check_refs(c)
        if not ok:
            return "reject", f"tham_chieu_sai:{reason}"
        ok, reason = self.validate_synthetically(c)
        if not ok:
            return "reject", f"kiem_chung:{reason}"

        bullets = all_bullets(playbook)
        dup, why = self.is_structural_duplicate(c, bullets)
        if dup:
            return "reject", f"trung_cau_truc:{why}"
        dup, why = self.is_duplicate_lexical(c, bullets)
        if dup:
            return "reject", why
        return "add", "ok"


# ─────────────────────────────── curator ───────────────────────────────

class Curator:
    """Quality gate → hạn ngạch cụm → hạn ngạch mục → chèn bullet."""

    def __init__(self, gate: QualityGate, max_bullets_per_cluster=3):
        self.gate = gate
        self.max_per_cluster = max_bullets_per_cluster
        self.bullet_cluster: dict[str, str] = {}

    def _section_over_quota(self, playbook, section):
        counts = section_counts(playbook)
        total = sum(counts.values())
        if total < SECTION_QUOTA_MIN_BULLETS:
            return False
        return counts.get(section, 0) >= total * SECTION_QUOTA_PCT

    @staticmethod
    def _fallback_section(playbook, exclude):
        counts = section_counts(playbook)
        cands = [s for s in SECTION_FALLBACK_ORDER if s != exclude]
        cands.sort(key=lambda s: (counts.get(s, 0), SECTION_FALLBACK_ORDER.index(s)))
        return cands[0]

    def __call__(self, strategy, error_type, playbook, cluster_id=None):
        """Trả ``(playbook, action, lý do, bullet_id)``."""
        if cluster_id and sum(1 for c in self.bullet_cluster.values()
                              if c == cluster_id) >= self.max_per_cluster:
            return playbook, "reject", f"cum_{cluster_id}_da_du_bullet", None

        action, reason = self.gate(strategy, playbook)
        if action == "reject":
            return playbook, "reject", reason, None

        section = ERROR_TO_SECTION.get(error_type, "loi_thuong_gap")
        if self._section_over_quota(playbook, section):
            section = self._fallback_section(playbook, section)

        slug = SECTION_SLUGS[section]
        bid = f"{slug}-{next_id_map(playbook).get(slug, 0) + 1:05d}"
        playbook = inject_bullet(playbook, section, bid, strategy.strip())
        if cluster_id:
            self.bullet_cluster[bid] = cluster_id
        return playbook, "add", "da_them", bid
