# -*- coding: utf-8 -*-
"""Vòng lặp ACE: học playbook từ lỗi của model trên tập train.

Mỗi vòng xử lý một mini-batch và gồm năm bước, **tất cả đều gom lô qua vLLM** nên
nhanh hơn nhiều so với vòng lặp tuần tự từng mẫu của bản ACE gốc:

1. **Generator** — chạy đúng pipeline của nấc đang xét, chấm PA/EA.
2. **Thống kê bullet** — cập nhật ``used / pa_ok / ea_ok`` cho bullet đã được truy hồi.
3. **Reflector** — chọn tối đa ``max_reflect`` mẫu sai, ưu tiên trải đều các cụm lỗi.
4. **Verify** — sinh lại lời giải kèm bullet ứng viên; chỉ bullet **thực sự sửa được**
   mẫu mới đi tiếp. Đây là thứ giữ playbook khỏi đầy quy tắc nghe hợp lý nhưng sai.
5. **Curator** — quality gate → chèn → ép trần dung lượng → thăng hạng Tier-1.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict

from ..dsl import check_ea, check_pa, execute_program, extract_program_answer
from ..pipeline import run_pipeline
from .clusters import cluster_id_for_sample
from .playbook import all_bullets, render_playbook, update_bullet_counts
from .reflector import diagnose


def _noi_dung_khoa(text: str) -> str:
    """Khoá nhận dạng một đề xuất, bỏ qua khác biệt hoa/thường và khoảng trắng."""
    return hashlib.md5(re.sub(r"\s+", " ", (text or "").strip().lower())
                       .encode("utf-8")).hexdigest()[:12]


__all__ = ["AceTrainer"]


class AceTrainer:
    def __init__(self, prompt_kit, generate_fn, retriever, curator, reflector, *,
                 prompt_level="engineered", use_selfeval=True,
                 sp_step1=None, sp_step2=None, round_size=32, max_reflect=12,
                 use_verify=True, verify_require_pa=False, log_path=None,
                 verify_lan=2):
        self.prompt_kit = prompt_kit
        self.generate_fn = generate_fn
        self.retriever = retriever
        self.curator = curator
        self.reflector = reflector
        self.prompt_level = prompt_level
        self.use_selfeval = use_selfeval
        self.sp_step1, self.sp_step2 = sp_step1, sp_step2
        self.round_size, self.max_reflect = round_size, max_reflect
        self.use_verify, self.verify_require_pa = use_verify, verify_require_pa
        self.verify_lan = max(1, int(verify_lan))
        self.log_path = log_path

        self.playbook = render_playbook([])
        self.history: list[dict] = []
        self.qg_reasons: Counter = Counter()
        # Nội dung đã bị loại thì đừng xét lại: Reflector không nhớ, sẽ đề xuất lặp,
        # mỗi lần lặp tốn một lượt verify (một lượt sinh của model). Bản ACE gốc có
        # cơ chế này, bản port trước đây thiếu.
        self.quarantine: set[str] = set()
        self.stats: Counter = Counter()

    @staticmethod
    def _pick_failures(rows, limit):
        """Trải đều theo cụm lỗi để playbook không lệch về một họ."""
        fails = [r for r in rows if not r["ea"]]
        by_cluster = defaultdict(list)
        for r in fails:
            by_cluster[r["_cluster"]].append(r)
        picked, ring = [], list(by_cluster)
        while ring and len(picked) < limit:
            for cid in list(ring):
                if by_cluster[cid]:
                    picked.append(by_cluster[cid].pop(0))
                    if len(picked) >= limit:
                        break
                else:
                    ring.remove(cid)
        return picked

    def verify_candidates(self, candidates):
        """Sinh lại lời giải kèm bullet ứng viên — gom một lô."""
        if not candidates:
            return candidates
        if not self.use_verify:
            for c in candidates:
                c["passed"], c["verify_note"] = True, "bo_qua_verify"
            return candidates

        # Sinh LẠI 2 lần rồi lấy kết quả tốt nhất. Ở temperature 0.1 một lượt sinh vẫn
        # ngẫu nhiên, mà lượt trước verify loại tới 55 ứng viên — trong đó chắc chắn có
        # những bullet tốt bị trượt oan. Thêm một lượt sinh rẻ hơn nhiều so với mất bullet.
        for c in candidates:
            c["passed"], c["verify_note"] = False, ""
        for lan in range(self.verify_lan):
            con_lai = [c for c in candidates if not c["passed"]]
            if not con_lai:
                break
            prompts = [self.prompt_kit.step1(
                           c["sample"],
                           (c["bullets_text"] + "\n- " + c["strategy"]).strip("\n"),
                           level=self.prompt_level)
                       for c in con_lai]
            outs = self.generate_fn(prompts, self.sp_step1, desc=f"verify{lan + 1}")
            for c, raw in zip(con_lai, outs):
                prog, _ = extract_program_answer(raw)
                value = execute_program(prog, c["sample"].get("table") or []) if prog else None
                ea = check_ea(value, c["sample"]["qa"].get("exe_ans"))
                pa_s, _ = check_pa(prog, c["sample"]["qa"].get("program", ""))
                c["passed"] = bool(pa_s) if self.verify_require_pa else bool(ea or pa_s)
                c["verify_note"] = f"lần{lan + 1} ea={ea} pa={pa_s} val={value}"
        return candidates

    def run_round(self, batch, round_idx):
        """Một vòng ACE hoàn chỉnh. Trả ``(rows, info)``."""
        rows = run_pipeline(batch, self.prompt_kit, self.generate_fn,
                            prompt_level=self.prompt_level,
                            use_selfeval=self.use_selfeval,
                            playbook=self.playbook, retriever=self.retriever,
                            sp_step1=self.sp_step1, sp_step2=self.sp_step2,
                            desc=f"vòng{round_idx}", record_usage=True, keep_raw=True)
        for r, s in zip(rows, batch):
            r["_sample"] = s
            r["_cluster"] = cluster_id_for_sample(s)
            self.retriever.record_outcome(r["used_bullets"], r["pa_strict"], r["ea"])

        failures = self._pick_failures(rows, self.max_reflect)
        # Ca ĐÚNG cùng cụm, lấy ngay trong lô này — không tốn thêm lượt sinh nào.
        dung_theo_cum = {}
        for r in rows:
            if r["ea"] and r["_cluster"] not in dung_theo_cum:
                dung_theo_cum[r["_cluster"]] = r

        items = [{"sample": r["_sample"], "pred_prog": r["final_program"],
                  "row_dung": dung_theo_cum.get(r["_cluster"]),
                  "pred_value": r["pred_value"], "bullets_text": r["bullets_text"],
                  "cluster_id": r["_cluster"],
                  "diag": diagnose(r["raw_step2"] or r["raw_step1"], r["final_program"],
                                   r["pred_value"], r["gold_program"], r["gold_answer"])}
                 for r in failures]
        refs = self.reflector(items, self.playbook)

        for ref in refs:
            if ref.get("bullet_tags"):
                self.playbook = update_bullet_counts(self.playbook, ref["bullet_tags"])

        candidates = []
        for it, ref in zip(items, refs):
            strategy = (ref.get("new_strategy") or "").strip()
            if not strategy:
                self.qg_reasons["reflector_khong_de_xuat"] += 1
                continue
            _key = _noi_dung_khoa(strategy)
            if _key in self.quarantine:          # đã bị loại rồi, đừng verify lại
                self.qg_reasons["bi_cach_ly"] += 1
                continue
            action, reason = self.curator.gate(strategy, self.playbook)
            if action == "reject":
                self.qg_reasons[reason] += 1
                self.quarantine.add(_key)
                continue
            candidates.append({"sample": it["sample"], "strategy": strategy,
                               "bullets_text": it["bullets_text"],
                               "error_type": ref.get("error_type", "khac"),
                               "cluster_id": it["cluster_id"]})
        candidates = self.verify_candidates(candidates)

        added = []
        for c in candidates:
            if not c["passed"]:
                # KHÔNG cách ly: verify chạy ở temperature 0.1 nên có yếu tố ngẫu nhiên,
                # trượt một lần không có nghĩa bullet vô dụng.
                self.qg_reasons["verify_khong_sua_duoc"] += 1
                continue
            self.playbook, action, reason, bid = self.curator(
                c["strategy"], c["error_type"], self.playbook, c["cluster_id"])
            if action == "add":
                added.append((bid, c["strategy"]))
            else:
                # KHÔNG cách ly: đây là hạn ngạch cụm/mục, phụ thuộc trạng thái playbook —
                # bullet bị chặn lúc này có thể hợp lệ sau khi có bullet khác bị trục xuất.
                self.qg_reasons[reason] += 1

        self.playbook, evicted = self.retriever.enforce_budget(self.playbook)
        baseline_pa = ((sum(h["pa_strict"] for h in self.history) / len(self.history))
                       if self.history else 0.0)
        promoted = self.retriever.maybe_promote_tier1(self.playbook, baseline_pa)

        self.stats["reflected"] += len(items)
        self.stats["added"] += len(added)
        self.stats["evicted"] += len(evicted)

        self.history.extend(rows)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(
                        {k: v for k, v in r.items()
                         if k not in ("_sample", "raw_step1", "raw_step2")},
                        ensure_ascii=False, default=str) + "\n")

        return rows, {"added": added, "evicted": evicted, "promoted": promoted,
                      "n_fail": sum(1 for r in rows if not r["ea"])}
