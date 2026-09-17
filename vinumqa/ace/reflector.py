# -*- coding: utf-8 -*-
"""Reflector: đọc một ca sai rồi rút ra chiến lược TỔNG QUÁT (tiếng Việt).

Ba backend:

* ``"slm"`` (mặc định) — dùng chính model đang chạy, **không cần API key**, và mọi lời gọi
  trong một vòng được gom thành MỘT lô vLLM;
* ``"openai"`` — giống bản ACE gốc (``gpt-4o-mini``), cho bullet chất lượng cao hơn
  nhưng phải trả phí.

Lưu ý về thiết lập: cấu hình *constrained* cấm dùng LLM/API ngoài ở cả train lẫn
inference. Vậy nên **chỉ backend ``"slm"`` mới nằm trong thiết lập constrained**; dùng
``"openai"`` thì kết quả phải báo cáo ở nhóm *unconstrained*.
"""
from __future__ import annotations

import json
import os
import re

from .clusters import CLUSTER_BY_ID
from ..dsl import OP_DETECT_RE, first_op, n_ops

__all__ = ["DIAG_TYPES", "diagnose", "REFLECTOR_PROMPT",
           "build_reflector_prompt", "parse_reflector_json", "Reflector"]

DIAG_TYPES = ["lay_sai_so_lieu", "sai_phep_toan", "sai_don_vi", "sai_ham_bang", "thieu_buoc",
              "thua_buoc", "khong_co_program", "sai_dau", "sai_bac_do_lon", "sai_dinh_dang",
              "so_lieu_trong_text", "suy_luan_sai", "khac"]

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def diagnose(raw_text, pred_prog, pred_value, gold_prog, gold_ans) -> str:
    """Chẩn đoán nhanh bằng luật — làm gợi ý đầu vào cho Reflector."""
    if not pred_prog:
        return ("sai_dinh_dang"
                if ("```" in (raw_text or "") or "program" in (raw_text or "").lower())
                else "khong_co_program")
    if pred_value is None:
        return "sai_dinh_dang"
    try:
        p, g = float(pred_value), float(gold_ans)
        if g != 0:
            if p * g < 0 and abs(abs(p) - abs(g)) / abs(g) < 0.01:
                return "sai_dau"
            if p != 0:
                ratio = abs(p / g)
                for f in (10, 100, 1000, 10000, 1000000):
                    if abs(ratio - f) / f < 0.05 or abs(ratio - 1 / f) * f < 0.05:
                        return "sai_bac_do_lon"
    except (TypeError, ValueError):
        pass
    np_, ng = n_ops(pred_prog), n_ops(gold_prog)
    if ng and np_ and np_ < ng:
        return "thieu_buoc"
    if ng and np_ > ng:
        return "thua_buoc"
    if first_op(gold_prog).startswith("table_") != first_op(pred_prog).startswith("table_"):
        return "sai_ham_bang"
    if first_op(gold_prog) != first_op(pred_prog):
        return "sai_phep_toan"
    return "lay_sai_so_lieu"


REFLECTOR_PROMPT = """Bạn đang phân tích vì sao một mô hình trả lời sai bài toán suy luận số học trên báo cáo tài chính tiếng Việt.
Nhiệm vụ: rút ra MỘT CHIẾN LƯỢC TỔNG QUÁT giúp làm đúng các bài TƯƠNG TỰ — không phải lời giải cho riêng câu này.

=== MODEL ĐÃ ĐƯỢC DẶN SẴN NHỮNG ĐIỀU NÀY ===
{luat_da_co}

⚠ ĐỪNG đề xuất lại bất cứ điều gì đã nằm trong danh sách trên — nó đã có hiệu lực rồi,
nhắc lại không giúp gì. Chỉ đề xuất điều mà danh sách đó CHƯA nói tới.
⚠ NGOẠI LỆ: phần "LUẬT DSL" ngay bên dưới mới là luật ĐÚNG. Chỗ nào danh sách trên mâu
thuẫn với LUẬT DSL thì LUẬT DSL thắng, và bạn ĐƯỢC PHÉP đề xuất chiến lược theo LUẬT DSL.

=== MỘT CA CÙNG LOẠI MÀ MODEL ĐÃ LÀM ĐÚNG ===
{ca_dung}

Hãy so ca sai với ca đúng này để tìm ra ĐIỂM KHÁC BIỆT cụ thể, rồi khái quát nó lên.

=== LUẬT DSL CỦA ViNumQA ===
- Các phép: add(a,b), subtract(a,b), multiply(a,b), divide(a,b), table_max(nhãn, none), table_min(nhãn, none), table_average(nhãn, none), table_sum(nhãn, none)
- Viết phẳng, KHÔNG lồng nhau: viết "add(1, 0.15), divide(5310, #0)", KHÔNG viết "divide(5310, add(1, 0.15))"
- #0 là kết quả phép thứ nhất, #1 phép thứ hai... Ở phép thứ i chỉ được tham chiếu #N với N < i.
- KHÔNG có hằng số const_*; dùng số thường. Toán hạng dạng 20% nghĩa là 0.2.
- Tỷ lệ/phần trăm trả về SỐ THẬP PHÂN (0.15), TUYỆT ĐỐI không multiply(#n, 100).
- table_* nhận NHÃN HÀNG ở ô đầu mỗi dòng (đo trên gold: 618/618), không nhận tên cột hay năm.
- Hai số khác đơn vị phải quy đổi bằng multiply/divide NGAY TRONG program.

=== CỤM BÀI TOÁN ===
Cụm: {cluster_id}
Đặc điểm: {cluster_lesson}
Mẫu chuẩn: {cluster_pattern}
Các kiểu làm sai hay gặp ở cụm này:
{cluster_wrong}

=== CA ĐANG XÉT ===
Câu hỏi: {question}
Ngữ cảnh (rút gọn): {context}
Program vàng: {gold_prog}   →  Đáp án vàng: {gold_ans}
Program của model: {pred_prog}   →  Kết quả: {pred_value}
Loại lỗi đã chẩn đoán: {diag}
Các bullet đã được đưa vào prompt lúc model trả lời:
{bullets}

=== CÁC BULLET ĐÃ CÓ TRONG PLAYBOOK (KHÔNG ĐƯỢC TRÙNG) ===
{existing}

=== ĐẦU RA ===
Chỉ in ra DUY NHẤT một JSON, không thêm lời dẫn:
{{
  "error_type": "một trong: {diag_types}",
  "root_cause": "một câu nêu đúng bản chất lỗi",
  "new_strategy": "một câu quy tắc tổng quát, xem RÀNG BUỘC bên dưới",
  "bullet_tags": [{{"id": "cl-00001", "tag": "helpful|harmful|neutral"}}]
}}

=== RÀNG BUỘC BẮT BUỘC CHO new_strategy ===
1. Bắt đầu bằng "Khi ..." nêu rõ điều kiện kích hoạt (bám từ khoá của cụm {cluster_id}).
2. Là quy tắc TỔNG QUÁT, áp dụng được cho nhiều câu, KHÔNG phải lời giải của câu này.
3. TUYỆT ĐỐI không chứa số liệu cụ thể của ca này và không chứa năm. Chỉ được dùng các số 1, 2, 3, 10, 100, 1000, 1000000.
4. Phải chứa ÍT NHẤT MỘT phép toán DSL kèm dấu ngoặc. Bài MỘT phép chiếm 64 % tập test, nên chiến lược một phép — ví dụ table_max(ten_chi_tieu, none) — cũng rất đáng giá; đừng gò cho đủ hai phép.
5. Toán hạng dùng tên biến tiếng Việt không dấu: gia_tri_moi, gia_tri_cu, gia_tri_hien_tai, tong, phan, tu_so, mau_so, ty_le_tang, he_so, ten_chi_tieu, a, b, v1, v2.
6. Nêu RÕ mẫu số của mọi phép divide.
7. Không nhắc tên công ty, tên cột, nhãn hàng cụ thể.
8. Không quá 200 ký tự, viết bằng tiếng Việt.
9. Không lặp lại ý của các bullet đã có ở trên.

=== MẪU THAM KHẢO (chỉ để học văn phong) ===
- Khi hỏi TỶ LỆ tăng/giảm giữa hai kỳ, dùng subtract(gia_tri_moi, gia_tri_cu), divide(#0, gia_tri_cu) và giữ nguyên dấu âm nếu giảm.
- Khi hai số liệu khác đơn vị, quy đổi trước bằng multiply(gia_tri, 1000) rồi mới divide(phan, #0).
- Khi hỏi giá trị lớn nhất của một chỉ tiêu trong bảng, dùng table_max(ten_chi_tieu, none) rồi mới đưa vào add/subtract nếu cần."""


def summarize_playbook(playbook, all_bullets_fn, max_show=14) -> str:
    bullets = all_bullets_fn(playbook)
    if not bullets:
        return "(playbook đang rỗng)"
    lines = []
    for b in bullets[:max_show]:
        ops = "->".join(o for o in OP_DETECT_RE.findall(b["content"])[:4]) or "khong_co_op"
        m = re.match(r"^\s*(?:khi|nếu|với)\s+([^,.;:]{1,45})", b["content"], re.I)
        trig = m.group(1).strip() if m else "(không rõ điều kiện)"
        lines.append(f"  - [{b['id']}] '{trig}' -> {ops}")
    return "\n".join(lines)


#: Chốt phòng hờ. Prompt dùng thật đã sửa chỗ dạy ngược về ``table_*``, nên danh sách
#: này thường lọc ra 0 dòng. Giữ lại để nếu ai đó lỡ đưa bản gốc chưa sửa vào thì
#: Reflector vẫn không bị cấm đề xuất đúng luật nhãn-hàng.
_DONG_DAY_NGUOC = ("table_max(tên cột", "table_min(tên cột", "table_average(tên cột",
                   "table_sum(tên cột", "tổng của cột", "tên của cột",
                   "chỉ nhận đúng 1 cột", "nhất trong cột")

_MUC_TO_ATTR = {"basic": "BASIC_SYSTEM_PROMPT",
                "no_fewshot": "NO_FEWSHOT_SYSTEM_PROMPT",
                "engineered": "ENGINEERED_SYSTEM_PROMPT"}


def luat_dang_ap_dung(prompt_kit, level="engineered", gioi_han=1800) -> str:
    """Trích phần ánh xạ từ khoá → phép toán của system prompt ĐANG DÙNG THẬT.

    Reflector trước đây KHÔNG biết system prompt đã dặn model những gì, nên nó đề xuất
    lại chính các quy tắc đã có — bullet đúng nhưng thừa. Đưa phần này vào để nó tránh.

    Hai chỗ bản trước làm sai, đều đã đo:

    * Nó đọc cứng ``ENGINEERED_SYSTEM_PROMPT`` bất kể nấc nào đang chạy. Ở **nấc 5c**
      model chỉ nhận prompt ``basic`` — vốn KHÔNG có khối ánh xạ từ khoá — nhưng
      Reflector vẫn bị bảo "model đã được dặn 14 mục này rồi, đừng nhắc lại". Thế là
      triệt tiêu đúng lý do tồn tại của 5c.
    * Khối đó chứa các dòng dạy NGƯỢC về ``table_*``. Cấm Reflector nhắc lại chúng
      nghĩa là cấm luôn nó phát hiện ra luật nhãn-hàng.
    """
    p = getattr(prompt_kit, _MUC_TO_ATTR.get(level, "ENGINEERED_SYSTEM_PROMPT"), "") or ""
    i = p.find("=== HƯỚNG DẪN CHỌN PHÉP TOÁN THEO TỪ KHÓA ===")
    j = p.find("=== VÍ DỤ ===")
    if not (0 <= i < j):
        # `basic` không có khối này → nói thẳng là CHƯA dặn gì, đừng bịa ra luật đã có.
        return ("(system prompt ở nấc này KHÔNG chứa hướng dẫn chọn phép toán nào — "
                "mọi quy tắc chọn phép đều còn trống, cứ đề xuất)")
    doan = p[i:j].strip()[:gioi_han]
    giu = [ln for ln in doan.splitlines()
           if not any(x in ln for x in _DONG_DAY_NGUOC)]
    return chr(10).join(giu)


def ca_lam_dung(row, prompt_kit, gioi_han=420) -> str:
    """Một ca CÙNG CỤM LỖI mà model đã làm đúng — để Reflector thấy 'đúng trông thế nào'.

    Bản ACE gốc gọi là counterfactual retrieval; bản port trước đây thiếu hẳn.
    """
    if not row:
        return "(chưa có ca đúng nào cùng cụm trong lô này)"
    q = row.get("question", "")[:200]
    return f"Câu hỏi: {q}\nProgram đúng: {row.get('final_program', '')[:gioi_han]}"


def build_reflector_prompt(sample, pred_prog, pred_value, bullets_text, diag,
                           playbook, cluster_id, prompt_kit, all_bullets_fn,
                           row_dung=None, prompt_level="engineered") -> str:
    c = CLUSTER_BY_ID.get(cluster_id, CLUSTER_BY_ID["C11_khac"])
    pre, post, table = prompt_kit.context_block(sample)
    context = (f"{pre} {post}".strip()[:500] + "\n" + table[:400]).strip()
    return REFLECTOR_PROMPT.format(
        luat_da_co=luat_dang_ap_dung(prompt_kit, prompt_level),
        ca_dung=ca_lam_dung(row_dung, prompt_kit),
        cluster_id=c["id"], cluster_lesson=c["lesson"], cluster_pattern=c["pattern"],
        cluster_wrong="\n".join(f"  - {w}" for w in c["wrong"]) or "  (chưa ghi nhận)",
        question=sample["qa"]["question"], context=context,
        gold_prog=sample["qa"]["program"], gold_ans=sample["qa"].get("exe_ans"),
        pred_prog=pred_prog or "(không sinh được)", pred_value=pred_value,
        diag=diag, bullets=bullets_text or "(chưa có bullet nào)",
        existing=summarize_playbook(playbook, all_bullets_fn),
        diag_types=", ".join(DIAG_TYPES))


def parse_reflector_json(raw, fallback="khac") -> dict:
    default = {"error_type": fallback, "root_cause": "", "new_strategy": "", "bullet_tags": []}
    if not raw:
        return default
    text = _THINK_RE.sub("", raw).strip()
    candidates = [text]
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        candidates.append(m.group(1))
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            obj.setdefault("error_type", fallback)
            obj.setdefault("new_strategy", "")
            obj.setdefault("bullet_tags", [])
            if not isinstance(obj["bullet_tags"], list):
                obj["bullet_tags"] = []
            return obj
    return default


class Reflector:
    """Gọi Reflector theo lô.

    Parameters
    ----------
    backend : ``"slm"`` | ``"openai"``
    generate_fn : hàm ``(prompts, sampling_params) -> list[str]`` — chỉ dùng cho backend slm.
    """

    def __init__(self, prompt_kit, all_bullets_fn, backend="slm", generate_fn=None,
                 sampling_params=None, api_model="gpt-4o-mini",
                 temperature=0.0, max_tokens=512, prompt_level="engineered"):
        self.prompt_kit = prompt_kit
        self.all_bullets_fn = all_bullets_fn
        self.backend = backend
        self.generate_fn = generate_fn
        self.sampling_params = sampling_params
        self.api_model = api_model
        self.temperature, self.max_tokens = temperature, max_tokens
        self.prompt_level = prompt_level
        self.n_calls = 0

    def build_prompts(self, items, playbook):
        return [build_reflector_prompt(it["sample"], it["pred_prog"], it["pred_value"],
                                       it["bullets_text"], it["diag"], playbook,
                                       it["cluster_id"], self.prompt_kit,
                                       self.all_bullets_fn, it.get("row_dung"),
                                       prompt_level=self.prompt_level)
                for it in items]

    def __call__(self, items, playbook) -> list[dict]:
        """``items``: list dict(sample, pred_prog, pred_value, bullets_text, diag, cluster_id)."""
        if not items:
            return []
        prompts = self.build_prompts(items, playbook)
        self.n_calls += len(prompts)

        if self.backend == "openai":
            outs = self._call_openai(prompts)
        else:
            formatted = [self.prompt_kit.chat([{"role": "user", "content": p}])
                         for p in prompts]
            outs = self.generate_fn(formatted, self.sampling_params)

        return [parse_reflector_json(o, fallback=it["diag"]) for o, it in zip(outs, items)]

    def _call_openai(self, prompts):
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        outs = []
        for p in prompts:
            try:
                r = client.chat.completions.create(
                    model=self.api_model, messages=[{"role": "user", "content": p}],
                    temperature=self.temperature, response_format={"type": "json_object"},
                    max_tokens=self.max_tokens)
                outs.append(r.choices[0].message.content)
            except Exception as e:                        # noqa: BLE001
                print(f"[REFLECT] ⚠ OpenAI lỗi: {e}")
                outs.append(None)
        return outs
