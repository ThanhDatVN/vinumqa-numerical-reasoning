# -*- coding: utf-8 -*-
"""Thang prompt LỒNG NHAU (basic ⊂ no_fewshot ⊂ engineered), cộng prompt self-eval.

Ba mức cắt ra từ CHÍNH prompt hoàn chỉnh (:func:`PromptKit.bo_vi_du`,
:func:`PromptKit.bo_huong_dan_tu_khoa`), nên phần dùng chung giống nhau từng ký tự và
mỗi bước là một phép **chèn thuần** — hiệu số giữa hai nấc kề nhau vì thế đo đúng phần
vừa thêm, không lẫn chuyện viết lại câu chữ. Có test canh điều đó.

======================  ==========================================================
``basic``               Nấc 1. Mở đầu + danh sách phép toán (có mô tả từng phép) +
                        toàn bộ yêu cầu định dạng đầu ra.
``no_fewshot``          Bậc giữa, KHÔNG phải nấc phải chạy. = ``basic`` + ánh xạ
                        **từ khoá tiếng Việt → phép toán** (14 mục).
``engineered``          Nấc 2 trở đi. = ``no_fewshot`` + 2 ví dụ mẫu có khung.
``self_eval``           Prompt bước 2: đưa lại ngữ cảnh + lời giải bước 1, yêu
                        cầu model tự soát và sửa.
======================  ==========================================================

Prompt gốc nằm trong :mod:`vinumqa._prompt_text`, chép **nguyên văn** từ
``reference/original_notebooks/inference_with_difference_models.ipynb`` — có test canh
từng ký tự. Nhưng bản gốc có **ba chỗ nói SAI so với chính dữ liệu gold**, nên
:func:`PromptKit.theo_du_lieu` sửa chúng trước khi dùng. Mỗi chỗ sửa neo vào một con số
đếm được trên toàn bộ 4 074 mẫu train+valid+test, không phải ý kiến:

=========================  ==========================================  ==============
bản gốc nói                dữ liệu gold nói                            số đo
=========================  ==========================================  ==============
``table_*`` nhận TÊN CỘT   nhận NHÃN HÀNG (ô đầu mỗi dòng)             618/618 khớp
                                                                       nhãn hàng,
                                                                       0 khớp tên cột
``add(#0,c), add(#0,d)``   mỗi bước tham chiếu bước NGAY TRƯỚC         2053 lần ngay
                                                                       trước / 109 lùi
                                                                       xa hơn
"giảm ⇒ kết quả luôn âm"   chỉ đúng khi hỏi TỶ LỆ giảm (có divide);    tỷ lệ: 61/74
                           hỏi MỨC giảm tuyệt đối thì lấy dương        giữ âm (82 %)
                                                                       mức: 45/74 lấy
                                                                       dương (61 %)
=========================  ==========================================  ==============

Sửa ở **cả** bước 1 lẫn bước 2 — sửa bước 1 mà bỏ bước 2 thì bước 2 dạy lại điều sai
ngay sau khi bước 1 vừa làm đúng.
"""
from __future__ import annotations

from ._prompt_text import (SYSTEM_PROMPT_STEP_1, SYSTEM_PROMPT_STEP_2,
                           table_to_str as _table_to_str)

__all__ = ["PromptKit", "strip_assistant",
           "MEMORY_ASK", "MEMORY_PREFIX", "MEMORY_SUFFIX"]

# ─────────────────── lượt "hồi tưởng" để chèn playbook ACE ───────────────────

MEMORY_ASK = ("Trước khi vào câu hỏi thật, bạn hãy nhắc lại những chiến lược đã rút ra "
              "từ các bài toán tài chính tương tự.")

MEMORY_PREFIX = ("Đây là những chiến lược tôi đã rút ra từ các bài toán tài chính số học "
                 "tương tự. Tôi sẽ dùng chúng như gợi ý phụ; các quy tắc trong system prompt "
                 "vẫn được ưu tiên cao nhất.\n\n")

MEMORY_SUFFIX = "\n\nBây giờ tôi đã sẵn sàng nhận câu hỏi."


class PromptKit:
    """Dựng prompt cho một tokenizer cụ thể, ở mức chi tiết tuỳ chọn.

    Parameters
    ----------
    tokenizer : tokenizer của model; ``None`` thì dùng định dạng ``<<SYSTEM>>`` thuần.
    model_name : dùng để tự tắt thinking-mode với Qwen3.
    max_ctx_chars, max_prev_chars : trần ký tự cho ngữ cảnh / lời giải bước 1.
        ``None`` = không cắt. Chỉ cần đặt khi chạy trên GPU nhỏ.
    """

    #: Mức chạy được. ``no_fewshot`` là BẬC GIỮA — không phải nấc phải chạy, nó ở đây
    #: để bộ kiểm chứng minh thang prompt là phép CHÈN THUẦN từng bước.
    LEVELS = ("basic", "no_fewshot", "engineered")

    def __init__(self, tokenizer=None, model_name: str = "",
                 enable_thinking: bool | None = None,
                 max_ctx_chars: int | None = None, max_prev_chars: int | None = None):
        # Prompt DÙNG THẬT = bản gốc đã sửa ba chỗ mà dữ liệu gold bác bỏ.
        # Bản gốc chưa sửa vẫn nằm nguyên ở `_prompt_text` để truy xuất xứ.
        self.ENGINEERED_SYSTEM_PROMPT = self.theo_du_lieu(SYSTEM_PROMPT_STEP_1)
        self.SELF_EVAL_SYSTEM_PROMPT = self.theo_du_lieu_step2(SYSTEM_PROMPT_STEP_2)
        self.NO_FEWSHOT_SYSTEM_PROMPT = self.bo_vi_du(self.ENGINEERED_SYSTEM_PROMPT)
        # Thang prompt LỒNG NHAU — mỗi nấc thêm đúng một khối, cắt ra từ chính prompt
        # hoàn chỉnh nên phần dùng chung giống nhau từng ký tự. Nhờ vậy hiệu số giữa hai
        # nấc kề nhau đo đúng phần vừa thêm, không lẫn chữ nghĩa viết lại.
        #
        #   basic       = mở đầu + DANH SÁCH PHÉP TOÁN + yêu cầu định dạng
        #   no_fewshot  = basic + HƯỚNG DẪN CHỌN PHÉP THEO TỪ KHÓA (14 mục)
        #   engineered  = no_fewshot + 2 ví dụ mẫu có khung
        self.BASIC_SYSTEM_PROMPT = self.bo_vi_du(
            self.bo_huong_dan_tu_khoa(self.ENGINEERED_SYSTEM_PROMPT))
        self.table_to_str = staticmethod(_table_to_str).__func__

        self.tokenizer = tokenizer
        self.model_name = model_name
        # None = KHÔNG truyền enable_thinking, để chat template tự quyết — đúng như bản
        # tham chiếu gọi ``apply_chat_template(messages, tokenize=False,
        # add_generation_prompt=True)``. Với Qwen3, mặc định của template là **bật** suy
        # nghĩ.
        #
        # ⚠ Đừng ép False ở đây. Bản trước ép tắt cho Qwen3 và mất 10 điểm PA_loose
        # (41.05 % thay vì ~51 %), vì bài toán suy luận số cần chuỗi suy luận. Dấu hiệu
        # nhận ra: 497 mẫu chạy xong trong ~1 phút thay vì ~9 phút.
        self.enable_thinking = enable_thinking

        self.max_ctx_chars = max_ctx_chars
        self.max_prev_chars = max_prev_chars
        self.n_truncated = 0

    # ── tiện ích ──
    def has_chat_template(self) -> bool:
        return getattr(self.tokenizer, "chat_template", None) is not None

    def chat(self, messages) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self.enable_thinking is not None:
            kwargs["enable_thinking"] = self.enable_thinking
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def context_block(self, sample) -> tuple[str, str, str]:
        pre = " ".join(sample.get("pre_text", []) or []).strip()
        post = " ".join(sample.get("post_text", []) or []).strip()
        table = self.table_to_str(sample.get("table", []) or [])

        if self.max_ctx_chars and len(pre) + len(post) + len(table) > self.max_ctx_chars:
            self.n_truncated += 1
            budget = max(0, self.max_ctx_chars - len(table))   # bảng được giữ nguyên
            if budget <= 0:
                pre, post = "", ""
            else:
                half = budget // 2
                if len(pre) > half:
                    pre = pre[:half] + " […đã cắt bớt]"
                keep = budget - min(len(pre), half)
                if len(post) > keep:
                    post = ("[…đã cắt bớt] " + post[-keep:]) if keep > 0 else ""
        return pre, post, table

    @staticmethod
    def _memory_turns(bullets_text: str) -> list[dict]:
        if not bullets_text or not bullets_text.strip():
            return []
        return [{"role": "user", "content": MEMORY_ASK},
                {"role": "assistant",
                 "content": MEMORY_PREFIX + bullets_text.strip() + MEMORY_SUFFIX}]

    def _render(self, system_prompt: str, user_message: str, bullets_text: str = "") -> str:
        if self.tokenizer is not None and self.has_chat_template():
            messages = ([{"role": "system", "content": system_prompt}]
                        + self._memory_turns(bullets_text)
                        + [{"role": "user", "content": user_message}])
            return self.chat(messages)
        memory = (MEMORY_PREFIX + bullets_text + MEMORY_SUFFIX) if bullets_text.strip() else ""
        return (f"<<SYSTEM>>\n{system_prompt}\n{memory}\n\n"
                f"<<USER>>\n{user_message}\n\n<<ASSISTANT>>\n")

    # ── nội dung user message ──
    def _user_engineered(self, sample) -> str:
        """Giữ nguyên user message của bản tham chiếu."""
        pre, post, table = self.context_block(sample)
        return f"""Câu hỏi: {sample["qa"]["question"]}
Câu trả lời phải tuân theo định dạng:
```plaintext
program:.....
answer:....
```
Đây là nội dung liên quan đến câu hỏi:
Pre-text: {pre}
Post-text: {post}
Bảng:
{table}

Phân tích bằng tiếng việt, dừng trả lời sau khi đưa ra câu trả lời cuối cùng trong khối ```plaintext ... ```
Output:"""

    # ── API chính ──
    def step1(self, sample, bullets_text: str = "", level: str = "engineered",
              vi_du_dong: str | None = None) -> str:
        """Prompt sinh program. ``level`` ∈ :attr:`LEVELS` (thang lồng nhau).

        ``vi_du_dong`` — khối ví dụ TRUY HỒI (xem :mod:`vinumqa.fewshot`). Truyền vào
        thì 2 ví dụ cố định bị **thay** bằng nó; mọi phần khác của prompt giữ nguyên
        từng ký tự, nên hiệu số so với nấc 2 quy đúng về chuyện đổi ví dụ.
        Chỉ có nghĩa với ``engineered`` — ``basic``/``no_fewshot`` vốn không có khối ví dụ.
        """
        if level == "engineered":
            sp = self.ENGINEERED_SYSTEM_PROMPT
            if vi_du_dong:
                sp = self.thay_vi_du(sp, vi_du_dong)
            return self._render(sp, self._user_engineered(sample), bullets_text)
        if level in ("basic", "no_fewshot"):
            if vi_du_dong:
                raise ValueError(f"level {level!r} không có khối ví dụ để thay — "
                                 f"ví dụ động chỉ dùng được với 'engineered'")
            sp = (self.BASIC_SYSTEM_PROMPT if level == "basic"
                  else self.NO_FEWSHOT_SYSTEM_PROMPT)
            return self._render(sp, self._user_engineered(sample), bullets_text)
        raise ValueError(f"level lạ: {level!r}, chọn trong {self.LEVELS}")

    # ── biến thể bỏ ví dụ mẫu ──
    _MOC_VI_DU = "=== VÍ DỤ ==="
    _MOC_SAU_VI_DU = "==== CÂU HỎI ===="
    #: Chỗ bản gốc dạy NGƯỢC về ``table_*``. Đo trên 4 074 mẫu: 618/618 tham số khớp
    #: NHÃN HÀNG, 0 khớp tên cột. Xem :func:`theo_du_lieu`.
    _SUA_TABLE = (
        ("- table_max(column, none): Giá trị lớn nhất của cột.",
         "- table_max(nhãn_hàng, none): Giá trị lớn nhất của HÀNG mang nhãn đó."),
        ("- table_min(column, none): Giá trị nhỏ nhất của cột.",
         "- table_min(nhãn_hàng, none): Giá trị nhỏ nhất của HÀNG mang nhãn đó."),
        ("- table_average(column, none): Giá trị trung bình của cột.",
         "- table_average(nhãn_hàng, none): Giá trị trung bình của HÀNG mang nhãn đó."),
        ("- table_sum(column, none): Tổng cả cột (chỉ dùng khi hỏi tổng cả cột/hàng, "
         "không dùng cho 2-3 ô).",
         "- table_sum(nhãn_hàng, none): Tổng cả HÀNG mang nhãn đó (chỉ dùng khi hỏi tổng "
         "cả hàng, không dùng cho 2-3 ô)."),
        ("→ Lưu ý quan trọng: table_ functions chỉ nhận đúng 1 cột, không thêm hàng.",
         "→ ⚠ THAM SỐ LÀ NHÃN HÀNG, KHÔNG PHẢI TÊN CỘT. Nhãn hàng là chuỗi nằm ở Ô ĐẦU "
         "TIÊN của mỗi dòng trong bảng. Chép lại NGUYÊN VĂN nhãn đó, không rút gọn, "
         "không dịch, không thêm đơn vị. Tuyệt đối đừng truyền tên cột hay năm."),
        ("   → table_max(tên cột, none)", "   → table_max(nhãn hàng, none)"),
        ("   → table_min(tên cột, none)", "   → table_min(nhãn hàng, none)"),
        ("   → table_average(tên cột, none)", "   → table_average(nhãn hàng, none)"),
        ("11. Các cụm từ KHÔNG dùng table_sum (vì chỉ tính tổng 1 cột/hàng):",
         "11. Các cụm từ KHÔNG dùng table_sum (vì nó chỉ tính tổng ĐÚNG MỘT hàng):"),
        ("    - “tổng của cột X” → mới dùng table_sum(tên cột, none)",
         "    - “tổng của hàng X” → mới dùng table_sum(nhãn hàng X, none)"),
        ("12. Các từ khóa thường dùng table_ (không cần add/subtract thủ công):",
         "12. Các từ khóa thường dùng table_ (không cần add/subtract thủ công) — nhớ "
         "truyền NHÃN HÀNG:"),
        ("13. Khi cần tìm số lớn nhất trong cột nhưng số đó được dùng để tính toán",
         "13. Khi cần tìm số lớn nhất trong hàng nhưng số đó được dùng để tính toán"),
        ("    - cao nhất, lớn nhất, thấp nhất, nhỏ nhất, trung bình, bình quân, tổng của "
         "cột, sử dụng tên của cột chứ không cho mảng vào",
         "    - cao nhất, lớn nhất, thấp nhất, nhỏ nhất, trung bình, bình quân, tổng của "
         "hàng; truyền NHÃN HÀNG (ô đầu dòng) chứ không cho mảng vào"),
    )

    #: Chỗ bản gốc SAI TOÁN: ``add(#0,d)`` cộng dồn bỏ mất số hạng ``c``. Đo trên gold:
    #: 2 053 tham chiếu trỏ bước NGAY TRƯỚC, chỉ 109 lùi xa hơn. Có ở CẢ hai prompt.
    _SUA_CHUOI = (
        ("   → Dùng add liên tiếp: add(a,b), add(#0,c), add(#0,d)...",
         "   → Dùng add liên tiếp: add(a,b), add(#0,c), add(#1,d)... "
         "(mỗi bước tham chiếu kết quả NGAY TRƯỚC nó, đừng quay lại #0)"),
    )

    #: Chỗ bản gốc nói dấu SAI. Bản gốc: "giảm ⇒ kết quả sẽ âm", tuyệt đối. Đo trên
    #: gold thì phụ thuộc dạng câu: hỏi TỶ LỆ giảm (có divide) → 61/74 giữ dấu âm (82 %);
    #: hỏi MỨC giảm tuyệt đối (không divide) → 45/74 lấy giá trị dương (61 %).
    _SUA_DAU_GIAM = (
        ("2. Giảm bao nhiêu % / Giảm …% yoy / -…% yoy\n"
         "   → Vẫn dùng đúng pattern trên (kết quả sẽ âm → đúng bản chất giảm)",
         "2. Giảm bao nhiêu % / Giảm …% yoy / -…% yoy — DẤU phụ thuộc dạng câu:\n"
         "   → Hỏi TỶ LỆ giảm (có chia): subtract(mới, cũ), divide(#0, cũ) — GIỮ dấu âm\n"
         "   → Hỏi MỨC giảm tuyệt đối (không chia), nhất là khi trừ hai số vốn đã là %:\n"
         "     subtract(số_lớn, số_nhỏ) — trả về giá trị DƯƠNG"),
    )

    #: Bước 2 dùng chung 12/13 mốc với bước 1; riêng mục 13 viết khác nên phải nêu riêng.
    _SUA_TABLE_STEP2 = (
        ("13. Khi cần tìm số lớn nhất hoặc nhỏ nhất trong cột nhưng số đó được dùng để "
         "tính toán",
         "13. Khi cần tìm số lớn nhất hoặc nhỏ nhất trong hàng nhưng số đó được dùng để "
         "tính toán"),
    )

    @staticmethod
    def _ap_dung_sua(prompt: str, cap) -> str:
        """Thay từng cặp (cũ → mới), NÉM LỖI nếu mốc không còn — đừng sửa im lặng."""
        for cu, moi in cap:
            if cu not in prompt:
                raise ValueError(f"không thấy mốc cần sửa: {cu[:60]!r}")
            prompt = prompt.replace(cu, moi, 1)
        return prompt

    @classmethod
    def theo_du_lieu(cls, prompt: str) -> str:
        """Sửa BƯỚC 1 ở ba chỗ bản gốc nói sai so với chính dữ liệu gold.

        Không phải chỉnh văn phong — mỗi chỗ đều neo vào một con số đếm được trên toàn
        bộ 4 074 mẫu train+valid+test:

        * ``table_*`` — bản gốc nói tham số là TÊN CỘT. Executor
          (:func:`vinumqa.dsl.table_row_values`) đọc theo NHÃN HÀNG, và gold cũng vậy:
          **618/618** tham số khớp nhãn hàng, **0** khớp tên cột. Bản gốc còn viết
          "chỉ nhận đúng 1 cột, không thêm hàng" — cấm thẳng thứ cần làm.
        * **chuỗi ``#N``** — bản gốc dạy ``add(#0,c), add(#0,d)``, tức bỏ mất số hạng
          ``c``. Gold: **2 053** tham chiếu trỏ bước ngay trước, **109** lùi xa hơn.
        * **dấu của câu "giảm"** — bản gốc nói kết quả *luôn* âm. Gold chia hai nhánh
          rõ rệt theo dạng câu; xem :attr:`_SUA_DAU_GIAM`.

        Bản gốc chưa sửa vẫn nằm nguyên ở :mod:`vinumqa._prompt_text` để truy xuất xứ,
        có test canh khớp từng ký tự với notebook tham chiếu.
        """
        return cls._ap_dung_sua(prompt,
                                cls._SUA_TABLE + cls._SUA_CHUOI + cls._SUA_DAU_GIAM)

    @classmethod
    def theo_du_lieu_step2(cls, prompt: str) -> str:
        """Y như :func:`theo_du_lieu` nhưng cho prompt BƯỚC 2.

        Bước 2 chép gần nguyên khối hướng dẫn của bước 1 nên mang theo **toàn bộ** chỗ
        nói sai. Sửa bước 1 mà bỏ bước 2 thì bước 2 dạy lại điều sai ngay sau khi bước 1
        vừa làm đúng.
        """
        cap = [(cu, moi) for cu, moi in cls._SUA_TABLE if cu in prompt]
        # Đã đo: đúng 12/13 mốc của bước 1 có mặt ở bước 2. Chốt con số lại để lần sau
        # ai sửa prompt gốc thì vỡ ở đây, chứ không im lặng bỏ sót một chỗ nói sai.
        if len(cap) != len(cls._SUA_TABLE) - 1:
            raise ValueError(f"bước 2 khớp {len(cap)} mốc, chờ {len(cls._SUA_TABLE) - 1} "
                             f"— prompt gốc đã đổi, kiểm lại _SUA_TABLE_STEP2")
        return cls._ap_dung_sua(prompt, tuple(cap) + cls._SUA_TABLE_STEP2
                                + cls._SUA_CHUOI + cls._SUA_DAU_GIAM)

    _MOC_HUONG_DAN = "=== HƯỚNG DẪN CHỌN PHÉP TOÁN THEO TỪ KHÓA ==="
    _MOC_DINH_DANG = "- Trả lời đúng 2 dòng:"

    @classmethod
    def bo_huong_dan_tu_khoa(cls, prompt: str) -> str:
        """Cắt khối 14 mục ánh xạ từ khoá → phép toán, giữ nguyên phần còn lại.

        Phần còn lại vẫn có: mở đầu, DANH SÁCH PHÉP TOÁN (mô tả từng phép) và TOÀN BỘ
        yêu cầu định dạng đầu ra. Đó là "prompt cơ bản": model biết có những phép nào
        và phải trả lời ra sao, nhưng không được mách chọn phép nào cho loại câu hỏi nào.

        Khác hẳn ``PLAIN_SYSTEM_PROMPT`` (530 ký tự) — bản đó thiếu cả quy tắc định
        dạng nên 32 % mẫu sinh ra program không chạy được, tức là một cái sàn hỏng:
        hiệu số so với nấc 2 khi ấy chủ yếu là "model mới biết viết đúng cú pháp",
        chứ không đo được prompt engineering.
        """
        i, j = prompt.find(cls._MOC_HUONG_DAN), prompt.find(cls._MOC_DINH_DANG)
        if i == -1 or j == -1 or j <= i:
            raise ValueError("không thấy khối hướng dẫn từ khoá — prompt đã đổi, "
                             "kiểm tra lại mốc")
        return prompt[:i] + prompt[j:]

    @classmethod
    def bo_vi_du(cls, prompt: str) -> str:
        """Cắt khối 2 ví dụ mẫu ra khỏi prompt engineered, giữ nguyên phần còn lại.

        Tách được vì khối ví dụ nằm gọn giữa hai mốc tiêu đề. Mọi thứ khác — danh sách
        phép toán, ánh xạ từ khoá tiếng Việt, quy tắc bắt buộc — giữ nguyên từng ký tự.
        Nhờ vậy hiệu số so với nấc 2 đo đúng phần đóng góp của **ví dụ mẫu**, không lẫn
        thứ gì khác.
        """
        i, j = prompt.find(cls._MOC_VI_DU), prompt.find(cls._MOC_SAU_VI_DU)
        if i == -1 or j == -1 or j <= i:
            raise ValueError("không thấy khối ví dụ — prompt đã đổi, kiểm tra lại mốc")
        return prompt[:i] + prompt[j:]

    @classmethod
    def thay_vi_du(cls, prompt: str, van_ban: str) -> str:
        """Thay RUỘT khối ``=== VÍ DỤ ===`` bằng ví dụ truy hồi, giữ nguyên phần khác.

        Dùng chung hai mốc với :func:`bo_vi_du`, nên chỗ thay luôn nằm đúng khối ví dụ
        và không bao giờ liếm sang danh sách phép toán hay quy tắc định dạng.
        """
        i, j = prompt.find(cls._MOC_VI_DU), prompt.find(cls._MOC_SAU_VI_DU)
        if i == -1 or j == -1 or j <= i:
            raise ValueError("không thấy khối ví dụ — prompt đã đổi, kiểm tra lại mốc")
        if not (van_ban or "").strip():
            raise ValueError("ví dụ động rỗng — truyền None nếu muốn giữ ví dụ cố định")
        return (prompt[:i] + cls._MOC_VI_DU + "\n" + van_ban.strip() + "\n" + prompt[j:])

    def step2(self, sample, initial_response: str, bullets_text: str = "",
              gia_tri_buoc1=...) -> str:
        """Prompt self-evaluation.

        ``gia_tri_buoc1`` là KẾT QUẢ THỰC THI chương trình bước 1 (``...`` = không truyền,
        giữ đúng bản tham chiếu). Đây là thông tin bước 2 vốn KHÔNG có: nó chỉ thấy văn
        bản chương trình, không biết chương trình đó chạy ra số bao nhiêu. Mà độ lớn của
        con số là chỗ lộ lỗi rõ nhất — "tỷ lệ tăng trưởng = 15000" thì sai ngay từ cái
        nhìn đầu, dù chương trình viết đúng cú pháp.
        """
        question = sample["qa"]["question"]
        pre, post, table = self.context_block(sample)
        if self.max_prev_chars and len(initial_response) > self.max_prev_chars:
            # Giữ PHẦN CUỐI: khối ```plaintext nằm ở đó.
            initial_response = ("[…phần đầu đã cắt bớt]\n"
                                + initial_response[-self.max_prev_chars:])

        if gia_tri_buoc1 is ...:
            _khoi_gia_tri = ""
        elif gia_tri_buoc1 is None:
            _khoi_gia_tri = ("\n⚙ Chạy thật chương trình trên bằng máy: KHÔNG CHẠY ĐƯỢC "
                             "(sai cú pháp, sai tham chiếu #N, hoặc nhãn bảng không khớp).\n")
        else:
            _khoi_gia_tri = (f"\n⚙ Chạy thật chương trình trên bằng máy thì ra: "
                             f"{gia_tri_buoc1}\n"
                             f"Hãy xem con số này có HỢP LÝ với câu hỏi không — sai độ lớn, "
                             f"sai dấu, hay tỷ lệ vượt quá mức có thể — rồi mới soát tiếp.\n")

        user_message = f"""Câu hỏi: {question}
Đây là nội dung liên quan đến câu hỏi:
Pre-text: {pre}
Post-text: {post}
Bảng:
{table}
Đây là phân tích và kết quả:
{initial_response}
{_khoi_gia_tri}
====NHIỆM VỤ CỦA BẠN====
Hãy phân tích và kiểm tra xem phân tích và kết quả trước đã chính xác thỏa mãn các điều kiện chưa, nếu sai thì hãy sửa lại, thêm bớt cho đúng:
Các điều cần chú ý khi phân tích:
1. không dùng multiply(#n, 100) để tính phần trăm, tỉ lệ, tóm lại nếu có multiply(#n, 100) mà 100 có ý nghĩa 100% thì loại bỏ nó.
2. viết chưa đúng định dạng program:... answer:... sau khi phân tích hãy viết lại đúng kết quả theo cấu trúc program:.... answer:... trong ```plaintext ... ```.
3. Nếu là số nguyên thì cần viết thành số nguyên chứ không thêm .00, ví dụ add(134.5, 100) chứ không phải add(134.5, 100.00), chỉ có 1 đầu ra answer, ví dụ nếu có nhiều phép tính thì lấy kết quả là đầu ra của phép cuối cùng, phép đó cần dùng #0, #1,.. nếu có nhiều phép tính để liên kết với các kết quả của các phép tính trước đó, không tính thừa, ví dụ add(1, 0.15), add(#0, 0.25), divide(#1, 2) là ra 1 answer.
4. Khi tính các giá trị khác đơn vị đo lường cần phải dùng multiply hoặc divide trong program để quy đổi ra cùng đơn vị ra trước, ví dụ cần trừ 1 tỷ cho 900 triệu thì nếu dùng đơn vị triệu thì cần dùng multiply(1, 1000) để quy đối sang triệu trước khi trừ: subtract(#0, 900), không được trực tiếp dùng subtract(1000, 900), ví dụ tỉ lệ của 800 triệu và 6.0 tỷ cần dùng multiply(6, 1000) trước.
5. Kiểm tra kỹ xem phân tích đã lấy ra đúng thông tin chưa, nếu sai, phân tích lại.
6. program phải đúng định dạng được yêu cầu, nếu có nhiều phép toán, cần sử dụng #0, #1, #2 hợp lý để chỉ có 1 đầu ra cuối cùng và không thừa phép tính, các phép toán không được lồng vào nhau, ví dụ: không dùng divide(30, add(1, 0.3)) mà cần sửa lại thành add(1, 0.3), divide(30, #0).
7. Chú ý đến các câu hỏi hỏi về tỉ lệ giảm, giảm bao nhiêu, khi đó cần tính số âm để tìm ra tỉ lệ giảm, ví dụ giảm từ 553 xuống còn 500, tính tỉ lệ giảm cần dùng: subtract(500, 553), divide(#0, 553) không phải subtract(553, 500), divide(#0, 553).
8. Bắt buộc phải đưa ra ```plaintext.
program:...
answer:...
```
9. Dừng trả lời khi đưa ra xong kết quả trong ```plaintext... ```
"""
        return self._render(self.SELF_EVAL_SYSTEM_PROMPT, user_message, bullets_text)

    #: Lý do executor từ chối → câu nhắc CỤ THỂ phải sửa gì. Nói chung chung
    #: ("program sai, sửa đi") thì model thường sinh lại đúng cái cũ.
    _NHAC_SUA = {
        "nhan_bang_khong_khop": "Nhãn truyền cho table_* KHÔNG khớp hàng nào. Nhãn phải "
                                "là chuỗi ở Ô ĐẦU TIÊN của một dòng trong bảng, chép "
                                "NGUYÊN VĂN. Đừng truyền tên cột hay năm.",
        "nhan_bang_mo_ho": "Nhãn khớp nhiều hàng có số liệu khác nhau. Chép nhãn đầy đủ "
                           "hơn để chỉ còn đúng một hàng.",
        "dung_table_khi_khong_co_bang": "Mẫu này KHÔNG có bảng — đừng dùng table_*, lấy "
                                        "số thẳng từ văn bản.",
        "table_sai_cu_phap": "table_* phải có đúng dạng table_max(nhãn hàng, none).",
        "phep_long_nhau": "Có phép toán LỒNG trong tham số. Tách ra viết lần lượt và nối "
                          "bằng #0, #1.",
        "tham_chieu_sai": "#N trỏ tới phép chưa tồn tại. Ở phép thứ i chỉ được dùng #N "
                          "với N < i.",
        "tham_so_khong_phai_so": "Có tham số không đọc được thành số. Chỉ dùng số lấy "
                                 "thẳng từ văn bản/bảng, hoặc #N.",
        "sai_so_tham_so": "Mỗi phép cần ĐÚNG 2 tham số.",
        "cu_phap_di_dang": "Cú pháp hỏng. Mỗi phép viết dạng ten_phep(a, b), ngăn nhau "
                           "bằng dấu phẩy, không xuống dòng giữa chừng.",
        "chia_cho_0": "Có phép chia cho 0 — kiểm lại mẫu số.",
        "phep_toan_la": "Dùng phép không có trong danh sách cho phép.",
        "so_tran": "Kết quả tràn số — kiểm lại độ lớn các toán hạng.",
    }

    def step_sua(self, sample, program_hong: str, ly_do: str | None = None,
                 bullets_text: str = "") -> str:
        """Prompt SỬA: chương trình đã sinh không chạy được, nói rõ hỏng ở đâu.

        Khác self-eval ở chỗ đây **không** phải soát lại toàn bộ lời giải — máy đã biết
        chắc chương trình sai cú pháp/tham chiếu, nên chỉ yêu cầu viết lại cho chạy được.
        Chỉ những mẫu executor từ chối mới đi qua đây.
        """
        pre, post, table = self.context_block(sample)
        nhac = self._NHAC_SUA.get(ly_do or "", "Chương trình không chạy được.")
        user_message = f"""Câu hỏi: {sample["qa"]["question"]}
Đây là nội dung liên quan đến câu hỏi:
Pre-text: {pre}
Post-text: {post}
Bảng:
{table}

Chương trình đã sinh:
program: {program_hong}

⚙ Máy đã chạy thử chương trình trên và TỪ CHỐI. Lý do: {nhac}

Hãy viết lại chương trình cho CHẠY ĐƯỢC, giữ nguyên cách hiểu bài nếu nó vốn đúng.
Chỉ trả lời bằng đúng khối sau, không giải thích dài:
```plaintext
program: <các phép toán, ngăn bởi dấu phẩy>
answer: <kết quả cuối cùng>
```"""
        return self._render(self.ENGINEERED_SYSTEM_PROMPT, user_message, bullets_text)

    # ── dùng cho SFT: trả về messages thay vì chuỗi đã render ──
    def sft_messages(self, sample, target_text: str, level: str = "engineered") -> list[dict]:
        """Bộ ``messages`` để huấn luyện: system + user + assistant(target)."""
        system = {"basic": self.BASIC_SYSTEM_PROMPT,
                  "no_fewshot": self.NO_FEWSHOT_SYSTEM_PROMPT,
                  "engineered": self.ENGINEERED_SYSTEM_PROMPT}[level]
        # Mọi mức dùng CHUNG một user message, nên hiệu số giữa chúng chỉ do system
        # prompt — đúng điều kiện của phép ablation.
        user = self._user_engineered(sample)
        return [{"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": target_text}]


def strip_assistant(text: str) -> str:
    return text.split("assistant")[-1].strip() if "assistant" in text else text.strip()
