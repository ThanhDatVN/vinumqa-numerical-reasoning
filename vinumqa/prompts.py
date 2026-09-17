# -*- coding: utf-8 -*-
"""Thang prompt LỒNG NHAU (basic ⊂ no_fewshot ⊂ engineered), cộng prompt self-eval.

Ba mức đầu cắt ra từ CHÍNH prompt hoàn chỉnh (:func:`PromptKit.bo_vi_du`,
:func:`PromptKit.bo_huong_dan_tu_khoa`), nên phần dùng chung giống nhau từng ký tự và
mỗi bước là một phép **chèn thuần** — hiệu số giữa hai nấc kề nhau vì thế đo đúng phần
vừa thêm, không lẫn chuyện viết lại câu chữ. Có test canh điều đó.

======================  ==========================================================
``basic``               Nấc 1. Mở đầu + danh sách phép toán (có mô tả từng phép) +
                        toàn bộ yêu cầu định dạng đầu ra. 2 535 ký tự.
``no_fewshot``          Bậc giữa, KHÔNG phải nấc phải chạy. = ``basic`` + ánh xạ
                        **từ khoá tiếng Việt → phép toán** (14 mục). 5 408 ký tự.
``engineered``          Nấc 2 trở đi. = ``no_fewshot`` + 2 ví dụ mẫu có khung.
                        5 924 ký tự.
``self_eval``           Prompt bước 2: đưa lại ngữ cảnh + lời giải bước 1, yêu
                        cầu model tự soát và sửa.
``plain``               Prompt trần 530 ký tự, ĐÃ RỜI thang bậc: thiếu cả quy tắc
                        định dạng nên 32 % mẫu sinh ra program không chạy được —
                        một cái sàn hỏng. Giữ lại cho tương thích, đừng dùng để đo.
======================  ==========================================================

Prompt ``engineered`` và ``self_eval`` nằm trong :mod:`vinumqa._prompt_text`, chép
**nguyên văn** từ bản đã dùng ở lần chạy tham chiếu, nên kết quả mới luôn so sánh được với kết
quả cũ. Prompt ``plain`` là bản mới, cố tình viết tối giản để đo xem prompt engineering
đóng góp bao nhiêu.
"""
from __future__ import annotations

from ._prompt_text import (SYSTEM_PROMPT_STEP_1, SYSTEM_PROMPT_STEP_2,
                           table_to_str as _table_to_str)

__all__ = ["PromptKit", "PLAIN_SYSTEM_PROMPT", "strip_assistant",
           "MEMORY_ASK", "MEMORY_PREFIX", "MEMORY_SUFFIX"]

# ────────────────── prompt trần (đã rời thang bậc, xem README) ──────────────────

PLAIN_SYSTEM_PROMPT = """Bạn là trợ lý phân tích báo cáo tài chính. Hãy đọc văn bản và bảng số liệu rồi trả lời câu hỏi.

Thay vì tự tính nhẩm, hãy viết một chương trình tính toán dùng các phép sau:
add(a, b), subtract(a, b), multiply(a, b), divide(a, b),
table_max(nhãn, none), table_min(nhãn, none), table_sum(nhãn, none), table_average(nhãn, none)

Nếu cần nhiều phép, viết lần lượt và dùng #0, #1, ... để chỉ kết quả của phép trước.

Trả lời theo đúng định dạng:
```plaintext
program: <các phép toán, ngăn bởi dấu phẩy>
answer: <kết quả cuối cùng>
```"""

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
    repo_dir : không còn dùng, giữ để tương thích chữ ký cũ.
    tokenizer : tokenizer của model; ``None`` thì dùng định dạng ``<<SYSTEM>>`` thuần.
    model_name : dùng để tự tắt thinking-mode với Qwen3.
    max_ctx_chars, max_prev_chars : trần ký tự cho ngữ cảnh / lời giải bước 1.
        ``None`` = không cắt. Chỉ cần đặt khi chạy trên GPU nhỏ.
    """

    LEVELS = ("plain", "basic", "no_fewshot", "engineered")

    def __init__(self, repo_dir: str = "", tokenizer=None, model_name: str = "",
                 enable_thinking: bool | None = None,
                 max_ctx_chars: int | None = None, max_prev_chars: int | None = None):
        # repo_dir giữ lại cho tương thích với chữ ký cũ; prompt nay nằm trong package.
        self.ENGINEERED_SYSTEM_PROMPT = SYSTEM_PROMPT_STEP_1
        self.SELF_EVAL_SYSTEM_PROMPT = SYSTEM_PROMPT_STEP_2
        self.PLAIN_SYSTEM_PROMPT = PLAIN_SYSTEM_PROMPT
        self.NO_FEWSHOT_SYSTEM_PROMPT = self.bo_vi_du(SYSTEM_PROMPT_STEP_1)
        # Thang prompt LỒNG NHAU — mỗi nấc thêm đúng một khối, cắt ra từ chính prompt
        # hoàn chỉnh nên phần dùng chung giống nhau từng ký tự. Nhờ vậy hiệu số giữa hai
        # nấc kề nhau đo đúng phần vừa thêm, không lẫn chữ nghĩa viết lại.
        #
        #   basic       = mở đầu + DANH SÁCH PHÉP TOÁN + yêu cầu định dạng
        #   no_fewshot  = basic + HƯỚNG DẪN CHỌN PHÉP THEO TỪ KHÓA (14 mục)
        #   engineered  = no_fewshot + 2 ví dụ mẫu có khung
        self.BASIC_SYSTEM_PROMPT = self.bo_vi_du(
            self.bo_huong_dan_tu_khoa(SYSTEM_PROMPT_STEP_1))
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
    def _user_plain(self, sample) -> str:
        pre, post, table = self.context_block(sample)
        return f"""Câu hỏi: {sample["qa"]["question"]}

Văn bản trước bảng: {pre}
Văn bản sau bảng: {post}
Bảng:
{table}
"""

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
    def step1(self, sample, bullets_text: str = "", level: str = "engineered") -> str:
        """Prompt sinh program. ``level`` ∈ :attr:`LEVELS` (thang lồng nhau)."""
        if level == "plain":
            return self._render(self.PLAIN_SYSTEM_PROMPT, self._user_plain(sample),
                                bullets_text)
        if level == "engineered":
            return self._render(self.ENGINEERED_SYSTEM_PROMPT,
                                self._user_engineered(sample), bullets_text)
        if level == "basic":
            return self._render(self.BASIC_SYSTEM_PROMPT,
                                self._user_engineered(sample), bullets_text)
        if level == "no_fewshot":
            return self._render(self.NO_FEWSHOT_SYSTEM_PROMPT,
                                self._user_engineered(sample), bullets_text)
        raise ValueError(f"level lạ: {level!r}, chọn trong {self.LEVELS}")

    # ── biến thể bỏ ví dụ mẫu ──
    _MOC_VI_DU = "=== VÍ DỤ ==="
    _MOC_SAU_VI_DU = "==== CÂU HỎI ===="
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

    def step2(self, sample, initial_response: str, bullets_text: str = "") -> str:
        """Prompt self-evaluation — giữ nguyên bản tham chiếu."""
        question = sample["qa"]["question"]
        pre, post, table = self.context_block(sample)
        if self.max_prev_chars and len(initial_response) > self.max_prev_chars:
            # Giữ PHẦN CUỐI: khối ```plaintext nằm ở đó.
            initial_response = ("[…phần đầu đã cắt bớt]\n"
                                + initial_response[-self.max_prev_chars:])

        user_message = f"""Câu hỏi: {question}
Đây là nội dung liên quan đến câu hỏi:
Pre-text: {pre}
Post-text: {post}
Bảng:
{table}
Đây là phân tích và kết quả:
{initial_response}

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

    # ── dùng cho SFT: trả về messages thay vì chuỗi đã render ──
    def sft_messages(self, sample, target_text: str, level: str = "engineered") -> list[dict]:
        """Bộ ``messages`` để huấn luyện: system + user + assistant(target)."""
        system = {"plain": self.PLAIN_SYSTEM_PROMPT,
                  "basic": self.BASIC_SYSTEM_PROMPT,
                  "no_fewshot": self.NO_FEWSHOT_SYSTEM_PROMPT,
                  "engineered": self.ENGINEERED_SYSTEM_PROMPT}[level]
        # Chỉ ``plain`` dùng user message riêng; ba mức còn lại dùng chung một bản, nên
        # hiệu số giữa chúng chỉ do system prompt — đúng điều kiện của phép ablation.
        user = (self._user_plain(sample) if level == "plain"
                else self._user_engineered(sample))
        return [{"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": target_text}]


def strip_assistant(text: str) -> str:
    return text.split("assistant")[-1].strip() if "assistant" in text else text.strip()
