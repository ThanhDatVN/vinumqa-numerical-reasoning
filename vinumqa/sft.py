# -*- coding: utf-8 -*-
"""SFT cho Qwen3-8B: dựng dữ liệu bằng rejection sampling + cấu hình huấn luyện.

Vì sao **không** fine-tune thẳng trên gold
------------------------------------------
Lần chạy trước đã thử (Stage 2: LoRA Mistral-7B trên gold thô) và **thất bại** — train loss
giảm đều nhưng val loss tăng ngay từ step 50. Hai nguyên nhân được nêu: nhãn vàng có
5–10 % không nhất quán, và **đích huấn luyện chỉ là program trần, không có chuỗi suy
luận**, nên model không học được cách suy luận, chỉ học thuộc.

Cách làm ở đây: **rejection sampling / self-distillation**

1. Chạy chính Qwen3-8B (chưa fine-tune) với prompt có cấu trúc trên tập train.
2. Giữ lại những mẫu model làm **đúng** (PA hoặc EA).
3. Dùng chính lời giải đó — *có đầy đủ chuỗi suy luận tiếng Việt* — làm đích huấn luyện.

Ưu điểm so với cách SFT thẳng trên gold:

* Đích huấn luyện **có chuỗi suy luận**, đúng văn phong mà model vốn sinh ra được.
* **Không cần API ngoài** (cách cũ phải dùng Gemini-2.5-flash sửa 600 mẫu) → vẫn nằm
  trong thiết lập *constrained-resource*.
* Tự động loại mẫu có nhãn nhiễu: mẫu mà gold sai thì model khó "làm đúng" theo gold,
  nên phần lớn tự rơi ra ngoài.

Hạn chế phải nêu khi báo cáo: chỉ học từ những gì model **đã làm được**, nên khó dạy
được kiểu bài mà model chưa bao giờ giải đúng. Tuỳ chọn ``add_gold_fallback`` bù lại
một phần bằng cách dựng đích từ gold cho các mẫu model luôn sai.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

from .data import is_noisy_gold
from .dsl import n_ops

__all__ = ["build_sft_records", "write_jsonl", "sft_data_stats",
           "lora_config", "training_config", "GOLD_TARGET_TEMPLATE"]


GOLD_TARGET_TEMPLATE = """Phân tích câu hỏi và số liệu liên quan:
{reasoning}

```plaintext
program: {program}
answer: {answer}
```"""


def _clean_target(raw_text: str) -> str | None:
    """Cắt gọn lời giải của model: bỏ phần thừa sau khối ```plaintext cuối cùng."""
    if not raw_text:
        return None
    text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.S).strip()
    # giữ tới hết khối plaintext cuối cùng
    blocks = list(re.finditer(r"```(?:plaintext|text)?\s*\n.*?\n```", text, re.S | re.I))
    if not blocks:
        return None
    text = text[:blocks[-1].end()].strip()
    return text or None


def build_sft_records(rows, samples, prompt_kit, *, level="engineered",
                      accept="pa_or_ea", add_gold_fallback=False,
                      drop_noisy_gold=True, max_chars=20000):
    """Dựng dữ liệu SFT từ kết quả chạy trên tập train.

    Parameters
    ----------
    rows : output của :func:`vinumqa.pipeline.run_pipeline` trên tập train
        (phải chạy với ``keep_raw=True``).
    samples : danh sách mẫu train tương ứng, cùng thứ tự với ``rows``.
    accept : ``'pa'`` | ``'ea'`` | ``'pa_or_ea'`` — tiêu chí coi là "model làm đúng".
        ``'pa'`` chặt nhất (đúng cả chương trình), ``'pa_or_ea'`` cho nhiều dữ liệu hơn.
    add_gold_fallback : với mẫu model làm sai, dựng đích từ gold program (không có
        chuỗi suy luận của model). Mặc định tắt vì đây chính là thứ gây overfit ở lần chạy trước.
    drop_noisy_gold : bỏ mẫu có nhãn vàng nhiễu (``multiply(#n,100)`` hoặc gold không
        thực thi được) — xem :func:`vinumqa.data.is_noisy_gold`.
    """
    assert len(rows) == len(samples), "rows và samples phải cùng thứ tự, cùng độ dài"

    records, stats = [], Counter()
    for row, sample in zip(rows, samples):
        stats["tong"] += 1

        if drop_noisy_gold and is_noisy_gold(sample):
            stats["bo_nhan_nhieu"] += 1
            continue

        if accept == "pa":
            ok = row["pa_strict"]
        elif accept == "ea":
            ok = row["ea"]
        else:
            ok = row["pa_strict"] or row["ea"]

        target = None
        source = None
        if ok:
            target = _clean_target(row.get("raw_step1") or "")
            source = "model"
            if target is None:
                stats["bo_khong_trich_duoc"] += 1
        elif add_gold_fallback:
            qa = sample["qa"]
            target = GOLD_TARGET_TEMPLATE.format(
                reasoning="Xác định các số liệu cần dùng từ văn bản và bảng, "
                          "rồi viết chương trình tính toán tương ứng.",
                program=qa["program"], answer=qa.get("exe_ans", ""))
            source = "gold"
            stats["dung_gold"] += 1
        else:
            stats["bo_model_lam_sai"] += 1
            continue

        if target is None:
            continue

        messages = prompt_kit.sft_messages(sample, target, level=level)
        total_chars = sum(len(m["content"]) for m in messages)
        if total_chars > max_chars:
            stats["bo_qua_dai"] += 1
            continue

        records.append({
            "messages": messages,
            "id": sample["id"],
            "source": source,
            "n_ops_gold": n_ops(sample["qa"]["program"]),
            "chars": total_chars,
        })
        stats[f"nhan_tu_{source}"] += 1

    return records, dict(stats)


def write_jsonl(records, path: str, keep_only_messages=True) -> str:
    """Ghi ra JSONL đúng định dạng ``{"messages": [...]}`` mà SFTTrainer đọc được."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            obj = {"messages": r["messages"]} if keep_only_messages else r
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return path


def sft_data_stats(records) -> dict:
    """Thống kê để in báo cáo và kiểm tra dữ liệu có lệch không."""
    if not records:
        return {"n": 0}
    by_ops = Counter(r["n_ops_gold"] for r in records)
    by_src = Counter(r["source"] for r in records)
    chars = sorted(r["chars"] for r in records)
    return {
        "n": len(records),
        "theo_so_phep": dict(sorted(by_ops.items())),
        "theo_nguon": dict(by_src),
        "ky_tu_p50": chars[len(chars) // 2],
        "ky_tu_p95": chars[int(len(chars) * 0.95)],
        "ky_tu_max": chars[-1],
    }


# ─────────────────────────── cấu hình huấn luyện ───────────────────────────

def lora_config() -> dict:
    """LoRA — giữ đúng tham số của notebook fine-tune cũ trong repo."""
    return dict(
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )


def training_config(vram_gb: float, n_train: int, *, epochs=3, output_dir="outputs",
                    bf16=True, max_seq: int = 8192, vocab: int = 151936) -> dict:
    """Tham số ``TrainingArguments``, chia batch theo NGÂN SÁCH TOKEN.

    Đếm theo SỐ CHUỖI là sai, và sai rất đắt. Thứ quyết định bộ nhớ khi huấn luyện
    causal LM là tensor logits — ``tokens × vocab × 2 byte`` cho bf16, rồi cross-entropy
    còn upcast lên fp32 nữa, tức nhân ba. Cùng ``batch=16``:

    * chuỗi 512 token  →  8.192 token/micro-batch  →  2,3 GB logits
    * chuỗi 8.192 token → 131.072 token/micro-batch → 37 GB logits, 111 GB kèm fp32

    Bảng cũ lấy từ notebook fine-tune trước (chuỗi ngắn) nên cho ``bs=16`` trên A100
    80GB, và tràn ngay ở step đầu với bản ghi ~3.400 token của bộ này. Giờ chốt ngân
    sách token rồi mới suy ra ``bs``; ``accum`` bù lại để **batch hiệu dụng vẫn là 16**
    (learning rate 2e-4 chỉ hợp lệ ở mức đó).

    ``max_seq`` dùng làm TRẦN chứ không phải độ dài trung bình: một chuỗi dài có thể
    rơi vào bất kỳ batch nào, và ``packing=False`` thì cả batch đệm theo chuỗi dài nhất.
    """
    if vram_gb < 18:              # T4 15GB
        ngan_sach_token = 2048
    elif vram_gb < 30:            # L4 24GB
        ngan_sach_token = 4096
    elif vram_gb < 50:            # A100 40GB
        ngan_sach_token = 8192
    else:                         # A100 80GB
        ngan_sach_token = 16384

    bs = max(1, min(16, ngan_sach_token // max(1, int(max_seq))))
    accum = max(1, round(16 / bs))
    eval_bs = bs

    token_moi_lo = bs * int(max_seq)
    logits_gb = token_moi_lo * vocab * 2 / 1024 ** 3

    effective = bs * accum
    steps_per_epoch = max(1, n_train // effective)
    total_steps = steps_per_epoch * epochs
    # chấm eval/lưu ~10 lần mỗi lần chạy, tối thiểu mỗi 10 step
    interval = max(10, total_steps // 10)

    return dict(
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=accum,
        per_device_eval_batch_size=eval_bs,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        learning_rate=2e-4,
        logging_steps=max(5, interval // 2),
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        weight_decay=0.05,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=interval,
        eval_strategy="steps",
        eval_steps=interval,
        bf16=bf16,
        fp16=not bf16,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        output_dir=output_dir,
        report_to="none",
        seed=3407,
        _meta=dict(effective_batch=effective, steps_per_epoch=steps_per_epoch,
                   total_steps=total_steps, eval_every=interval,
                   token_moi_lo=token_moi_lo, ngan_sach_token=ngan_sach_token,
                   logits_gb=round(logits_gb, 1),
                   # False = ngay ở bs=1 cũng không lọt. Không phải "chật" mà là
                   # KHÔNG CHẠY ĐƯỢC: T4 15GB với chuỗi 8192 rơi vào đây.
                   vua_vram=bool(logits_gb <= vram_gb / 8)),
    )
