# -*- coding: utf-8 -*-
"""Kiểm thử CPU cho ``vinumqa.sft`` — dựng dữ liệu SFT bằng rejection sampling.

Chạy được không cần GPU, không cần model:  python -m pytest tests/test_sft.py -v
"""
from __future__ import annotations

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from vinumqa import data, sft

DATA_DIR = os.path.join(REPO, "data")

PLAINTEXT_OK = "Phân tích mẫu.\n```plaintext\nprogram: {prog}\nanswer: {ans}\n```\nThừa ở cuối."


@pytest.fixture(scope="module")
def splits():
    return {name: data.load_split(DATA_DIR, name) for name in ("train", "test")}


@pytest.fixture(scope="module")
def test_set(splits):
    return splits["test"]


class _FakePromptKit:
    """Đủ dùng cho sft.build_sft_records — không cần tokenizer."""

    def sft_messages(self, sample, target_text, level="engineered"):
        return [{"role": "system", "content": f"SYS:{level}"},
                {"role": "user", "content": f"Câu hỏi: {sample['qa']['question']}"},
                {"role": "assistant", "content": target_text}]


def _rows_and_samples(test_set, n=12):
    """2/3 mẫu model làm đúng, 1/3 làm sai."""
    samples = test_set[:n]
    rows = []
    for i, s in enumerate(samples):
        correct = i % 3 != 0
        rows.append({
            "id": s["id"],
            "pa_strict": correct,
            "ea": correct,
            "raw_step1": PLAINTEXT_OK.format(prog=s["qa"]["program"],
                                             ans=s["qa"]["exe_ans"]),
        })
    return rows, samples


class TestBuildRecords:
    def test_chi_lay_mau_model_lam_dung(self, test_set):
        rows, samples = _rows_and_samples(test_set)
        recs, stats = sft.build_sft_records(rows, samples, _FakePromptKit(),
                                            drop_noisy_gold=False)
        n_ok = sum(1 for r in rows if r["pa_strict"])
        assert len(recs) == n_ok
        assert stats["bo_model_lam_sai"] == len(rows) - n_ok
        assert all(r["source"] == "model" for r in recs)

    def test_cat_phan_thua_sau_khoi_plaintext(self, test_set):
        """Đích huấn luyện phải kết thúc ngay sau khối ```plaintext."""
        rows, samples = _rows_and_samples(test_set)
        recs, _ = sft.build_sft_records(rows, samples, _FakePromptKit(),
                                        drop_noisy_gold=False)
        target = recs[0]["messages"][-1]["content"]
        assert target.rstrip().endswith("```")
        assert "Thừa ở cuối" not in target
        assert "program:" in target            # vẫn giữ nguyên nội dung cần học

    def test_giu_chuoi_suy_luan(self, test_set):
        """Khác với SFT thẳng trên gold: đích PHẢI có phần suy luận, không chỉ program trần."""
        rows, samples = _rows_and_samples(test_set)
        recs, _ = sft.build_sft_records(rows, samples, _FakePromptKit(),
                                        drop_noisy_gold=False)
        target = recs[0]["messages"][-1]["content"]
        assert target.index("Phân tích mẫu") < target.index("```plaintext")

    def test_dinh_dang_messages(self, test_set):
        rows, samples = _rows_and_samples(test_set)
        recs, _ = sft.build_sft_records(rows, samples, _FakePromptKit(),
                                        drop_noisy_gold=False)
        msgs = recs[0]["messages"]
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert all(m["content"].strip() for m in msgs)

    def test_bo_qua_khi_khong_trich_duoc(self, test_set):
        samples = test_set[:4]
        rows = [{"id": s["id"], "pa_strict": True, "ea": True,
                 "raw_step1": "Nói lan man, không có khối plaintext nào."}
                for s in samples]
        recs, stats = sft.build_sft_records(rows, samples, _FakePromptKit(),
                                            drop_noisy_gold=False)
        assert recs == []
        assert stats["bo_khong_trich_duoc"] == 4

    def test_gold_fallback(self, test_set):
        rows, samples = _rows_and_samples(test_set)
        recs, stats = sft.build_sft_records(rows, samples, _FakePromptKit(),
                                            add_gold_fallback=True, drop_noisy_gold=False)
        assert len(recs) == len(samples)
        assert stats["dung_gold"] == sum(1 for r in rows if not r["pa_strict"])
        assert {r["source"] for r in recs} == {"model", "gold"}

    def test_loc_nhan_nhieu(self, splits):
        """Mẫu gold có multiply(#n,100) phải bị loại khỏi dữ liệu huấn luyện."""
        noisy = [s for s in splits["train"] if data.uses_multiply_100(s)][:5]
        assert noisy, "không tìm thấy mẫu nhiễu trong train"
        rows = [{"id": s["id"], "pa_strict": True, "ea": True,
                 "raw_step1": PLAINTEXT_OK.format(prog="add(1,2)", ans="3")}
                for s in noisy]
        recs, stats = sft.build_sft_records(rows, noisy, _FakePromptKit(),
                                            drop_noisy_gold=True)
        assert recs == []
        assert stats["bo_nhan_nhieu"] == len(noisy)

    def test_tieu_chi_chap_nhan(self, test_set):
        samples = test_set[:6]
        rows = [{"id": s["id"], "pa_strict": i < 2, "ea": i < 4,
                 "raw_step1": PLAINTEXT_OK.format(prog="add(1,2)", ans="3")}
                for i, s in enumerate(samples)]
        kit = _FakePromptKit()
        n_pa = len(sft.build_sft_records(rows, samples, kit, accept="pa",
                                         drop_noisy_gold=False)[0])
        n_ea = len(sft.build_sft_records(rows, samples, kit, accept="ea",
                                         drop_noisy_gold=False)[0])
        n_or = len(sft.build_sft_records(rows, samples, kit, accept="pa_or_ea",
                                         drop_noisy_gold=False)[0])
        assert (n_pa, n_ea, n_or) == (2, 4, 4)

    def test_bo_mau_qua_dai(self, test_set):
        rows, samples = _rows_and_samples(test_set)
        recs, stats = sft.build_sft_records(rows, samples, _FakePromptKit(),
                                            drop_noisy_gold=False, max_chars=10)
        assert recs == []
        assert stats["bo_qua_dai"] > 0

    def test_lech_do_dai_thi_bao_loi(self, test_set):
        rows, samples = _rows_and_samples(test_set)
        with pytest.raises(AssertionError):
            sft.build_sft_records(rows[:3], samples, _FakePromptKit())


class TestIO:
    def test_ghi_jsonl(self, tmp_path, test_set):
        rows, samples = _rows_and_samples(test_set)
        recs, _ = sft.build_sft_records(rows, samples, _FakePromptKit(),
                                        drop_noisy_gold=False)
        p = sft.write_jsonl(recs, str(tmp_path / "sft.jsonl"))
        with open(p, encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        assert len(lines) == len(recs)
        obj = json.loads(lines[0])
        assert list(obj.keys()) == ["messages"]      # đúng định dạng SFTTrainer đọc
        assert len(obj["messages"]) == 3

    def test_thong_ke(self, test_set):
        rows, samples = _rows_and_samples(test_set)
        recs, _ = sft.build_sft_records(rows, samples, _FakePromptKit(),
                                        drop_noisy_gold=False)
        st = sft.sft_data_stats(recs)
        assert st["n"] == len(recs)
        assert st["ky_tu_p50"] <= st["ky_tu_p95"] <= st["ky_tu_max"]
        assert sum(st["theo_so_phep"].values()) == len(recs)
        assert st["theo_nguon"] == {"model": len(recs)}

    def test_thong_ke_rong(self):
        assert sft.sft_data_stats([]) == {"n": 0}


class TestTrainingConfig:
    def test_lora_khop_notebook_cu(self):
        """Giữ đúng tham số LoRA của reference/original_notebooks/finetune_phi4.ipynb."""
        cfg = sft.lora_config()
        assert cfg["r"] == 16
        assert cfg["lora_alpha"] == 32
        assert cfg["lora_dropout"] == 0
        assert cfg["bias"] == "none"
        assert cfg["use_gradient_checkpointing"] == "unsloth"
        assert cfg["random_state"] == 3407
        assert cfg["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"]

    @pytest.mark.parametrize("vram", [15, 24, 40, 80])
    def test_batch_hieu_dung_on_dinh_moi_gpu(self, vram):
        """Batch hiệu dụng phải không đổi giữa các GPU.

        Nếu batch hiệu dụng thay đổi theo máy thì learning rate 2e-4 lấy từ notebook cũ
        sẽ không còn phù hợp, và kết quả SFT giữa các máy không so được với nhau.
        """
        cfg = sft.training_config(vram, n_train=2000)
        assert cfg["_meta"]["effective_batch"] == 16
        assert cfg["learning_rate"] == 2e-4
        assert cfg["optim"] == "paged_adamw_8bit"
        assert cfg["lr_scheduler_type"] == "cosine"
        assert cfg["weight_decay"] == 0.05
        assert cfg["warmup_ratio"] == 0.1

    def test_gpu_nho_thi_batch_nho_accum_lon(self):
        t4 = sft.training_config(15, 2000)
        a100 = sft.training_config(80, 2000)
        assert t4["per_device_train_batch_size"] < a100["per_device_train_batch_size"]
        assert t4["gradient_accumulation_steps"] > a100["gradient_accumulation_steps"]

    def test_bf16_fp16_loai_tru_nhau(self):
        assert sft.training_config(24, 2000, bf16=True)["fp16"] is False
        assert sft.training_config(15, 2000, bf16=False)["fp16"] is True

    def test_lich_eval_hop_ly(self):
        cfg = sft.training_config(24, n_train=2000, epochs=3)
        meta = cfg["_meta"]
        assert meta["total_steps"] == meta["steps_per_epoch"] * 3
        assert cfg["eval_steps"] == cfg["save_steps"] >= 10
        assert cfg["load_best_model_at_end"] is True
        assert cfg["metric_for_best_model"] == "eval_loss"
        assert cfg["greater_is_better"] is False

    @pytest.mark.parametrize("vram", [15, 24, 40, 80])
    def test_logits_khong_vuot_ngan_sach_vram(self, vram):
        """Tensor logits phải nằm gọn trong VRAM, ở ĐỘ DÀI TRẦN.

        Bảng cũ chia batch theo SỐ CHUỖI nên trên A100 80GB cho ``bs=16``; với chuỗi
        8192 token thì riêng logits là 37 GB bf16, và cross-entropy upcast fp32 thành
        ~111 GB. Tràn ngay step đầu. Giới hạn ở đây: logits bf16 ≤ 1/8 VRAM, tức còn
        đủ chỗ cho bản fp32 (×2) lẫn model, activation và optimizer.
        """
        m = sft.training_config(vram, 2000, max_seq=8192)["_meta"]
        if m["vua_vram"]:
            assert m["logits_gb"] <= vram / 8, (
                f"{m['logits_gb']} GB logits trên GPU {vram} GB — "
                f"bs={m['token_moi_lo'] // 8192} quá lớn cho chuỗi 8192 token")
        else:
            # Không lọt ngay cả ở bs=1 thì phải NÓI RA, để notebook chặn trước khi
            # huấn luyện chứ không tràn giữa chừng. T4 15GB rơi vào nhánh này.
            assert vram < 18 and m["token_moi_lo"] == 8192

    def test_chuoi_dai_hon_thi_batch_nho_hon(self):
        """Cùng một GPU: nhân đôi độ dài thì batch phải giảm, batch hiệu dụng giữ nguyên."""
        ngan = sft.training_config(80, 2000, max_seq=2048)
        dai = sft.training_config(80, 2000, max_seq=8192)
        assert ngan["per_device_train_batch_size"] > dai["per_device_train_batch_size"]
        assert (ngan["_meta"]["effective_batch"]
                == dai["_meta"]["effective_batch"] == 16)

    def test_du_lieu_nho_khong_chia_cho_0(self):
        cfg = sft.training_config(24, n_train=3, epochs=1)
        assert cfg["_meta"]["steps_per_epoch"] >= 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
