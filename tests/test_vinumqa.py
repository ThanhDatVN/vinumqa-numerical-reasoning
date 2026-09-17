# -*- coding: utf-8 -*-
"""Kiểm thử CPU cho package ``vinumqa`` — chạy được không cần GPU, không cần model.

    python -m pytest tests/ -v          (hoặc:  python tests/test_ace_vinumqa.py)

Bộ test này là thứ bảo đảm "dự án chuẩn": nó chạy trên DỮ LIỆU THẬT và sẽ đỏ nếu
executor, quality gate, hay pipeline bị chỉnh sai — trước khi tốn giờ GPU trên Colab.
"""
from __future__ import annotations

import os
import random
import re
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from vinumqa import data, dsl, io_utils, pipeline, prompts as prompts_mod, sft, stats
from vinumqa.ace import clusters, playbook, reflector
from vinumqa.ace import trainer as ace_trainer

DATA_DIR = os.path.join(REPO, "data")


# ═══════════════════════════ fixtures ═══════════════════════════

@pytest.fixture(scope="session")
def splits():
    return {name: data.load_split(DATA_DIR, name) for name in ("train", "valid", "test")}


@pytest.fixture(scope="session")
def test_set(splits):
    return splits["test"]


TABLE_FIXTURE = [
    ["", "2018", "2019", "2020"],
    ["Lãi ròng", "104", "(79)", chr(8212)],
    ["Biên lãi gộp", "26% ( 26 % )", "21.0% ( 21.0 % )", "19.8% ( 19.8 % )"],
    ["P/E", "22.0x", "16.6x", "9.5x"],
]


# ═══════════════════════════ 1. executor DSL ═══════════════════════════

class TestExecutor:
    def test_phep_toan_co_ban(self):
        assert dsl.execute_program("add(1, 2)", []) == 3
        assert dsl.execute_program("subtract(9829, 642)", []) == 9187
        assert dsl.execute_program("multiply(16000, 20%)", []) == pytest.approx(3200.0)
        assert dsl.execute_program("divide(892, 18600)", []) == pytest.approx(0.04795698, rel=1e-6)

    def test_chuoi_va_tham_chieu(self):
        assert dsl.execute_program("add(1, 0.15), divide(5310, #0)", []) == \
            pytest.approx(4617.3913043, rel=1e-9)
        assert dsl.execute_program("subtract(108.50, 100), divide(#0, 100)", []) == \
            pytest.approx(0.085)

    def test_toan_hang_phan_tram(self):
        """ViNumQA dùng dạng '20%' nghĩa là 0.2 — 61 gold program trong train dùng dạng này."""
        assert dsl.execute_program("subtract(36.5%, 19.3%)", []) == pytest.approx(0.172)
        assert dsl.execute_program("add(853.5, 4.9%)", []) == pytest.approx(853.549)

    def test_khong_co_const_finqa(self):
        """ViNumQA KHÔNG dùng const_* — phải trả None chứ không âm thầm coi là 0."""
        assert dsl.execute_program("divide(637, const_5)", []) is None

    def test_fail_closed(self):
        assert dsl.execute_program("divide(5, 0)", []) is None          # chia 0
        assert dsl.execute_program("add(#3, 1)", []) is None            # tham chiếu tiến
        assert dsl.execute_program("divide(5310, add(1, 0.15))", []) is None   # lồng nhau
        assert dsl.execute_program("add(1)", []) is None                # thiếu tham số
        assert dsl.execute_program("khong_phai_phep(1, 2)", []) is None
        assert dsl.execute_program("", []) is None
        assert dsl.execute_program(None, []) is None

    def test_table_theo_nhan_hang(self):
        """table_* lấy theo NHÃN HÀNG (cột 0), không phải tên cột — điểm khác FinQA."""
        assert dsl.execute_program("table_max(Lãi ròng, none)", TABLE_FIXTURE) == 104.0
        assert dsl.execute_program("table_min(Lãi ròng, none)", TABLE_FIXTURE) == -79.0
        assert dsl.execute_program("table_max(2019, none)", TABLE_FIXTURE) is None

    def test_o_bang_dac_biet(self):
        assert dsl.cell_to_number("(79)") == -79.0          # ngoặc = số âm
        assert dsl.cell_to_number(chr(8212)) is None        # em dash = ô trống
        assert dsl.cell_to_number("n/a") is None
        assert dsl.cell_to_number("26% ( 26 % )") == pytest.approx(0.26)
        assert dsl.cell_to_number("22.0x") == pytest.approx(22.0)
        assert dsl.cell_to_number("$ 1,234") == 1234.0
        assert dsl.execute_program("table_average(Biên lãi gộp, none)", TABLE_FIXTURE) == \
            pytest.approx(0.2226666667)

    def test_table_nhan_mo_ho_thi_tu_choi(self):
        tbl = [["", "a"], ["Doanh thu", "1"], ["Doanh thu thuần", "2"]]
        # 'Doanh thu' khớp chính xác hàng 1 → chấp nhận
        assert dsl.execute_program("table_max(Doanh thu, none)", tbl) == 1.0
        # nhãn không khớp duy nhất → fail-closed
        assert dsl.execute_program("table_max(Doanh, none)", tbl) is None


# ═══════════════════════════ 2. trích xuất ═══════════════════════════

class TestExtraction:
    def test_dinh_dang_chuan(self):
        raw = ("Phân tích...\n```plaintext\n"
               "program: subtract(108.50, 100), divide(#0, 100)\nanswer: 0.085\n```")
        assert dsl.extract_program_answer(raw) == \
            ("subtract(108.50, 100), divide(#0, 100)", "0.085")

    def test_lay_khoi_cuoi_cung(self):
        raw = ("Thử lần 1:\n```plaintext\nprogram: divide(1,2)\n```\n"
               "Sửa lại:\n```plaintext\nprogram: divide(3,4)\nanswer: 0.75\n```")
        assert dsl.extract_program_answer(raw) == ("divide(3,4)", "0.75")

    def test_khong_co_tien_to_program(self):
        assert dsl.extract_program_answer("```plaintext\ndivide(60, 243)\n```")[0] == \
            "divide(60, 243)"

    def test_bo_think_cua_qwen3(self):
        raw = "<think>nghĩ lung tung divide(9,9)</think>\n```plaintext\nprogram: add(1,2)\n```"
        assert dsl.extract_program_answer(raw)[0] == "add(1,2)"

    def test_khong_tim_thay(self):
        assert dsl.extract_program_answer("không có gì ở đây")[0] is None
        assert dsl.extract_program_answer("```plaintext\nchỉ là văn bản\n```")[0] is None
        assert dsl.extract_program_answer("")[0] is None
        assert dsl.extract_program_answer(None) == (None, None)


# ═══════════════════════════ 3. PA / EA ═══════════════════════════

class TestMetrics:
    def test_pa_giao_hoan(self):
        assert dsl.check_pa("add(108.50, 100)", "add(100, 108.5)")[0]
        assert dsl.check_pa("multiply(2, 3)", "multiply(3, 2)")[0]

    def test_pa_khong_giao_hoan_voi_subtract_divide(self):
        assert not dsl.check_pa("subtract(100, 124.6)", "subtract(124.6, 100)")[0]
        assert not dsl.check_pa("divide(2, 3)", "divide(3, 2)")[0]

    def test_pa_chuan_hoa_so(self):
        assert dsl.check_pa("subtract(124.6, 100.00)", "subtract(124.6, 100)")[0]
        assert dsl.check_pa("add(35.5%, 14.5%)", "add(0.355, 0.145)")[0]

    def test_ea_lam_tron_5_chu_so(self):
        """exe_ans của ViNumQA đã làm tròn 5 chữ số — chấm ở 4 chữ số sẽ sai."""
        assert dsl.check_ea(0.6066481994, "0.60665")
        assert dsl.check_ea(40192.92604501608, "40192.92605")
        assert not dsl.check_ea(0.085, "0.086")
        assert not dsl.check_ea(None, "0.085")

    def test_ea_gia_tri_chuoi(self):
        assert dsl.check_ea("yes", "yes")
        assert not dsl.check_ea("yes", "no")

    def test_classify_outcome(self):
        assert dsl.classify_outcome(True, True, "add(1,2)", 3) == "dung"
        assert dsl.classify_outcome(True, False, "add(1,2)", 3) == "dung_nhung_khac_program"
        assert dsl.classify_outcome(False, False, "", None) == "khong_co_program"
        assert dsl.classify_outcome(False, False, "add(1,2)", None) == "program_khong_chay_duoc"


# ═══════════════════════════ 4. đối chiếu với DỮ LIỆU THẬT ═══════════════════════════

class TestAgainstRealData:
    """Test quan trọng nhất: executor phải tái tạo được exe_ans của nhãn vàng."""

    @pytest.mark.parametrize("split,min_acc", [("train", 0.99), ("valid", 0.99), ("test", 1.0)])
    def test_executor_tai_tao_exe_ans(self, splits, split, min_acc):
        samples = splits[split]
        ok = sum(1 for s in samples
                 if dsl.check_ea(dsl.execute_program(s["qa"]["program"], s.get("table") or []),
                                 s["qa"].get("exe_ans")))
        acc = ok / len(samples)
        assert acc >= min_acc, f"{split}: chỉ {acc:.4f} gold tái tạo được exe_ans"

    def test_table_ops_deu_chay_duoc(self, splits):
        """Mọi gold program dùng table_* phải thực thi ra đúng exe_ans."""
        for split in ("train", "valid", "test"):
            bad = [s["id"] for s in splits[split]
                   if "table_" in s["qa"]["program"]
                   and not dsl.check_ea(
                       dsl.execute_program(s["qa"]["program"], s.get("table") or []),
                       s["qa"].get("exe_ans"))]
            assert not bad, f"{split}: {len(bad)} gold table_* không chạy đúng: {bad[:3]}"

    def test_khong_gold_nao_dung_const(self, splits):
        for split, samples in splits.items():
            assert not [s for s in samples if "const_" in s["qa"]["program"]], \
                f"{split} có gold dùng const_* — giả định 'ViNumQA không có const_' sai rồi"

    def test_tach_nguon_du_lieu(self, splits):
        """FinQA-Vi / Vi Data tách được từ id, và tỉ lệ khớp mốc tham chiếu (~49/51)."""
        for split, samples in splits.items():
            by_src = data.split_by_source(samples)
            assert set(by_src) == {"FinQA-Vi", "ViData"}, f"{split}: {list(by_src)}"
        total = sum(len(s) for s in splits.values())
        finqa = sum(len(data.split_by_source(s).get("FinQA-Vi", [])) for s in splits.values())
        assert 0.45 < finqa / total < 0.55


# ═══════════════════════════ 5. audit nhiễu nhãn ═══════════════════════════

class TestGoldAudit:
    def test_multiply100_chi_o_train(self, splits):
        """Phát hiện then chốt: nhiễu nhãn nằm đúng ở tập mà ACE học."""
        n_train = sum(data.uses_multiply_100(s) for s in splits["train"])
        n_valid = sum(data.uses_multiply_100(s) for s in splits["valid"])
        n_test = sum(data.uses_multiply_100(s) for s in splits["test"])
        assert n_train > 50, f"train chỉ có {n_train} mẫu multiply(100)?"
        assert n_valid <= 5 and n_test == 0, f"valid={n_valid} test={n_test}"

    def test_audit_tra_ve_du_truong(self, splits):
        rep = data.audit_gold(splits["valid"], "valid")
        for key in ("n", "by_source", "multiply_100", "gold_not_executable",
                    "noisy", "noisy_pct", "executable_pct", "table_ops"):
            assert key in rep
        assert rep["n"] == len(splits["valid"])

    def test_is_noisy_gold(self, splits):
        noisy = [s for s in splits["train"] if data.is_noisy_gold(s)]
        assert 50 < len(noisy) < 300, f"{len(noisy)} mẫu nhiễu — ngoài khoảng mong đợi"
        assert not any(data.is_noisy_gold(s) for s in splits["test"])


# ═══════════════════════════ 6. lấy mẫu ═══════════════════════════

class TestSampling:
    def test_stratified_dung_so_luong(self, splits):
        for total in (40, 120, 600):
            sub = data.stratified_sample(splits["train"], total, seed=42)
            assert len(sub) == total

    def test_stratified_on_dinh_theo_seed(self, splits):
        a = data.stratified_sample(splits["train"], 50, seed=1)
        b = data.stratified_sample(splits["train"], 50, seed=1)
        c = data.stratified_sample(splits["train"], 50, seed=2)
        assert [s["id"] for s in a] == [s["id"] for s in b]
        assert [s["id"] for s in a] != [s["id"] for s in c]

    def test_stratified_co_bai_nhieu_buoc(self, splits):
        sub = data.stratified_sample(splits["train"], 200, seed=42)
        hard = sum(1 for s in sub if dsl.n_ops(s["qa"]["program"]) >= 3)
        assert hard >= 20, f"chỉ {hard} bài ≥3 bước — phân tầng không nâng được bài khó"

    def test_total_lon_hon_du_lieu(self, splits):
        sub = data.stratified_sample(splits["valid"], 10_000)
        assert len(sub) == len(splits["valid"])


# ═══════════════════════════ 7. cụm lỗi ═══════════════════════════

class TestClusters:
    @pytest.mark.parametrize("question,program,expect", [
        ("Tỷ lệ tăng trưởng doanh thu 2019 so với 2018?", "subtract(5, 4), divide(#0, 4)",
         "C2_tang_truong"),
        ("Doanh thu lớn nhất giai đoạn này là bao nhiêu?", "table_max(Doanh thu, none)",
         "C7_max_min_bang"),
        ("Tỷ trọng chi phí quản lý trong tổng doanh thu?", "divide(892, 18600)",
         "C1_ty_trong"),
        ("Bình quân lợi nhuận là bao nhiêu?", "table_average(LNST, none)",
         "C8_trung_binh_bang"),
    ])
    def test_phan_cum(self, question, program, expect):
        assert clusters.cluster_id_for_sample(
            {"qa": {"question": question, "program": program}}) == expect

    def test_moi_cum_deu_co_mau_thuc_te(self, splits):
        """Cụm không bao giờ khớp mẫu nào là cụm chết — regex cần sửa."""
        dist = clusters.cluster_distribution(splits["train"])
        dead = [c["id"] for c in clusters.VI_CLUSTERS if dist.get(c["id"], 0) == 0]
        assert not dead, f"cụm không khớp mẫu nào: {dead}"

    def test_catch_all_khong_nuot_qua_nua(self, splits):
        dist = clusters.cluster_distribution(splits["train"])
        n = sum(dist.values())
        assert dist.get("C11_khac", 0) / n < 0.45


# ═══════════════════════════ 8. playbook ═══════════════════════════

GOOD_BULLET = ("Khi hỏi tốc độ tăng trưởng giữa hai kỳ, dùng "
               "subtract(gia_tri_moi, gia_tri_cu), divide(#0, gia_tri_cu).")


class TestPlaybookText:
    def test_playbook_rong(self):
        pb = playbook.empty_playbook()
        assert playbook.all_bullets(pb) == []
        assert all(t in pb for t in playbook.SECTION_TITLES.values())

    def test_them_va_doc_lai(self):
        pb = playbook.inject_bullet(playbook.empty_playbook(),
                                    "sinh_program", "sp-00001", GOOD_BULLET)
        bullets = playbook.all_bullets(pb)
        assert len(bullets) == 1
        assert bullets[0]["id"] == "sp-00001"
        assert bullets[0]["section"] == "sinh_program"
        assert bullets[0]["content"] == GOOD_BULLET

    def test_render_on_dinh(self):
        """Dựng lại nhiều lần phải ra đúng một chuỗi — bố cục không được trôi."""
        pb = playbook.inject_bullet(playbook.empty_playbook(),
                                    "doc_bang", "db-00001", GOOD_BULLET)
        assert playbook.render_playbook(playbook.all_bullets(pb)) == pb

    def test_bo_dem_helpful_harmful(self):
        pb = playbook.inject_bullet(playbook.empty_playbook(),
                                    "sinh_program", "sp-00001", GOOD_BULLET)
        pb = playbook.update_bullet_counts(pb, [{"id": "sp-00001", "tag": "helpful"}])
        pb = playbook.update_bullet_counts(pb, [{"id": "sp-00001", "tag": "helpful"}])
        pb = playbook.update_bullet_counts(pb, [{"id": "sp-00001", "tag": "harmful"}])
        b = playbook.all_bullets(pb)[0]
        assert (b["helpful"], b["harmful"]) == (2, 1)
        assert b["content"] == GOOD_BULLET       # nội dung không bị dính hậu tố đếm

    def test_next_id_map(self):
        pb = playbook.empty_playbook()
        pb = playbook.inject_bullet(pb, "sinh_program", "sp-00001", GOOD_BULLET)
        pb = playbook.inject_bullet(pb, "sinh_program", "sp-00007", GOOD_BULLET + " Biến thể.")
        assert playbook.next_id_map(pb)["sp"] == 7


# ═══════════════════════════ 9. quality gate ═══════════════════════════

class TestQualityGate:
    @pytest.fixture
    def gate(self):
        return playbook.QualityGate(embedder=None, retriever=None)

    @pytest.fixture
    def pb(self):
        return playbook.empty_playbook()

    def test_nhan_bullet_tot(self, gate, pb):
        assert gate(GOOD_BULLET, pb) == ("add", "ok")

    @pytest.mark.parametrize("bullet,reason", [
        ("Khi tính tỷ lệ tăng trưởng, dùng subtract(4060, 2527), divide(#0, 2527).",
         "chua_so_lieu_cu_the"),
        ("Khi tính doanh thu năm 2019, dùng subtract(a, b), divide(#0, b).",
         "chua_nam_cu_the"),
        ("Khi tính tỷ lệ, dùng divide(phan, tong), multiply(#0, 100).",
         "multiply_100_tinh_phan_tram"),
        ("Khi tính giá trị gốc, dùng divide(gia_tri_hien_tai, add(1, ty_le_tang)) cho nhanh.",
         "phep_toan_long_nhau"),
        ("Khi gặp bài khó, hãy suy nghĩ thật cẩn thận rồi mới trả lời nhé bạn.",
         "qua_chung_chung"),
        ("Khi tính tỷ lệ tăng, dùng subtract(gia_tri_moi, gia_tri_cu), divide(#1, gia_tri_cu).",
         "tham_chieu_sai"),
    ])
    def test_loai_bullet_xau(self, gate, pb, bullet, reason):
        action, why = gate(bullet, pb)
        assert action == "reject"
        assert why.startswith(reason), f"kỳ vọng {reason}, nhận {why}"

    def test_nhan_bullet_mot_phep_toan(self, gate, pb):
        """64 % tập test là câu MỘT phép — ép ≥2 là chặn lời khuyên cho nhóm lớn nhất.

        Bản ACE gốc ép 2 vì FinQA tiếng Anh khác phân bố; ta để 1.
        """
        assert gate("Khi hỏi tỷ trọng của khoản mục, chỉ cần dùng divide(phan, tong).",
                    pb)[0] == "add"

    def test_van_loai_bullet_khong_co_phep_toan_nao(self, gate, pb):
        assert gate("Khi gặp câu hỏi khó về bảng số liệu thì nên đọc kỹ đề bài hơn.",
                    pb)[1].startswith("duoi_1_phep_toan")

    def test_nguong_dedup_phai_hop_voi_embedder(self):
        """Ngưỡng tương đồng không chuyển được giữa các embedder.

        e5 nén mọi cặp vào dải ~0.70–0.90; giữ 0.85 của MiniLM/bge là loại oan.
        """
        assert playbook.QualityGate().dedup_thresh >= 0.90

    def test_do_dai(self, gate, pb):
        assert gate("Khi tính, dùng add(a,b).", pb)[1] == "qua_ngan"
        assert gate("Khi hỏi tỷ lệ tăng trưởng " + "rất " * 80 +
                    "thì dùng subtract(a, b), divide(#0, b).", pb)[1] == "qua_dai"

    def test_kiem_chung_tong_hop_bat_cong_thuc_nguoc(self, gate):
        ok, _ = gate.validate_synthetically(
            "Khi hỏi tỷ lệ tăng trưởng, dùng subtract(gia_tri_cu, gia_tri_moi), "
            "divide(#0, gia_tri_moi).")
        assert not ok, "công thức đảo ngược phải bị bắt"
        ok, _ = gate.validate_synthetically(GOOD_BULLET)
        assert ok

    def test_chong_trung(self, gate):
        pb = playbook.inject_bullet(playbook.empty_playbook(),
                                    "sinh_program", "sp-00001", GOOD_BULLET)
        assert gate(GOOD_BULLET, pb)[0] == "reject"

    def test_tat_gate_cho_ablation(self, pb):
        off = playbook.QualityGate(enabled=False)
        assert off("bất kỳ thứ gì", pb)[0] == "add"
        assert off("", pb)[0] == "reject"


# ═══════════════════════════ 10. truy hồi + curator ═══════════════════════════

class TestRetrieverCurator:
    def _pb_with(self, n=6):
        pb = playbook.empty_playbook()
        texts = [
            ("sinh_program", "Khi hỏi tốc độ tăng trưởng, dùng subtract(gia_tri_moi, "
                             "gia_tri_cu), divide(#0, gia_tri_cu)."),
            ("doc_bang", "Khi hỏi giá trị lớn nhất của chỉ tiêu, dùng "
                         "table_max(ten_chi_tieu, none) rồi mới add(#0, gia_tri)."),
            ("chien_luoc_so_hoc", "Khi hai số khác đơn vị, quy đổi bằng "
                                  "multiply(gia_tri, 1000) rồi mới divide(phan, #0)."),
            ("meo_ngu_canh", "Khi số liệu nằm trong pre-text, lấy trực tiếp vào "
                             "subtract(a, b), divide(#0, b)."),
            ("loi_thuong_gap", "Khi đề hỏi chênh lệch, chỉ dùng subtract(a, b), "
                               "không thêm divide(#0, b)."),
            ("sinh_program", "Khi cộng nhiều kỳ, dùng add(a, b), add(#0, c) nối tiếp."),
        ]
        for i, (sec, txt) in enumerate(texts[:n], 1):
            pb = playbook.inject_bullet(pb, sec, f"{playbook.SECTION_SLUGS[sec]}-{i:05d}", txt)
        return pb

    def test_truy_hoi_khong_embedding(self):
        r = playbook.Retriever(embedder=None, k_tier1=2, k_tier2=2)
        text, ids = r.retrieve("Tỷ lệ tăng trưởng doanh thu năm nay?", self._pb_with())
        assert len(ids) == 4
        assert text.count("\n- ") == 3

    def test_playbook_rong_tra_ve_rong(self):
        r = playbook.Retriever(embedder=None)
        assert r.retrieve("câu hỏi", playbook.empty_playbook()) == ("", [])

    def test_k_bang_0(self):
        r = playbook.Retriever(embedder=None)
        assert r.retrieve("câu hỏi", self._pb_with(), k_tier1=0, k_tier2=0) == ("", [])

    def test_ghi_nhan_luot_dung(self):
        r = playbook.Retriever(embedder=None, k_tier1=1, k_tier2=1)
        _, ids = r.retrieve("tăng trưởng", self._pb_with(), record=True)
        assert all(r.retrieval_count[i] == 1 for i in ids)

    def test_thang_hang_tier1(self):
        r = playbook.Retriever(embedder=None, tier1_min_uses=3, tier1_min_lift=0.01,
                               tier1_max=2)
        pb = self._pb_with()
        ids = [b["id"] for b in playbook.all_bullets(pb)]
        for _ in range(5):                       # bullet 0 luôn đúng, bullet 1 luôn sai
            r.record_outcome([ids[0]], pa_ok=True, ea_ok=True)
            r.record_outcome([ids[1]], pa_ok=False, ea_ok=False)
        promoted = r.maybe_promote_tier1(pb, baseline_pa=0.5)
        assert [p[0] for p in promoted] == [ids[0]]
        assert ids[1] not in r.tier1_ids

    def test_tier1_luon_duoc_truy_hoi(self):
        r = playbook.Retriever(embedder=None, k_tier1=1, k_tier2=1)
        pb = self._pb_with()
        target = playbook.all_bullets(pb)[3]["id"]
        r.tier1_ids.add(target)
        for q in ("tăng trưởng doanh thu", "giá trị lớn nhất", "quy đổi đơn vị"):
            _, ids = r.retrieve(q, pb)
            assert target in ids, f"Tier-1 phải luôn có mặt, thiếu ở '{q}'"

    def test_ep_tran_dung_luong(self):
        r = playbook.Retriever(embedder=None, max_bullets=3)
        pb = self._pb_with(6)
        pb2, dropped = r.enforce_budget(pb)
        assert len(playbook.all_bullets(pb2)) == 3
        assert len(dropped) == 3

    def test_ep_tran_khong_bo_tier1(self):
        r = playbook.Retriever(embedder=None, max_bullets=2)
        pb = self._pb_with(6)
        keep = [b["id"] for b in playbook.all_bullets(pb)[:2]]
        r.tier1_ids.update(keep)
        pb2, _ = r.enforce_budget(pb)
        remaining = {b["id"] for b in playbook.all_bullets(pb2)}
        assert set(keep) <= remaining

    def test_curator_chen_dung_muc(self):
        cur = playbook.Curator(playbook.QualityGate(), max_bullets_per_cluster=3)
        pb, action, why, bid = cur(GOOD_BULLET, "sai_phep_toan",
                                   playbook.empty_playbook(), "C2_tang_truong")
        assert (action, bid) == ("add", "sp-00001")
        assert playbook.all_bullets(pb)[0]["section"] == "sinh_program"

    def test_curator_han_ngach_cum(self):
        cur = playbook.Curator(playbook.QualityGate(), max_bullets_per_cluster=1)
        pb = playbook.empty_playbook()
        pb, _, _, _ = cur(GOOD_BULLET, "sai_phep_toan", pb, "C2_tang_truong")
        pb, action, why, _ = cur(
            "Khi cộng nhiều kỳ liên tiếp, dùng add(a, b), add(#0, c) nối bằng tham chiếu.",
            "sai_phep_toan", pb, "C2_tang_truong")
        assert action == "reject" and "da_du_bullet" in why


# ═══════════════════════════ 11. reflector ═══════════════════════════

class TestReflector:
    def test_diagnose(self):
        assert reflector.diagnose("", None, None, "add(1,2)", "3") == "khong_co_program"
        assert reflector.diagnose("```plaintext```", None, None, "add(1,2)", "3") == \
            "sai_dinh_dang"
        assert reflector.diagnose("x", "subtract(100, 553)", -453.0,
                                  "subtract(553, 100)", "453.0") == "sai_dau"
        assert reflector.diagnose("x", "divide(1,2)", 50.0, "divide(1,2)", "0.5") == \
            "sai_bac_do_lon"
        assert reflector.diagnose("x", "divide(a,b)", 1.0,
                                  "subtract(a,b), divide(#0,b)", "2.0") == "thieu_buoc"

    def test_parse_json_lung_tung(self):
        r = reflector.parse_reflector_json(
            'nói lăng nhăng {"error_type": "sai_phep_toan", '
            '"new_strategy": "Khi ... divide(a, b), add(#0, c)", "bullet_tags": []} hết')
        assert r["error_type"] == "sai_phep_toan"
        assert r["new_strategy"].startswith("Khi")

    def test_parse_json_trong_code_fence(self):
        r = reflector.parse_reflector_json(
            '```json\n{"error_type": "sai_don_vi", "new_strategy": "Khi quy đổi..."}\n```')
        assert r["error_type"] == "sai_don_vi"

    def test_parse_hong_tra_ve_mac_dinh(self):
        r = reflector.parse_reflector_json("hoàn toàn không phải JSON", fallback="thieu_buoc")
        assert r["error_type"] == "thieu_buoc" and r["new_strategy"] == ""
        assert reflector.parse_reflector_json(None)["bullet_tags"] == []

    def test_prompt_co_thong_tin_cum(self, test_set):
        kit = _FakePromptKit()
        p = reflector.build_reflector_prompt(
            test_set[0], "divide(1,2)", 0.5, "", "sai_phep_toan",
            playbook.empty_playbook(), "C2_tang_truong", kit, playbook.all_bullets)
        assert "C2_tang_truong" in p
        assert "subtract(gia_tri_moi, gia_tri_cu)" in p
        assert "const_" not in p.replace("KHÔNG có hằng số const_*", "")


# ═══════════════════════════ 12. pipeline (generator giả) ═══════════════════════════

class _FakePromptKit:
    """PromptKit tối giản, không cần tokenizer/model."""

    def step1(self, sample, bullets_text="", level="engineered"):
        return (f"[STEP1:{level}]\nBULLETS:{bullets_text}\n"
                f"Q:{sample['qa']['question']}\nID:{sample['id']}")

    def step2(self, sample, initial_response, bullets_text="", gia_tri_buoc1=...):
        _gt = "" if gia_tri_buoc1 is ... else f"\nVAL:{gia_tri_buoc1}"
        return (f"[STEP2]\nBULLETS:{bullets_text}\nPREV:{initial_response}"
                f"{_gt}\nID:{sample['id']}")

    def sft_messages(self, sample, target_text, level="engineered"):
        return [{"role": "system", "content": f"SYS:{level}"},
                {"role": "user", "content": f"ID:{sample['id']}"},
                {"role": "assistant", "content": target_text}]

    def chat(self, messages):
        return "\n".join(m["content"] for m in messages)

    def context_block(self, sample):
        return ("pre", "post", "bảng")


def _make_generator(samples, correct_rate=0.6, seed=7, boost_with_bullets=True):
    """Generator giả: trả gold cho một phần mẫu, còn lại sai theo kiểu thật."""
    by_id = {s["id"]: s for s in samples}
    rng = random.Random(seed)

    def gen(prompts, sampling_params=None, desc=None, **kw):
        outs = []
        for p in prompts:
            m = re.search(r"ID:(\S+)", p)
            s = by_id.get(m.group(1)) if m else None
            if s is None:
                outs.append("không rõ")
                continue
            gold = s["qa"]["program"]
            rate = correct_rate
            if boost_with_bullets and "gia_tri_cu" in p:
                rate = min(1.0, correct_rate + 0.35)
            r = rng.random()
            if r < rate:
                prog = gold
            elif r < rate + 0.15:
                prog = re.sub(r"subtract\(([^,]+),\s*([^)]+)\)", r"subtract(\2, \1)",
                              gold, count=1)
            elif r < rate + 0.25:
                outs.append("Tôi không chắc lắm.")
                continue
            else:
                prog = "divide(1, 7)"
            val = dsl.execute_program(prog, s.get("table") or [])
            outs.append(f"Phân tích.\n```plaintext\nprogram: {prog}\nanswer: {val}\n```")
        return outs

    return gen


class TestPipeline:
    def test_run_pipeline_khong_playbook(self, test_set):
        samples = test_set[:20]
        rows = pipeline.run_pipeline(samples, _FakePromptKit(),
                                     _make_generator(samples, correct_rate=1.0),
                                     use_selfeval=False)
        assert len(rows) == 20
        assert all(r["ea"] and r["pa_strict"] for r in rows)
        assert all(r["used_bullets"] == [] for r in rows)
        assert [r["id"] for r in rows] == [s["id"] for s in samples]

    def test_run_pipeline_uu_tien_step2(self, test_set):
        """Có program ở step 2 thì phải lấy step 2, không lấy step 1."""
        samples = test_set[:5]

        def gen(prompts, sp=None, desc=None, **kw):
            if "[STEP1" in prompts[0]:
                return ["```plaintext\nprogram: divide(1, 7)\n```"] * len(prompts)
            return ["```plaintext\nprogram: add(2, 3)\nanswer: 5\n```"] * len(prompts)

        rows = pipeline.run_pipeline(samples, _FakePromptKit(), gen, use_selfeval=True)
        assert all(r["program_step1"] == "divide(1, 7)" for r in rows)
        assert all(r["final_program"] == "add(2, 3)" for r in rows)

    def test_step2_khong_ra_program_thi_lui_ve_step1(self, test_set):
        samples = test_set[:5]

        def gen(prompts, sp=None, desc=None, **kw):
            if "[STEP1" in prompts[0]:
                return ["```plaintext\nprogram: add(2, 3)\n```"] * len(prompts)
            return ["Tôi không sửa được gì."] * len(prompts)

        rows = pipeline.run_pipeline(samples, _FakePromptKit(), gen, use_selfeval=True)
        assert all(r["final_program"] == "add(2, 3)" for r in rows)

    def test_summarize(self, test_set):
        samples = test_set[:30]
        rows = pipeline.run_pipeline(samples, _FakePromptKit(),
                                     _make_generator(samples, 0.5), use_selfeval=False)
        m = pipeline.summarize(rows, "thử")
        assert m["n"] == 30
        assert 0.0 <= m["EA"] <= 1.0 and 0.0 <= m["PA_strict"] <= 1.0
        assert sum(v[0] for v in m["by_steps"].values()) == 30
        assert sum(m["outcome"].values()) == 30

    def test_mot_vong_ace_day_du(self, test_set):
        """Chạy trọn một vòng Generator → Reflector → Verify → Curator."""
        samples = test_set[:24]
        gate = playbook.QualityGate(embedder=None)
        ret = playbook.Retriever(embedder=None, k_tier1=2, k_tier2=2)
        cur = playbook.Curator(gate)

        class _Refl:
            def __call__(self, items, pb):
                return [{"error_type": "sai_phep_toan",
                         "root_cause": "x",
                         "new_strategy": GOOD_BULLET,
                         "bullet_tags": []} for _ in items]

        trainer = ace_trainer.AceTrainer(
            _FakePromptKit(), _make_generator(samples, 0.4), ret, cur, _Refl(),
            round_size=24, max_reflect=4, use_selfeval=False, use_verify=True)
        rows, info = trainer.run_round(samples, 0)

        assert len(rows) == 24
        assert len(trainer.history) == 24
        assert all("_cluster" in r for r in rows)
        # Reflector luôn đề xuất cùng một bullet → chỉ 1 cái được thêm, phần còn lại bị loại trùng
        assert len(playbook.all_bullets(trainer.playbook)) <= 1
        assert info["n_fail"] >= 0

    def test_verify_chan_bullet_vo_dung(self, test_set):
        """Bullet không sửa được mẫu thì không được vào playbook."""
        samples = test_set[:8]
        gate = playbook.QualityGate(embedder=None)
        ret = playbook.Retriever(embedder=None)
        cur = playbook.Curator(gate)

        class _Refl:
            def __call__(self, items, pb):
                return [{"error_type": "sai_phep_toan", "new_strategy": GOOD_BULLET,
                         "bullet_tags": []} for _ in items]

        def always_wrong(prompts, sp=None, desc=None, **kw):
            return ["```plaintext\nprogram: divide(1, 7)\n```"] * len(prompts)

        trainer = ace_trainer.AceTrainer(_FakePromptKit(), always_wrong, ret, cur, _Refl(),
                                         round_size=8, max_reflect=4, use_selfeval=False,
                                         use_verify=True)
        trainer.run_round(samples, 0)
        assert playbook.all_bullets(trainer.playbook) == []
        assert trainer.qg_reasons["verify_khong_sua_duoc"] > 0


# ═══════════════════════════ 13. thống kê ═══════════════════════════

class TestStats:
    def test_mcnemar_khong_bat_dong(self):
        r = stats.mcnemar_exact([True, False, True], [True, False, True])
        assert r["n_discordant"] == 0 and r["p_value"] == 1.0

    def test_mcnemar_lech_han(self):
        a = [False] * 20
        b = [True] * 20
        r = stats.mcnemar_exact(a, b)
        assert r["b"] == 0 and r["c"] == 20
        assert r["p_value"] < 0.001

    def test_mcnemar_can_bang(self):
        a = [True] * 10 + [False] * 10
        b = [False] * 10 + [True] * 10
        assert stats.mcnemar_exact(a, b)["p_value"] == pytest.approx(1.0, abs=0.01)

    def test_mcnemar_doi_xung(self):
        a = [True, False, True, False, False]
        b = [False, True, True, False, True]
        r1 = stats.mcnemar_exact(a, b)
        r2 = stats.mcnemar_exact(b, a)
        assert r1["p_value"] == pytest.approx(r2["p_value"])
        assert (r1["b"], r1["c"]) == (r2["c"], r2["b"])

    def test_bootstrap_ci_bao_delta(self):
        a = [False] * 50 + [True] * 50
        b = [True] * 80 + [False] * 20
        lo, hi = stats.bootstrap_delta_ci(a, b, n_boot=2000, seed=1)
        delta = sum(b) / len(b) - sum(a) / len(a)
        assert lo <= delta <= hi

    def test_bootstrap_ci_chua_0_khi_khong_khac(self):
        rng = random.Random(3)
        a = [rng.random() < 0.6 for _ in range(400)]
        b = list(a)
        lo, hi = stats.bootstrap_delta_ci(a, b, n_boot=2000, seed=1)
        assert lo <= 0 <= hi

    def test_compare_pair(self, test_set):
        base = [{"id": s["id"], "ea": False} for s in test_set[:50]]
        var = [{"id": s["id"], "ea": True} for s in test_set[:50]]
        out = stats.compare_pair(base, var, key="ea", verbose=False)
        assert out["delta"] == pytest.approx(1.0)
        assert out["p_value"] < 0.001
        assert "✅" in out["verdict"]

    def test_compare_pair_bat_lech_thu_tu(self, test_set):
        base = [{"id": s["id"], "ea": True} for s in test_set[:10]]
        var = [{"id": s["id"], "ea": True} for s in reversed(test_set[:10])]
        with pytest.raises(ValueError):
            stats.compare_pair(base, var, verbose=False)


# ═══════════════════════════ 14. io_utils ═══════════════════════════

class TestIO:
    def test_cham_lai_du_doan_da_luu(self, test_set):
        """Tái lập PA tham chiếu của Qwen3-8B từ file CSV có sẵn."""
        csv_path = os.path.join(REPO, "reference", "baseline_results", "Qwen3-8B_program.csv")
        if not os.path.exists(csv_path):
            pytest.skip("chưa có file dự đoán Qwen3-8B")
        rows = io_utils.score_saved_predictions(csv_path, test_set, "Qwen3-8B")
        m = pipeline.summarize(rows, "qwen3-baseline")
        assert m["n"] == len(test_set)
        assert m["PA_loose"] * 100 == pytest.approx(
            io_utils.BASELINE_RESULTS["Qwen3-8B"]["PA"], abs=0.1), \
            "PA_loose phải tái lập đúng mốc tham chiếu"
        assert m["EA"] * 100 > io_utils.BASELINE_RESULTS["Qwen3-8B"]["EA"], \
            "EA chấm lại phải cao hơn (mốc tham chiếu bị hụt do lỗi table_*)"

    def test_thu_tu_row_bam_theo_samples(self, test_set):
        csv_path = os.path.join(REPO, "reference", "baseline_results", "Qwen3-8B_program.csv")
        if not os.path.exists(csv_path):
            pytest.skip("chưa có file dự đoán Qwen3-8B")
        rows = io_utils.score_saved_predictions(csv_path, test_set[:20], "x")
        assert [r["id"] for r in rows] == [s["id"] for s in test_set[:20]]

    def test_ghi_doc_csv(self, tmp_path, test_set):
        rows = [{"id": s["id"], "question": s["qa"]["question"],
                 "gold_program": s["qa"]["program"], "gold_answer": s["qa"]["exe_ans"],
                 "program_step1": "add(1,2)", "program_step2": "add(1,3)",
                 "thua": "bỏ qua"} for s in test_set[:5]]
        p = io_utils.save_details_csv(rows, str(tmp_path / "x_program.csv"))
        back = io_utils.load_predictions(p)
        assert len(back) == 5
        assert list(back[0].keys()) == io_utils.LEGACY_COLS

    def test_ghi_jsonl(self, tmp_path, test_set):
        rows = [{"id": "a", "ea": True, "raw_step1": "dài", "bullets_text": "x"}]
        p = io_utils.save_full_jsonl(rows, str(tmp_path / "f.jsonl"))
        import json as _json
        with open(p, encoding="utf-8") as f:
            rec = _json.loads(f.read().strip())
        assert "raw_step1" not in rec and rec["ea"] is True


# ═══════════════════════════ 15. prompts ═══════════════════════════

class TestPrompts:
    def test_import_prompt_tu_repo(self):
        from vinumqa.prompts import PromptKit
        kit = PromptKit(REPO, tokenizer=None, model_name="test")
        assert len(kit.ENGINEERED_SYSTEM_PROMPT) > 1000
        assert len(kit.SELF_EVAL_SYSTEM_PROMPT) > 1000
        assert "DANH SÁCH PHÉP TOÁN" in kit.ENGINEERED_SYSTEM_PROMPT

    def test_prompt_chua_cau_hoi_va_bullet(self, test_set):
        from vinumqa.prompts import PromptKit
        kit = PromptKit(REPO, tokenizer=None, model_name="test")
        p = kit.step1(test_set[0], "- Bullet thử nghiệm")
        assert test_set[0]["qa"]["question"] in p
        assert "Bullet thử nghiệm" in p
        assert "plaintext" in p

    def test_khong_bullet_thi_khong_co_khoi_hoi_tuong(self, test_set):
        from vinumqa.prompts import PromptKit, MEMORY_ASK
        kit = PromptKit(REPO, tokenizer=None, model_name="test")
        assert MEMORY_ASK not in kit.step1(test_set[0], "")

    def test_step2_chua_ket_qua_buoc_1(self, test_set):
        from vinumqa.prompts import PromptKit
        kit = PromptKit(REPO, tokenizer=None, model_name="test")
        p = kit.step2(test_set[0], "KẾT QUẢ BƯỚC MỘT", "")
        assert "KẾT QUẢ BƯỚC MỘT" in p


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))


class TestCheDoSuyNghi:
    """Chế độ suy nghĩ phải để chat template tự quyết, đúng như bản tham chiếu.

    Bản trước ép enable_thinking=False cho Qwen3 và mất ~10 điểm PA_loose.
    """

    def test_mac_dinh_khong_ep_tat_cho_qwen3(self):
        kit = prompts_mod.PromptKit(model_name="unsloth/Qwen3-8B")
        assert kit.enable_thinking is None, (
            "Qwen3 phải để template tự quyết (mặc định BẬT suy nghĩ), không ép tắt")

    def test_mac_dinh_giong_nhau_moi_model(self):
        for name in ("unsloth/Qwen3-8B", "microsoft/phi-4", "mistralai/Mistral-7B", ""):
            assert prompts_mod.PromptKit(model_name=name).enable_thinking is None

    def test_van_ep_duoc_khi_can(self):
        assert prompts_mod.PromptKit(model_name="unsloth/Qwen3-8B",
                                 enable_thinking=False).enable_thinking is False
        assert prompts_mod.PromptKit(model_name="unsloth/Qwen3-8B",
                                 enable_thinking=True).enable_thinking is True

    def test_chat_khong_truyen_enable_thinking_khi_None(self):
        ghi = {}

        class _Tok:
            chat_template = "x"

            def apply_chat_template(self, messages, **kw):
                ghi.update(kw)
                return "PROMPT"

        kit = prompts_mod.PromptKit(tokenizer=_Tok(), model_name="unsloth/Qwen3-8B")
        kit.chat([{"role": "user", "content": "hi"}])
        assert "enable_thinking" not in ghi, (
            "Phải gọi apply_chat_template y như bản tham chiếu: không truyền tham số này")


class TestTrichXuatKhiBatSuyNghi:
    """Output của Qwen3 ở chế độ suy nghĩ — ba dạng thẻ <think> đều phải xử lý đúng."""

    KHOI = "```plaintext\nprogram: subtract(100, 40)\nanswer: 60\n```"

    def test_du_cap_the(self):
        raw = "<think>\nTa lấy 100 trừ 40. program: add(1,2) thử xem\n</think>\n\n" + self.KHOI
        prog, ans = dsl.extract_program_answer(raw)
        assert prog == "subtract(100, 40)", "phải bỏ phần suy luận, lấy khối chốt"
        assert ans == "60"

    def test_chi_co_the_dong(self):
        """Template mở sẵn <think> nên phần sinh ra chỉ có thẻ đóng."""
        raw = "Ta lấy 100 trừ 40. program: add(1,2) thử xem\n</think>\n\n" + self.KHOI
        prog, _ = dsl.extract_program_answer(raw)
        assert prog == "subtract(100, 40)", "phần trước </think> là suy luận, phải bỏ"

    def test_chi_co_the_mo_thi_coi_nhu_khong_sinh_duoc(self):
        """Chạm max_tokens giữa lúc suy luận — không có câu trả lời chốt."""
        raw = "<think>\nTôi đang nghĩ, chưa xong thì bị cắt"
        prog, _ = dsl.extract_program_answer(raw)
        assert prog is None

    def test_suy_luan_dai_nhieu_khoi_lay_khoi_cuoi(self):
        raw = ("<think>\nThử ```plaintext\nprogram: add(1, 2)\nanswer: 3\n``` xem sao\n"
               "</think>\n\nSau khi soát lại:\n" + self.KHOI)
        prog, _ = dsl.extract_program_answer(raw)
        assert prog == "subtract(100, 40)"


class TestMucNoFewshot:
    """Nấc 2b: đúng prompt engineered nhưng bỏ 2 ví dụ mẫu."""

    def test_bo_dung_khoi_vi_du(self):
        kit = prompts_mod.PromptKit()
        day_du, bo_vd = kit.ENGINEERED_SYSTEM_PROMPT, kit.NO_FEWSHOT_SYSTEM_PROMPT
        assert "Ví dụ 1:" in day_du and "Ví dụ 2:" in day_du
        assert "Ví dụ 1:" not in bo_vd and "Ví dụ 2:" not in bo_vd
        assert 0 < len(day_du) - len(bo_vd) < 800, "chỉ được bỏ khối ví dụ"

    def test_giu_nguyen_phan_con_lai(self):
        kit = prompts_mod.PromptKit()
        bo_vd = kit.NO_FEWSHOT_SYSTEM_PROMPT
        for moc in ("DANH SÁCH PHÉP TOÁN", "HƯỚNG DẪN CHỌN PHÉP TOÁN THEO TỪ KHÓA",
                    "==== CÂU HỎI ===="):
            assert moc in bo_vd, f"mất mục {moc!r} — hiệu số 2−2b sẽ lẫn thứ khác"

    def test_step1_chay_duoc_voi_muc_moi(self, test_set):
        kit = prompts_mod.PromptKit()
        mau = test_set[0]
        p = kit.step1(mau, level="no_fewshot")
        assert "Ví dụ 1:" not in p and mau["qa"]["question"] in p

    def test_muc_la_van_bao_loi(self, test_set):
        with pytest.raises(ValueError):
            prompts_mod.PromptKit().step1(test_set[0], level="khong_ton_tai")


class TestThangPromptLongNhau:
    """Nấc 1 mới = ``basic``: biết phép toán và định dạng, chưa được mách chọn phép.

    Thang phải LỒNG NHAU (basic ⊂ no_fewshot ⊂ engineered) thì hiệu số giữa hai nấc
    kề nhau mới quy được về phần vừa thêm. Sửa chữ ở phần dùng chung là hỏng phép đo,
    nên có test canh.
    """

    MUC = ("basic", "no_fewshot", "engineered")

    def _thang(self):
        kit = prompts_mod.PromptKit()
        return kit, [getattr(kit, f"{m.upper()}_SYSTEM_PROMPT") for m in self.MUC]

    def test_basic_co_phep_toan_va_dinh_dang(self):
        _, (basic, _, _) = self._thang()
        for moc in ("=== DANH SÁCH PHÉP TOÁN===", "- Trả lời đúng 2 dòng:",
                    "==== CÂU HỎI ===="):
            assert moc in basic, f"prompt cơ bản thiếu {moc!r}"

    def test_basic_khong_co_huong_dan_tu_khoa_va_vi_du(self):
        _, (basic, _, _) = self._thang()
        assert "=== HƯỚNG DẪN CHỌN PHÉP TOÁN THEO TỪ KHÓA ===" not in basic
        assert "Ví dụ 1:" not in basic and "Ví dụ 2:" not in basic

    def test_thang_long_nhau_moi_buoc_la_chen_thuan(self):
        import difflib
        _, ps = self._thang()
        for (na, a), (nb, b) in zip(zip(self.MUC, ps), zip(self.MUC[1:], ps[1:])):
            ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False)
                   .get_opcodes() if o[0] != "equal"]
            assert len(ops) == 1 and ops[0][0] == "insert", (
                f"{na} → {nb} không phải chèn thuần ({[o[0] for o in ops]}) — "
                "phần dùng chung bị sửa thì hiệu số hai nấc không còn quy về phần thêm")

    def test_do_dai_tang_dan(self):
        _, ps = self._thang()
        assert len(ps[0]) < len(ps[1]) < len(ps[2])

    def test_step1_basic_dung_chung_user_message_voi_engineered(self, test_set):
        """Chỉ system prompt được khác — khác cả user message là lẫn biến."""
        kit = prompts_mod.PromptKit()
        mau = test_set[0]
        pb_, pe = kit.step1(mau, level="basic"), kit.step1(mau, level="engineered")
        duoi = kit._user_engineered(mau)
        assert duoi in pb_ and duoi in pe

    def test_basic_nam_trong_LEVELS(self):
        assert "basic" in prompts_mod.PromptKit.LEVELS

    def test_sft_messages_nhan_du_bon_muc(self, test_set):
        kit = prompts_mod.PromptKit()
        for muc in ("plain",) + self.MUC:
            msgs = kit.sft_messages(test_set[0], "program: add(1, 2)", level=muc)
            assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
            assert msgs[0]["content"], f"mức {muc} không có system prompt"

    def test_moc_doi_thi_bao_loi_chu_khong_cat_bua(self):
        with pytest.raises(ValueError):
            prompts_mod.PromptKit.bo_huong_dan_tu_khoa("prompt không có mốc nào cả")


class TestPhanLoaiSai:
    """Ô "sai" nay là ô lớn nhất; tách theo KIỂU sai mới biết phải chữa gì."""

    def _mot(self, model, gold):
        return pipeline.phan_loai_sai([{"outcome": "sai", "final_program": model,
                                        "gold_program": gold, "question": "q"}])

    @pytest.mark.parametrize("model,gold,mong", [
        ("divide(500, 100)", "subtract(500, 100), divide(#0, 100)", "thieu_buoc"),
        ("subtract(5,1), divide(#0,1), add(#1,0)", "subtract(5, 1)", "thua_buoc"),
        ("subtract(500, 100)", "divide(500, 100)", "dung_so_buoc_sai_phep"),
        ("divide(500, 100)", "divide(600, 100)", "dung_phep_sai_so_lieu"),
        ("add(1, 2)", "table_max(doanh thu, none)", "bo_qua_table"),
        ("table_max(doanh thu, none)", "add(1, 2)", "lam_dung_table"),
    ])
    def test_tung_kieu(self, model, gold, mong):
        assert list(self._mot(model, gold)["theo_kieu"]) == [mong]

    def test_chi_dem_mau_sai(self):
        rows = [{"outcome": "dung", "final_program": "add(1,2)", "gold_program": "add(3,4)"}]
        assert pipeline.phan_loai_sai(rows) is None

    def test_giu_vi_du_that(self):
        d = self._mot("subtract(500, 100)", "divide(500, 100)")
        assert d["vi_du"]["dung_so_buoc_sai_phep"][0]["gold"] == "divide(500, 100)"


class TestSoSanhHaiBuoc:
    """Δ EA = 0 vẫn còn hai cách giải thích; phải đếm số program bước 2 đã đổi."""

    def _mot(self, p1, p2, test_set):
        rows = [{"id": test_set[0]["id"], "program_step1": p1, "program_step2": p2}]
        return pipeline.so_sanh_hai_buoc(rows, test_set)

    def test_chep_lai_y_nguyen(self, test_set):
        d = self._mot("divide(500, 100)", "divide(500, 100)", test_set)
        assert d["theo_nhom"] == {"chep_lai_y_nguyen": 1} and d["so_program_bi_doi"] == 0

    def test_khac_chu_nhung_cung_gia_tri(self, test_set):
        """`add(3,7)` vs `add(7,3)`: khác văn bản, cùng giá trị — phải tách riêng."""
        d = self._mot("add(3, 7)", "add(7, 3)", test_set)
        assert list(d["theo_nhom"]) == ["doi_nhung_gia_tri_giu_nguyen"]

    def test_doi_ca_gia_tri(self, test_set):
        d = self._mot("divide(500, 100)", "divide(500, 50)", test_set)
        assert list(d["theo_nhom"]) == ["doi_va_doi_ca_gia_tri"]

    def test_buoc2_bo_trong_khong_tinh_la_doi(self, test_set):
        rows = [{"id": test_set[0]["id"], "program_step1": "add(1,2)", "program_step2": ""},
                {"id": test_set[0]["id"], "program_step1": "add(1,2)",
                 "program_step2": "add(1,3)"}]
        d = pipeline.so_sanh_hai_buoc(rows, test_set)
        assert d["so_program_bi_doi"] == 1, "bỏ trống rồi lùi về bước 1 không phải là đổi"

    def test_nac_khong_co_buoc2(self, test_set):
        assert pipeline.so_sanh_hai_buoc(
            [{"id": "x", "program_step1": "add(1,2)", "program_step2": ""}], test_set) is None


class TestVotMauBiCat:
    """Mẫu chạm trần token được sinh lại với suy nghĩ TẮT."""

    CAT = "<think>đang nghĩ thì hết chỗ"
    XONG = "<think>xong</think>\n```plaintext\nprogram: add(1, 2)\nanswer: 3\n```"

    def _chay(self, test_set, vot, dap_lai_duoc=True):
        kit = _FakePromptKit()
        kit.enable_thinking = None
        luot = {"n": 0, "thinking": []}

        def gen(prompts, sp=None, desc=None, batch_size=None):
            luot["n"] += 1
            luot["thinking"].append(kit.enable_thinking)
            if luot["n"] == 1:
                return [self.CAT] * len(prompts)
            return [(self.XONG if dap_lai_duoc else self.CAT)] * len(prompts)

        rows = pipeline.run_pipeline(test_set[:4], kit, gen, prompt_level="basic",
                                     use_selfeval=False, vot_mau_bi_cat=vot)
        return rows, luot

    def test_co_vot_thi_cuu_duoc(self, test_set):
        rows, luot = self._chay(test_set, vot=True)
        assert luot["n"] == 2, "phải có lượt sinh thứ hai để vớt"
        assert luot["thinking"][1] is False, "lượt vớt phải TẮT suy nghĩ"
        assert all(r["final_program"] for r in rows)

    def test_tat_vot_thi_giu_nguyen(self, test_set):
        rows, luot = self._chay(test_set, vot=False)
        assert luot["n"] == 1 and not any(r["final_program"] for r in rows)

    def test_vot_that_bai_thi_khong_de_len(self, test_set):
        """Lượt vớt cũng hỏng thì giữ raw cũ, để con số 'bị cắt' vẫn đúng sự thật."""
        rows, _ = self._chay(test_set, vot=True, dap_lai_duoc=False)
        assert not any(r["final_program"] for r in rows)
        m = pipeline.summarize(rows, "x")
        assert m["vi_sao_khong_co_program"]["bi_cat_giua_suy_nghi"] == len(rows)

    def test_khoi_phuc_co_thinking_sau_khi_vot(self, test_set):
        kit = _FakePromptKit()
        kit.enable_thinking = None
        pipeline.run_pipeline(
            test_set[:2], kit,
            lambda ps, sp=None, desc=None, batch_size=None: [self.CAT] * len(ps),
            prompt_level="basic", use_selfeval=False)
        assert kit.enable_thinking is None, "phải trả lại cờ suy nghĩ như cũ"


class TestSelfEvalBietGiaTri:
    """Bước 2 phải được cho biết chương trình bước 1 chạy ra số bao nhiêu."""

    def test_gia_tri_duoc_dua_vao_prompt(self, test_set):
        kit = _FakePromptKit()
        thay = {}

        def gen(prompts, sp=None, desc=None, batch_size=None):
            if desc and desc.endswith("step2"):
                thay["p2"] = prompts[0]
                return ["```plaintext\nprogram: add(1, 2)\nanswer: 3\n```"] * len(prompts)
            return ["```plaintext\nprogram: divide(500, 100)\nanswer: 5\n```"] * len(prompts)

        pipeline.run_pipeline(test_set[:2], kit, gen, use_selfeval=True,
                              vot_mau_bi_cat=False)
        assert "VAL:5.0" in thay["p2"], "bước 2 phải thấy giá trị thực thi 500/100"

    def test_tat_co_thi_khong_dua(self, test_set):
        kit = _FakePromptKit()
        thay = {}

        def gen(prompts, sp=None, desc=None, batch_size=None):
            if desc and desc.endswith("step2"):
                thay["p2"] = prompts[0]
            return ["```plaintext\nprogram: divide(500, 100)\nanswer: 5\n```"] * len(prompts)

        pipeline.run_pipeline(test_set[:2], kit, gen, use_selfeval=True,
                              vot_mau_bi_cat=False, bao_gia_tri_cho_buoc2=False)
        assert "VAL:" not in thay["p2"]

    def test_prompt_that_co_khoi_gia_tri(self, test_set):
        kit = prompts_mod.PromptKit()
        co = kit.step2(test_set[0], "phân tích", gia_tri_buoc1=15000.0)
        khong = kit.step2(test_set[0], "phân tích")
        hong = kit.step2(test_set[0], "phân tích", gia_tri_buoc1=None)
        assert "15000.0" in co and "HỢP LÝ" in co
        assert "Chạy thật chương trình" not in khong, "không truyền thì giữ bản tham chiếu"
        assert "KHÔNG CHẠY ĐƯỢC" in hong


class TestReflectorCaiTien:
    """Hai thứ Reflector trước đây không được biết, nên đề xuất thừa."""

    def test_biet_luat_da_co_de_khong_de_xuat_lai(self):
        kit = prompts_mod.PromptKit()
        luat = reflector.luat_dang_ap_dung(kit)
        assert "TỪ KHÓA" in luat and 200 < len(luat) <= 1800

    def test_ca_lam_dung_duoc_dua_vao(self, test_set):
        kit = prompts_mod.PromptKit()
        p = reflector.build_reflector_prompt(
            test_set[0], "divide(1,2)", 0.5, "", "sai_phep_toan",
            playbook.empty_playbook(), "C11_khac", kit, playbook.all_bullets,
            row_dung={"question": "Doanh thu lớn nhất?",
                      "final_program": "table_max(Doanh thu, none)"})
        assert "table_max(Doanh thu, none)" in p
        assert "ĐỪNG đề xuất lại" in p

    def test_khong_co_ca_dung_van_chay(self, test_set):
        p = reflector.build_reflector_prompt(
            test_set[0], "divide(1,2)", 0.5, "", "sai_phep_toan",
            playbook.empty_playbook(), "C11_khac", prompts_mod.PromptKit(),
            playbook.all_bullets, row_dung=None)
        assert "chưa có ca đúng nào" in p


class TestVerifyHaiLan:
    """Verify sinh lại 2 lần: temperature 0.1 vẫn ngẫu nhiên, trượt 1 lần chưa phải vô dụng."""

    def _trainer(self, ket_qua):
        """generate_fn giả: lần 1 trả sai, lần 2 trả đúng."""
        goi = {"n": 0}

        def gen(prompts, sp=None, desc=None, batch_size=None):
            goi["n"] += 1
            prog = ket_qua[min(goi["n"], len(ket_qua)) - 1]
            return ["```plaintext\nprogram: " + prog + "\nanswer: 0\n```"] * len(prompts)

        t = ace_trainer.AceTrainer(prompts_mod.PromptKit(), gen, None, None, None,
                                   verify_lan=2)
        return t, goi

    def _ung_vien(self, test_set):
        s = next(x for x in test_set if x["qa"].get("program"))
        return [{"sample": s, "strategy": "dùng divide(a, b) cho tỷ trọng",
                 "bullets_text": "", "error_type": "khac", "cluster_id": "C11_khac"}]

    def test_lan_hai_cuu_duoc_ung_vien(self, test_set):
        uv = self._ung_vien(test_set)
        dung = uv[0]["sample"]["qa"]["program"]
        t, goi = self._trainer(["add(9999, 1)", dung])
        out = t.verify_candidates(uv)
        assert out[0]["passed"], "lần 2 đúng thì phải được nhận"
        assert goi["n"] == 2, "phải sinh đúng 2 lần"

    def test_dung_ngay_lan_dau_thi_khong_sinh_them(self, test_set):
        uv = self._ung_vien(test_set)
        dung = uv[0]["sample"]["qa"]["program"]
        t, goi = self._trainer([dung, dung])
        out = t.verify_candidates(uv)
        assert out[0]["passed"] and goi["n"] == 1, "đúng ngay thì khỏi sinh lần 2"

    def test_sai_ca_hai_lan_thi_loai(self, test_set):
        uv = self._ung_vien(test_set)
        t, goi = self._trainer(["add(9999, 1)", "add(8888, 2)"])
        out = t.verify_candidates(uv)
        assert not out[0]["passed"] and goi["n"] == 2


class TestViSaoKhongCoProgram:
    """no_program gộp hai nguyên nhân cần cách chữa khác nhau — phải tách được."""

    def _row(self, prog, raw):
        return {"final_program": prog, "raw_step1": raw, "raw_step2": "",
                "ea": False, "pa_strict": False, "pa_loose": False,
                "ea_tol1e-3": False, "n_ops_gold": 1, "outcome": "x",
                "pred_value": None}

    def test_the_think_khuyet_la_bi_cat(self):
        r = pipeline.phan_loai_khong_co_program(
            [self._row("", "<think>đang nghĩ thì hết token")])
        assert r["bi_cat_giua_suy_nghi"] == 1 and r["sai_dinh_dang"] == 0

    def test_nghi_xong_ma_thieu_khoi_la_sai_dinh_dang(self):
        r = pipeline.phan_loai_khong_co_program(
            [self._row("", "<think>xong rồi</think> quên mất khối program")])
        assert r["bi_cat_giua_suy_nghi"] == 0 and r["sai_dinh_dang"] == 1

    def test_mau_co_program_khong_bi_dem(self):
        r = pipeline.phan_loai_khong_co_program(
            [self._row("add(1, 2)", "<think>chưa đóng thẻ")])
        assert r["bi_cat_giua_suy_nghi"] == 0 and r["sai_dinh_dang"] == 0

    def test_khong_giu_raw_thi_tra_None(self):
        assert pipeline.phan_loai_khong_co_program([self._row("", "")]) is None

    def test_summarize_co_kem_phan_loai(self, test_set):
        rows = [{"id": s["id"], "question": s["qa"]["question"],
                 "gold_program": s["qa"]["program"], "gold_answer": s["qa"].get("exe_ans"),
                 "program_step1": "", "program_step2": "", "final_program": "",
                 "pred_value": None, "pred_answer_text": None, "ea": False,
                 "ea_tol1e-3": False, "pa_strict": False, "pa_loose": False,
                 "n_ops_gold": 1, "outcome": "khong_co_program", "used_bullets": [],
                 "raw_step1": "<think>bị cắt", "raw_step2": ""} for s in test_set[:4]]
        m = pipeline.summarize(rows, "thử")
        assert m["vi_sao_khong_co_program"]["bi_cat_giua_suy_nghi"] == 4
        assert m["no_program"] == 1.0
