# -*- coding: utf-8 -*-
"""Bộ CHỌN: gom ứng viên, đọc lựa chọn, và vòng chọn phải FAIL-CLOSED.

Vì sao cần bộ chọn: ở nấc 9, bỏ phiếu theo số đông đúng 49/116 câu có từ 2 giá trị
phân biệt, trong khi 89/116 câu CÓ mẫu đúng nằm đâu đó. Cả 40 câu chênh lệch đều là
trường hợp đáp án đúng thuộc THIỂU SỐ (1/5 hoặc 2/5 mẫu) — đếm phiếu không thắng được.

Điều kiện bắt buộc của phép đo này: nó chỉ được phép tốt lên hoặc đứng yên vì lỗi kỹ
thuật, KHÔNG được tụt vì lỗi kỹ thuật. Mọi nhánh hỏng phải giữ nguyên đáp án bỏ phiếu.
"""
import pytest

from vinumqa import pipeline
from vinumqa.prompts import PromptKit


def _mau(qid="x", gold="add(1,2)", ans="3.0"):
    return {"id": qid, "table": [], "pre_text": ["p"], "post_text": ["q"],
            "qa": {"question": "Tổng hai số là bao nhiêu?", "program": gold,
                   "exe_ans": ans}}


def _row(progs, vals, final=None):
    return {"id": "x", "question": "q", "gold_program": "add(1,2)", "gold_answer": "3.0",
            "cac_program": progs, "cac_gia_tri": vals,
            "final_program": final if final is not None else (progs[0] if progs else ""),
            "pred_value": vals[0] if vals else None,
            "program_step1": "", "program_step2": "", "ea": False, "ea_tol1e-3": False,
            "pa_strict": False, "pa_loose": False, "n_ops_gold": 1, "outcome": "sai",
            "ly_do_khong_chay": None}


class TestGomUngVien:
    def test_gom_theo_gia_tri_khong_theo_van_ban(self):
        """Hai program khác chữ mà cùng giá trị là CÙNG một lựa chọn."""
        u = pipeline.nhom_ung_vien(_row(["add(1,2)", "add(2,1)", "add(1,3)"],
                                        [3.0, 3.0, 4.0]))
        assert len(u) == 2
        assert u[0]["so_mau"] == 2 and u[1]["so_mau"] == 1

    def test_bo_mau_khong_chay_duoc_va_mau_rong(self):
        u = pipeline.nhom_ung_vien(_row(["add(1,2)", "hong", ""], [3.0, None, None]))
        assert len(u) == 1 and u[0]["program"] == "add(1,2)"

    def test_thu_tu_theo_lan_xuat_hien_DAU_khong_theo_so_phieu(self):
        """Xếp theo số phiếu là rò rỉ tín hiệu số đông vào chỗ đứng trong prompt —
        đúng thứ tín hiệu mà bỏ phiếu đã dùng và đã sai ở 58 % nhóm này."""
        u = pipeline.nhom_ung_vien(_row(["a()", "b()", "b()", "b()"],
                                        [1.0, 2.0, 2.0, 2.0]))
        assert [x["gia_tri"] for x in u] == [1.0, 2.0]

    def test_khong_co_mau_nao_chay_duoc(self):
        assert pipeline.nhom_ung_vien(_row(["x", "y"], [None, None])) == []


class TestDocLuaChon:
    @pytest.mark.parametrize("raw,mong", [("chon: 2", 1), ("chon:1", 0),
                                          ("CHON = 3", 2), ("  chon : 2  ", 1)])
    def test_doc_duoc_cac_dang_viet(self, raw, mong):
        assert pipeline.doc_lua_chon(raw, 3) == mong

    def test_lay_lan_khop_CUOI(self):
        """Phần suy nghĩ hay nhắc lại đề bài có thể chứa 'chon: 1' — chốt là cái cuối."""
        assert pipeline.doc_lua_chon("cân nhắc chon: 1 ... kết luận chon: 3", 3) == 2

    @pytest.mark.parametrize("raw", ["chon: 0", "chon: 9", "không chọn gì", "", None])
    def test_ngoai_pham_vi_hoac_khong_doc_duoc_tra_None(self, raw):
        assert pipeline.doc_lua_chon(raw, 3) is None


class TestPromptChon:
    def test_can_it_nhat_2_ung_vien(self):
        with pytest.raises(ValueError):
            PromptKit().step_chon(_mau(), [{"program": "add(1,2)", "gia_tri": 3.0,
                                            "so_mau": 5}])

    def test_co_du_moi_ung_vien_va_gia_tri_cua_no(self):
        uv = [{"program": "add(1,2)", "gia_tri": 3.0, "so_mau": 4},
              {"program": "add(1,3)", "gia_tri": 4.0, "so_mau": 1}]
        t = PromptKit().step_chon(_mau(), uv)
        for u in uv:
            assert u["program"] in t and str(u["gia_tri"]) in t
        assert "[1]" in t and "[2]" in t

    def test_KHONG_lo_so_phieu(self):
        """Đưa số phiếu vào là mời model neo theo số đông — đúng thứ đang sai."""
        uv = [{"program": "add(1,2)", "gia_tri": 3.0, "so_mau": 4},
              {"program": "add(1,3)", "gia_tri": 4.0, "so_mau": 1}]
        t = PromptKit().step_chon(_mau(), uv)
        assert "so_mau" not in t
        assert "4 mẫu" not in t and "số phiếu" not in t.lower()


class TestVongChonFailClosed:
    """Phép đo chỉ được tốt lên hoặc đứng yên vì lỗi kỹ thuật, không được tụt."""

    def _chay(self, tra_loi):
        rows = [_row(["add(1,2)", "add(1,3)"], [3.0, 4.0], final="add(1,3)")]
        rows[0]["pred_value"] = 4.0
        return pipeline.chon_bang_model(
            rows, [_mau()], PromptKit(), lambda ps, sp=None, desc=None: tra_loi,
            desc="thu")

    def test_chon_dung_thi_cham_lai_diem(self):
        ra = self._chay(["chon: 1"])
        assert ra[0]["da_chon"] and ra[0]["final_program"] == "add(1,2)"
        assert ra[0]["ea"] and ra[0]["pa_strict"]

    @pytest.mark.parametrize("tl", [["chon: 7"], ["không rõ"], [""]])
    def test_khong_doc_duoc_thi_GIU_NGUYEN_bo_phieu(self, tl):
        ra = self._chay(tl)
        assert ra[0]["da_chon"] is False
        assert ra[0]["final_program"] == "add(1,3)" and ra[0]["ea"] is False

    def test_cau_chi_co_1_ung_vien_khong_sinh_prompt_nao(self):
        goi = []

        def gen(ps, sp=None, desc=None):
            goi.append(len(ps))
            return ["chon: 1"] * len(ps)

        rows = [_row(["add(1,2)", "add(2,1)"], [3.0, 3.0])]
        ra = pipeline.chon_bang_model(rows, [_mau()], PromptKit(), gen, desc="thu")
        assert goi == [] and ra[0]["da_chon"] is False

    def test_so_luong_va_thu_tu_hang_duoc_giu_nguyen(self):
        rows = [_row(["a()", "b()"], [1.0, 2.0]), _row(["c()"], [5.0]),
                _row(["d()", "e()"], [7.0, 8.0])]
        ra = pipeline.chon_bang_model(
            rows, [_mau(), _mau(), _mau()], PromptKit(),
            lambda ps, sp=None, desc=None: ["chon: 1"] * len(ps), desc="thu")
        assert len(ra) == 3
        assert [r["id"] for r in ra] == [r["id"] for r in rows]
        assert [r["n_ung_vien"] for r in ra] == [2, 1, 2]
