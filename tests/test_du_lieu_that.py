# -*- coding: utf-8 -*-
"""Kiểm mã nguồn trên DỮ LIỆU THẬT, không phải fixture tự bịa.

Vì sao cần riêng file này: mọi test khác dùng mẫu tự dựng, nên một khiếm khuyết
HÌNH DẠNG DỮ LIỆU chỉ tồn tại trong một tập cụ thể là vô hình. Đúng chuyện đã xảy ra:
tập ``train`` có 5 nhãn vàng cụt dấu ngoặc (``divide(#0, 1682`` — thiếu ``)``), ``valid``
và ``test`` thì sạch. Vì notebook 01/02/08 chỉ chạy trên ``test``, còn 03 là notebook
DUY NHẤT chấm điểm trên ``train``, lỗi chỉ nổ ở 03 — sau 43 phút sinh trên A100.

Nguyên tắc rút ra: hàm PHÂN LOẠI để báo cáo phải fail-closed. Nhãn vàng hỏng được phép
làm sai một con số, không được phép giết cả lượt chạy.
"""
import os

import pytest

from vinumqa import data, dsl, pipeline

GOC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def tap():
    return data.load_all(os.path.join(GOC, "data"))


TEN_TAP = ("train", "valid", "test")


def _gold(tap, ten):
    return [(s.get("id", ""), (s.get("qa", {}).get("program") or "")) for s in tap[ten]]


@pytest.mark.parametrize("ten", TEN_TAP)
def test_nhom_phep_khong_bao_gio_nem_loi(tap, ten):
    """Hồi quy trực tiếp cho lỗi đã nổ ở notebook 03."""
    for sid, prog in _gold(tap, ten):
        assert isinstance(pipeline.nhom_phep(prog), str), sid


@pytest.mark.parametrize("ten", TEN_TAP)
def test_executor_va_chuan_hoa_khong_bao_gio_nem_loi(tap, ten):
    """``execute_program`` trả None, ``normalize_program_strict`` trả '' — không ném."""
    for s in tap[ten]:
        prog = s.get("qa", {}).get("program") or ""
        dsl.execute_program(prog, s.get("table") or [])
        assert isinstance(dsl.normalize_program_strict(prog), str), s.get("id")


def test_gold_cut_ngoac_van_con_dung_5_mau_va_deu_bi_loc(tap):
    """Chốt con số. Nếu dữ liệu đổi, test này phải đổi theo một cách CÓ Ý THỨC.

    Quan trọng hơn con số: cả 5 mẫu phải bị ``is_noisy_gold`` bắt, vì đó là thứ ngăn
    chúng lọt vào dữ liệu huấn luyện SFT và vào kho ví dụ few-shot. Vá dấu ngoặc thì
    chương trình chạy ra 1.347 trong khi ``exe_ans`` là 2266 — nhãn tự mâu thuẫn,
    không phải chỉ cụt chữ, nên loại bỏ mới đúng chứ không phải sửa.
    """
    hong = [s for s in tap["train"]
            if _nem_loi(s.get("qa", {}).get("program") or "")]
    assert len(hong) == 5, [s.get("id") for s in hong]
    for s in hong:
        assert pipeline.nhom_phep(s["qa"]["program"]) == "?"
        assert data.is_noisy_gold(s), f"{s.get('id')} lọt qua bộ lọc nhãn nhiễu"

    for ten in ("valid", "test"):
        assert not [s for s in tap[ten]
                    if _nem_loi(s.get("qa", {}).get("program") or "")]


def _nem_loi(prog: str) -> bool:
    try:
        dsl.split_dsl_items(prog.strip())
        return False
    except ValueError:
        return True


def test_summarize_chay_duoc_tren_ca_tap_train(tap):
    """Đúng đường mà notebook 03 đi: chấm điểm trên train, không phải test."""
    rows = [{
        "id": s.get("id", ""), "ea": False, "ea_tol1e-3": False,
        "pa_strict": False, "pa_loose": False,
        "gold_program": s.get("qa", {}).get("program") or "",
        "final_program": "divide(1,2)", "program_step2": "", "lay_buoc2": False,
        "pred_value": 0.5, "n_ops_gold": dsl.n_ops(s.get("qa", {}).get("program") or ""),
        "outcome": "sai", "ly_do_khong_chay": None,
    } for s in tap["train"]]
    m = pipeline.summarize(rows, "train")
    assert m["n"] == len(tap["train"])
    assert "?" in m["by_phep"], "5 nhãn vàng hỏng phải rơi vào nhóm '?'"
    assert pipeline.phan_loai_sai(rows) is not None
