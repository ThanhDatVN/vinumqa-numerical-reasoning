# -*- coding: utf-8 -*-
"""ACE — Agentic Context Engineering cho ViNumQA (nấc 5 của lộ trình).

ACE không đổi trọng số. Nó học một **playbook** — danh sách bullet chiến lược rút ra
từ chính lỗi của model trên tập train — rồi **truy hồi top-k bullet liên quan cho từng
câu hỏi** và chèn vào prompt lúc inference.

Nguồn gốc: ``reference/original_notebooks/02_ace_finqa_ENGLISH.ipynb`` (FinQA, tiếng Anh). Tám điểm đã chỉnh cho
ViNumQA được ghi trong docstring từng module — quan trọng nhất là bỏ ``const_*``,
``table_*`` đọc theo nhãn hàng, và embedding đa ngữ.
"""
from . import clusters, playbook, reflector, trainer

__all__ = ["clusters", "playbook", "reflector", "trainer"]
