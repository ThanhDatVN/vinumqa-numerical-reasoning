# -*- coding: utf-8 -*-
"""ViNumQA — suy luận số học trên báo cáo tài chính tiếng Việt với Qwen3-8B.

Lõi dùng chung của lộ trình thí nghiệm 5 nấc trong ``notebooks/``:

===  ==========================  ==============================================
Nấc  Kỹ thuật                    Notebook
===  ==========================  ==============================================
1    Prompt cơ bản               ``01_baseline_basic.ipynb``
2    Prompt hoàn chỉnh           ``02_prompt_engineering.ipynb``
3    SFT trên Qwen3-8B           ``03_sft_qwen3.ipynb``
4    Self-evaluation 2 bước      ``04_self_evaluation.ipynb``
5    ACE (playbook + truy hồi)   ``05_ace.ipynb``
\-    Ma trận tổ hợp 2×2×2        ``06_combination.ipynb``
===  ==========================  ==============================================

Bố cục
------
========================  =====================================================
``dsl``                   Executor DSL + chấm PA/EA (đã đối chiếu nhãn vàng)
``data``                  Nạp ViNumQA, tách nguồn, audit nhiễu nhãn
``prompts``               Thang prompt lồng nhau: basic ⊂ no_fewshot ⊂ engineered
``pipeline``              ``run_pipeline`` dùng chung cho mọi nấc
``sft``                   Dựng dữ liệu SFT (rejection sampling) + cấu hình LoRA
``stats``                 McNemar theo cặp + bootstrap CI
``io_utils``              Ghi artifact, chấm lại dự đoán đã lưu
``ace``                   Playbook, truy hồi, Reflector, vòng lặp ACE
========================  =====================================================

Phần cần GPU nhận ``generate_fn`` **tiêm từ ngoài vào**, nên toàn bộ package test
được trên CPU mà không cần model.
"""

__version__ = "2.0.0"

from . import data, dsl, io_utils, pipeline, prompts, sft, stats

__all__ = ["dsl", "data", "prompts", "pipeline", "sft", "stats", "io_utils",
           "ace", "__version__"]
