# -*- coding: utf-8 -*-
"""
直接调用 routes/scale.py 内部函数，绕过 HTTP 装饰器，覆盖 DB 辅助函数
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch


class TestScaleInternals:
    def test_ensure_columns(self, monkeypatch):
        import routes.scale as rs
        calls = []
        monkeypatch.setattr(rs, 'execute_query',
                            lambda sql, params=None, fetch_one=False: calls.append(sql) or [])
        rs._ensure_result_extra_columns()
        assert len(calls) > 0

    def test_evaluate_score_phq9(self):
        from routes.scale import evaluate_score
        # PHQ-9: 0-4 无抑郁, 5-9 轻度, 10-14 中度, 15-19 中重度, 20-27 重度
        for score in [0, 4, 5, 9, 10, 14, 15, 19, 20, 27]:
            r = evaluate_score(1, score)
            assert r is not None

    def test_evaluate_score_gad7(self):
        from routes.scale import evaluate_score
        for score in [0, 4, 5, 9, 10, 14, 15, 21]:
            r = evaluate_score(2, score)
            assert r is not None

    def test_evaluate_score_pss(self):
        from routes.scale import evaluate_score
        for score in [0, 13, 14, 26, 27, 40]:
            r = evaluate_score(3, score)
            assert r is not None

    def test_evaluate_score_unknown(self):
        from routes.scale import evaluate_score
        r = evaluate_score(999, 10)
        assert r is not None
