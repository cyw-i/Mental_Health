# -*- coding: utf-8 -*-
"""
routes/scale.py 的 evaluate_score 纯函数测试
覆盖 PHQ-9 / GAD-7 / PSS 三档边界
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from routes.scale import evaluate_score, SCALE_RULES


class TestPHQ9:
    @pytest.mark.parametrize("score,level", [
        (0, '无抑郁'), (4, '无抑郁'),
        (5, '轻度抑郁'), (9, '轻度抑郁'),
        (10, '中度抑郁'), (14, '中度抑郁'),
        (15, '中重度抑郁'), (19, '中重度抑郁'),
        (20, '重度抑郁'), (27, '重度抑郁'),
    ])
    def test_levels(self, score, level):
        lv, _, _, _ = evaluate_score(1, score)
        assert lv == level

    def test_returns_4_tuple(self):
        result = evaluate_score(1, 10)
        assert len(result) == 4
        level, desc, analysis, advice = result
        assert desc and analysis and advice


class TestGAD7:
    @pytest.mark.parametrize("score,level", [
        (0, '无焦虑'), (4, '无焦虑'),
        (5, '轻度焦虑'), (9, '轻度焦虑'),
        (10, '中度焦虑'), (14, '中度焦虑'),
        (15, '重度焦虑'), (21, '重度焦虑'),
    ])
    def test_levels(self, score, level):
        lv, _, _, _ = evaluate_score(2, score)
        assert lv == level


class TestPSS:
    @pytest.mark.parametrize("score,level", [
        (0, '压力较低'), (4, '压力较低'),
        (5, '压力适中'), (9, '压力适中'),
        (10, '压力较高'), (12, '压力较高'),
        (13, '压力过高'), (15, '压力过高'),
    ])
    def test_levels(self, score, level):
        lv, _, _, _ = evaluate_score(3, score)
        assert lv == level


class TestUnknown:
    def test_unknown_scale(self):
        lv, desc, analysis, advice = evaluate_score(99, 10)
        assert lv == '未知'

    def test_out_of_range(self):
        # PHQ-9 满分 27，传 30 不在任何区间
        lv, _, _, _ = evaluate_score(1, 30)
        assert lv == '未知'
