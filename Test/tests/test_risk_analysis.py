# -*- coding: utf-8 -*-
"""
risk_analysis.py 纯函数单元测试
覆盖：归一化、风险分级、线性趋势、综合洞察、建议生成
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from utils.risk_analysis import (
    _normalize_score, _risk_level, _linear_slope, _trend_label,
    _predict_next, compute_risk_insight, _build_suggestions, _fmt_date,
    SCALE_META, MOOD_WEIGHT,
)


class TestNormalizeScore:
    def test_normal(self):
        assert _normalize_score(10, 20) == 50

    def test_zero_max(self):
        assert _normalize_score(10, 0) == 0

    def test_clamp_high(self):
        assert _normalize_score(30, 20) == 100

    def test_clamp_low(self):
        assert _normalize_score(-5, 20) == 0

    def test_round(self):
        assert _normalize_score(1, 3) == 33


class TestRiskLevel:
    @pytest.mark.parametrize("score,level", [
        (0, '低风险'), (29, '低风险'),
        (30, '中等风险'), (54, '中等风险'),
        (55, '较高风险'), (74, '较高风险'),
        (75, '高风险'), (100, '高风险'),
    ])
    def test_boundaries(self, score, level):
        lv, _, _ = _risk_level(score)
        assert lv == level

    def test_color(self):
        assert _risk_level(10)[2] == '#52c41a'
        assert _risk_level(90)[2] == '#f5222d'


class TestLinearSlope:
    def test_flat(self):
        assert _linear_slope([(0, 50), (1, 50), (2, 50)]) == 0

    def test_up(self):
        s = _linear_slope([(0, 10), (1, 20), (2, 30)])
        assert s > 9

    def test_down(self):
        s = _linear_slope([(0, 30), (1, 20), (2, 10)])
        assert s < -9

    def test_single_point(self):
        assert _linear_slope([(0, 50)]) == 0

    def test_vertical_x(self):
        # xs 全相同 → den=0
        assert _linear_slope([(5, 10), (5, 20), (5, 30)]) == 0


class TestTrendLabel:
    def test_improving(self):
        assert _trend_label(-1.0) == ('改善中', 'down')

    def test_watch(self):
        assert _trend_label(1.0) == ('需关注', 'up')

    def test_stable(self):
        assert _trend_label(0.0) == ('较稳定', 'stable')

    def test_threshold(self):
        assert _trend_label(0.3) == ('需关注', 'up')
        assert _trend_label(-0.3) == ('改善中', 'down')


class TestPredictNext:
    def test_normal(self):
        assert _predict_next(50, 5) == 55.0

    def test_floor_zero(self):
        assert _predict_next(2, -10) == 0


class TestComputeRiskInsight:
    def test_no_data(self):
        r = compute_risk_insight([], [])
        assert r['has_data'] is False
        assert r['risk_score'] is None
        assert 'suggestions' in r

    def test_only_mood(self):
        moods = [{'mood_level': 5, 'created_at': '2026-11-01'}]
        r = compute_risk_insight([], moods)
        assert r['has_data'] is True
        # mood 5 → risk 0
        assert r['risk_score'] == 0
        assert r['risk_level'] == '低风险'

    def test_only_scale(self):
        records = [{'scale_id': 1, 'total_score': 20, 'created_at': '2026-11-01'}]
        r = compute_risk_insight(records, [])
        assert r['has_data'] is True
        # PHQ-9 20/27 = 74%
        assert r['factors'][0]['risk_percent'] == 74

    def test_full_data_with_trend(self):
        records = [
            {'scale_id': 1, 'total_score': 5, 'created_at': '2026-10-01'},
            {'scale_id': 1, 'total_score': 10, 'created_at': '2026-10-15'},
            {'scale_id': 1, 'total_score': 15, 'created_at': '2026-11-01'},
        ]
        moods = [{'mood_level': 3, 'created_at': f'2026-10-{d:02d}'} for d in range(1, 8)]
        r = compute_risk_insight(records, moods)
        assert r['has_data'] is True
        assert r['risk_score'] > 0
        # 量表连续上升 → up
        scale_factor = [f for f in r['factors'] if f['key'] == 'scale_1'][0]
        assert 'trend' in scale_factor
        assert 'predictions' in r

    def test_unknown_scale_ignored(self):
        records = [{'scale_id': 99, 'total_score': 10, 'created_at': '2026-11-01'}]
        r = compute_risk_insight(records, [])
        assert r['has_data'] is False

    def test_mood_empty_levels(self):
        moods = [{'mood_level': None, 'created_at': '2026-11-01'}]
        r = compute_risk_insight([], moods)
        # 无有效 mood_level → 仍无数据
        assert r['has_data'] is False

    def test_overall_trend_declining(self):
        # 两个量表都上升
        records = [
            {'scale_id': 1, 'total_score': 5, 'created_at': '2026-10-01'},
            {'scale_id': 1, 'total_score': 15, 'created_at': '2026-11-01'},
            {'scale_id': 2, 'total_score': 3, 'created_at': '2026-10-01'},
            {'scale_id': 2, 'total_score': 12, 'created_at': '2026-11-01'},
        ]
        r = compute_risk_insight(records, [])
        assert r['overall_trend'] == 'declining'


class TestBuildSuggestions:
    def test_high_risk(self):
        tips = _build_suggestions(80, [{'key': 'scale_1', 'name': 'PHQ-9抑郁', 'risk_percent': 80}], 'declining')
        assert any('预约心理咨询' in t for t in tips)
        assert any('下滑' in t for t in tips)

    def test_low_risk(self):
        tips = _build_suggestions(10, [{'key': 'scale_1', 'risk_percent': 10}], 'stable')
        assert any('保持' in t for t in tips)

    def test_max_four(self):
        tips = _build_suggestions(80,
            [{'key': 'scale_1', 'name': 'PHQ-9', 'risk_percent': 90},
             {'key': 'scale_2', 'name': 'GAD-7', 'risk_percent': 80}],
            'declining')
        assert len(tips) <= 4


class TestFmtDate:
    def test_none(self):
        assert _fmt_date(None) == ''

    def test_datetime(self):
        import datetime
        dt = datetime.datetime(2026, 11, 10)
        assert _fmt_date(dt) == '2026-11-10'

    def test_string(self):
        assert _fmt_date('2026-11-10T10:00:00') == '2026-11-10'
