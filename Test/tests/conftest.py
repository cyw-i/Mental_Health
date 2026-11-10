"""
pytest 共享 fixtures
为 RAG 相关测试提供 mock 环境：Flask app、数据库、LLM、Embedding API

重要：为避免 Python from-import 闭包引用问题，conftest.py 不直接 import app。
app fixture 在函数体内延迟 import，并在 import 前先 monkeypatch utils.auth。
"""
import json
import sys
import os
import functools
import flask
import pytest

# 确保 server/ 目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 模拟数据
# ============================================================

MOCK_EMBEDDING_768 = [0.1 * (i % 10 + 1) for i in range(768)]  # 768 维向量 (nomic-embed-text)

MOCK_EMBEDDING_QUERY = [0.05 * (i % 20 + 1) for i in range(768)]  # 查询向量

MOCK_KNOWLEDGE_CHUNKS = [
    {
        'id': 1,
        'source_type': 'scale_overview',
        'source_id': 1,
        'title': 'PHQ-9 抑郁自评量表 - 量表说明',
        'content': '【PHQ-9 抑郁自评量表】\n量表说明：PHQ-9 是用于筛查抑郁症状严重程度的自评量表，共9道题，请根据近两周的感受作答。\n该量表用于心理健康自评。',
        'embedding': json.dumps([0.1 * (i % 10 + 1) for i in range(768)]),
        'metadata': json.dumps({'scale_id': 1, 'scale_name': 'PHQ-9 抑郁自评量表', 'type': 'overview'}, ensure_ascii=False),
    },
    {
        'id': 2,
        'source_type': 'scale_overview',
        'source_id': 2,
        'title': 'GAD-7 焦虑自评量表 - 量表说明',
        'content': '【GAD-7 焦虑自评量表】\n量表说明：GAD-7 是用于筛查焦虑症状的自评量表，共7道题。',
        'embedding': json.dumps([0.2 * (i % 5 + 1) for i in range(768)]),
        'metadata': json.dumps({'scale_id': 2, 'scale_name': 'GAD-7 焦虑自评量表', 'type': 'overview'}, ensure_ascii=False),
    },
    {
        'id': 3,
        'source_type': 'scoring_rule',
        'source_id': 1,
        'title': 'PHQ-9 抑郁自评量表 - 中度抑郁（10-14分）分析',
        'content': '【PHQ-9 抑郁自评量表 · 中度抑郁（10-14分）】\n您可能存在中度抑郁症状，建议寻求心理咨询帮助。\n\n详细分析：测评显示您近期抑郁症状已达到中度水平，可能包括较明显的情绪低落、动力下降、注意力难以集中、睡眠或食欲改变等。',
        'embedding': json.dumps([0.08 * (i % 12 + 1) for i in range(768)]),
        'metadata': json.dumps({'scale_id': 1, 'scale_name': 'PHQ-9 抑郁自评量表', 'level': '中度抑郁', 'type': 'analysis'}, ensure_ascii=False),
    },
    {
        'id': 4,
        'source_type': 'scoring_rule',
        'source_id': 1,
        'title': 'PHQ-9 抑郁自评量表 - 中度抑郁（10-14分）建议',
        'content': '【PHQ-9 抑郁自评量表 · 中度抑郁（10-14分）· 改善建议】\n1. 建议尽快预约校内心理咨询师进行面对面评估；\n2. 向信任的家人、朋友或辅导员适度倾诉；\n3. 减少自我批评，从小目标开始恢复日常节奏。',
        'embedding': json.dumps([0.07 * (i % 14 + 1) for i in range(768)]),
        'metadata': json.dumps({'scale_id': 1, 'scale_name': 'PHQ-9 抑郁自评量表', 'level': '中度抑郁', 'type': 'advice'}, ensure_ascii=False),
    },
    {
        'id': 5,
        'source_type': 'scale_questions',
        'source_id': 3,
        'title': '压力感知量表 - 全部题目',
        'content': '【压力感知量表】共 5 道题目：\n第1题: 在过去的一个月里，你是否经常因为发生意外的事情而感到心烦？ [选项: 从不(0分) | 偶尔(1分) | 有时(2分) | 时常(3分) | 总是(4分)]\n...（共5题）',
        'embedding': json.dumps([0.03 * (i % 25 + 1) for i in range(768)]),
        'metadata': json.dumps({'scale_id': 3, 'scale_name': '压力感知量表', 'type': 'questions'}, ensure_ascii=False),
    },
]

MOCK_AI_REPLY = '我理解你最近的感受。根据PHQ-9抑郁自评量表的标准，持续的情绪低落和兴趣减退值得关注。建议你在平台完成一次PHQ-9测评，这样我们可以更准确地了解你的状态。同时，预约校内心理咨询师进行面对面评估也会很有帮助。'

MOCK_USER = {
    'id': 1,
    'username': 'student1',
    'role': 'student',
}

MOCK_SESSIONS = [
    {'id': 1, 'title': '最近心情不太好', 'created_at': '2026-07-18 10:00:00', 'updated_at': '2026-07-18 10:30:00'},
]

MOCK_MESSAGES = [
    {'id': 1, 'role': 'assistant', 'content': '你好，我是AI心理助手小暖 🌿', 'created_at': '2026-07-18 10:00:00'},
    {'id': 2, 'role': 'user', 'content': '最近做事总提不起劲，是不是抑郁了？', 'created_at': '2026-07-18 10:01:00'},
]


# ============================================================
# Mock 辅助函数
# ============================================================

def mock_embedding_response(embedding_list):
    """构造 OpenAI 兼容的 embedding API 响应"""
    return type('Response', (), {
        'status_code': 200,
        'json': lambda self: {
            'object': 'list',
            'data': [
                {'object': 'embedding', 'index': i, 'embedding': emb}
                for i, emb in enumerate(embedding_list)
            ],
            'model': 'nomic-embed-text:v1.5',
            'usage': {'prompt_tokens': 10, 'total_tokens': 10},
        },
    })()


def mock_chat_response(content):
    """构造 OpenAI 兼容的 chat completion API 响应"""
    return type('Response', (), {
        'status_code': 200,
        'json': lambda self: {
            'choices': [{'message': {'role': 'assistant', 'content': content}}],
            'model': 'qwen2.5:7b',
        },
    })()


def _mock_login_required(roles=None):
    """Mock JWT 认证装饰器 —— 直接注入测试用户"""
    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            flask.request.current_user = MOCK_USER
            return f(*args, **kwargs)
        return decorated
    return decorator


# ============================================================
# Flask 应用 fixtures
# ============================================================

@pytest.fixture
def app(monkeypatch):
    """
    创建 Flask 测试应用实例。

    关键设计：在 import create_app（进而 import routes）之前，
    先 monkeypatch utils.auth.login_required，确保 routes/ai_chat.py
    中的 from utils.auth import login_required 拿到的是 mock 版本。
    """
    # Step 1: Mock auth BEFORE importing any route modules
    monkeypatch.setattr('utils.auth.login_required', _mock_login_required)

    # Step 2: Now import create_app (which triggers route blueprints import)
    from app import create_app
    app = create_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Flask 测试客户端"""
    return app.test_client()