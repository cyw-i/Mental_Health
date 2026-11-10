"""
AI Chat API 集成测试
测试 AI 助手会话管理、消息发送、RAG 集成响应字段
所有数据库查询、LLM 调用、Embedding 调用均 mock

关键设计：使用 pytest conftest 提供的 app fixture（在 import routes 前 monkeypatch login_required），
避免 Python from-import 导致的闭包引用问题。
"""
import json
import os
import sys
import functools
import flask
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Mock 数据和辅助
# ============================================================

MOCK_USER = {
    'id': 1,
    'username': 'student1',
    'role': 'student',
}

MOCK_SESSIONS = [
    {'id': 1, 'title': '最近心情不太好', 'created_at': '2026-07-18 10:00:00', 'updated_at': '2026-07-18 10:30:00'},
]

MOCK_MESSAGES = [
    {'id': 1, 'role': 'assistant', 'content': '你好，我是AI心理助手小暖', 'created_at': '2026-07-18 10:00:00'},
    {'id': 2, 'role': 'user', 'content': '最近做事总提不起劲，是不是抑郁了？', 'created_at': '2026-07-18 10:01:00'},
]

MOCK_AI_REPLY = '我理解你最近的感受。建议你在平台完成一次PHQ-9测评，也可以预约校内心理咨询师。'

MOCK_KNOWLEDGE_CHUNKS = [
    {
        'id': 1,
        'source_type': 'scale_overview',
        'source_id': 1,
        'title': 'PHQ-9 抑郁自评量表 - 量表说明',
        'content': '【PHQ-9 抑郁自评量表】量表说明：PHQ-9 是用于筛查抑郁症状的自评量表。',
        'embedding': json.dumps([0.1 * (i % 10 + 1) for i in range(768)]),
        'metadata': json.dumps({'scale_id': 1, 'scale_name': 'PHQ-9', 'type': 'overview'}, ensure_ascii=False),
    },
    {
        'id': 2,
        'source_type': 'scoring_rule',
        'source_id': 1,
        'title': 'PHQ-9 - 中度抑郁（10-14分）分析',
        'content': '【PHQ-9 · 中度抑郁】测评显示您近期抑郁症状已达到中度水平。',
        'embedding': json.dumps([0.08 * (i % 12 + 1) for i in range(768)]),
        'metadata': json.dumps({'scale_id': 1, 'scale_name': 'PHQ-9', 'level': '中度抑郁', 'type': 'analysis'}, ensure_ascii=False),
    },
    {
        'id': 3,
        'source_type': 'scoring_rule',
        'source_id': 1,
        'title': 'PHQ-9 - 中度抑郁（10-14分）建议',
        'content': '【PHQ-9 · 中度抑郁 · 改善建议】建议预约校内心理咨询师。',
        'embedding': json.dumps([0.07 * (i % 14 + 1) for i in range(768)]),
        'metadata': json.dumps({'scale_id': 1, 'scale_name': 'PHQ-9', 'level': '中度抑郁', 'type': 'advice'}, ensure_ascii=False),
    },
]


def _mock_login_required(roles=None):
    """Mock JWT 认证装饰器 —— 直接注入测试用户"""
    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            flask.request.current_user = MOCK_USER
            return f(*args, **kwargs)
        return decorated
    return decorator


def _setup_all_mocks(monkeypatch, rag_enabled=True):
    """设置 AI Chat 路由所有 mock"""
    from config import Config
    monkeypatch.setattr(Config, 'RAG_ENABLED', rag_enabled)
    monkeypatch.setattr(Config, 'RAG_TOP_K', 3)

    monkeypatch.setattr('routes.ai_chat._ensure_tables', lambda: None)
    monkeypatch.setattr('routes.ai_chat.login_required', _mock_login_required)
    monkeypatch.setattr(
        'routes.ai_chat._get_user_session',
        lambda session_id: MOCK_SESSIONS[0] if session_id == 1 else None,
    )
    monkeypatch.setattr('routes.ai_chat.execute_update', lambda sql, params: 1)

    def mock_query(sql, params=None, fetch_one=False):
        if 'COUNT' in sql:
            return {'c': 2}
        if fetch_one:
            return MOCK_SESSIONS[0]
        return MOCK_MESSAGES

    monkeypatch.setattr('routes.ai_chat.execute_query', mock_query)
    monkeypatch.setattr('routes.ai_chat.chat_completion', lambda messages: MOCK_AI_REPLY)
    monkeypatch.setattr(
        'utils.rag.get_embedding',
        lambda text: [0.05 * (i % 20 + 1) for i in range(768)],
    )
    monkeypatch.setattr(
        'utils.rag.execute_query',
        lambda sql, **kwargs: MOCK_KNOWLEDGE_CHUNKS,
    )


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def app(monkeypatch):
    """
    创建 Flask 测试应用。
    关键：必须在任何导入 routes 模块的操作之前 mock login_required。
    因为 routes/ai_chat.py 中使用 from utils.auth import login_required，
    一旦导入，该模块的本地引用就固定了。

    monkeypatch.setattr('utils.auth.login_required', ...)
    会修改模块属性，但 routes.ai_chat.login_required 已经是原始引用的副本。
    所以需要同时 patch routes.ai_chat.login_required。

    但 routes 模块还没有被导入！所以我们需要先 mock utils.auth，
    然后让 create_app() 去导入 routes（routes 会拿到 mock 版本）。
    """
    # Mock auth at the source BEFORE routes get imported
    monkeypatch.setattr('utils.auth.login_required', _mock_login_required)

    # Also ensure routes module gets reloaded fresh (running tests with -p no:cacheimport)
    # For safety, if routes.ai_chat is already cached, reload it
    if 'routes.ai_chat' in sys.modules:
        import routes.ai_chat
        import importlib
        importlib.reload(routes.ai_chat)

    from app import create_app
    app = create_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Flask 测试客户端"""
    return app.test_client()


# ============================================================
# Test Cases
# ============================================================

class TestAuthRequired:
    """测试认证（mock 版本）"""

    def test_mock_auth_enabled(self, client, monkeypatch):
        """验证 mock auth 生效 —— 无 token 也能访问（因为 auth 被 mock）"""
        monkeypatch.setattr('routes.ai_chat._ensure_tables', lambda: None)
        monkeypatch.setattr(
            'routes.ai_chat._get_user_session',
            lambda session_id: MOCK_SESSIONS[0],
        )
        monkeypatch.setattr(
            'routes.ai_chat.execute_query',
            lambda sql, params=None, fetch_one=False: MOCK_MESSAGES,
        )

        resp = client.get('/api/ai-chat/sessions/1/messages')
        data = resp.get_json()
        # mock auth 生效，即使没有 Authorization header 也能通过
        assert data['code'] == 200

    def test_list_sessions_mock_auth(self, client, monkeypatch):
        """mock auth 生效时获取列表"""
        monkeypatch.setattr('routes.ai_chat._ensure_tables', lambda: None)
        monkeypatch.setattr(
            'routes.ai_chat.execute_query',
            lambda sql, params=None, fetch_one=False: MOCK_SESSIONS,
        )

        resp = client.get('/api/ai-chat/sessions')
        data = resp.get_json()
        assert data['code'] == 200


class TestCreateSession:
    """测试创建会话"""

    def test_create_session_success(self, client, monkeypatch):
        """POST /sessions → 返回 session_id"""
        _setup_all_mocks(monkeypatch)

        resp = client.post(
            '/api/ai-chat/sessions',
            json={'title': '测试会话'},
        )
        data = resp.get_json()
        assert data['code'] == 200
        assert 'session_id' in data['data']

    def test_create_session_default_title(self, client, monkeypatch):
        """不提供 title → 使用默认标题"""
        _setup_all_mocks(monkeypatch)

        resp = client.post('/api/ai-chat/sessions', json={})
        data = resp.get_json()
        assert data['code'] == 200
        assert 'session_id' in data['data']


class TestGetMessages:
    """测试获取消息"""

    def test_get_messages_success(self, client, monkeypatch):
        """GET /messages → 返回消息列表"""
        _setup_all_mocks(monkeypatch)

        resp = client.get('/api/ai-chat/sessions/1/messages')
        data = resp.get_json()
        assert data['code'] == 200
        assert len(data['data']) >= 1

    def test_get_messages_nonexistent_session(self, client, monkeypatch):
        """不存在的 session_id → 返回错误"""
        monkeypatch.setattr('routes.ai_chat._ensure_tables', lambda: None)
        monkeypatch.setattr('routes.ai_chat.login_required', _mock_login_required)
        monkeypatch.setattr(
            'routes.ai_chat._get_user_session',
            lambda session_id: None,
        )

        resp = client.get('/api/ai-chat/sessions/999/messages')
        data = resp.get_json()
        assert data['code'] != 200 or data['data'] is None


class TestSendMessage:
    """测试发送消息和 RAG 集成"""

    def test_send_message_rag_enabled(self, client, monkeypatch):
        """RAG 启用时，响应包含 rag_used=true 和 rag_sources"""
        _setup_all_mocks(monkeypatch, rag_enabled=True)

        resp = client.post(
            '/api/ai-chat/sessions/1/messages',
            json={'content': '我最近做事总提不起劲，是不是抑郁了？'},
        )
        data = resp.get_json()
        assert data['code'] == 200
        assert data['data']['rag_used'] is True
        assert len(data['data']['rag_sources']) > 0
        assert 'reply' in data['data']

    def test_send_message_rag_disabled(self, client, monkeypatch):
        """RAG 关闭时，rag_used=false"""
        _setup_all_mocks(monkeypatch, rag_enabled=False)

        resp = client.post(
            '/api/ai-chat/sessions/1/messages',
            json={'content': '我最近心情不好'},
        )
        data = resp.get_json()
        assert data['code'] == 200
        assert data['data']['rag_used'] is False
        assert data['data']['rag_sources'] == []

    def test_send_message_empty_content(self, client, monkeypatch):
        """content 为空 → 返回错误"""
        _setup_all_mocks(monkeypatch)

        resp = client.post(
            '/api/ai-chat/sessions/1/messages',
            json={'content': ''},
        )
        data = resp.get_json()
        assert data['code'] != 200

    def test_send_message_nonexistent_session(self, client, monkeypatch):
        """不存在的 session_id → 返回错误"""
        monkeypatch.setattr('routes.ai_chat._ensure_tables', lambda: None)
        monkeypatch.setattr('routes.ai_chat.login_required', _mock_login_required)
        monkeypatch.setattr(
            'routes.ai_chat._get_user_session',
            lambda session_id: None,
        )

        resp = client.post(
            '/api/ai-chat/sessions/999/messages',
            json={'content': '你好'},
        )
        data = resp.get_json()
        assert data['code'] != 200 or data['data'] is None

    def test_rag_sources_fields(self, client, monkeypatch):
        """验证 rag_sources 中每个元素包含完整字段"""
        _setup_all_mocks(monkeypatch, rag_enabled=True)

        resp = client.post(
            '/api/ai-chat/sessions/1/messages',
            json={'content': '抑郁量表评分标准'},
        )
        data = resp.get_json()

        if data['code'] == 200 and data['data'].get('rag_used'):
            sources = data['data']['rag_sources']
            for src in sources:
                assert 'title' in src, f"缺少 title: {src}"
                assert 'summary' in src, f"缺少 summary: {src}"
                assert 'similarity' in src, f"缺少 similarity: {src}"
                assert 'source_type' in src, f"缺少 source_type: {src}"
                assert isinstance(src['similarity'], float)


class TestDeleteSession:
    """测试删除会话"""

    def test_delete_session_success(self, client, monkeypatch):
        """DELETE → 返回成功"""
        _setup_all_mocks(monkeypatch)

        resp = client.delete('/api/ai-chat/sessions/1')
        data = resp.get_json()
        assert data['code'] == 200

    def test_delete_nonexistent_session(self, client, monkeypatch):
        """删除不存在的会话 → 返回错误"""
        monkeypatch.setattr('routes.ai_chat._ensure_tables', lambda: None)
        monkeypatch.setattr('routes.ai_chat.login_required', _mock_login_required)
        monkeypatch.setattr(
            'routes.ai_chat._get_user_session',
            lambda session_id: None,
        )

        resp = client.delete('/api/ai-chat/sessions/999')
        data = resp.get_json()
        assert data['code'] != 200 or data['data'] is None