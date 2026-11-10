# -*- coding: utf-8 -*-
"""
通用 API 路由测试：mock 数据库层，批量打各蓝图 GET 端点
覆盖 announcement / scale / article / relaxation / mood / appointment /
tree_hole / notification / consultation / admin / insight 等
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ---------- 宽松字典：任何 key 访问都返回安全默认值，避免 KeyError ----------
class LazyDict(dict):
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return 0


def fake_execute_query(sql, params=None, fetch_one=False):
    sql_low = sql.lower()
    if 'count(*)' in sql_low or 'count(1)' in sql_low:
        return {'count': 3}
    if 'distinct' in sql_low:
        return []
    if fetch_one:
        return LazyDict({'id': 1, 'title': 'x', 'name': 'x', 'content': 'x',
                         'username': 'student1', 'real_name': '学生',
                         'role': 'student', 'status': 1, 'is_read': 0,
                         'like_count': 0, 'favorite_count': 0, 'view_count': 0,
                         'mood_level': 4, 'category': 'general',
                         'publisher_name': '管理员', 'author_name': '咨询师',
                         'user_name': '学生', 'created_at': '2026-11-01'})
    # 列表查询默认空列表（最安全，不进循环）
    return []


@pytest.fixture
def client_with_db(app, monkeypatch):
    import routes.announcement
    import routes.scale
    import routes.article
    import routes.relaxation
    import routes.mood
    import routes.appointment
    import routes.tree_hole
    import routes.notification
    import routes.auth as routes_auth
    import routes.consultation
    import routes.admin
    import routes.insight
    import flask, functools
    MOCK_USER = {'id': 1, 'username': 'student1', 'real_name': '学生',
                 'role': 'student', 'status': 1}
    def fake_login_required(roles=None):
        def deco(f):
            @functools.wraps(f)
            def w(*a, **kw):
                flask.request.current_user = MOCK_USER
                return f(*a, **kw)
            return w
        return deco
    for mod in [routes.announcement, routes.scale, routes.article,
                routes.relaxation, routes.mood, routes.appointment,
                routes.tree_hole, routes.notification, routes_auth,
                routes.consultation, routes.admin, routes.insight]:
        if hasattr(mod, 'execute_query'):
            monkeypatch.setattr(mod, 'execute_query', fake_execute_query)
        if hasattr(mod, 'execute_update'):
            monkeypatch.setattr(mod, 'execute_update', lambda sql, params=None: 1)
        if hasattr(mod, 'login_required'):
            monkeypatch.setattr(mod, 'login_required', fake_login_required)
    return app.test_client()


# ---------- 公告 ----------
class TestAnnouncement:
    def test_list(self, client_with_db):
        r = client_with_db.get('/api/announcements')
        assert r.status_code == 200

    def test_detail(self, client_with_db):
        r = client_with_db.get('/api/announcements/1')
        assert r.status_code == 200

    def test_detail_not_found(self, client_with_db, monkeypatch):
        import routes.announcement as ra
        monkeypatch.setattr(ra, 'execute_query', lambda sql, params=None, fetch_one=False: None)
        r = client_with_db.get('/api/announcements/999')
        assert r.status_code == 400


# ---------- 文章 ----------
class TestArticle:
    def test_list(self, client_with_db):
        r = client_with_db.get('/api/articles')
        assert r.status_code == 200

    def test_detail(self, client_with_db):
        r = client_with_db.get('/api/articles/1')
        assert r.status_code == 200


# ---------- 放松音频 ----------
class TestRelaxation:
    def test_list(self, client_with_db):
        r = client_with_db.get('/api/relaxation/audios')
        assert r.status_code == 200


# ---------- 心情日记 ----------
class TestMood:
    def test_list(self, client_with_db):
        r = client_with_db.get('/api/mood')
        assert r.status_code == 200

    def test_labels(self, client_with_db):
        r = client_with_db.get('/api/mood/labels')
        assert r.status_code == 200


# ---------- 预约 ----------
class TestAppointment:
    def test_list(self, client_with_db):
        r = client_with_db.get('/api/appointments')
        assert r.status_code == 200


# ---------- 树洞 ----------
class TestTreeHole:
    def test_list(self, client_with_db):
        r = client_with_db.get('/api/tree-hole')
        assert r.status_code == 200


# ---------- 通知 ----------
class TestNotification:
    def test_list(self, client_with_db):
        r = client_with_db.get('/api/notifications')
        assert r.status_code == 200


# ---------- 咨询会话 ----------
class TestConsultation:
    def test_list(self, client_with_db):
        r = client_with_db.get('/api/consultations')
        assert r.status_code == 200


# ---------- 数据分析 ----------
class TestInsight:
    def test_insight(self, client_with_db):
        r = client_with_db.get('/api/insight/risk')
        assert r.status_code == 200


# ---------- 健康检查 ----------
class TestHealth:
    def test_health_ok(self, client_with_db, monkeypatch):
        class FakeConn:
            def cursor(self):
                class C:
                    def __enter__(self): return self
                    def __exit__(self, *a): pass
                    def execute(self, *a): pass
                    def fetchone(self): return {'1': 1}
                return C()
            def close(self): pass
        monkeypatch.setattr('utils.db.get_db_connection', lambda: FakeConn())
        r = client_with_db.get('/api/health')
        assert r.status_code == 200
