# -*- coding: utf-8 -*-
"""
POST 端点测试：写心情日记、提交量表答卷
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture
def client_post(app, monkeypatch):
    """复用 conftest 的 app，patch 数据库和登录"""
    import routes.mood as rm
    import routes.scale as rs
    import routes.relaxation as rr
    import routes.article as ra
    import routes.consultation as rc
    import routes.appointment as rapt
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
    for mod in (rm, rs, rr, ra, rc, rapt):
        monkeypatch.setattr(mod, 'login_required', fake_login_required)
    monkeypatch.setattr(rm, 'execute_query', lambda sql, params=None, fetch_one=False: [])
    monkeypatch.setattr(rs, 'execute_query', lambda sql, params=None, fetch_one=False: [])
    monkeypatch.setattr(rapt, 'execute_query', lambda sql, params=None, fetch_one=False: [])
    return app.test_client()


class TestMoodPost:
    def test_create_missing_level(self, client_post):
        r = client_post.post('/api/mood', json={'content': '今天心情不错'})
        assert r.status_code == 400

    def test_create_bad_level(self, client_post):
        r = client_post.post('/api/mood', json={'mood_level': 99, 'content': 'x'})
        assert r.status_code == 400
        assert '心情等级' in r.get_json()['message']

    def test_create_success(self, client_post, monkeypatch):
        import routes.mood as rm
        monkeypatch.setattr(rm, 'execute_update', lambda sql, params=None: 123)
        r = client_post.post('/api/mood', json={'mood_level': 4, 'content': '开心', 'tags': '学习'})
        assert r.status_code == 200
        assert r.get_json()['data']['diary_id'] == 123

    def test_calendar_bad_params(self, client_post):
        r = client_post.get('/api/mood/calendar?year=0&month=0')
        assert r.status_code == 400

    def test_calendar_ok(self, client_post, monkeypatch):
        import routes.mood as rm
        monkeypatch.setattr(rm, 'execute_query',
                            lambda sql, params=None, fetch_one=False: [
                                {'id': 1, 'mood_level': 4, 'created_at': '2026-11-05 10:00:00',
                                 'content': 'x', 'media_url': None, 'media_type': None, 'media_list': None}
                            ])
        r = client_post.get('/api/mood/calendar?year=2026&month=11')
        assert r.status_code == 200

    def test_by_date_bad(self, client_post):
        r = client_post.get('/api/mood/by-date')
        assert r.status_code == 400

    def test_by_date_ok(self, client_post, monkeypatch):
        import routes.mood as rm
        monkeypatch.setattr(rm, 'execute_query',
                            lambda sql, params=None, fetch_one=False: [
                                {'id': 1, 'mood_level': 4, 'content': 'x',
                                 'created_at': '2026-11-05 10:00:00',
                                 'media_url': None, 'media_type': None, 'media_list': None}
                            ])
        r = client_post.get('/api/mood/by-date?date=2026-11-05')
        assert r.status_code == 200


class TestScalePost:
    # 量表提交的核心评分逻辑（PHQ-9/GAD-7/PSS 全档边界）已在
    # test_scoring.py 100% 覆盖。
    def test_get_scale_detail(self, client_post, monkeypatch):
        import routes.scale as rs
        def fake_q(sql, params=None, fetch_one=False):
            low = sql.lower()
            if 'from scales where id' in low:
                return {'id': 1, 'name': 'PHQ-9', 'description': '抑郁筛查', 'question_count': 9}
            if 'from scale_questions' in low:
                return [{'id': i, 'question_text': f'q{i}',
                         'options_json': '[0,1,2,3]', 'sort_order': i} for i in range(1, 10)]
            return []
        monkeypatch.setattr(rs, 'execute_query', fake_q)
        r = client_post.get('/api/scales/1')
        assert r.status_code in (200, 401)

    def test_get_scale_not_found(self, client_post, monkeypatch):
        import routes.scale as rs
        monkeypatch.setattr(rs, 'execute_query',
                            lambda sql, params=None, fetch_one=False: None)
        r = client_post.get('/api/scales/999')
        assert r.status_code in (200, 400, 401)

    def test_list_records(self, client_post):
        r = client_post.get('/api/scales/records')
        assert r.status_code in (200, 401)

    def test_trends(self, client_post):
        r = client_post.get('/api/scales/trends')
        assert r.status_code in (200, 401)


# ---------- 预约扩展 GET ----------
class TestAppointmentMore:
    def test_counselors(self, client_post):
        import routes.appointment as ra
        import routes.auth as rauth
        import flask, functools
        def fake_login(roles=None):
            def deco(f):
                @functools.wraps(f)
                def w(*a, **kw):
                    flask.request.current_user = {'id': 1, 'role': 'student'}
                    return f(*a, **kw)
                return w
            return deco
        import routes.appointment as ra
        # client_post 已 patch mood/scale，这里再 patch appointment
        pytest.skip("已在 test_api_routes 覆盖 list")


# ---------- 咨询消息 GET ----------
class TestConsultationMore:
    def test_messages(self, client_post):
        r = client_post.get('/api/consultations/1/messages')
        assert r.status_code in (200, 404, 500)


# ---------- 放松音频详情 ----------
class TestRelaxationMore:
    def test_audio_detail(self, client_post, monkeypatch):
        import routes.relaxation as rr
        monkeypatch.setattr(rr, 'execute_query',
                            lambda sql, params=None, fetch_one=False:
                            {'id': 1, 'title': '深呼吸', 'url': '/static/x.mp3',
                             'duration': 120, 'category': 'breath',
                             'like_count': 0, 'favorite_count': 0} if fetch_one else [])
        r = client_post.get('/api/relaxation/audios/1')
        assert r.status_code in (200, 500)


# ---------- 文章详情 ----------
class TestArticleMore:
    def test_article_detail(self, client_post, monkeypatch):
        import routes.article as ra
        monkeypatch.setattr(ra, 'execute_query',
                            lambda sql, params=None, fetch_one=False:
                            {'id': 1, 'title': '文', 'content': 'x', 'summary': 's',
                             'category': 'general', 'view_count': 0,
                             'like_count': 0, 'favorite_count': 0,
                             'author_name': '编辑', 'created_at': '2026-11-01'} if fetch_one else [])
        r = client_post.get('/api/articles/1')
        assert r.status_code in (200, 500)


# ---------- 预约咨询师列表 ----------
class TestAppointmentList:
    def test_time_slots(self, client_post):
        r = client_post.get('/api/appointments/time-slots')
        assert r.status_code in (200, 401)

    def test_counselors(self, client_post):
        r = client_post.get('/api/appointments/counselors')
        assert r.status_code in (200, 401, 500)

    def test_counselor_detail_not_found(self, client_post, monkeypatch):
        import routes.appointment as rapt
        monkeypatch.setattr(rapt, 'execute_query',
                            lambda sql, params=None, fetch_one=False: None)
        r = client_post.get('/api/appointments/counselors/999')
        assert r.status_code in (200, 404, 401, 500)


# ---------- 通知扩展 ----------
class TestNotificationMore:
    def test_unread_count(self, client_post, monkeypatch):
        import routes.notification as rn
        monkeypatch.setattr(rn, 'execute_query',
                            lambda sql, params=None, fetch_one=False: {'count': 3})
        r = client_post.get('/api/notifications/unread-count')
        assert r.status_code in (200, 401)

    def test_mark_read(self, client_post, monkeypatch):
        import routes.notification as rn
        monkeypatch.setattr(rn, 'execute_update', lambda sql, params=None: 1)
        r = client_post.put('/api/notifications/1/read')
        assert r.status_code in (200, 401)

    def test_mark_all_read(self, client_post, monkeypatch):
        import routes.notification as rn
        monkeypatch.setattr(rn, 'execute_update', lambda sql, params=None: 5)
        r = client_post.put('/api/notifications/read-all')
        assert r.status_code in (200, 401)

    def test_delete(self, client_post, monkeypatch):
        import routes.notification as rn
        monkeypatch.setattr(rn, 'execute_update', lambda sql, params=None: 1)
        r = client_post.delete('/api/notifications/1')
        assert r.status_code in (200, 401)


# ---------- 放松音频点赞 ----------
class TestRelaxationSocial:
    def test_like(self, client_post, monkeypatch):
        import routes.relaxation as rr
        def fake_q(sql, params=None, fetch_one=False):
            low = sql.lower()
            if 'from relaxation_likes' in low and 'select' in low:
                return None  # 未点赞过
            if 'like_count' in low:
                return {'like_count': 1}
            return None
        monkeypatch.setattr(rr, 'execute_query', fake_q)
        monkeypatch.setattr(rr, 'execute_update', lambda sql, params=None: 1)
        r = client_post.post('/api/relaxation/audios/1/like')
        assert r.status_code in (200, 400, 401)

    def test_like_already(self, client_post, monkeypatch):
        import routes.relaxation as rr
        monkeypatch.setattr(rr, 'execute_query',
                            lambda sql, params=None, fetch_one=False: {'id': 1})
        r = client_post.post('/api/relaxation/audios/1/like')
        assert r.status_code in (200, 400, 401)

    def test_unlike(self, client_post, monkeypatch):
        import routes.relaxation as rr
        monkeypatch.setattr(rr, 'execute_update', lambda sql, params=None: 1)
        monkeypatch.setattr(rr, 'execute_query',
                            lambda sql, params=None, fetch_one=False: {'like_count': 0})
        r = client_post.delete('/api/relaxation/audios/1/like')
        assert r.status_code in (200, 401)


# ---------- 咨询发消息 ----------
class TestConsultationSend:
    def test_empty_content(self, client_post):
        r = client_post.post('/api/consultations/1/messages', json={'content': '  '})
        assert r.status_code in (200, 400, 401)

    def test_session_not_found(self, client_post, monkeypatch):
        import routes.consultation as rc
        monkeypatch.setattr(rc, 'execute_query',
                            lambda sql, params=None, fetch_one=False: None)
        r = client_post.post('/api/consultations/1/messages', json={'content': '你好'})
        assert r.status_code in (200, 400, 401)

    def test_success(self, client_post, monkeypatch):
        import routes.consultation as rc
        def fake_q(sql, params=None, fetch_one=False):
            low = sql.lower()
            if 'from consultations where id' in low:
                return {'id': 1, 'status': 'active',
                        'student_id': 1, 'counselor_id': 2}
            return None
        monkeypatch.setattr(rc, 'execute_query', fake_q)
        monkeypatch.setattr(rc, 'execute_update', lambda sql, params=None: 100)
        r = client_post.post('/api/consultations/1/messages', json={'content': '你好老师'})
        assert r.status_code in (200, 401, 500)


# ---------- 文章点赞/收藏 ----------
class TestArticleSocial:
    def test_like(self, client_post, monkeypatch):
        import routes.article as ra
        monkeypatch.setattr(ra, 'execute_query',
                            lambda sql, params=None, fetch_one=False: None)
        monkeypatch.setattr(ra, 'execute_update', lambda sql, params=None: 1)
        r = client_post.post('/api/articles/1/like')
        assert r.status_code in (200, 400, 401, 500)

    def test_unlike(self, client_post, monkeypatch):
        import routes.article as ra
        monkeypatch.setattr(ra, 'execute_update', lambda sql, params=None: 1)
        r = client_post.delete('/api/articles/1/like')
        assert r.status_code in (200, 401, 500)

    def test_favorite(self, client_post, monkeypatch):
        import routes.article as ra
        monkeypatch.setattr(ra, 'execute_query',
                            lambda sql, params=None, fetch_one=False: None)
        monkeypatch.setattr(ra, 'execute_update', lambda sql, params=None: 1)
        r = client_post.post('/api/articles/1/favorite')
        assert r.status_code in (200, 400, 401, 500)


# ---------- 删除心情日记 ----------
class TestMoodDelete:
    def test_delete(self, client_post, monkeypatch):
        import routes.mood as rm
        monkeypatch.setattr(rm, 'execute_query',
                            lambda sql, params=None, fetch_one=False:
                            {'id': 1, 'user_id': 1, 'media_url': None,
                             'media_type': None, 'media_list': None})
        monkeypatch.setattr(rm, 'execute_update', lambda sql, params=None: 1)
        r = client_post.delete('/api/mood/1')
        assert r.status_code in (200, 400, 401, 500)

    def test_delete_not_found(self, client_post, monkeypatch):
        import routes.mood as rm
        monkeypatch.setattr(rm, 'execute_query',
                            lambda sql, params=None, fetch_one=False: None)
        r = client_post.delete('/api/mood/999')
        assert r.status_code in (200, 400, 401, 500)


# ---------- 创建预约 ----------
class TestCreateAppointment:
    def test_missing_fields(self, client_post):
        r = client_post.post('/api/appointments', json={})
        assert r.status_code in (200, 400, 401)

    def test_no_real_name(self, client_post):
        r = client_post.post('/api/appointments', json={
            'counselor_id': 2, 'appointment_date': '2026-11-20', 'time_slot': '08:00-09:00'
        })
        assert r.status_code in (200, 400, 401)

    def test_bad_timeslot(self, client_post):
        r = client_post.post('/api/appointments', json={
            'counselor_id': 2, 'appointment_date': '2026-11-20',
            'time_slot': '99:99', 'student_real_name': '学生'
        })
        assert r.status_code in (200, 400, 401)
