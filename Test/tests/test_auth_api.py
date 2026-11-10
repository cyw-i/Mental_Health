# -*- coding: utf-8 -*-
"""
auth 路由 API 测试：登录成功/失败分支
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from utils.auth import md5_encrypt

FAKE_USER = {
    'id': 1, 'username': 'student1', 'password': md5_encrypt('123456'),
    'real_name': '学生', 'nickname': '学', 'role': 'student',
    'phone': '', 'email': 's@x.com', 'gender': 1, 'avatar': '',
    'status': 1, 'security_question': 'q',
}


def make_fake_query(user=None):
    def fake(sql, params=None, fetch_one=False):
        sql_low = sql.lower()
        if 'count(*)' in sql_low:
            return {'count': 1}
        if fetch_one:
            if 'from users where username' in sql_low:
                return user
            # ensure_* 探测列是否存在 → 返回非 None 表示存在
            return {'id': 1}
        return []
    return fake


@pytest.fixture
def auth_client(app, monkeypatch):
    import routes.auth as ra
    monkeypatch.setattr(ra, 'execute_query', make_fake_query(dict(FAKE_USER)))
    monkeypatch.setattr(ra, 'execute_update', lambda sql, params=None: 1)
    return app.test_client()


class TestLogin:
    def test_missing_fields(self, auth_client):
        r = auth_client.post('/api/auth/login', json={})
        assert r.status_code == 400
        assert '不能为空' in r.get_json()['message']

    def test_user_not_found(self, app, monkeypatch):
        import routes.auth as ra
        monkeypatch.setattr(ra, 'execute_query', make_fake_query(None))
        monkeypatch.setattr(ra, 'execute_update', lambda sql, params=None: 1)
        c = app.test_client()
        r = c.post('/api/auth/login', json={'username': 'nope', 'password': 'x'})
        assert r.status_code == 400
        assert '不存在' in r.get_json()['message']

    def test_wrong_password(self, auth_client):
        r = auth_client.post('/api/auth/login', json={'username': 'student1', 'password': 'wrong'})
        assert r.status_code == 400
        assert '密码' in r.get_json()['message']

    def test_success(self, auth_client):
        r = auth_client.post('/api/auth/login', json={'username': 'student1', 'password': '123456'})
        assert r.status_code == 200
        body = r.get_json()
        assert body['code'] == 200
        assert 'token' in body['data']
        assert body['data']['user']['username'] == 'student1'

    def test_disabled_user(self, app, monkeypatch):
        disabled = dict(FAKE_USER)
        disabled['status'] = 0
        import routes.auth as ra
        monkeypatch.setattr(ra, 'execute_query', make_fake_query(disabled))
        monkeypatch.setattr(ra, 'execute_update', lambda sql, params=None: 1)
        c = app.test_client()
        r = c.post('/api/auth/login', json={'username': 'student1', 'password': '123456'})
        assert r.status_code == 400
        assert '禁用' in r.get_json()['message']


class TestRegister:
    def _base_data(self):
        return {
            'username': 'newstudent', 'password': 'secret123',
            'student_no': '20230001', 'real_name': '新生',
            'email': 'new@chd.edu.cn', 'security_question': '你的母校？',
            'security_answer': '长大', 'phone': '13800000000', 'gender': 1,
        }

    def test_missing_fields(self, auth_client):
        d = self._base_data()
        d['username'] = ''
        r = auth_client.post('/api/auth/register', json=d)
        assert r.status_code == 400

    def test_bad_email(self, auth_client):
        d = self._base_data()
        d['email'] = 'not-an-email'
        r = auth_client.post('/api/auth/register', json=d)
        assert r.status_code == 400
        assert '邮箱' in r.get_json()['message']

    def test_short_password(self, auth_client):
        d = self._base_data()
        d['password'] = '123'
        r = auth_client.post('/api/auth/register', json=d)
        assert r.status_code == 400
        assert '密码' in r.get_json()['message']

    def test_success(self, app, monkeypatch):
        import routes.auth as ra
        registry = {'student_no': '20230001', 'real_name': '新生',
                    'gender': 1, 'is_registered': 0, 'department': '计科'}
        calls = []
        def fake_q(sql, params=None, fetch_one=False):
            sql_low = sql.lower()
            if 'count(*)' in sql_low:
                return {'count': 1}
            if 'from student_registry where student_no' in sql_low:
                return registry
            if 'from users where username' in sql_low:
                return None  # 用户名不重复
            if 'from users where student_no' in sql_low:
                return None
            if 'from users where email' in sql_low:
                return None
            return {'id': 1}
        monkeypatch.setattr(ra, 'execute_query', fake_q)
        monkeypatch.setattr(ra, 'execute_update', lambda sql, params=None: 42)
        c = app.test_client()
        r = c.post('/api/auth/register', json=self._base_data())
        assert r.status_code == 200
        assert r.get_json()['data']['user_id'] == 42

    def test_duplicate_username(self, app, monkeypatch):
        import routes.auth as ra
        def fake_q(sql, params=None, fetch_one=False):
            sql_low = sql.lower()
            if 'count(*)' in sql_low:
                return {'count': 1}
            if 'from student_registry' in sql_low:
                return {'student_no': 'x', 'real_name': '新生', 'is_registered': 0}
            if 'from users where username' in sql_low:
                return {'id': 99}
            return None
        monkeypatch.setattr(ra, 'execute_query', fake_q)
        monkeypatch.setattr(ra, 'execute_update', lambda sql, params=None: 1)
        c = app.test_client()
        r = c.post('/api/auth/register', json=self._base_data())
        assert r.status_code == 400
        assert '已存在' in r.get_json()['message']
