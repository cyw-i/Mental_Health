# -*- coding: utf-8 -*-
"""
utils/auth.py 纯函数测试：MD5、JWT 生成/解析
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import jwt as pyjwt
import pytest
from utils.auth import md5_encrypt, generate_token, decode_token
from config import Config


class TestMd5:
    def test_known_hash(self):
        # 123456 的 md5
        assert md5_encrypt('123456') == 'e10adc3949ba59abbe56e057f20f883e'

    def test_consistency(self):
        assert md5_encrypt('abc') == md5_encrypt('abc')

    def test_different_passwords(self):
        assert md5_encrypt('abc') != md5_encrypt('abd')


class TestToken:
    def test_roundtrip(self):
        t = generate_token(1, 'student1', 'student')
        payload = decode_token(t)
        assert payload is not None
        assert payload['user_id'] == 1
        assert payload['username'] == 'student1'
        assert payload['role'] == 'student'

    def test_invalid_token(self):
        assert decode_token('not.a.jwt') is None

    def test_expired_token(self):
        # 手工构造一个已过期的 token
        payload = {
            'user_id': 1, 'username': 'x', 'role': 'student',
            'exp': int(time.time()) - 100,
        }
        t = pyjwt.encode(payload, Config.JWT_SECRET, algorithm='HS256')
        assert decode_token(t) is None

    def test_wrong_secret(self):
        payload = {'user_id': 1, 'exp': int(time.time()) + 1000}
        t = pyjwt.encode(payload, 'wrong-secret', algorithm='HS256')
        assert decode_token(t) is None
