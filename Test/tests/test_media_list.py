# -*- coding: utf-8 -*-
"""
utils/media_list.py 纯函数测试
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from utils.media_list import (
    _parse_json_list, validate_media_list, pack_media_for_db,
    attach_media_list, MAX_MEDIA_COUNT,
)


class TestParseJson:
    def test_none(self):
        assert _parse_json_list(None) == []

    def test_list_passthrough(self):
        assert _parse_json_list([1, 2]) == [1, 2]

    def test_json_string(self):
        assert _parse_json_list('[1,2,3]') == [1, 2, 3]

    def test_invalid_json(self):
        assert _parse_json_list('not json') == []

    def test_json_dict(self):
        # 不是 list
        assert _parse_json_list('{"a":1}') == []


class TestValidate:
    def test_none(self):
        assert validate_media_list(None, '/static/') == []

    def test_not_list(self):
        with pytest.raises(ValueError):
            validate_media_list('bad', '/static/')

    def test_too_many(self):
        items = [{'media_url': f'/static/{i}.jpg', 'media_type': 'image'} for i in range(MAX_MEDIA_COUNT + 1)]
        with pytest.raises(ValueError):
            validate_media_list(items, '/static/')

    def test_not_dict(self):
        with pytest.raises(ValueError):
            validate_media_list(['x'], '/static/')

    def test_missing_fields(self):
        with pytest.raises(ValueError):
            validate_media_list([{}], '/static/')

    def test_bad_prefix(self):
        with pytest.raises(ValueError):
            validate_media_list([{'media_url': '/etc/passwd', 'media_type': 'image'}], '/static/')

    def test_bad_type(self):
        with pytest.raises(ValueError):
            validate_media_list([{'media_url': '/static/a.png', 'media_type': 'audio'}], '/static/')

    def test_ok(self):
        items = [{'media_url': '/static/a.png', 'media_type': 'image'}]
        out = validate_media_list(items, '/static/')
        assert out == items


class TestPack:
    def test_empty(self):
        assert pack_media_for_db([]) == (None, None, None)

    def test_pack(self):
        items = [{'media_url': '/static/a.png', 'media_type': 'image'}]
        j, url, t = pack_media_for_db(items)
        assert url == '/static/a.png'
        assert t == 'image'
        assert 'a.png' in j


class TestAttach:
    def test_none_row(self):
        assert attach_media_list(None) is None

    def test_from_json(self):
        row = {'media_list': '[{"media_url":"/static/a.png","media_type":"image"}]'}
        out = attach_media_list(row)
        assert out['media_list'][0]['media_url'] == '/static/a.png'

    def test_legacy_fields(self):
        row = {'media_url': '/static/b.png', 'media_type': 'image'}
        out = attach_media_list(row)
        assert out['media_list'][0]['media_url'] == '/static/b.png'
