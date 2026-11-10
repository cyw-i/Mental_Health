# -*- coding: utf-8 -*-
"""
utils/llm.py 测试：mock requests，覆盖成功/超时/认证错误/余额不足/本地模型
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
from utils.llm import chat_completion, LLMError, _friendly_error, _is_local_llm


class TestFriendlyError:
    def test_balance(self):
        assert '余额不足' in _friendly_error('Insufficient balance')

    def test_invalid_key(self):
        assert 'API Key' in _friendly_error('Invalid API key')

    def test_model_not_found(self):
        r = _friendly_error('model gpt-9 not found')
        assert '模型名称' in r

    def test_unknown(self):
        r = _friendly_error('some weird error')
        assert '请求失败' in r


class TestIsLocal:
    def test_localhost(self):
        assert _is_local_llm('http://localhost:11434') is True

    def test_127(self):
        assert _is_local_llm('http://127.0.0.1:8000') is True

    def test_remote(self):
        assert _is_local_llm('https://api.deepseek.com') is False

    def test_none(self):
        assert _is_local_llm(None) is False


class TestChatCompletion:
    @patch('utils.llm._get_llm_config')
    @patch('utils.llm.requests.post')
    def test_success(self, mock_post, mock_cfg):
        mock_cfg.return_value = {
            'api_key': 'sk-test', 'api_base': 'https://api.deepseek.com/v1',
            'model': 'deepseek-chat', 'timeout': 60,
        }
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            'choices': [{'message': {'content': '  你好呀  '}}]
        }
        mock_post.return_value = resp
        out = chat_completion([{'role': 'user', 'content': 'hi'}])
        assert out == '你好呀'

    @patch('utils.llm._get_llm_config')
    def test_no_api_key_remote(self, mock_cfg):
        mock_cfg.return_value = {
            'api_key': '', 'api_base': 'https://api.deepseek.com/v1',
            'model': 'deepseek-chat', 'timeout': 60,
        }
        with pytest.raises(LLMError, match='未配置'):
            chat_completion([{'role': 'user', 'content': 'hi'}])

    @patch('utils.llm._get_llm_config')
    @patch('utils.llm.requests.post')
    def test_local_no_key_ok(self, mock_post, mock_cfg):
        mock_cfg.return_value = {
            'api_key': '', 'api_base': 'http://localhost:11434/v1',
            'model': 'qwen', 'timeout': 60,
        }
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {'choices': [{'message': {'content': 'ok'}}]}
        mock_post.return_value = resp
        assert chat_completion([{'role': 'user', 'content': 'hi'}]) == 'ok'

    @patch('utils.llm._get_llm_config')
    @patch('utils.llm.requests.post')
    def test_timeout(self, mock_post, mock_cfg):
        import requests as req
        mock_cfg.return_value = {
            'api_key': 'k', 'api_base': 'https://api.x.com/v1',
            'model': 'm', 'timeout': 1,
        }
        mock_post.side_effect = req.Timeout()
        with pytest.raises(LLMError, match='超时'):
            chat_completion([{'role': 'user', 'content': 'hi'}])

    @patch('utils.llm._get_llm_config')
    @patch('utils.llm.requests.post')
    def test_api_error(self, mock_post, mock_cfg):
        mock_cfg.return_value = {
            'api_key': 'k', 'api_base': 'https://api.x.com/v1',
            'model': 'm', 'timeout': 60,
        }
        resp = MagicMock()
        resp.status_code = 401
        resp.json.return_value = {'error': {'message': 'Invalid API key'}}
        mock_post.return_value = resp
        with pytest.raises(LLMError, match='API Key'):
            chat_completion([{'role': 'user', 'content': 'hi'}])

    @patch('utils.llm._get_llm_config')
    @patch('utils.llm.requests.post')
    def test_conn_error(self, mock_post, mock_cfg):
        import requests as req
        mock_cfg.return_value = {
            'api_key': 'k', 'api_base': 'https://api.x.com/v1',
            'model': 'm', 'timeout': 60,
        }
        mock_post.side_effect = req.ConnectionError('refused')
        with pytest.raises(LLMError, match='无法连接'):
            chat_completion([{'role': 'user', 'content': 'hi'}])
