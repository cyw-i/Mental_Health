# -*- coding: utf-8 -*-
"""
utils/db.py 错误分支测试
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
from pymysql.err import OperationalError
from utils.db import get_db_connection, DatabaseError


class TestDbErrors:
    @patch('utils.db.pymysql.connect')
    def test_auth_fail(self, mock_connect):
        mock_connect.side_effect = OperationalError(1045, 'Access denied')
        with pytest.raises(DatabaseError, match='认证失败'):
            get_db_connection()

    @patch('utils.db.pymysql.connect')
    def test_db_not_exist(self, mock_connect):
        mock_connect.side_effect = OperationalError(1049, 'Unknown database')
        with pytest.raises(DatabaseError, match='不存在'):
            get_db_connection()

    @patch('utils.db.pymysql.connect')
    def test_conn_refused(self, mock_connect):
        mock_connect.side_effect = OperationalError(2003, 'Can not connect')
        with pytest.raises(DatabaseError, match='无法连接'):
            get_db_connection()

    @patch('utils.db.pymysql.connect')
    def test_other_error(self, mock_connect):
        mock_connect.side_effect = OperationalError(9999, 'weird')
        with pytest.raises(DatabaseError, match='连接失败'):
            get_db_connection()
