"""
rag.py 单元测试
测试余弦相似度、search_similar 检索、build_rag_context、build_rag_sources
所有数据库查询和 embedding 调用均 mock
"""
import json
import math
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.rag import (
    _cosine_similarity,
    search_similar,
    build_rag_context,
    build_rag_sources,
)
from utils.embedding import EmbeddingError

# 从 conftest 导入 mock 数据
from tests.conftest import (
    MOCK_KNOWLEDGE_CHUNKS,
    MOCK_EMBEDDING_768,
)


class TestCosineSimilarity:
    """测试 _cosine_similarity 函数"""

    def test_identical_vectors(self):
        """相同向量 → 相似度 ≈ 1.0"""
        v = [0.1, 0.2, 0.3, 0.4]
        sim = _cosine_similarity(v, v)
        assert math.isclose(sim, 1.0, rel_tol=1e-6)

    def test_orthogonal_vectors(self):
        """正交向量 → 相似度 ≈ 0.0"""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        sim = _cosine_similarity(a, b)
        assert math.isclose(sim, 0.0, abs_tol=1e-6)

    def test_opposite_vectors(self):
        """反向向量 → 相似度 ≈ -1.0"""
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        sim = _cosine_similarity(a, b)
        assert math.isclose(sim, -1.0, rel_tol=1e-6)

    def test_different_length_vectors(self):
        """不同长度向量 → 返回 0.0"""
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0]
        sim = _cosine_similarity(a, b)
        assert sim == 0.0

    def test_zero_vector(self):
        """零向量 → 返回 0.0（避免除零）"""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        sim = _cosine_similarity(a, b)
        assert sim == 0.0

    def test_both_zero_vectors(self):
        """两个都是零向量 → 返回 0.0"""
        a = [0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        sim = _cosine_similarity(a, b)
        assert sim == 0.0

    def test_positive_similarity(self):
        """同方向但不相同 → 0 < sim < 1"""
        a = [1.0, 2.0, 3.0]
        b = [1.5, 2.5, 3.5]
        sim = _cosine_similarity(a, b)
        assert 0.9 < sim < 1.0

    def test_single_element_vectors(self):
        """单元素向量"""
        a = [5.0]
        b = [10.0]
        sim = _cosine_similarity(a, b)
        assert math.isclose(sim, 1.0, rel_tol=1e-6)


class TestSearchSimilar:
    """测试 search_similar 检索函数"""

    def test_rag_disabled_returns_empty(self, monkeypatch):
        """RAG_ENABLED=False → 返回空列表"""
        monkeypatch.setattr(Config, 'RAG_ENABLED', False)
        results = search_similar('抑郁')
        assert results == []

    def test_search_returns_top_k(self, monkeypatch):
        """知识库有 5 条，top_k=3 → 返回 3 条"""
        monkeypatch.setattr(Config, 'RAG_ENABLED', True)
        monkeypatch.setattr(Config, 'RAG_TOP_K', 5)

        # Mock get_embedding 返回查询向量
        monkeypatch.setattr(
            'utils.rag.get_embedding',
            lambda text: [0.05 * (i % 20 + 1) for i in range(768)],
        )
        # Mock execute_query 返回知识库数据
        monkeypatch.setattr(
            'utils.rag.execute_query',
            lambda sql, **kwargs: MOCK_KNOWLEDGE_CHUNKS,
        )

        results = search_similar('抑郁症状', top_k=3)
        assert len(results) == 3

    def test_search_results_sorted_by_similarity(self, monkeypatch):
        """返回结果按相似度降序排列"""
        monkeypatch.setattr(Config, 'RAG_ENABLED', True)
        monkeypatch.setattr(Config, 'RAG_TOP_K', 5)

        # 使用与 MOCK_KNOWLEDGE_CHUNKS 第一条接近的查询向量
        monkeypatch.setattr(
            'utils.rag.get_embedding',
            lambda text: [0.1 * (i % 10 + 1) for i in range(768)],
        )
        monkeypatch.setattr(
            'utils.rag.execute_query',
            lambda sql, **kwargs: MOCK_KNOWLEDGE_CHUNKS,
        )

        results = search_similar('PHQ-9 抑郁量表说明')
        # 验证降序排列
        similarities = [r['similarity'] for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_search_empty_db(self, monkeypatch):
        """知识库为空 → 返回空列表"""
        monkeypatch.setattr(Config, 'RAG_ENABLED', True)

        monkeypatch.setattr(
            'utils.rag.get_embedding',
            lambda text: MOCK_EMBEDDING_768,
        )
        monkeypatch.setattr(
            'utils.rag.execute_query',
            lambda sql, **kwargs: [],
        )

        results = search_similar('抑郁')
        assert results == []

    def test_search_embedding_error_graceful(self, monkeypatch):
        """Embedding 失败 → 返回空列表（静默降级）"""
        monkeypatch.setattr(Config, 'RAG_ENABLED', True)

        def mock_embed_error(text):
            raise EmbeddingError('模拟嵌入失败')

        monkeypatch.setattr('utils.rag.get_embedding', mock_embed_error)

        results = search_similar('测试')
        assert results == []

    def test_search_db_error_graceful(self, monkeypatch):
        """数据库查询失败 → 返回空列表"""
        monkeypatch.setattr(Config, 'RAG_ENABLED', True)

        monkeypatch.setattr(
            'utils.rag.get_embedding',
            lambda text: MOCK_EMBEDDING_768,
        )
        monkeypatch.setattr(
            'utils.rag.execute_query',
            lambda sql, **kwargs: (_ for _ in ()).throw(Exception('DB error')),
        )

        results = search_similar('测试')
        assert results == []

    def test_search_result_structure(self, monkeypatch):
        """验证返回结果的字典结构"""
        monkeypatch.setattr(Config, 'RAG_ENABLED', True)

        monkeypatch.setattr(
            'utils.rag.get_embedding',
            lambda text: [0.05 * (i % 20 + 1) for i in range(768)],
        )
        monkeypatch.setattr(
            'utils.rag.execute_query',
            lambda sql, **kwargs: MOCK_KNOWLEDGE_CHUNKS,
        )

        results = search_similar('抑郁')
        for r in results:
            assert 'id' in r
            assert 'content' in r
            assert 'title' in r
            assert 'source_type' in r
            assert 'similarity' in r
            assert isinstance(r['similarity'], float)
            assert -1.0 <= r['similarity'] <= 1.0


class TestBuildRagContext:
    """测试 build_rag_context 上下文构建"""

    def test_context_contains_expected_sections(self, monkeypatch):
        """验证上下文包含标题和相关度"""
        monkeypatch.setattr(Config, 'RAG_ENABLED', True)

        monkeypatch.setattr(
            'utils.rag.get_embedding',
            lambda text: [0.05 * (i % 20 + 1) for i in range(768)],
        )
        monkeypatch.setattr(
            'utils.rag.execute_query',
            lambda sql, **kwargs: MOCK_KNOWLEDGE_CHUNKS,
        )

        ctx = build_rag_context('抑郁症状')
        assert ctx is not None
        assert '相关知识库信息' in ctx
        assert 'PHQ-9' in ctx
        assert '相关度' in ctx

    def test_context_no_results(self, monkeypatch):
        """无检索结果 → 返回 None"""
        monkeypatch.setattr(Config, 'RAG_ENABLED', True)

        monkeypatch.setattr(
            'utils.rag.get_embedding',
            lambda text: MOCK_EMBEDDING_768,
        )
        monkeypatch.setattr(
            'utils.rag.execute_query',
            lambda sql, **kwargs: [],
        )

        ctx = build_rag_context('无关联查询')
        assert ctx is None

    def test_context_rag_disabled(self, monkeypatch):
        """RAG 关闭 → 返回 None"""
        monkeypatch.setattr(Config, 'RAG_ENABLED', False)
        ctx = build_rag_context('测试')
        assert ctx is None


class TestBuildRagSources:
    """测试 build_rag_sources 来源构建（前端展示用）"""

    def test_sources_structure(self, monkeypatch):
        """验证返回的 source 包含必要字段"""
        monkeypatch.setattr(Config, 'RAG_ENABLED', True)

        monkeypatch.setattr(
            'utils.rag.get_embedding',
            lambda text: [0.05 * (i % 20 + 1) for i in range(768)],
        )
        monkeypatch.setattr(
            'utils.rag.execute_query',
            lambda sql, **kwargs: MOCK_KNOWLEDGE_CHUNKS,
        )

        sources = build_rag_sources('抑郁')
        assert len(sources) > 0
        for src in sources:
            assert 'title' in src
            assert 'summary' in src
            assert 'similarity' in src
            assert 'source_type' in src
            assert isinstance(src['similarity'], float)

    def test_sources_no_results(self, monkeypatch):
        """无结果 → 返回空列表"""
        monkeypatch.setattr(Config, 'RAG_ENABLED', True)

        monkeypatch.setattr(
            'utils.rag.get_embedding',
            lambda text: MOCK_EMBEDDING_768,
        )
        monkeypatch.setattr(
            'utils.rag.execute_query',
            lambda sql, **kwargs: [],
        )

        sources = build_rag_sources('无关联查询')
        assert sources == []

    def test_sources_summary_truncation(self, monkeypatch):
        """长内容摘要被截断到 80 字以内"""
        monkeypatch.setattr(Config, 'RAG_ENABLED', True)

        long_content_chunk = [{
            'id': 1,
            'source_type': 'scale_overview',
            'source_id': 1,
            'title': '测试量表',
            'content': 'A' * 200 + '\n' + 'B' * 100,
            'embedding': json.dumps([0.1] * 768),
            'metadata': '{}',
        }]

        monkeypatch.setattr(
            'utils.rag.get_embedding',
            lambda text: [0.1] * 768,
        )
        monkeypatch.setattr(
            'utils.rag.execute_query',
            lambda sql, **kwargs: long_content_chunk,
        )

        sources = build_rag_sources('测试')
        assert len(sources) == 1
        # 第一行超过 80 字，应该被截断到 80 字内
        assert len(sources[0]['summary']) <= 80

    def test_sources_respect_top_k(self, monkeypatch):
        """验证 top_k 参数生效"""
        monkeypatch.setattr(Config, 'RAG_ENABLED', True)

        monkeypatch.setattr(
            'utils.rag.get_embedding',
            lambda text: [0.05 * (i % 20 + 1) for i in range(768)],
        )
        monkeypatch.setattr(
            'utils.rag.execute_query',
            lambda sql, **kwargs: MOCK_KNOWLEDGE_CHUNKS,
        )

        sources = build_rag_sources('抑郁', top_k=2)
        assert len(sources) <= 2
