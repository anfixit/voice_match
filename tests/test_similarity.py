"""Тесты описательного скоринга эмбеддингов."""

import numpy as np
import pytest

from voice_match.scoring.similarity import (
    cosine_similarity,
    normalize_embedding,
    summarize_embeddings,
)


def test_cosine_similarity_returns_one_for_same_direction() -> None:
    first = np.array([1.0, 2.0, 3.0])
    second = np.array([2.0, 4.0, 6.0])

    result = cosine_similarity(first, second)

    assert result == pytest.approx(1.0)


def test_cosine_similarity_returns_zero_for_orthogonal_vectors() -> None:
    first = np.array([1.0, 0.0])
    second = np.array([0.0, 1.0])

    result = cosine_similarity(first, second)

    assert result == pytest.approx(0.0)


def test_normalize_embedding_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match='Нулевой эмбеддинг'):
        normalize_embedding(np.zeros(3))


def test_summarize_embeddings_reports_pairwise_statistics() -> None:
    first = np.array([[1.0, 0.0], [0.8, 0.2]])
    second = np.array([[1.0, 0.0], [0.6, 0.4]])

    result = summarize_embeddings(first, second)

    assert result.pair_count == 4
    assert result.centroid_score > 0.9
    assert result.minimum <= result.median_pair_score <= result.maximum
    assert result.standard_deviation >= 0.0


def test_summarize_embeddings_rejects_dimension_mismatch() -> None:
    first = np.ones((2, 3))
    second = np.ones((2, 4))

    with pytest.raises(ValueError, match='Размерности'):
        summarize_embeddings(first, second)
