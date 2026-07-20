"""Чистые функции для расчёта сходства эмбеддингов."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SimilaritySummary:
    """Описательная статистика сырых cosine score."""

    centroid_score: float
    median_pair_score: float
    mean_pair_score: float
    standard_deviation: float
    minimum: float
    maximum: float
    pair_count: int


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Нормализовать эмбеддинг по L2-норме.

    Raises:
        ValueError: Если вектор пустой, нечисловой или нулевой.
    """
    vector = np.asarray(embedding, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError('Эмбеддинг должен содержать конечные числа.')

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError('Нулевой эмбеддинг нельзя нормализовать.')

    return vector / norm


def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Вычислить cosine score двух эмбеддингов."""
    first_normalized = normalize_embedding(first)
    second_normalized = normalize_embedding(second)
    return float(np.dot(first_normalized, second_normalized))


def summarize_embeddings(
    first_embeddings: np.ndarray,
    second_embeddings: np.ndarray,
) -> SimilaritySummary:
    """Суммировать сходство двух наборов сегментных эмбеддингов.

    Значения являются сырыми cosine score. Они не являются
    вероятностями и не должны интерпретироваться как LLR.

    Raises:
        ValueError: Если наборы пусты или имеют разные размерности.
    """
    first = _normalize_matrix(first_embeddings)
    second = _normalize_matrix(second_embeddings)

    if first.shape[1] != second.shape[1]:
        raise ValueError('Размерности эмбеддингов должны совпадать.')

    first_centroid = normalize_embedding(first.mean(axis=0))
    second_centroid = normalize_embedding(second.mean(axis=0))
    centroid_score = float(np.dot(first_centroid, second_centroid))

    pair_scores = (first @ second.T).reshape(-1)
    return SimilaritySummary(
        centroid_score=centroid_score,
        median_pair_score=float(np.median(pair_scores)),
        mean_pair_score=float(np.mean(pair_scores)),
        standard_deviation=float(np.std(pair_scores)),
        minimum=float(np.min(pair_scores)),
        maximum=float(np.max(pair_scores)),
        pair_count=int(pair_scores.size),
    )


def _normalize_matrix(embeddings: np.ndarray) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError('Ожидается непустая матрица эмбеддингов.')
    if not np.all(np.isfinite(matrix)):
        raise ValueError('Эмбеддинги должны содержать конечные числа.')

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError('Матрица содержит нулевой эмбеддинг.')

    return matrix / norms
