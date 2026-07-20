"""Методы скоринга voice_match."""

from voice_match.scoring.similarity import (
    SimilaritySummary,
    cosine_similarity,
    normalize_embedding,
    summarize_embeddings,
)

__all__ = [
    'SimilaritySummary',
    'cosine_similarity',
    'normalize_embedding',
    'summarize_embeddings',
]
