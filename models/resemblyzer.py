from resemblyzer import VoiceEncoder
import torch
import numpy as np
from typing import Dict, List, Union
from app.log import setup_logger

log = setup_logger("resemblyzer_model")


class EnhancedResemblyzer:
    """Улучшенная версия Resemblyzer с дополнительными функциями для судебной экспертизы"""

    def __init__(self):
        """Инициализирует улучшенную модель Resemblyzer"""
        self.model = None
        self.embedding_dim = 256  # Размерность d-вектора Resemblyzer
        self.similarity_threshold = 0.75  # Порог для принятия решения о совпадении голосов

    def load_model(self):
        """Загружает предобученную модель Resemblyzer"""
        try:
            self.model = VoiceEncoder()
            log.info("Модель Resemblyzer успешно загружена")
        except Exception as e:
            log.error(f"Ошибка при загрузке модели Resemblyzer: {e}")
            raise RuntimeError(f"Не удалось загрузить модель Resemblyzer: {e}")

    def embed_utterance(self, wav: np.ndarray, return_partials: bool = False) -> Union[
        np.ndarray, Dict[str, np.ndarray]]:
        """
        Извлекает d-вектор из аудиосигнала.

        Args:
            wav: Аудиосигнал в формате numpy array
            return_partials: Флаг возврата промежуточных результатов

        Returns:
            D-вектор или словарь с d-вектором и промежуточными результатами
        """
        if self.model is None:
            self.load_model()

        try:
            if return_partials:
                # Извлечение вектора с промежуточными результатами
                embedding, partial_embeds, wave_slices = self.model.embed_utterance(
                    wav, return_partials=True
                )

                return {
                    "embedding": embedding,
                    "partial_embeds": partial_embeds,
                    "wave_slices": wave_slices
                }
            else:
                # Стандартное извлечение вектора
                embedding = self.model.embed_utterance(wav)
                return embedding

        except Exception as e:
            log.error(f"Ошибка при извлечении d-вектора: {e}")
            # Возвращаем нулевой вектор при ошибке
            return np.zeros(self.embedding_dim) if not return_partials else {
                "embedding": np.zeros(self.embedding_dim),
                "partial_embeds": np.zeros((0, self.embedding_dim)),
                "wave_slices": []
            }

    def embed_segments(self, segments: List[np.ndarray]) -> np.ndarray:
        """
        Извлекает d-векторы из нескольких сегментов аудио.

        Args:
            segments: Список сегментов аудиоданных

        Returns:
            Массив с d-векторами формы [num_segments, embedding_dim]
        """
        if not segments:
            return np.zeros((0, self.embedding_dim))

        # Извлечение d-векторов для каждого сегмента
        embeddings = []

        for segment in segments:
            embedding = self.embed_utterance(segment)
            embeddings.append(embedding)

        # Объединение результатов
        return np.vstack(embeddings)

    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> Dict[str, float]:
        """
        Сравнивает два d-вектора и возвращает метрики сходства.

        Args:
            emb1: Первый d-вектор
            emb2: Второй d-вектор

        Returns:
            Словарь с метриками сходства
        """
        if self.model is None:
            self.load_model()

        # Косинусное сходство
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        # Вероятность того же диктора (сигмоидное масштабирование)
        probability = 1.0 / (1.0 + np.exp(-10.0 * (cosine_sim - 0.5)))

        # Уверенность
        confidence = abs(probability - 0.5) * 2.0

        # Определение класса
        is_same_speaker = cosine_sim >= self.similarity_threshold

        return {
            "cosine_similarity": float(cosine_sim),
            "probability": float(probability),
            "confidence": float(confidence),
            "is_same_speaker": bool(is_same_speaker)
        }

    def compare_segments(self, segments1: List[np.ndarray], segments2: List[np.ndarray]) -> Dict[
        str, Union[float, List[float]]]:
        """
        Сравнивает наборы сегментов от двух дикторов.

        Args:
            segments1: Список сегментов первого диктора
            segments2: Список сегментов второго диктора

        Returns:
            Словарь с метриками сходства
        """
        # Извлечение эмбеддингов
        embs1 = self.embed_segments(segments1)
        embs2 = self.embed_segments(segments2)

        # Проверка на пустые наборы
        if embs1.shape[0] == 0 or embs2.shape[0] == 0:
            return {
                "mean_cosine_similarity": 0.0,
                "max_cosine_similarity": 0.0,
                "min_cosine_similarity": 0.0,
                "std_cosine_similarity": 0.0,
                "probability": 0.0,
                "confidence": 0.0,
                "is_same_speaker": False
            }

        # Вычисление всех попарных сходств
        similarities = []

        for i in range(embs1.shape[0]):
            for j in range(embs2.shape[0]):
                cos_sim = np.dot(embs1[i], embs2[j]) / (np.linalg.norm(embs1[i]) * np.linalg.norm(embs2[j]))
                similarities.append(cos_sim)

        # Статистика
        similarities = np.array(similarities)
        mean_sim = float(np.mean(similarities))
        max_sim = float(np.max(similarities))
        min_sim = float(np.min(similarities))
        std_sim = float(np.std(similarities))

        # Вероятность того же диктора
        probability = 1.0 / (1.0 + np.exp(-10.0 * (mean_sim - 0.5)))

        # Уверенность
        confidence = abs(probability - 0.5) * 2.0

        # Определение класса
        is_same_speaker = mean_sim >= self.similarity_threshold

        return {
            "mean_cosine_similarity": mean_sim,
            "max_cosine_similarity": max_sim,
            "min_cosine_similarity": min_sim,
            "std_cosine_similarity": std_sim,
            "all_similarities": similarities.tolist(),
            "probability": float(probability),
            "confidence": float(confidence),
            "is_same_speaker": bool(is_same_speaker)
        }

    def analyze_temporal_stability(self, wav: np.ndarray) -> Dict[str, Union[float, List[float]]]:
        """
        Анализирует временную стабильность голосовых характеристик.

        Args:
            wav: Аудиосигнал в формате numpy array

        Returns:
            Словарь с анализом стабильности
        """
        if self.model is None:
            self.load_model()

        try:
            # Извлечение вектора с промежуточными результатами
            result = self.embed_utterance(wav, return_partials=True)

            embedding = result["embedding"]
            partial_embeds = result["partial_embeds"]

            # Если недостаточно сегментов
            if len(partial_embeds) < 2:
                return {
                    "temporal_stability": 1.0,
                    "voice_consistency": 1.0,
                    "segment_similarities": []
                }

            # Сравнение каждого частичного эмбеддинга с общим
            segment_similarities = []

            for i, partial in enumerate(partial_embeds):
                sim = np.dot(partial, embedding) / (np.linalg.norm(partial) * np.linalg.norm(embedding))
                segment_similarities.append(float(sim))

            # Среднее сходство и стандартное отклонение
            mean_sim = float(np.mean(segment_similarities))
            std_sim = float(np.std(segment_similarities))

            # Стабильность во времени
            temporal_stability = mean_sim

            # Консистентность голоса (обратно пропорционально стандартному отклонению)
            voice_consistency = 1.0 - min(1.0, std_sim * 5.0)

            return {
                "temporal_stability": temporal_stability,
                "voice_consistency": voice_consistency,
                "segment_similarities": segment_similarities,
                "num_segments": len(partial_embeds)
            }

        except Exception as e:
            log.error(f"Ошибка при анализе временной стабильности: {e}")
            return {
                "temporal_stability": 1.0,
                "voice_consistency": 1.0,
                "segment_similarities": [],
                "error": str(e)
            }


def get_resemblyzer() -> EnhancedResemblyzer:
    """
    Возвращает экземпляр улучшенной модели Resemblyzer.

    Returns:
        Экземпляр EnhancedResemblyzer
    """
    model = EnhancedResemblyzer()
    model.load_model()
    return model
