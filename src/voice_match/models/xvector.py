from collections.abc import Callable

import numpy as np
import torch

from pyannote.audio import Model

from voice_match.log import setup_logger

log = setup_logger("xvector_model")


class EnhancedXVector:
    """Улучшенная модель X-vector с дополнительными возможностями для судебной экспертизы"""

    def __init__(self, use_auth_token: bool = True):
        """
        Инициализирует улучшенную модель X-vector.

        Args:
            use_auth_token: Флаг использования токена аутентификации для загрузки модели
        """
        self.model = None
        self.use_auth_token = use_auth_token
        self.embedding_dim = 512  # Размерность X-vector эмбеддинга
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        """Загружает предобученную модель X-vector"""
        try:
            self.model = Model.from_pretrained(
                "pyannote/embedding",
                use_auth_token=self.use_auth_token
            )
            self.model = self.model.to(self.device)
            log.info("Модель X-vector успешно загружена")
        except Exception as e:
            log.error('Ошибка при загрузке модели X-vector: %s', e)
            raise RuntimeError(f"Не удалось загрузить модель X-vector: {e}") from e

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Извлекает X-vector эмбеддинг из аудиосигнала.

        Args:
            waveform: Тензор с аудиоданными формы [1, samples]

        Returns:
            Тензор с эмбеддингом формы [1, embedding_dim]
        """
        if self.model is None:
            self.load_model()

        # Убеждаемся, что тензор на правильном устройстве
        if waveform.device != torch.device(self.device):
            waveform = waveform.to(self.device)

        with torch.no_grad():
            # Подготовка входных данных для модели
            inputs = {
                "waveform": waveform,
                "sample_rate": 16000
            }

            # Извлечение эмбеддинга
            embedding = self.model(inputs)

            # Нормализация для косинусного сравнения
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

            return embedding

    def extract_batch(self, segments: list[np.ndarray]) -> np.ndarray:
        """
        Извлекает эмбеддинги из нескольких сегментов аудио.

        Args:
            segments: Список сегментов аудиоданных

        Returns:
            Массив с эмбеддингами формы [num_segments, embedding_dim]
        """
        if not segments:
            return np.zeros((0, self.embedding_dim))

        # Преобразование numpy arrays в тензоры
        embeddings = []

        for segment in segments:
            waveform = torch.FloatTensor(segment).unsqueeze(0).to(self.device)
            embedding = self(waveform)
            embeddings.append(embedding.cpu().numpy())

        # Объединение результатов и сжатие размерности batch
        return np.vstack([emb.squeeze(0) for emb in embeddings])

    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> dict[str, float]:
        """
        Сравнивает два эмбеддинга и возвращает метрики сходства.

        Args:
            emb1: Первый эмбеддинг
            emb2: Второй эмбеддинг

        Returns:
            Словарь с метриками сходства
        """
        # Косинусное сходство
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        # Евклидово расстояние (нормализованное)
        euclidean_dist = np.linalg.norm(emb1 - emb2)
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)

        # PLDA-подобная оценка (масштабирование косинусного сходства)
        # В реальной системе здесь может быть настоящая PLDA
        plda_score = 2.0 * cosine_sim - 1.0

        # Вероятность того же диктора
        probability = 1.0 / (1.0 + np.exp(-5.0 * plda_score))

        # Уверенность
        confidence = abs(probability - 0.5) * 2.0

        return {
            "cosine_similarity": float(cosine_sim),
            "euclidean_similarity": float(euclidean_sim),
            "plda_score": float(plda_score),
            "probability": float(probability),
            "confidence": float(confidence),
            "is_same_speaker": bool(probability >= 0.5)
        }

    def compare_segments(self, segments1: list[np.ndarray], segments2: list[np.ndarray]) -> dict[
        str, float | list[float]]:
        """
        Сравнивает наборы сегментов от двух дикторов.

        Args:
            segments1: Список сегментов первого диктора
            segments2: Список сегментов второго диктора

        Returns:
            Словарь с метриками сходства
        """
        # Извлечение эмбеддингов
        embs1 = self.extract_batch(segments1)
        embs2 = self.extract_batch(segments2)

        # Проверка на пустые наборы
        if embs1.shape[0] == 0 or embs2.shape[0] == 0:
            return {
                "mean_cosine_similarity": 0.0,
                "max_cosine_similarity": 0.0,
                "min_cosine_similarity": 0.0,
                "std_cosine_similarity": 0.0,
                "plda_score": 0.0,
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

        # PLDA-подобная оценка
        plda_score = 2.0 * mean_sim - 1.0

        # Вероятность того же диктора
        probability = 1.0 / (1.0 + np.exp(-5.0 * plda_score))

        # Уверенность
        confidence = abs(probability - 0.5) * 2.0

        return {
            "mean_cosine_similarity": mean_sim,
            "max_cosine_similarity": max_sim,
            "min_cosine_similarity": min_sim,
            "std_cosine_similarity": std_sim,
            "all_similarities": similarities.tolist(),
            "plda_score": float(plda_score),
            "probability": float(probability),
            "confidence": float(confidence),
            "is_same_speaker": bool(probability >= 0.5)
        }


def get_xvector() -> Callable:
    """
    Возвращает функцию для извлечения X-vector эмбеддингов.

    Returns:
        Функция для извлечения эмбеддингов
    """
    model = EnhancedXVector()
    model.load_model()

    def extract(tensor: torch.Tensor) -> torch.Tensor:
        """
        Извлекает X-vector эмбеддинг из аудиосигнала.

        Args:
            tensor: Тензор с аудиоданными формы [1, samples]

        Returns:
            Тензор с эмбеддингом формы [1, embedding_dim]
        """
        return model(tensor)

    return extract
