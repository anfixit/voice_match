from speechbrain.pretrained import EncoderClassifier
import torch
import numpy as np
from typing import Dict, List, Union
from app.log import setup_logger

log = setup_logger("ecapa_model")


class EnhancedEcapa:
    """Расширенная версия ECAPA-TDNN с дополнительными возможностями для судебной экспертизы"""

    def __init__(self, device: str = "cpu"):
        """
        Инициализирует улучшенную модель ECAPA-TDNN.

        Args:
            device: Устройство для вычислений (cpu/cuda)
        """
        self.device = torch.device(device)
        self.model = None
        self.embedding_dim = 192  # Размерность ECAPA-TDNN эмбеддинга
        self.confidence_threshold = 0.85  # Порог уверенного сравнения

    def load_model(self):
        """Загружает предобученную модель ECAPA-TDNN"""
        try:
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/ecapa",
                run_opts={"device": self.device}
            )
            log.info("Модель ECAPA-TDNN успешно загружена")
        except Exception as e:
            log.error(f"Ошибка при загрузке модели ECAPA-TDNN: {e}")
            raise RuntimeError(f"Не удалось загрузить модель ECAPA-TDNN: {e}")

    def encode_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Извлекает эмбеддинги из аудиосигналов.

        Args:
            waveforms: Тензор с аудиоданными формы [batch_size, samples]

        Returns:
            Тензор с эмбеддингами формы [batch_size, embedding_dim]
        """
        if self.model is None:
            self.load_model()

        with torch.no_grad():
            # Убеждаемся, что тензор на правильном устройстве
            if waveforms.device != self.device:
                waveforms = waveforms.to(self.device)

            # Извлечение эмбеддингов
            embeddings = self.model.encode_batch(waveforms)

            # Нормализация эмбеддингов для косинусного сравнения
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings

    def encode_segments(self, segments: List[np.ndarray]) -> torch.Tensor:
        """
        Извлекает эмбеддинги из нескольких сегментов аудио.

        Args:
            segments: Список сегментов аудиоданных

        Returns:
            Тензор с эмбеддингами формы [num_segments, embedding_dim]
        """
        if not segments:
            return torch.zeros((0, self.embedding_dim), device=self.device)

        # Преобразование numpy arrays в тензоры
        waveforms = [torch.FloatTensor(segment).unsqueeze(0) for segment in segments]

        # Обработка каждого сегмента отдельно
        embeddings = []
        for waveform in waveforms:
            embedding = self.encode_batch(waveform)
            embeddings.append(embedding.squeeze(0))

        # Объединение результатов
        return torch.stack(embeddings)

    def compare_embeddings(self, emb1: torch.Tensor, emb2: torch.Tensor) -> Dict[str, float]:
        """
        Сравнивает два эмбеддинга и возвращает детальные метрики сходства.

        Args:
            emb1: Первый эмбеддинг
            emb2: Второй эмбеддинг

        Returns:
            Словарь с метриками сходства
        """
        # Косинусное сходство
        cosine_sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()

        # Евклидово расстояние (нормализованное)
        euclidean_dist = torch.norm(emb1 - emb2).item()
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)

        # Пороговая вероятность (sigmoid масштабирование)
        # Косинусное сходство -> [0, 1]
        probability = 1.0 / (1.0 + torch.exp(-12.0 * (torch.tensor(cosine_sim) - 0.5))).item()

        # Уверенность в решении
        confidence = abs(probability - 0.5) * 2.0  # [0, 1]

        # Определение класса (тот же / разные дикторы)
        is_same_speaker = probability >= 0.5

        return {
            "cosine_similarity": cosine_sim,
            "euclidean_similarity": euclidean_sim,
            "probability": probability,
            "confidence": confidence,
            "is_same_speaker": is_same_speaker,
            "decision_reliable": confidence >= self.confidence_threshold
        }

    def compare_embeddings_multiple(self, embs1: torch.Tensor, embs2: torch.Tensor) -> Dict[
        str, Union[float, List[float]]]:
        """
        Сравнивает два набора эмбеддингов (от нескольких сегментов).

        Args:
            embs1: Набор первых эмбеддингов [num_segments, embedding_dim]
            embs2: Набор вторых эмбеддингов [num_segments, embedding_dim]

        Returns:
            Словарь с метриками сходства и их статистикой
        """
        # Убеждаемся, что у нас есть эмбеддинги для сравнения
        if embs1.shape[0] == 0 or embs2.shape[0] == 0:
            return {
                "mean_cosine_similarity": 0.0,
                "median_cosine_similarity": 0.0,
                "max_cosine_similarity": 0.0,
                "min_cosine_similarity": 0.0,
                "std_cosine_similarity": 0.0,
                "all_similarities": [],
                "probability": 0.0,
                "confidence": 0.0,
                "is_same_speaker": False,
                "decision_reliable": False
            }

        # Вычисление всех косинусных сходств попарно
        cosine_similarities = []

        for i in range(embs1.shape[0]):
            for j in range(embs2.shape[0]):
                sim = torch.nn.functional.cosine_similarity(
                    embs1[i].unsqueeze(0),
                    embs2[j].unsqueeze(0)
                ).item()
                cosine_similarities.append(sim)

        # Конвертация в numpy для статистических вычислений
        similarities = np.array(cosine_similarities)

        # Статистика
        mean_sim = float(np.mean(similarities))
        median_sim = float(np.median(similarities))
        max_sim = float(np.max(similarities))
        min_sim = float(np.min(similarities))
        std_sim = float(np.std(similarities))

        # Вероятность того же диктора
        probability = 1.0 / (1.0 + np.exp(-12.0 * (mean_sim - 0.5)))

        # Уверенность в решении
        confidence = float(abs(probability - 0.5) * 2.0)

        # Определение класса
        is_same_speaker = probability >= 0.5

        # Надежность решения
        decision_reliable = confidence >= self.confidence_threshold

        return {
            "mean_cosine_similarity": mean_sim,
            "median_cosine_similarity": median_sim,
            "max_cosine_similarity": max_sim,
            "min_cosine_similarity": min_sim,
            "std_cosine_similarity": std_sim,
            "all_similarities": cosine_similarities,
            "probability": float(probability),
            "confidence": confidence,
            "is_same_speaker": bool(is_same_speaker),
            "decision_reliable": decision_reliable
        }

    def get_embedding_layer_outputs(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Извлекает промежуточные признаки из разных слоев ECAPA-TDNN.
        Полезно для более глубокого анализа и интерпретации.

        Args:
            waveform: Тензор с аудиоданными формы [1, samples]

        Returns:
            Словарь с признаками из разных слоев
        """
        if self.model is None:
            self.load_model()

        if waveform.device != self.device:
            waveform = waveform.to(self.device)

        # Получение промежуточных активаций требует доступа к внутренностям модели
        # Это зависит от конкретной реализации SpeechBrain, поэтому
        # предоставляем упрощенный интерфейс с основными слоями
        try:
            with torch.no_grad():
                # Извлечение признаков
                feats = self.model.mods.compute_features(waveform)

                # Нормализация
                feats = self.model.mods.mean_var_norm(feats, torch.ones(1).to(self.device))

                # Извлечение промежуточных признаков
                layer_outputs = {}

                # ECAPA модель
                x = self.model.mods.encoder.conv1(feats)
                layer_outputs["conv1"] = x.detach().clone()

                x = self.model.mods.encoder.res2(x)
                layer_outputs["res2"] = x.detach().clone()

                x = self.model.mods.encoder.res3(x)
                layer_outputs["res3"] = x.detach().clone()

                x = self.model.mods.encoder.res4(x)
                layer_outputs["res4"] = x.detach().clone()

                x = self.model.mods.encoder.res5(x)
                layer_outputs["res5"] = x.detach().clone()

                # Финальный эмбеддинг
                embeddings = self.model.encode_batch(waveform)
                layer_outputs["embedding"] = embeddings.detach().clone()

                return layer_outputs

        except Exception as e:
            log.error(f"Ошибка при извлечении промежуточных признаков: {e}")
            return {"embedding": self.encode_batch(waveform)}


def get_ecapa(device: str = "cpu") -> EnhancedEcapa:
    """
    Возвращает экземпляр улучшенной модели ECAPA-TDNN.

    Args:
        device: Устройство для вычислений (cpu/cuda)

    Returns:
        Экземпляр EnhancedEcapa
    """
    model = EnhancedEcapa(device=device)
    model.load_model()
    return model
