
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from voice_match.log import setup_logger

log = setup_logger("yamnet_model")


class EnhancedYAMNet:
    """Улучшенная модель YAMNet с дополнительными функциями для анализа восприятия звука"""

    def __init__(self):
        """Инициализирует улучшенную модель YAMNet"""
        self.model = None
        self.class_names = None
        self.embedding_dim = 1024  # Размерность YAMNet эмбеддинга
        self.sample_rate = 16000  # Частота дискретизации для YAMNet

    def load_model(self):
        """Загружает предобученную модель YAMNet"""
        try:
            self.model = hub.load("https://tfhub.dev/google/yamnet/1")

            # Загрузка имен классов
            class_map_path = tf.keras.utils.get_file(
                "yamnet_class_map.csv",
                "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
            )
            with open(class_map_path) as f:
                self.class_names = [line.strip().split(",")[2] for line in f]

            log.info("Модель YAMNet успешно загружена")
        except Exception as e:
            log.error(f"Ошибка при загрузке модели YAMNet: {e}")
            raise RuntimeError(f"Не удалось загрузить модель YAMNet: {e}") from e

    def __call__(self, waveform: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Прямой вызов модели YAMNet для получения предсказаний.

        Args:
            waveform: Тензор с аудиоданными

        Returns:
            Кортеж из scores, embeddings, log_mel_spectrogram
        """
        if self.model is None:
            self.load_model()

        return self.model(waveform)

    def extract_embedding(self, wav: np.ndarray) -> np.ndarray:
        """
        Извлекает YAMNet эмбеддинг из аудиосигнала.

        Args:
            wav: Аудиосигнал в формате numpy array

        Returns:
            YAMNet эмбеддинг (усредненный по всем фреймам)
        """
        if self.model is None:
            self.load_model()

        # Проверка и преобразование частоты дискретизации
        if len(wav.shape) != 1:
            wav = np.mean(wav, axis=0)

        # Нормализация аудио
        wav = librosa.util.normalize(wav)

        # Преобразование в тензор
        waveform = tf.convert_to_tensor(wav, dtype=tf.float32)

        # Получение эмбеддингов
        _, embeddings, _ = self.model(waveform)

        # Усреднение по времени
        embedding = tf.reduce_mean(embeddings, axis=0)

        return embedding.numpy()

    def extract_with_timestamps(self, wav: np.ndarray, sr: int = 16000) -> dict[str, np.ndarray | list[float]]:
        """
        Извлекает YAMNet эмбеддинги и классы с временными метками.

        Args:
            wav: Аудиосигнал в формате numpy array
            sr: Частота дискретизации входного сигнала

        Returns:
            Словарь с эмбеддингами, классами и временными метками
        """
        if self.model is None:
            self.load_model()

        # Ресемплирование, если нужно
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)

        # Нормализация
        wav = librosa.util.normalize(wav)

        # Преобразование в тензор
        waveform = tf.convert_to_tensor(wav, dtype=tf.float32)

        # Получение предсказаний
        scores, embeddings, _log_mel_spectrogram = self.model(waveform)

        # Преобразование в numpy для дальнейшей обработки
        scores_np = scores.numpy()
        embeddings_np = embeddings.numpy()

        # Создание временных меток
        # YAMNet использует перекрывающиеся фреймы по 0.96 секунд с шагом 0.48 секунд
        hop_duration = 0.48
        timestamps = [i * hop_duration for i in range(len(scores_np))]

        # Получение основных классов для каждого фрейма
        top_classes = []

        for i in range(len(scores_np)):
            top_class_indices = np.argsort(scores_np[i])[-3:][::-1]  # Топ-3 классов
            top_classes_frame = [
                {"class": self.class_names[idx], "score": float(scores_np[i, idx])}
                for idx in top_class_indices
            ]
            top_classes.append(top_classes_frame)

        return {
            "embeddings": embeddings_np,
            "top_classes": top_classes,
            "timestamps": timestamps,
            "mean_embedding": np.mean(embeddings_np, axis=0)
        }

    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> dict[str, float]:
        """
        Сравнивает два YAMNet эмбеддинга.

        Args:
            emb1: Первый YAMNet эмбеддинг
            emb2: Второй YAMNet эмбеддинг

        Returns:
            Словарь с метриками сходства
        """
        # Косинусное сходство
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        # Евклидово расстояние (нормализованное)
        euclidean_dist = np.linalg.norm(emb1 - emb2)
        euclidean_sim = 1.0 / (1.0 + euclidean_dist / 10.0)  # Нормализация для YAMNet

        # Коэффициент корреляции (измеряет сходство паттернов)
        correlation = np.corrcoef(emb1, emb2)[0, 1]

        return {
            "cosine_similarity": float(cosine_sim),
            "euclidean_similarity": float(euclidean_sim),
            "correlation": float(correlation if not np.isnan(correlation) else 0.0)
        }

    def compare_segments(self, segments1: list[np.ndarray], segments2: list[np.ndarray]) -> dict[
        str, float | list[float]]:
        """
        Сравнивает наборы сегментов от двух записей.

        Args:
            segments1: Список сегментов первой записи
            segments2: Список сегментов второй записи

        Returns:
            Словарь с метриками сходства
        """
        # Извлечение эмбеддингов
        embeddings1 = [self.extract_embedding(segment) for segment in segments1]
        embeddings2 = [self.extract_embedding(segment) for segment in segments2]

        # Проверка на пустые наборы
        if not embeddings1 or not embeddings2:
            return {
                "mean_cosine_similarity": 0.0,
                "max_cosine_similarity": 0.0,
                "mean_correlation": 0.0,
                "acoustic_environment_match": 0.0
            }

        # Среднее по сегментам
        mean_emb1 = np.mean(embeddings1, axis=0)
        mean_emb2 = np.mean(embeddings2, axis=0)

        # Сравнение средних эмбеддингов
        comparison = self.compare_embeddings(mean_emb1, mean_emb2)

        # Вычисление всех попарных сходств
        all_similarities = []
        all_correlations = []

        for emb1 in embeddings1:
            for emb2 in embeddings2:
                pair_comparison = self.compare_embeddings(emb1, emb2)
                all_similarities.append(pair_comparison["cosine_similarity"])
                all_correlations.append(pair_comparison["correlation"])

        # Детекция акустической среды
        acoustic_match = comparison["cosine_similarity"]

        # Для применения в общей биометрической модели: мера сходства акустической среды
        # Если среды слишком разные, это может указывать на разные источники записи
        acoustic_environment_match = acoustic_match

        return {
            "mean_cosine_similarity": comparison["cosine_similarity"],
            "max_cosine_similarity": float(np.max(all_similarities)),
            "mean_correlation": float(np.mean(all_correlations)),
            "acoustic_environment_match": acoustic_environment_match,
            "all_similarities": all_similarities
        }

    def detect_voice_masking(self, wav: np.ndarray) -> dict[str, float]:
        """
        Обнаруживает признаки маскировки голоса или искусственной среды.

        Args:
            wav: Аудиосигнал в формате numpy array

        Returns:
            Словарь с вероятностями различных типов маскировки
        """
        if self.model is None:
            self.load_model()

        # Нормализация
        wav = librosa.util.normalize(wav)

        # Получение классов YAMNet
        result = self.extract_with_timestamps(wav)
        np.array([frame[0]["score"] for frame in result["top_classes"]])
        top_classes = [frame[0]["class"] for frame in result["top_classes"]]

        # Поиск специфических звуков, указывающих на маскировку
        masking_indicators = {
            "synthetic": ["Synthesizer", "Theremin", "Synthesized voice", "Speech synthesizer"],
            "distorted": ["Distortion", "Echo", "Reverberation", "Noise"],
            "mechanical": ["Mechanical fan", "Engine", "Machinery", "Machine"],
            "telephone": ["Telephone", "Telephone dialing", "Ringtone", "Dial tone"]
        }

        # Подсчет встречаемости каждого типа индикатора
        indicator_counts = {
            mask_type: sum(1 for cls in top_classes if any(ind in cls for ind in indicators))
            for mask_type, indicators in masking_indicators.items()
        }

        # Нормализация к количеству кадров
        total_frames = len(top_classes)
        masking_probabilities = {
            mask_type: min(1.0, count / max(1, total_frames) * 3.0)  # Умножаем на 3 для усиления сигнала
            for mask_type, count in indicator_counts.items()
        }

        # Общая вероятность маскировки
        masking_probabilities["overall"] = min(
            1.0,
            sum(masking_probabilities.values()) / len(masking_probabilities)
        )

        return masking_probabilities


def get_yamnet() -> EnhancedYAMNet:
    """
    Возвращает экземпляр улучшенной модели YAMNet.

    Returns:
        Экземпляр EnhancedYAMNet
    """
    model = EnhancedYAMNet()
    model.load_model()
    return model
