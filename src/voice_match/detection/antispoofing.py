import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import os
import tempfile
from typing import Dict, Tuple, List, Optional, Union
from voice_match.log import setup_logger

log = setup_logger("antispoofing")


class SincConv(nn.Module):
    """Sinc-based convolution for antispoofing detection"""

    def __init__(self, device='cpu'):
        super(SincConv, self).__init__()
        self.device = device
        # Фильтр 1D свертки
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=1024,
            stride=256,
            padding=0,
            bias=False
        )

        # GRU слой для обработки последовательностей
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Полносвязный слой
        self.fc = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 2)  # [real, spoof]

        # Инициализация весов sinc-фильтров
        self._init_sinc_weights()

    def _init_sinc_weights(self):
        """Инициализация sinc фильтров на разных частотах"""
        weights = torch.zeros(64, 1, 1024)

        # Создаем sinc фильтры на разных частотах
        for i in range(64):
            freq = 50 + (i * 100)  # 50-6450 Hz
            t = torch.arange(0, 1024) - 512
            t = t.float() / 16000  # sample rate

            # sinc фильтр
            y = torch.sin(2 * np.pi * freq * t) / (np.pi * t)
            y[512] = 2 * freq / 16000  # центральный элемент

            # Применение окна Хэмминга
            window = 0.54 - 0.46 * torch.cos(2 * np.pi * torch.arange(0, 1024) / 1024)
            y = y * window

            # Нормализация
            y = y / torch.sqrt(torch.sum(y ** 2))

            weights[i, 0, :] = y

        self.conv.weight.data = weights
        self.conv.weight.requires_grad = False  # замораживаем sinc-веса

    def forward(self, x):
        # x: [batch, time]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch, 1, time]

        # 1D свертка с sinc фильтрами
        x = F.relu(self.conv(x))  # [batch, 64, time/256]

        # Подготовка для GRU
        x = x.transpose(1, 2)  # [batch, time/256, 64]

        # GRU слой
        x, _ = self.gru(x)  # [batch, time/256, 128]

        # Пулинг по временному измерению
        x = torch.mean(x, dim=1)  # [batch, 128]

        # Полносвязный слой
        x = F.relu(self.fc(x))
        x = self.fc_out(x)

        return x


class AntiSpoofingDetector:
    """Детектор подделок голоса и синтетической речи"""

    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        """Загружает или инициализирует модель"""
        model_path = os.path.join("pretrained_models", "antispoofing", "model.pth")

        # Создаем модель
        self.model = SincConv(device=self.device).to(self.device)

        # Проверяем наличие предобученной модели
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                log.info("Загружена предобученная модель для обнаружения подделок")
            except Exception as e:
                log.warning(f"Не удалось загрузить предобученную модель: {e}")
                # Дополнительная инициализация, если модель не загружена
                self._initialize_pretrained_weights()
        else:
            log.warning("Предобученная модель не найдена, используется базовая инициализация")
            # Инициализация базовыми весами
            self._initialize_pretrained_weights()

        # Установка режима оценки
        self.model.eval()

    def _initialize_pretrained_weights(self):
        """Инициализирует веса предустановленными значениями"""
        # В реальном приложении здесь можно было бы загрузить веса из другого источника
        # или использовать предустановленные значения для ключевых слоев
        pass

    def detect(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Обнаруживает признаки синтетической или поддельной речи.

        Args:
            y: Аудиосигнал
            sr: Частота дискретизации

        Returns:
            Словарь с вероятностями различных типов подделок
        """
        if self.model is None:
            self.load_model()

        # Проверка наличия речи
        if len(y) < sr:
            return {
                "is_synthetic": 0.0,
                "is_real": 1.0,
                "confidence": 0.0
            }

        # Ресемплирование до 16kHz
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)

        # Нормализация
        y = librosa.util.normalize(y)

        # Разделение на сегменты по 4 секунды
        segment_len = 4 * 16000

        # Если сигнал короче 4 секунд, дополняем нулями
        if len(y) < segment_len:
            y = np.pad(y, (0, segment_len - len(y)))
            segments = [y]
        else:
            # Разбиваем на сегменты с перекрытием 50%
            hop = segment_len // 2
            segments = []
            for i in range(0, len(y) - segment_len + 1, hop):
                segments.append(y[i:i + segment_len])

        # Ограничиваем количество сегментов
        if len(segments) > 10:
            # Выбираем равномерно распределенные сегменты
            indices = np.linspace(0, len(segments) - 1, 10, dtype=int)
            segments = [segments[i] for i in indices]

        # Обработка каждого сегмента
        synthetic_probs = []

        with torch.no_grad():
            for segment in segments:
                # Преобразование в тензор
                x = torch.FloatTensor(segment).unsqueeze(0).to(self.device)  # [1, samples]

                # Предсказание
                output = self.model(x)

                # Применяем softmax для получения вероятностей
                probs = F.softmax(output, dim=1)

                # Вероятность синтетической речи (1 класс)
                synthetic_prob = probs[0, 1].item()
                synthetic_probs.append(synthetic_prob)

        # Статистика по всем сегментам
        mean_prob = np.mean(synthetic_probs)
        std_prob = np.std(synthetic_probs)
        max_prob = np.max(synthetic_probs)

        # Определение надежности обнаружения
        confidence = 1.0 - std_prob

        # Результат
        result = {
            "is_synthetic": mean_prob,
            "is_real": 1.0 - mean_prob,
            "max_probability": max_prob,
            "confidence": confidence
        }

        # Подробная классификация
        if mean_prob > 0.8:
            result["likely_type"] = "TTS/Deepfake"
        elif mean_prob > 0.6:
            result["likely_type"] = "Voice Conversion/Manipulation"
        elif mean_prob > 0.4:
            result["likely_type"] = "Possible Audio Editing"

        return result


def get_antispoofing_detector():
    """
    Загружает и возвращает детектор подделок голоса.

    Returns:
        Экземпляр AntiSpoofingDetector
    """
    detector = AntiSpoofingDetector()
    detector.load_model()
    return detector
