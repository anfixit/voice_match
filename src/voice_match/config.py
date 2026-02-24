"""
Центральный файл конфигурации для voice_match.

Содержит все настройки приложения, пути к моделям, параметры обработки.
"""

import os
from pathlib import Path

# Базовые пути
BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "pretrained_models"
TEMP_DIR = BASE_DIR / "tmp"

# Создаём директории если их нет
LOGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Аудио параметры
SAMPLE_RATE = 16000
AUDIO_FORMATS = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
MAX_FILE_SIZE_MB = 50
MIN_AUDIO_DURATION = 3.0
MAX_AUDIO_DURATION = 300.0

# Параметры сегментации
SEGMENT_COUNT = 5
SEGMENT_DURATION = 5.0
MIN_SPEECH_RATIO = 0.4

# Параметры моделей
MODEL_PATHS = {
    "ecapa": MODELS_DIR / "ecapa",
    "xvector": MODELS_DIR / "xvector",
    "yamnet": MODELS_DIR / "yamnet",
    "resemblyzer": MODELS_DIR / "resemblyzer",
    "antispoofing": MODELS_DIR / "antispoofing",
}

# Веса моделей для итоговой оценки
MODEL_WEIGHTS = {
    "ecapa": 1.5,
    "xvector": 1.5,
    "resemblyzer": 1.2,
    "formant": 1.0,
    "yamnet": 0.8,
    "voice_features": 1.0,
}

# Пороги для вердиктов
THRESHOLDS = {
    "very_high": 0.95,
    "high": 0.88,
    "probable": 0.80,
    "similar": 0.70,
    "different": 0.70,
}

# Параметры формантного анализа
FORMANT_CONFIG = {
    "n_formants": 4,
    "max_frequency": 5500,
    "window_length": 0.025,
    "pre_emphasis": 0.97,
}

# Параметры логирования
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "app.log"

# Параметры веб-интерфейса
GRADIO_SERVER_NAME = "0.0.0.0"
GRADIO_SERVER_PORT = 7860
GRADIO_SHARE = False
GRADIO_DEBUG = True

# Параметры Flask/FastAPI (для будущего фронтенда)
API_HOST = "0.0.0.0"
API_PORT = 8000
API_DEBUG = True
API_WORKERS = 4

# Параметры безопасности
ALLOWED_HOSTS = ["localhost", "127.0.0.1", "voice-match.ru"]
MAX_REQUESTS_PER_MINUTE = 10

# Параметры обработки
USE_GPU = False
NUM_THREADS = 4
BATCH_SIZE = 8

# Параметры отчётов
REPORT_TEMPLATE = "expert_report.html"
INCLUDE_VISUALIZATIONS = True
INCLUDE_TECHNICAL_DETAILS = True

# Параметры детекции подделок
ANTISPOOFING_ENABLED = True
MODIFICATION_DETECTION_ENABLED = True
CONFIDENCE_THRESHOLD = 0.7

# Переменные окружения
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
DATABASE_URL = os.getenv("DATABASE_URL", None)
REDIS_URL = os.getenv("REDIS_URL", None)
