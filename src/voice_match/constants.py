"""Все константы проекта voice_match.

Единственный источник правды для числовых параметров,
порогов, диапазонов и конфигурации.
"""

from pathlib import Path

# ── Пути ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = BASE_DIR / 'logs'
MODELS_DIR = BASE_DIR / 'pretrained_models'
TEMP_DIR = BASE_DIR / 'tmp'
WEIGHTS_PATH = BASE_DIR / 'weights.json'

# ── Аудио ────────────────────────────────────────
SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
SUPPORTED_EXTENSIONS = frozenset(
    {'.wav', '.mp3', '.m4a', '.flac', '.ogg'},
)
MAX_FILE_SIZE_MB = 20
MIN_AUDIO_DURATION = 3.0
MAX_AUDIO_DURATION = 300.0

# ── Сегментация ──────────────────────────────────
SEGMENT_COUNT = 5
SEGMENT_DURATION = 30.0
MIN_SPEECH_RATIO = 0.4

# ── VAD (Voice Activity Detection) ───────────────
VAD_FRAME_MS = 30
VAD_SPEECH_THRESHOLD = 0.15

# ── Спектральный анализ ──────────────────────────
FRAME_LENGTH = 2048
HOP_LENGTH = 512
FRAME_DURATION_S = 0.025
HOP_DURATION_S = 0.010
PRE_EMPHASIS = 0.97
PITCH_FMIN = 50
PITCH_FMAX = 400

# ── Аугментация ──────────────────────────────────
BANDPASS_MIN_CENTER_FREQ = 100
BANDPASS_MAX_CENTER_FREQ = 7500

# ── Форманты ─────────────────────────────────────
FORMANT_COUNT = 4
FORMANT_MAX_FREQUENCY = 5500
FORMANT_LIMITS: dict[str, tuple[int, int]] = {
    'F1': (100, 1_000),
    'F2': (800, 2_500),
    'F3': (1_500, 3_500),
    'F4': (3_000, 5_000),
}

# ── Джиттер / Шиммер ────────────────────────────
JITTER_NORMAL_RANGE = (0.0, 1.04)
SHIMMER_NORMAL_RANGE = (0.0, 3.81)

# ── Диапазоны основного тона (F0) ────────────────
PITCH_RANGE_MALE = (50, 180)
PITCH_RANGE_FEMALE = (150, 300)

# ── Детекция модификаций голоса ───────────────────
PITCH_JUMP_THRESHOLD_SEMITONES = 5.0
ROBOT_VOICE_STD_THRESHOLD = 0.1
ROBOT_VOICE_RATIO = 0.5
VOICE_MODIFIERS: dict[str, tuple[float, float] | dict] = {
    'pitch_shift': (-12, 12),
    'formant_shift': (0.5, 2.0),
    'robot': {'harmonic_intensity': (0.7, 1.0)},
}

# ── Антиспуфинг ──────────────────────────────────
SINC_FILTER_COUNT = 64
SINC_FILTER_LENGTH = 1024
SINC_BASE_FREQ = 50
SINC_FREQ_STEP = 100

# ── Весовые коэффициенты (по умолчанию) ──────────
DEFAULT_WEIGHTS: dict[str, float] = {
    'ecapa': 1.5,
    'xvec': 1.5,
    'res': 1.2,
    'formant': 1.0,
    'formant_dynamics': 1.5,
    'fricative': 1.2,
    'nasal': 1.3,
    'vocal_tract': 1.8,
    'jitter_shimmer': 1.1,
    'yamnet': 0.8,
}

# ── Пороги вердиктов ────────────────────────────
CONFIDENCE_LEVEL = 0.95
THRESHOLDS: dict[str, float] = {
    'very_high': 0.95,
    'high': 0.88,
    'probable': 0.80,
    'similar': 0.70,
    'different': 0.70,
}

# ── Антиспуфинг / детекция ───────────────────────
ANTISPOOFING_ENABLED = True
MODIFICATION_DETECTION_ENABLED = True
CONFIDENCE_THRESHOLD = 0.7

# ── Пути к предобученным моделям ─────────────────
MODEL_PATHS: dict[str, Path] = {
    'ecapa': MODELS_DIR / 'ecapa',
    'xvector': MODELS_DIR / 'xvector',
    'yamnet': MODELS_DIR / 'yamnet',
    'resemblyzer': MODELS_DIR / 'resemblyzer',
    'antispoofing': MODELS_DIR / 'antispoofing',
}

# ── Gradio UI ────────────────────────────────────
GRADIO_SERVER_NAME = '0.0.0.0'
GRADIO_SERVER_PORT = 7860

# ── Производительность ───────────────────────────
USE_GPU = False
NUM_THREADS = 4

# ── Отчёты ───────────────────────────────────────
INCLUDE_VISUALIZATIONS = True
INCLUDE_TECHNICAL_DETAILS = True
