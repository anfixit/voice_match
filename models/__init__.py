"""
Пакет моделей для анализа голоса.

Содержит модели машинного обучения и алгоритмы извлечения признаков
для биометрического сравнения голосов.
"""

from models.ecapa import get_ecapa, EnhancedEcapa
from models.xvector import get_xvector, EnhancedXVector
from models.resemblyzer import get_resemblyzer, EnhancedResemblyzer
from models.yamnet import get_yamnet, EnhancedYAMNet
from models.antispoofing import get_antispoofing_detector, AntiSpoofingDetector
from models.formant_tracker import get_formant_tracker, FormantTracker
from models.voice_features import VoiceFeatureExtractor, get_voice_feature_extractor

__all__ = [
    "get_ecapa",
    "EnhancedEcapa",
    "get_xvector",
    "EnhancedXVector",
    "get_resemblyzer",
    "EnhancedResemblyzer",
    "get_yamnet",
    "EnhancedYAMNet",
    "get_antispoofing_detector",
    "AntiSpoofingDetector",
    "get_formant_tracker",
    "FormantTracker",
    "VoiceFeatureExtractor",
    "get_voice_feature_extractor",
]
