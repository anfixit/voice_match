"""
Модуль формантного анализа для voice_match.

Форманты - это резонансные частоты голосового тракта,
уникальные биометрические характеристики человека.

Компоненты:
- formant_core: ядро LPC-анализа и извлечения формант
- formant_constraints: проверка физиологической корректности
- formant_comparison: сравнение формант двух голосов
- formant_statistics: статистический анализ формант
- formant_visualization: визуализация формантных треков
"""

from models.formant.formant_core import FormantExtractor
from models.formant.formant_constraints import FormantConstraints
from models.formant.formant_comparison import FormantComparator
from models.formant.formant_statistics import FormantStatistics
from models.formant.formant_visualization import FormantVisualizer

__all__ = [
    "FormantExtractor",
    "FormantConstraints",
    "FormantComparator",
    "FormantStatistics",
    "FormantVisualizer",
]
