#!/bin/bash
# ============================================================
# Шаг 2: Реорганизация структуры проекта voice_match
# Запускать из корня проекта: bash restructure.sh
# ============================================================
set -e

echo "🔧 Начинаем реорганизацию структуры проекта..."
echo ""

# ── 1. Создаём новую структуру каталогов ──
echo "📁 Создаём каталоги..."
mkdir -p src/voice_match/models/formant
mkdir -p src/voice_match/features
mkdir -p src/voice_match/scoring
mkdir -p src/voice_match/services
mkdir -p src/voice_match/detection
mkdir -p src/voice_match/ui

# ── 2. Перемещаем ядро из app/ ──
echo "📦 Перемещаем app/ → src/voice_match/..."

# Основная логика сравнения → services/
cp app/voice_compare_dual.py src/voice_match/services/comparison.py

# Утилиты обработки аудио → services/
cp app/utils.py src/voice_match/services/preprocessing.py

# Интерфейс → ui/
cp app/interface.py src/voice_match/ui/interface.py

# Визуализация из app/ → ui/
cp app/visualize.py src/voice_match/ui/visualization.py 2>/dev/null || touch src/voice_match/ui/visualization.py

# Логгер → корень пакета
cp app/log.py src/voice_match/log.py

# Константы → корень пакета
cp app/constants.py src/voice_match/constants.py

# modifiers из app/ (пустой) → detection/
cp app/modifiers.py src/voice_match/detection/modifiers.py 2>/dev/null || touch src/voice_match/detection/modifiers.py

# report из app/ → services/
cp app/report.py src/voice_match/services/report_legacy.py 2>/dev/null || touch src/voice_match/services/report_legacy.py

# ── 3. Перемещаем модели из models/ ──
echo "🧠 Перемещаем models/ → src/voice_match/models/..."

# Нейросетевые модели
cp models/ecapa.py src/voice_match/models/ecapa.py
cp models/xvector.py src/voice_match/models/xvector.py
cp models/resemblyzer.py src/voice_match/models/resemblyzer.py
cp models/yamnet.py src/voice_match/models/yamnet.py

# Формантный анализ — берём из models/formant/ (каноническая версия)
cp models/formant/formant_core.py src/voice_match/models/formant/core.py
cp models/formant/formant_constraints.py src/voice_match/models/formant/constraints.py
cp models/formant/formant_comparison.py src/voice_match/models/formant/comparison.py
cp models/formant/formant_statistics.py src/voice_match/models/formant/statistics.py
cp models/formant/formant_visualization.py src/voice_match/models/formant/visualization.py
cp models/formant_tracker.py src/voice_match/models/formant/tracker.py

# ── 4. Перемещаем features ──
echo "🎤 Перемещаем features → src/voice_match/features/..."
cp models/voice_features.py src/voice_match/features/voice_features.py
cp models/metadata.py src/voice_match/features/metadata.py

# ── 5. Перемещаем scoring ──
echo "📊 Перемещаем scoring → src/voice_match/scoring/..."
cp models/bayesian_scoring.py src/voice_match/scoring/bayesian.py
cp models/plda_scoring.py src/voice_match/scoring/plda.py

# ── 6. Перемещаем detection ──
echo "🔍 Перемещаем detection → src/voice_match/detection/..."
cp models/antispoofing.py src/voice_match/detection/antispoofing.py
cp models/modification_detector.py src/voice_match/detection/modification.py
cp models/nasal_analyzer.py src/voice_match/detection/nasal_analyzer.py
cp models/temporal_analyzer.py src/voice_match/detection/temporal_analyzer.py

# ── 7. Перемещаем report ──
echo "📋 Перемещаем report → src/voice_match/services/..."
cp models/report.py src/voice_match/services/report.py

# ── 8. Перемещаем config ──
echo "⚙️  Перемещаем config → src/voice_match/..."
cp config/settings.py src/voice_match/config.py

# ── 9. Перемещаем main.py ──
echo "🚀 Копируем main.py..."
cp main.py src/voice_match/__main__.py

# ── 10. Создаём __init__.py для всех пакетов ──
echo "📝 Создаём __init__.py..."

# Корневой __init__.py пакета
cat > src/voice_match/__init__.py << 'INIT'
"""voice_match — система экспертной верификации личности по голосу."""
INIT

# models/
cat > src/voice_match/models/__init__.py << 'INIT'
"""Нейросетевые модели для извлечения эмбеддингов."""
INIT

# models/formant/
cat > src/voice_match/models/formant/__init__.py << 'INIT'
"""Формантный анализ голосового тракта."""
INIT

# features/
cat > src/voice_match/features/__init__.py << 'INIT'
"""Извлечение голосовых биометрических признаков."""
INIT

# scoring/
cat > src/voice_match/scoring/__init__.py << 'INIT'
"""Методы скоринга и оценки сходства."""
INIT

# services/
cat > src/voice_match/services/__init__.py << 'INIT'
"""Бизнес-логика: сравнение, предобработка, отчёты."""
INIT

# detection/
cat > src/voice_match/detection/__init__.py << 'INIT'
"""Обнаружение подделок и модификаций голоса."""
INIT

# ui/
cat > src/voice_match/ui/__init__.py << 'INIT'
"""Пользовательский интерфейс (Gradio)."""
INIT

# ── 11. Обновляем main.py в корне как точку входа ──
echo "🎯 Создаём новый main.py (точка входа)..."
cat > main.py << 'MAIN'
"""Точка входа приложения voice_match."""

import sys
from pathlib import Path

# src/ должен быть в sys.path для импортов
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from voice_match.__main__ import main

if __name__ == '__main__':
    main()
MAIN

# ── 12. Обновляем __main__.py ──
echo "🔄 Обновляем __main__.py..."
cat > src/voice_match/__main__.py << 'MAINMOD'
"""Точка входа voice_match как модуля."""

import os
import sys
import importlib.util

from voice_match.log import setup_logger

log = setup_logger('main')


def check_environment() -> bool:
    """Проверяет наличие необходимых зависимостей."""
    try:
        required = [
            'librosa', 'gradio', 'pydub',
            'matplotlib', 'numpy', 'scipy',
        ]
        neural = [
            'torch', 'speechbrain', 'tensorflow',
            'tensorflow_hub', 'pyannote.audio',
        ]
        missing = [
            pkg for pkg in required
            if importlib.util.find_spec(pkg) is None
        ]
        missing_neural = [
            pkg for pkg in neural
            if importlib.util.find_spec(pkg) is None
        ]

        if missing:
            log.error(
                'Не установлены основные пакеты: %s',
                missing,
            )
            return False

        if missing_neural:
            log.warning(
                'Не установлены нейросетевые пакеты: %s',
                missing_neural,
            )

        return True

    except Exception:
        log.exception('Ошибка при проверке окружения')
        return False


def main() -> None:
    """Главная функция запуска приложения."""
    if not check_environment():
        log.critical(
            'Приложение не запущено из-за ошибок окружения.',
        )
        sys.exit(1)

    if importlib.util.find_spec('torch'):
        import torch
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        torch.set_num_threads(1)
        log.info('PyTorch настроен на CPU')

    from voice_match.ui.interface import launch_ui
    launch_ui()


if __name__ == '__main__':
    main()
MAINMOD

echo ""
echo "✅ Структура создана. Новое дерево src/voice_match/:"
echo ""
find src -type f | sort
echo ""
echo "⚠️  ВАЖНО: Старые каталоги app/, models/, config/ пока НЕ удалены."
echo "   Сначала убедись, что всё работает, потом удалим."
echo ""
echo "📋 Следующий шаг: обновить импорты во всех файлах в src/"
