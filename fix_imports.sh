#!/bin/bash
# ============================================================
# Шаг 2b: Обновление импортов в src/voice_match/
# Запускать из корня проекта: bash fix_imports.sh
# ============================================================
set -e

echo "🔧 Обновляем импорты в src/voice_match/..."
echo ""

# ── Утилита для замены через sed (macOS-совместимый) ──
# На macOS sed -i требует '' после -i
replace_in_file() {
    local file="$1"
    local old="$2"
    local new="$3"
    if grep -q "$old" "$file" 2>/dev/null; then
        sed -i '' "s|$old|$new|g" "$file"
        echo "  ✏️  $file: '$old' → '$new'"
    fi
}

# ══════════════════════════════════════════════════════
# 1. Все файлы: from app.log → from voice_match.log
# ══════════════════════════════════════════════════════
echo "📝 Замена: app.log → voice_match.log"
find src/voice_match -name '*.py' -exec grep -l 'from app\.log' {} \; | while read f; do
    replace_in_file "$f" "from app\.log" "from voice_match.log"
done
find src/voice_match -name '*.py' -exec grep -l 'from app.log' {} \; | while read f; do
    replace_in_file "$f" "from app.log" "from voice_match.log"
done

# ══════════════════════════════════════════════════════
# 2. from app.constants → from voice_match.constants
# ══════════════════════════════════════════════════════
echo "📝 Замена: app.constants → voice_match.constants"
find src/voice_match -name '*.py' -exec grep -l 'from app\.constants' {} \; | while read f; do
    replace_in_file "$f" "from app\.constants" "from voice_match.constants"
done
find src/voice_match -name '*.py' -exec grep -l 'from app.constants' {} \; | while read f; do
    replace_in_file "$f" "from app.constants" "from voice_match.constants"
done

# ══════════════════════════════════════════════════════
# 3. from app.utils → from voice_match.services.preprocessing
# ══════════════════════════════════════════════════════
echo "📝 Замена: app.utils → voice_match.services.preprocessing"
find src/voice_match -name '*.py' -exec grep -l 'from app\.utils' {} \; | while read f; do
    replace_in_file "$f" "from app\.utils" "from voice_match.services.preprocessing"
done
find src/voice_match -name '*.py' -exec grep -l 'from app.utils' {} \; | while read f; do
    replace_in_file "$f" "from app.utils" "from voice_match.services.preprocessing"
done

# ══════════════════════════════════════════════════════
# 4. from app.voice_compare_dual → from voice_match.services.comparison
# ══════════════════════════════════════════════════════
echo "📝 Замена: app.voice_compare_dual → voice_match.services.comparison"
find src/voice_match -name '*.py' -exec grep -l 'from app\.voice_compare_dual' {} \; | while read f; do
    replace_in_file "$f" "from app\.voice_compare_dual" "from voice_match.services.comparison"
done
find src/voice_match -name '*.py' -exec grep -l 'from app.voice_compare_dual' {} \; | while read f; do
    replace_in_file "$f" "from app.voice_compare_dual" "from voice_match.services.comparison"
done

# ══════════════════════════════════════════════════════
# 5. from app.interface → from voice_match.ui.interface
# ══════════════════════════════════════════════════════
echo "📝 Замена: app.interface → voice_match.ui.interface"
find src/voice_match -name '*.py' -exec grep -l 'from app\.interface' {} \; | while read f; do
    replace_in_file "$f" "from app\.interface" "from voice_match.ui.interface"
done
find src/voice_match -name '*.py' -exec grep -l 'from app.interface' {} \; | while read f; do
    replace_in_file "$f" "from app.interface" "from voice_match.ui.interface"
done

# ══════════════════════════════════════════════════════
# 6. from models.yamnet → from voice_match.models.yamnet (и т.д.)
# ══════════════════════════════════════════════════════
echo "📝 Замена: models.* → voice_match.models.*"

# Прямые модели
for mod in yamnet xvector ecapa resemblyzer; do
    find src/voice_match -name '*.py' -exec grep -l "from models\.${mod}" {} \; | while read f; do
        replace_in_file "$f" "from models\.${mod}" "from voice_match.models.${mod}"
    done
done

# ══════════════════════════════════════════════════════
# 7. from models.antispoofing → from voice_match.detection.antispoofing
# ══════════════════════════════════════════════════════
echo "📝 Замена: models.antispoofing → voice_match.detection.antispoofing"
find src/voice_match -name '*.py' -exec grep -l 'from models\.antispoofing' {} \; | while read f; do
    replace_in_file "$f" "from models\.antispoofing" "from voice_match.detection.antispoofing"
done
find src/voice_match -name '*.py' -exec grep -l 'from models.antispoofing' {} \; | while read f; do
    replace_in_file "$f" "from models.antispoofing" "from voice_match.detection.antispoofing"
done

# ══════════════════════════════════════════════════════
# 8. from models.formant_tracker → from voice_match.models.formant.tracker
# ══════════════════════════════════════════════════════
echo "📝 Замена: models.formant_tracker → voice_match.models.formant.tracker"
find src/voice_match -name '*.py' -exec grep -l 'from models\.formant_tracker' {} \; | while read f; do
    replace_in_file "$f" "from models\.formant_tracker" "from voice_match.models.formant.tracker"
done
find src/voice_match -name '*.py' -exec grep -l 'from models.formant_tracker' {} \; | while read f; do
    replace_in_file "$f" "from models.formant_tracker" "from voice_match.models.formant.tracker"
done

# ══════════════════════════════════════════════════════
# 9. from models.voice_features → from voice_match.features.voice_features
# ══════════════════════════════════════════════════════
echo "📝 Замена: models.voice_features → voice_match.features.voice_features"
find src/voice_match -name '*.py' -exec grep -l 'from models\.voice_features' {} \; | while read f; do
    replace_in_file "$f" "from models\.voice_features" "from voice_match.features.voice_features"
done
find src/voice_match -name '*.py' -exec grep -l 'from models.voice_features' {} \; | while read f; do
    replace_in_file "$f" "from models.voice_features" "from voice_match.features.voice_features"
done

# ══════════════════════════════════════════════════════
# 10. from models.formant_core → from voice_match.models.formant.core
# ══════════════════════════════════════════════════════
echo "📝 Замена: models.formant_core → voice_match.models.formant.core"
find src/voice_match -name '*.py' -exec grep -l 'from models\.formant_core' {} \; | while read f; do
    replace_in_file "$f" "from models\.formant_core" "from voice_match.models.formant.core"
done
find src/voice_match -name '*.py' -exec grep -l 'from models.formant_core' {} \; | while read f; do
    replace_in_file "$f" "from models.formant_core" "from voice_match.models.formant.core"
done

# ══════════════════════════════════════════════════════
# 11. from models.formant_statistics → from voice_match.models.formant.statistics
# ══════════════════════════════════════════════════════
echo "📝 Замена: models.formant_statistics → voice_match.models.formant.statistics"
find src/voice_match -name '*.py' -exec grep -l 'from models\.formant_statistics' {} \; | while read f; do
    replace_in_file "$f" "from models\.formant_statistics" "from voice_match.models.formant.statistics"
done
find src/voice_match -name '*.py' -exec grep -l 'from models.formant_statistics' {} \; | while read f; do
    replace_in_file "$f" "from models.formant_statistics" "from voice_match.models.formant.statistics"
done

# ══════════════════════════════════════════════════════
# 12. from models.formant.formant_* → from voice_match.models.formant.*
# (внутри formant/__init__.py и других файлов формантного пакета)
# ══════════════════════════════════════════════════════
echo "📝 Замена: models.formant.formant_* → voice_match.models.formant.*"
find src/voice_match -name '*.py' -exec grep -l 'from models\.formant\.formant_core' {} \; | while read f; do
    replace_in_file "$f" "from models\.formant\.formant_core" "from voice_match.models.formant.core"
done
find src/voice_match -name '*.py' -exec grep -l 'from models\.formant\.formant_constraints' {} \; | while read f; do
    replace_in_file "$f" "from models\.formant\.formant_constraints" "from voice_match.models.formant.constraints"
done
find src/voice_match -name '*.py' -exec grep -l 'from models\.formant\.formant_comparison' {} \; | while read f; do
    replace_in_file "$f" "from models\.formant\.formant_comparison" "from voice_match.models.formant.comparison"
done
find src/voice_match -name '*.py' -exec grep -l 'from models\.formant\.formant_statistics' {} \; | while read f; do
    replace_in_file "$f" "from models\.formant\.formant_statistics" "from voice_match.models.formant.statistics"
done
find src/voice_match -name '*.py' -exec grep -l 'from models\.formant\.formant_visualization' {} \; | while read f; do
    replace_in_file "$f" "from models\.formant\.formant_visualization" "from voice_match.models.formant.visualization"
done

# ══════════════════════════════════════════════════════
# 13. Замена имён классов в formant/__init__.py
# (FormantExtractor → FormantAnalyzer если нужно)
# ══════════════════════════════════════════════════════
echo "📝 Обновляем formant/__init__.py..."
cat > src/voice_match/models/formant/__init__.py << 'INIT'
"""Формантный анализ голосового тракта."""

from voice_match.models.formant.core import FormantAnalyzer
from voice_match.models.formant.constraints import FormantConstraints
from voice_match.models.formant.comparison import FormantComparator
from voice_match.models.formant.statistics import FormantStatistics
from voice_match.models.formant.visualization import FormantVisualizer

__all__ = [
    'FormantAnalyzer',
    'FormantConstraints',
    'FormantComparator',
    'FormantStatistics',
    'FormantVisualizer',
]
INIT

echo ""
echo "═══════════════════════════════════════════"
echo "✅ Импорты обновлены!"
echo ""
echo "📋 Проверяем, остались ли старые импорты:"
echo ""

# Проверка на оставшиеся старые импорты
OLD_IMPORTS=$(grep -rn "from app\.\|from models\.\|from config\." src/voice_match/ --include='*.py' 2>/dev/null | grep -v '__pycache__' || true)

if [ -z "$OLD_IMPORTS" ]; then
    echo "  ✅ Старых импортов не осталось!"
else
    echo "  ⚠️  Найдены оставшиеся старые импорты:"
    echo "$OLD_IMPORTS"
fi

echo ""
echo "📋 Следующий шаг: обновить pyproject.toml и проверить запуск"
