"""Точка входа приложения voice_match."""

import sys
from pathlib import Path

# src/ должен быть в sys.path для импортов
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from voice_match.__main__ import main

if __name__ == '__main__':
    main()
