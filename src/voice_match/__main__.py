"""Точка входа voice_match как модуля."""

import importlib.util
import os
import sys

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
