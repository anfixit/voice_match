"""CLI-точка входа voice_match."""

import importlib.util
import os

from voice_match.config import settings
from voice_match.log import setup_logger

log = setup_logger('main')

_REQUIRED_PACKAGES = (
    'gradio',
    'librosa',
    'matplotlib',
    'numpy',
    'pydub',
    'soundfile',
    'speechbrain',
    'torch',
    'webrtcvad',
)


def check_environment() -> tuple[str, ...]:
    """Вернуть список отсутствующих обязательных пакетов."""
    return tuple(
        package
        for package in _REQUIRED_PACKAGES
        if importlib.util.find_spec(package) is None
    )


def main() -> None:
    """Проверить окружение и запустить интерфейс."""
    missing = check_environment()
    if missing:
        packages = ', '.join(missing)
        raise RuntimeError(
            f'Не установлены обязательные зависимости: {packages}.'
        )

    settings.ensure_dirs()
    if not settings.use_gpu:
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

    import torch

    torch.set_num_threads(settings.num_threads)
    log.info(
        'Запуск voice_match: gpu=%s, threads=%s',
        settings.use_gpu,
        settings.num_threads,
    )

    from voice_match.ui.interface import launch_ui

    launch_ui()


if __name__ == '__main__':
    main()
