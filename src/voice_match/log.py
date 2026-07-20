"""Единая настройка логирования."""

import logging
import sys

from voice_match.config import settings

_LOG_FORMAT = (
    '%(asctime)s | %(levelname)-8s | '
    '%(name)s:%(lineno)d | %(message)s'
)


def setup_logger(name: str) -> logging.Logger:
    """Настроить логгер с консольным и файловым выводом."""
    settings.logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level.upper())
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(settings.log_level.upper())
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(
        settings.logs_dir / 'voice_match.log',
        encoding='utf-8',
    )
    file_handler.setLevel(settings.log_level.upper())
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger
