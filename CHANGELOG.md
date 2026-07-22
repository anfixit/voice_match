# Changelog

Формат основан на Keep a Changelog. Проект использует Semantic
Versioning.

## [Unreleased]

### Добавлено

- Строгий CSV protocol для target/nontarget trials.
- Воспроизводимый расчёт FAR, FRR, EER и minDCF.
- JSON и Markdown отчёты benchmark evaluator.
- Документация по разделению train/calibration/test.
- Псевдонимизированный recording manifest с проверкой split leakage.
- Детерминированный генератор cross-session target/nontarget trials.
- Протокол законного сбора и безопасного хранения голосовых записей.
- Пакетный ECAPA scoring с кэшированием уникальных записей.
- CLI для преобразования scoreless trial plan в benchmark CSV.

### Изменено

- Активное ядро заменено на прозрачный ECAPA baseline.
- Результат теперь показывает raw cosine score без псевдовероятности.
- Добавлен quality gate для длительности, речи, клиппинга и RMS.
- Речевые сегменты выделяются через WebRTC VAD.
- Конфигурация переведена на единый `pydantic-settings` источник.
- Gradio по умолчанию слушает только loopback.
- Для сетевого режима обязательна аутентификация.
- Docker переведён на Python 3.12, uv и непривилегированного
  пользователя.
- README полностью переписан под фактическое поведение проекта.

### Удалено

- Случайно инициализированный anti-spoofing detector.
- Псевдо-PLDA, псевдо-LLR и псевдобайесовская вероятность.
- Ручное смешивание несопоставимых акустических и ML-метрик.
- YAMNet, старый X-vector wrapper, Resemblyzer и формантный ансамбль
  из активного pipeline.
- Дублирующий `report_legacy.py`.
- Пустые модули и отслеживаемая `.idea`.
- Устаревший `requirements.txt` и `weights.json`.

### Безопасность

- Anti-spoofing теперь работает только с явно предоставленной
  TorchScript-моделью и завершает проверку безопасно без весов.
- Временные WAV удаляются после каждого запроса.
- Удалено логирование полных путей пользовательских файлов.
- Контейнер использует read-only filesystem, `cap_drop: ALL` и
  `no-new-privileges`.
- Manifest отклоняет абсолютные пути, traversal и дубли аудиофайлов.
- Один диктор не может одновременно попасть в разные dataset split.
- Scoring отклоняет неверные labels, split leakage и symlink escape.

### Тесты

- Добавлены unit-тесты manifest и генератора trial plan.
- Добавлены тесты scoreless plan, batch scoring и scoring CLI.
- Минимальное покрытие установлено на 70 процентов.
- CI запускает ruff, mypy, pytest, pip-audit и Docker build.

## [0.2.0]

- Переход на src-layout, Python 3.12, pyproject.toml и uv.
- Первичная реорганизация модулей приложения.

## [0.1.0] - 2025-01-17

- Первая публичная исследовательская версия.
