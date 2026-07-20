# voice_match

[![CI](https://github.com/anfixit/voice_match/actions/workflows/ci.yml/badge.svg)](https://github.com/anfixit/voice_match/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Status: research](https://img.shields.io/badge/status-research-orange.svg)](#статус-проекта)

Локальное open-source приложение для прозрачного сравнения
speaker embeddings двух голосовых записей.

Система проверяет качество входных данных, выделяет речевые сегменты,
получает ECAPA-TDNN embeddings и показывает сырой cosine score вместе
со статистикой сегментных сравнений.

## Статус проекта

`voice_match` находится в исследовательской стадии.

Текущая версия намеренно:

- не выдаёт процент совпадения личности;
- не называет cosine score вероятностью;
- не имитирует PLDA, LLR или байесовский posterior;
- не выносит автоматическое решение same/different;
- не формирует заключение судебного эксперта;
- отключает anti-spoofing без валидированных весов.

Это ограничение является частью корректной архитектуры, а не
недостающей красивой формулировкой. Решение о принадлежности голосов
можно добавлять только после калибровки на размеченных
`target/non-target` trials из целевого домена.

## Что уже работает

- загрузка WAV, MP3, M4A, FLAC и OGG;
- безопасная конвертация в mono PCM WAV 16 кГц;
- ограничение размера файла;
- контроль длительности, доли речи, клиппинга и уровня сигнала;
- WebRTC VAD и извлечение нескольких речевых сегментов;
- ECAPA-TDNN embeddings через SpeechBrain;
- L2-нормализация и cosine score центроидов;
- статистика всех попарных сегментных score;
- локальный Gradio UI;
- автоматическое удаление созданных временных WAV;
- сетевой запуск только с логином и паролем;
- Docker с непривилегированным пользователем;
- тесты, type checking, lint и dependency audit в CI;
- CSV evaluator для FAR, FRR, EER и minDCF.

## Как устроен анализ

```text
Аудиофайл 1                          Аудиофайл 2
     |                                    |
     +---------- декодирование -----------+
                      |
              mono PCM, 16 кГц
                      |
               quality gate
      duration / speech / clipping / RMS
                      |
                 WebRTC VAD
                      |
             речевые сегменты
                      |
                ECAPA-TDNN
                      |
          L2 speaker embeddings
                      |
       centroid cosine + pair statistics
                      |
             прозрачный отчёт
```

Главное числовое значение сейчас:

```text
ECAPA centroid cosine score
```

Это геометрическая близость двух нормализованных векторов.
Она не является вероятностью и не имеет универсального порога для
Telegram, телефонных звонков, диктофонов и студийных записей.

Подробнее:

- [Методология](docs/methodology.md)
- [Метрики и калибровка](docs/metrics.md)
- [Техническая архитектура](docs/technical_details.md)
- [Воспроизводимый benchmark](docs/benchmark.md)

## Быстрый запуск

### Требования

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- ffmpeg
- доступ к Hugging Face при первом запуске модели

На macOS:

```bash
brew install uv ffmpeg
```

На Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y ffmpeg libsndfile1
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Установка

```bash
git clone https://github.com/anfixit/voice_match.git
cd voice_match
uv sync --frozen --extra dev
cp .env.example .env
```

### Запуск

```bash
uv run voice-match
```

Интерфейс будет доступен по адресу:

```text
http://127.0.0.1:7860
```

При первом обращении SpeechBrain загрузит модель
`speechbrain/spkrec-ecapa-voxceleb` в каталог
`pretrained_models/speechbrain-ecapa`.

Для полностью изолированного запуска модель нужно заранее поместить
в этот каталог.

## Docker

Создайте `.env`:

```bash
cp .env.example .env
```

Обязательно задайте:

```dotenv
VM_GRADIO_AUTH_USER=anfi
VM_GRADIO_AUTH_PASSWORD=replace-with-a-long-random-password
```

Запуск:

```bash
docker compose up -d --build
```

Сервис публикуется только на loopback хоста:

```text
127.0.0.1:7860
```

Для внешнего доступа используйте reverse proxy с HTTPS.
Не публикуйте Gradio напрямую в интернет.

Контейнер:

- работает не от root;
- не получает Linux capabilities;
- использует `no-new-privileges`;
- имеет read-only root filesystem;
- хранит временные файлы в `tmpfs`;
- сохраняет только кэш модели в отдельном volume.

## Конфигурация

Все настройки читаются из переменных окружения с префиксом `VM_`.

| Переменная | По умолчанию | Назначение |
| --- | ---: | --- |
| `VM_GRADIO_HOST` | `127.0.0.1` | Адрес интерфейса |
| `VM_GRADIO_PORT` | `7860` | Порт интерфейса |
| `VM_GRADIO_AUTH_USER` | пусто | Логин для сетевого режима |
| `VM_GRADIO_AUTH_PASSWORD` | пусто | Пароль для сетевого режима |
| `VM_MAX_FILE_SIZE_MB` | `20` | Максимальный размер файла |
| `VM_MIN_AUDIO_DURATION` | `5.0` | Минимальная длительность |
| `VM_MIN_SPEECH_SECONDS` | `3.0` | Минимум обнаруженной речи |
| `VM_MIN_SPEECH_RATIO` | `0.25` | Минимальная доля речи |
| `VM_MAX_CLIPPING_RATIO` | `0.01` | Допустимый клиппинг |
| `VM_SEGMENT_COUNT` | `8` | Максимум сегментов |
| `VM_SEGMENT_DURATION` | `4.0` | Длительность сегмента |
| `VM_USE_GPU` | `false` | Использовать CUDA |

Если интерфейс слушает не `127.0.0.1` и не `localhost`, приложение
откажется запускаться без `VM_GRADIO_AUTH_USER` и
`VM_GRADIO_AUTH_PASSWORD`.

## Пример результата

```text
Автоматическое решение same/different не вынесено.
Сырой ECAPA cosine score: 0.734.
```

Отчёт дополнительно содержит:

- длительность каждой записи;
- количество обнаруженной речи;
- долю речи;
- долю клиппинга;
- количество сегментов;
- cosine score центроидов;
- среднее, медиану, минимум, максимум и разброс score;
- состояние anti-spoofing;
- обязательный дисклеймер об ограничениях.

## Почему удалён старый ансамбль

Предыдущая версия смешивала ECAPA, X-vector, Resemblyzer, YAMNet,
форманты, jitter, shimmer и вручную назначенные веса.

Проблемы такого подхода:

- разные score не имеют общей шкалы;
- YAMNet не является speaker verification моделью;
- cosine по сырым формантам систематически завышал сходство;
- sigmoid над cosine выдавался за вероятность;
- масштабированный cosine назывался PLDA и LLR;
- случайная anti-spoofing сеть формировала deepfake score;
- стандартное отклонение разных методов называлось доверительным
  интервалом.

Активный baseline оставляет только измеримое и воспроизводимое
поведение. Дополнительные модели вернутся только через обученный fusion
и отдельный benchmark.

## Anti-spoofing

Anti-spoofing по умолчанию выключен:

```dotenv
VM_ANTISPOOFING_ENABLED=false
```

Репозиторий не содержит проверенных anti-spoofing весов.
Если включить функцию без файла
`pretrained_models/antispoofing/model.pt`, система завершит эту проверку
безопасно и сообщит, что модель недоступна.

Для production-направления планируется отдельный pipeline на базе
официальных ASVspoof baselines. Его score не будет смешиваться с
speaker similarity.

## Воспроизводимая разработка

```bash
uv sync --frozen --extra dev
uv run ruff check .
uv run mypy src
uv run pytest
uv export --frozen --no-emit-project --no-hashes \
  --output-file requirements-audit.txt
uvx --from pip-audit pip-audit --strict \
  --requirement requirements-audit.txt
```

Тестовый набор проверяет:

- математику cosine score;
- обработку нулевых и несовместимых embeddings;
- quality gate;
- VAD-сегментацию;
- fail-closed anti-spoofing;
- запрет псевдо-PLDA и псевдовероятностей;
- ECAPA adapter;
- удаление временных WAV;
- оркестрацию сравнения без загрузки реальной модели;
- protocol validation и расчёт benchmark metrics.

## Структура проекта

```text
src/voice_match/
├── config.py                 настройки окружения
├── constants.py              неизменяемые технические значения
├── exceptions.py             доменные исключения
├── detection/
│   └── antispoofing.py        fail-closed интерфейс модели
├── models/
│   └── ecapa.py               SpeechBrain ECAPA adapter
├── evaluation/               benchmark protocol и метрики
├── scoring/
│   ├── similarity.py          чистая математика score
│   ├── plda.py                запрет fake PLDA
│   └── bayesian.py            запрет fake posterior
├── services/
│   ├── quality.py             quality gate и VAD
│   ├── preprocessing.py       безопасная конвертация
│   └── comparison.py          orchestration
└── ui/
    └── interface.py           Gradio frontend
```

## Roadmap к достоверному решению

### Этап 1. Честный baseline

- [x] удалить случайный anti-spoofing;
- [x] удалить псевдовероятности, fake PLDA и fake LLR;
- [x] сделать quality gate;
- [x] сделать ECAPA основным измеряемым backend;
- [x] добавить тесты, CI, безопасный Docker и новый README.

### Этап 2. Современный speaker backend

- [ ] добавить интерфейс `SpeakerEncoder`;
- [ ] интегрировать WeSpeaker как основной production-кандидат;
- [ ] сравнить ECAPA, ResNet, ERes2Net и новые WeSpeaker модели;
- [ ] добавить diarization для записей с несколькими людьми;
- [ ] хранить model cards и SHA-256 весов.

### Этап 3. Калибровка

- [x] добавить строгий target/nontarget CSV protocol;
- [x] считать FAR, FRR, EER и minDCF;
- [ ] собрать законный целевой датасет;
- [ ] разделить дикторов между train, calibration и test;
- [ ] построить target/non-target trials;
- [ ] добавить AS-Norm или другой нормализатор;
- [ ] обучить logistic calibration;
- [ ] публиковать EER, minDCF, CLLR, ROC и DET;
- [ ] выбрать operating points под конкретные риски.

### Этап 4. Anti-spoofing и SASV

- [ ] интегрировать официальный AASIST или другой ASVspoof baseline;
- [ ] проверить TTS, voice conversion, replay и codec attacks;
- [ ] измерять anti-spoof EER и minDCF отдельно;
- [ ] добавить SASV evaluation без смешивания несопоставимых score.

Актуальные upstream-проекты:

- [SpeechBrain](https://github.com/speechbrain/speechbrain)
- [WeSpeaker](https://github.com/wenet-e2e/wespeaker)
- [ASVspoof](https://www.asvspoof.org/)

## Конфиденциальность

Аудио не отправляется в облачные API приложения.
Модель SpeechBrain при первом запуске скачивается отдельно, но сами
пользовательские записи в запрос загрузки модели не входят.

Созданные приложением WAV удаляются в `finally` после обработки.
История сравнений и база голосовых отпечатков не ведутся.

Для чувствительных данных рекомендуется:

- запускать сервис в локальной или изолированной сети;
- заранее загрузить модель;
- отключить исходящий доступ контейнера после загрузки;
- использовать HTTPS и аутентификацию;
- не хранить application logs дольше необходимого;
- регулярно очищать model cache и временные volumes.

## Лицензия

Apache License 2.0. Полный текст находится в [LICENSE](LICENSE).

## Автор

Анфиса Ковганюк

- GitHub: [@anfixit](https://github.com/anfixit)
- Telegram: [@Anfikus](https://t.me/Anfikus)
