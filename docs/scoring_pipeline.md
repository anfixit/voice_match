# Пакетный scoring pipeline

## Назначение

Pipeline связывает подготовленный manifest и scoreless trial plan с
benchmark evaluator. Он рассчитывает только сырой ECAPA cosine score.
Вероятность личности, LLR и автоматический вердикт не формируются.

Полный поток:

```text
recordings.csv
      |
      v
generate_trials.py
      |
      v
trial-plan.csv
      |
      v
score_trials.py
      |
      v
scored-trials.csv
      |
      v
python -m voice_match.evaluation
      |
      v
FAR / FRR / EER / minDCF
```

## Требования к данным

Manifest должен соответствовать
[протоколу записи](recording_protocol.md). Аудиофайлы хранятся вне
Git, а поле `path` содержит относительный POSIX-путь от корня
датасета.

Каждый target trial обязан:

- содержать одного диктора;
- использовать две разные записи;
- использовать разные сессии;
- находиться внутри одного dataset split.

Каждый nontarget trial обязан содержать разных дикторов внутри одного
split.

Pipeline завершится ошибкой до загрузки модели, если trial plan
противоречит manifest.

## Генерация trial plan

```bash
uv run python -m voice_match.evaluation.generate_trials \
  recordings.csv \
  --split calibration \
  --nontarget-ratio 5 \
  --max-target-per-speaker 100 \
  --seed 42 \
  --output calibration-trial-plan.csv
```

Seed и все параметры нужно сохранять вместе с результатами
эксперимента.

## Расчёт score

```bash
uv run python -m voice_match.evaluation.score_trials \
  recordings.csv \
  calibration-trial-plan.csv \
  --dataset-root /secure/voice-dataset \
  --output calibration-scored.csv
```

Каждая уникальная запись кодируется один раз. Её сегментные ECAPA
embeddings нормализуются, усредняются и повторно L2-нормализуются.
Для каждой пары вычисляется cosine score двух centroid embeddings.

Выходной CSV совместим с benchmark evaluator:

```csv
enrollment_id,test_id,label,score,condition
speaker-a-1,speaker-a-2,target,0.812345,clean
speaker-a-1,speaker-b-1,nontarget,0.213456,cross-device
```

## Расчёт метрик

```bash
uv run python -m voice_match.evaluation \
  calibration-scored.csv \
  --format markdown \
  --output calibration-benchmark.md
```

Для test split создаётся отдельный trial plan и отдельный scored CSV.
Test нельзя использовать для выбора модели, параметров генератора,
порога или calibration mapping.

## Проверки безопасности

Перед кодированием pipeline:

- проверяет наличие всех recording ID;
- отклоняет дубли и перевёрнутые дубли пар;
- проверяет label по speaker ID;
- запрещает пары между разными split;
- разрешает только файлы внутри `dataset-root`;
- отклоняет symlink, который выводит путь за пределы датасета;
- не записывает embeddings на диск;
- атомарно заменяет итоговый CSV.

Scored protocol содержит псевдонимы и system score, но всё равно
является чувствительным исследовательским артефактом. Его нельзя
публиковать без оценки риска повторной идентификации.

## Воспроизводимость

Для каждого запуска сохраняйте:

- commit SHA приложения;
- SHA-256 весов модели;
- manifest и trial plan;
- параметры генератора и seed;
- конфигурацию quality gate;
- версию Python и lock-файл;
- scored protocol и benchmark report.

Без этих данных сравнение результатов разных запусков не считается
воспроизводимым.
