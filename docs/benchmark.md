# Воспроизводимый benchmark

## Назначение

Модуль `voice_match.evaluation` оценивает уже рассчитанные system
score на размеченном speaker verification protocol. Он не обучает
модель и не превращает cosine score в вероятность.

Каждая строка CSV описывает одну пару enrollment/test:

```csv
enrollment_id,test_id,label,score,condition
speaker-a-1,speaker-a-2,target,0.84,clean
speaker-a-1,speaker-b-1,nontarget,0.24,clean
```

Обязательные столбцы:

- `enrollment_id`: стабильный идентификатор enrollment-записи;
- `test_id`: стабильный идентификатор test-записи;
- `label`: только `target` или `nontarget`;
- `score`: конечное число, где большее значение означает более
  сильную поддержку target-гипотезы.

`condition` необязателен. Он предназначен для будущих срезов по
кодеку, устройству, шуму и длительности.

Пары `enrollment_id,test_id` должны быть уникальны. В одном protocol
обязаны присутствовать оба класса trials.

## Запуск

```bash
uv run python -m voice_match.evaluation \
  examples/trials.example.csv
```

JSON для автоматической обработки:

```bash
uv run python -m voice_match.evaluation \
  examples/trials.example.csv \
  --format json \
  --output benchmark.json
```

Дополнительный operating point:

```bash
uv run python -m voice_match.evaluation \
  examples/trials.example.csv \
  --threshold 0.65 \
  --target-prior 0.01 \
  --miss-cost 1 \
  --false-alarm-cost 1
```

Решение считается target при `score >= threshold`.

## Метрики

### FAR и FRR

```text
FRR = missed target trials / all target trials
FAR = accepted nontarget trials / all nontarget trials
```

В литературе FRR также называют miss probability, а FAR — false
alarm probability.

### EER

EER вычисляется линейной интерполяцией между соседними эмпирическими
operating points, между которыми меняется знак `FRR - FAR`.

Это описательная метрика ранжирования. Она не выбирает production
порог и не учитывает реальные цены ошибок.

### Detection Cost Function

```text
DCF = Cmiss * Ptarget * FRR
    + Cfa * (1 - Ptarget) * FAR
```

`minDCF` — минимальная DCF среди всех достижимых порогов, включая
тривиальное решение «отклонить все». Нормированное значение делится
на стоимость лучшего тривиального решения:

```text
min(Cmiss * Ptarget, Cfa * (1 - Ptarget))
```

По умолчанию используются:

```text
Ptarget = 0.01
Cmiss = 1
Cfa = 1
```

Отчёт всегда показывает raw и normalized minDCF, чтобы не смешивать
две шкалы.

## Разделение данных

Нельзя рассчитывать честный test benchmark на тех же дикторах и
записях, на которых подбирались порог, normalizer или calibration.

Минимальная схема:

1. `train`: обучение backend или fusion;
2. `calibration`: выбор преобразования score и operating point;
3. `test`: однократная итоговая оценка.

Дикторы между split не пересекаются. Дубликаты и фрагменты одной
исходной записи не должны попадать в разные split.

## Что отчёт не доказывает

Результат действует только для конкретного protocol и его условий.
Он не переносится автоматически между Telegram, телефонной сетью,
диктофоном, студийным микрофоном и другим языком.

EER и minDCF не являются вероятностью того, что два голоса
принадлежат одному человеку. Для калиброванного LLR нужен отдельный
calibration split, зафиксированный алгоритм и независимый test set.

## Первичные источники

- NIST, *2021 Speaker Recognition Evaluation Plan*: определения
  trial, miss/false alarm и cost-based evaluation.
- NIST, *Speaker Recognition Evaluation 2021*: официальный раздел
  программы оценки speaker recognition.
- Brümmer, de Villiers, *The BOSARIS Toolkit*: практическая оценка,
  калибровка и анализ speaker recognition systems.

Ссылки:

- <https://www.nist.gov/system/files/documents/2021/08/13/sre21_eval_plan_v7.pdf>
- <https://www.nist.gov/itl/iad/mig/speaker-recognition-evaluation-2021>
- <https://arxiv.org/abs/1304.2865>
