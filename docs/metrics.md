# Метрики и калибровка

## Raw cosine score

Для нормализованных embeddings используется скалярное произведение:

```text
score = dot(normalize(e1), normalize(e2))
```

Диапазон cosine score находится между `-1` и `1`.
Высокое значение означает близость направлений embeddings, но не
вероятность совпадения личности.

## Статистика сегментов

Отчёт содержит:

- centroid score;
- медиану попарных score;
- среднее;
- стандартное отклонение;
- минимум и максимум;
- число сравнённых пар.

Стандартное отклонение описывает разброс segment score.
Оно не называется доверительным интервалом и не измеряет уверенность
в гипотезе same speaker.

## Что нужно для решения same/different

Нужен отдельный evaluation protocol:

1. записи размечаются по speaker ID;
2. дикторы не пересекаются между train, calibration и test;
3. создаются target и non-target trials;
4. score normalization обучается только на разрешённых данных;
5. calibration обучается на calibration split;
6. operating point выбирается до просмотра test результатов.

## Рекомендуемые отчётные метрики

- Equal Error Rate, EER;
- minimum Detection Cost Function, minDCF;
- actual DCF;
- CLLR для калиброванных LLR;
- ROC и DET curves;
- False Match Rate;
- False Non-Match Rate;
- bootstrap intervals для benchmark metrics;
- срезы по длительности, кодеку, шуму и устройству.

Нельзя переносить EER upstream-модели на пользовательские записи без
повторного тестирования в целевом домене.
