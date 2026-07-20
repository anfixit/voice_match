"""CLI для расчёта speaker verification benchmark metrics."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from voice_match.evaluation.metrics import (
    BenchmarkSummary,
    DetectionCostModel,
    calculate_benchmark,
)
from voice_match.evaluation.protocol import load_trials_csv
from voice_match.evaluation.report import build_payload, render_markdown


def main(argv: Sequence[str] | None = None) -> int:
    """Загрузить trials и вывести воспроизводимый отчёт."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        trials = load_trials_csv(args.trials)
        cost_model = DetectionCostModel(
            target_prior=args.target_prior,
            miss_cost=args.miss_cost,
            false_alarm_cost=args.false_alarm_cost,
        )
        summary = calculate_benchmark(
            trials,
            cost_model=cost_model,
            threshold=args.threshold,
        )
        content = _serialize(summary, args.output_format)
        _write_output(content, args.output)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m voice_match.evaluation',
        description=(
            'Рассчитать EER, minDCF, FAR и FRR по размеченному CSV protocol.'
        ),
    )
    parser.add_argument(
        'trials',
        type=Path,
        help='CSV со столбцами enrollment_id,test_id,label,score.',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        help='Дополнительно рассчитать FAR/FRR при этом пороге.',
    )
    parser.add_argument(
        '--target-prior',
        type=float,
        default=0.01,
        help='P(target) для DCF. По умолчанию: 0.01.',
    )
    parser.add_argument(
        '--miss-cost',
        type=float,
        default=1.0,
        help='Стоимость false reject. По умолчанию: 1.',
    )
    parser.add_argument(
        '--false-alarm-cost',
        type=float,
        default=1.0,
        help='Стоимость false accept. По умолчанию: 1.',
    )
    parser.add_argument(
        '--format',
        dest='output_format',
        choices=('markdown', 'json'),
        default='markdown',
        help='Формат отчёта. По умолчанию: markdown.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Файл результата. Без параметра выводится в stdout.',
    )
    return parser


def _serialize(
    summary: BenchmarkSummary,
    output_format: str,
) -> str:
    if output_format == 'json':
        return (
            json.dumps(
                build_payload(summary),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + '\n'
        )
    return render_markdown(summary)


def _write_output(content: str, output: Path | None) -> None:
    if output is None:
        sys.stdout.write(content)
        return
    output.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
