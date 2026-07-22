"""CLI для генерации speaker verification trial plan."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from voice_match.evaluation.dataset import DatasetSplit, load_recordings_csv
from voice_match.evaluation.protocol import TrialLabel
from voice_match.evaluation.trial_generation import (
    TrialGenerationConfig,
    count_trial_labels,
    generate_trial_definitions,
    write_trial_definitions_csv,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Сформировать trial plan из псевдонимизированного manifest."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        recordings = load_recordings_csv(args.manifest)
        config = TrialGenerationConfig(
            split=DatasetSplit(args.split),
            nontarget_ratio=args.nontarget_ratio,
            max_target_trials_per_speaker=(
                None
                if args.max_target_per_speaker == 0
                else args.max_target_per_speaker
            ),
            seed=args.seed,
        )
        trials = generate_trial_definitions(
            recordings,
            config=config,
        )
        write_trial_definitions_csv(trials, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    counts = count_trial_labels(trials)
    sys.stdout.write(
        'Создан trial plan: '
        f'{counts[TrialLabel.TARGET]} target, '
        f'{counts[TrialLabel.NONTARGET]} nontarget.\n'
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m voice_match.evaluation.generate_trials',
        description=(
            'Сформировать target/nontarget trial plan без system score.'
        ),
    )
    parser.add_argument(
        'manifest',
        type=Path,
        help='CSV manifest голосовых записей.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Путь для сохранения trial plan CSV.',
    )
    parser.add_argument(
        '--split',
        choices=tuple(split.value for split in DatasetSplit),
        default=DatasetSplit.CALIBRATION.value,
        help='Dataset split. По умолчанию: calibration.',
    )
    parser.add_argument(
        '--nontarget-ratio',
        type=int,
        default=5,
        help='Nontarget trials на один target trial. По умолчанию: 5.',
    )
    parser.add_argument(
        '--max-target-per-speaker',
        type=int,
        default=100,
        help='Лимит target trials на диктора. 0 отключает лимит.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Seed детерминированной выборки. По умолчанию: 0.',
    )
    return parser


if __name__ == '__main__':
    raise SystemExit(main())
