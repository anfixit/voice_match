"""CLI пакетного расчёта score для speaker trial plan."""

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from voice_match.evaluation.dataset import load_recordings_csv
from voice_match.evaluation.protocol import TrialLabel
from voice_match.evaluation.scoring import (
    EcapaRecordingEncoder,
    score_trial_definitions,
    write_scored_trials_csv,
)
from voice_match.evaluation.trial_plan import (
    load_trial_definitions_csv,
)
from voice_match.exceptions import (
    AudioValidationError,
    ModelUnavailableError,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Рассчитать ECAPA cosine score для готового trial plan."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        recordings = load_recordings_csv(args.manifest)
        definitions = load_trial_definitions_csv(args.trial_plan)
        encoder = EcapaRecordingEncoder(args.dataset_root)
        trials = score_trial_definitions(
            recordings,
            definitions,
            encoder,
        )
        write_scored_trials_csv(trials, args.output)
    except (
        AudioValidationError,
        ModelUnavailableError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))

    counts = Counter(trial.label for trial in trials)
    sys.stdout.write(
        'Создан scored protocol: '
        f'{counts[TrialLabel.TARGET]} target, '
        f'{counts[TrialLabel.NONTARGET]} nontarget.\n',
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m voice_match.evaluation.score_trials',
        description=(
            'Рассчитать ECAPA cosine score и создать benchmark CSV.'
        ),
    )
    parser.add_argument(
        'manifest',
        type=Path,
        help='CSV manifest голосовых записей.',
    )
    parser.add_argument(
        'trial_plan',
        type=Path,
        help='Scoreless target/nontarget trial plan CSV.',
    )
    parser.add_argument(
        '--dataset-root',
        type=Path,
        required=True,
        help='Корневой каталог аудиофайлов из manifest.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Путь для scored protocol CSV.',
    )
    return parser


if __name__ == '__main__':
    raise SystemExit(main())
