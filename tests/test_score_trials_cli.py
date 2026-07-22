"""Тесты CLI пакетного speaker scoring."""

from pathlib import Path

import numpy as np

from voice_match.evaluation import score_trials
from voice_match.evaluation.dataset import Recording
from voice_match.evaluation.protocol import load_trials_csv


class FakeCliEncoder:
    """Recording encoder без загрузки ML-модели."""

    def encode(self, recording: Recording) -> np.ndarray:
        vectors = {
            'a-1': np.array([1.0, 0.0]),
            'a-2': np.array([0.9, 0.1]),
            'b-1': np.array([0.0, 1.0]),
        }
        return vectors[recording.recording_id]


def test_score_trials_cli_writes_benchmark_csv(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    manifest = tmp_path / 'recordings.csv'
    plan = tmp_path / 'trial-plan.csv'
    output = tmp_path / 'scored.csv'
    manifest.write_text(
        'recording_id,speaker_id,session_id,path,split,condition\n'
        'a-1,speaker-a,session-1,a-1.wav,calibration,clean\n'
        'a-2,speaker-a,session-2,a-2.wav,calibration,clean\n'
        'b-1,speaker-b,session-1,b-1.wav,calibration,clean\n',
        encoding='utf-8',
    )
    plan.write_text(
        'enrollment_id,test_id,label,condition\n'
        'a-1,a-2,target,clean\n'
        'a-1,b-1,nontarget,clean\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(
        score_trials,
        'EcapaRecordingEncoder',
        lambda _: FakeCliEncoder(),
    )

    exit_code = score_trials.main(
        [
            str(manifest),
            str(plan),
            '--dataset-root',
            str(tmp_path),
            '--output',
            str(output),
        ]
    )

    trials = load_trials_csv(output)
    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert len(trials) == 2
    assert trials[0].score > 0.9
    assert trials[1].score == 0.0
    assert '1 target, 1 nontarget' in stdout
