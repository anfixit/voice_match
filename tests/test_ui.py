"""Тесты безопасной обработки файлов в UI."""

from pathlib import Path

import pytest

from voice_match.ui import interface


def test_validate_upload_rejects_unknown_extension(tmp_path: Path) -> None:
    source = tmp_path / 'sample.txt'
    source.write_text('not audio')

    with pytest.raises(ValueError, match='Неподдерживаемый формат'):
        interface._validate_upload(source)


def test_process_files_removes_generated_wav(
    monkeypatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / 'first.wav'
    second = tmp_path / 'second.wav'
    generated_first = tmp_path / 'generated-first.wav'
    generated_second = tmp_path / 'generated-second.wav'
    for path in (first, second, generated_first, generated_second):
        path.write_bytes(b'audio')

    converted = iter(
        [
            (str(generated_first), 'converted 1'),
            (str(generated_second), 'converted 2'),
        ]
    )
    monkeypatch.setattr(
        interface,
        'convert_audio_to_wav',
        lambda _: next(converted),
    )
    monkeypatch.setattr(interface, 'visualize_audio', lambda *_: None)
    monkeypatch.setattr(
        interface,
        'compare_voices_dual',
        lambda *_: ('result', 'report'),
    )

    result, report, figure = interface.process_files(
        str(first),
        str(second),
    )

    assert result == 'result'
    assert 'converted 1' in report
    assert figure is None
    assert not generated_first.exists()
    assert not generated_second.exists()


def test_process_files_requires_both_inputs() -> None:
    result, report, figure = interface.process_files(None, None)

    assert result == 'Загрузите оба файла.'
    assert report == ''
    assert figure is None


def test_validate_upload_accepts_small_wav(tmp_path: Path) -> None:
    source = tmp_path / 'sample.wav'
    source.write_bytes(b'audio')

    interface._validate_upload(source)


def test_validate_upload_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match='Файл не найден'):
        interface._validate_upload(tmp_path / 'missing.wav')


def test_validate_upload_rejects_oversized_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / 'large.wav'
    source.write_bytes(b'audio')
    monkeypatch.setattr(interface.settings, 'max_file_size_mb', 0)

    with pytest.raises(ValueError, match='Максимум'):
        interface._validate_upload(source)
