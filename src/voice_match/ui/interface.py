"""Gradio-интерфейс voice_match."""

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from voice_match.config import settings
from voice_match.constants import SAMPLE_RATE, SUPPORTED_EXTENSIONS
from voice_match.log import setup_logger
from voice_match.services.comparison import compare_voices_dual
from voice_match.services.preprocessing import convert_audio_to_wav

log = setup_logger('interface')


def visualize_audio(
    first_path: str,
    second_path: str,
) -> Figure:
    """Построить диагностические графики без временного PNG."""
    first, _ = librosa.load(first_path, sr=SAMPLE_RATE, mono=True)
    second, _ = librosa.load(second_path, sr=SAMPLE_RATE, mono=True)

    figure, axes = plt.subplots(3, 2, figsize=(13, 11))
    first_db = librosa.amplitude_to_db(
        np.abs(librosa.stft(first)),
        ref=np.max,
    )
    second_db = librosa.amplitude_to_db(
        np.abs(librosa.stft(second)),
        ref=np.max,
    )
    librosa.display.specshow(
        first_db,
        sr=SAMPLE_RATE,
        x_axis='time',
        y_axis='log',
        ax=axes[0, 0],
    )
    librosa.display.specshow(
        second_db,
        sr=SAMPLE_RATE,
        x_axis='time',
        y_axis='log',
        ax=axes[0, 1],
    )
    axes[0, 0].set_title('Спектрограмма 1')
    axes[0, 1].set_title('Спектрограмма 2')

    axes[1, 0].plot(librosa.feature.rms(y=first)[0])
    axes[1, 1].plot(librosa.feature.rms(y=second)[0])
    axes[1, 0].set_title('Энергия 1')
    axes[1, 1].set_title('Энергия 2')

    first_mfcc = librosa.feature.mfcc(
        y=first,
        sr=SAMPLE_RATE,
        n_mfcc=13,
    )
    second_mfcc = librosa.feature.mfcc(
        y=second,
        sr=SAMPLE_RATE,
        n_mfcc=13,
    )
    librosa.display.specshow(
        first_mfcc,
        x_axis='time',
        ax=axes[2, 0],
    )
    librosa.display.specshow(
        second_mfcc,
        x_axis='time',
        ax=axes[2, 1],
    )
    axes[2, 0].set_title('MFCC 1')
    axes[2, 1].set_title('MFCC 2')
    figure.tight_layout()
    return figure


def process_files(
    first_file: str | None,
    second_file: str | None,
) -> tuple[str, str, Figure | None]:
    """Проверить, преобразовать и сравнить два файла."""
    if not first_file or not second_file:
        return 'Загрузите оба файла.', '', None

    source_paths = [Path(first_file), Path(second_file)]
    generated_paths: set[Path] = set()
    try:
        for source in source_paths:
            _validate_upload(source)

        first_wav, first_log = convert_audio_to_wav(first_file)
        second_wav, second_log = convert_audio_to_wav(second_file)
        for source, converted in zip(
            source_paths,
            (Path(first_wav), Path(second_wav)),
            strict=True,
        ):
            if converted.resolve() != source.resolve():
                generated_paths.add(converted)

        figure = visualize_audio(first_wav, second_wav)
        result, report = compare_voices_dual(first_wav, second_wav)
        details = '\n\n'.join(
            [
                f'Файл 1: {source_paths[0].name}\n{first_log}',
                f'Файл 2: {source_paths[1].name}\n{second_log}',
                report,
            ]
        )
        return result, details, figure
    except (OSError, RuntimeError, ValueError) as exc:
        log.exception('Ошибка обработки загруженных файлов')
        return 'Не удалось обработать файлы.', str(exc), None
    finally:
        for path in generated_paths:
            path.unlink(missing_ok=True)


def launch_ui() -> None:
    """Запустить локальный Gradio UI."""
    import gradio as gr

    settings.ensure_dirs()
    auth = settings.gradio_auth()
    if (
        settings.gradio_host not in {'127.0.0.1', 'localhost'}
        and auth is None
    ):
        raise ValueError(
            'Для сетевого доступа задайте VM_GRADIO_AUTH_USER '
            'и VM_GRADIO_AUTH_PASSWORD.'
        )

    supported = ', '.join(sorted(SUPPORTED_EXTENSIONS))
    description = (
        'Исследовательское сравнение speaker embeddings. '
        f'Форматы: {supported}. Файлы преобразуются в mono WAV '
        f'{SAMPLE_RATE} Гц и удаляются после запроса. '
        'Сервис не выдаёт вероятность личности или экспертное '
        'заключение без отдельной калибровки.'
    )

    with gr.Blocks(title='voice_match') as demo:
        gr.Markdown('# voice_match')
        gr.Markdown(description)

        with gr.Row():
            first_file = gr.Audio(label='Запись 1', type='filepath')
            second_file = gr.Audio(label='Запись 2', type='filepath')

        with gr.Row():
            compare_button = gr.Button('Сравнить', variant='primary')
            clear_button = gr.Button('Очистить')

        result = gr.Textbox(label='Результат')
        report = gr.Markdown(label='Детали анализа')
        visualization = gr.Plot(label='Диагностика аудио')

        compare_button.click(  # type: ignore[attr-defined]
            fn=process_files,
            inputs=[first_file, second_file],
            outputs=[result, report, visualization],
        )
        clear_button.click(  # type: ignore[attr-defined]
            fn=lambda: ('', '', None),
            inputs=[],
            outputs=[result, report, visualization],
        )

    demo.launch(
        server_name=settings.gradio_host,
        server_port=settings.gradio_port,
        auth=auth,
        share=False,
        debug=False,
        max_threads=1,
        show_error=False,
    )


def _validate_upload(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f'Файл не найден: {path.name}.')
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f'Неподдерживаемый формат: {path.suffix}.')

    size_mb = path.stat().st_size / 1024 / 1024
    if size_mb > settings.max_file_size_mb:
        raise ValueError(
            f'Файл {path.name} занимает {size_mb:.1f} МБ. '
            f'Максимум: {settings.max_file_size_mb} МБ.'
        )


__all__ = ['launch_ui', 'process_files', 'visualize_audio']
