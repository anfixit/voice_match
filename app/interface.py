import gradio as gr
import traceback
import os
import tempfile
import logging
import librosa
import numpy as np
import matplotlib.pyplot as plt
from app.log import setup_logger
from app.voice_compare_dual import compare_voices_dual
from app.utils import convert_audio_to_wav

# ──────────────── Логгер ────────────────
logging.basicConfig(level=logging.INFO)
log = setup_logger("interface")


def visualize_audio(wav_path_1, wav_path_2):
    try:
        temp_fd, temp_img_path = tempfile.mkstemp(suffix=".png")
        os.close(temp_fd)

        y1, sr1 = librosa.load(wav_path_1, sr=None)
        y2, sr2 = librosa.load(wav_path_2, sr=None)

        fig, ax = plt.subplots(4, 2, figsize=(14, 16))

        # Спектрограммы
        D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1)), ref=np.max)
        D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2)), ref=np.max)
        librosa.display.specshow(D1, sr=sr1, hop_length=512, x_axis='time', y_axis='log', ax=ax[0, 0])
        ax[0, 0].set_title('Спектрограмма №1')
        librosa.display.specshow(D2, sr=sr2, hop_length=512, x_axis='time', y_axis='log', ax=ax[0, 1])
        ax[0, 1].set_title('Спектрограмма №2')

        # Энергия
        energy1 = librosa.feature.rms(y=y1)[0]
        energy2 = librosa.feature.rms(y=y2)[0]
        ax[1, 0].plot(energy1)
        ax[1, 0].set_title('Энергия сигнала №1')
        ax[1, 1].plot(energy2)
        ax[1, 1].set_title('Энергия сигнала №2')

        # Pitch
        pitch1, _ = librosa.piptrack(y=y1, sr=sr1)
        pitch2, _ = librosa.piptrack(y=y2, sr=sr2)
        ax[2, 0].imshow(pitch1, aspect='auto', origin='lower', cmap='coolwarm')
        ax[2, 0].set_title('Pitch №1')
        ax[2, 1].imshow(pitch2, aspect='auto', origin='lower', cmap='coolwarm')
        ax[2, 1].set_title('Pitch №2')

        # MFCC
        mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)
        librosa.display.specshow(mfcc1, x_axis='time', ax=ax[3, 0])
        ax[3, 0].set_title('MFCC №1')
        librosa.display.specshow(mfcc2, x_axis='time', ax=ax[3, 1])
        ax[3, 1].set_title('MFCC №2')

        plt.tight_layout()
        plt.savefig(temp_img_path)
        plt.close()

        return temp_img_path
    except Exception as e:
        log.error(f"Ошибка визуализации: {e}")
        return None


def process_files(file1_path, file2_path):
    try:
        if not file1_path or not file2_path:
            return "⚠️ Загрузите оба файла.", "", None

        for file_path in [file1_path, file2_path]:
            if not os.path.exists(file_path):
                return f"⚠️ Ошибка: файл {file_path} не найден.", "", None

            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > 20:
                return f"⚠️ Файл слишком большой: {file_size_mb:.1f} МБ", "Используйте файлы до 20 МБ", None

        try:
            wav1_path, log1 = convert_audio_to_wav(file1_path)
            wav2_path, log2 = convert_audio_to_wav(file2_path)
        except Exception as e:
            return f"⚠️ Ошибка при конвертации: {str(e)}", "", None

        for wav_path in [wav1_path, wav2_path]:
            if not os.path.exists(wav_path):
                return f"⚠️ Ошибка: файл {wav_path} не найден после конвертации.", "", None

        viz_path = visualize_audio(wav1_path, wav2_path)

        result_msg, summary = compare_voices_dual(wav1_path, wav2_path)

        log_output = (
            f"📂 Файл 1: {os.path.basename(file1_path)}\n"
            f"{log1}\n\n"
            f"📂 Файл 2: {os.path.basename(file2_path)}\n"
            f"{log2}\n\n"
            f"{summary}"
        )

        return result_msg, log_output, viz_path

    except Exception as exc:
        log.error(f"Неожиданная ошибка: {exc}")
        log.debug(traceback.format_exc())
        return "⚠️ Произошла ошибка при обработке файлов.", str(exc), None


def launch_ui():
    description_text = (
        "🎧 <b>Поддерживаемые форматы:</b> .wav, .mp3, .m4a, .flac, .ogg<br>"
        "⚙️ <b>Автоконвертация:</b> аудио будет преобразовано в WAV (16kHz, mono)<br>"
        "⏱️ <b>Рекомендуемая длина:</b> от 5 сек до 1 мин. Музыка и шумы могут исказить результат.<br>"
        "🧠 <b>Технология:</b> ECAPA, Resemblyzer, X-vector и др."
    )

    with gr.Blocks(title="Сравнение голосов") as demo:
        gr.Markdown(f"### 🎙️ Сравнение голосов (voice_match)\n{description_text}", elem_id="intro")

        with gr.Row():
            file1 = gr.Audio(label="🔈 Голос №1", type="filepath")
            file2 = gr.Audio(label="🔉 Голос №2", type="filepath")

        with gr.Row():
            compare_button = gr.Button("🔍 Сравнить", interactive=True)
            clear_button = gr.Button("🧹 Очистить", interactive=True)

        with gr.Row():
            result = gr.Textbox(label="Результат")
        with gr.Row():
            log_output = gr.Textbox(label="Детали анализа", lines=8)
        with gr.Row():
            visualization = gr.Image(label="Визуализация спектрограмм", type="filepath")

        compare_button.click(
            fn=process_files,
            inputs=[file1, file2],
            outputs=[result, log_output, visualization]
        )

        clear_button.click(
            fn=lambda: ("", "", None),
            inputs=[],
            outputs=[result, log_output, visualization]
        )

        gr.Markdown(
            "📄 <a href='https://github.com/anfixit/voice_match' target='_blank'>Исходный код и документация на GitHub</a>",
            elem_id="footer"
        )

    demo.launch(share=False, debug=True, max_threads=1)
