import matplotlib.pyplot as plt
import numpy as np
import librosa
import scipy.signal
from typing import Dict, List, Tuple
import io
import base64


def create_advanced_visualization(y1, y2, sr1, sr2, formants1, formants2, jitter_shimmer1, jitter_shimmer2):
    """
    Создает расширенную визуализацию для судебного сравнения голосов.

    Args:
        y1, y2: Аудиосигналы
        sr1, sr2: Частоты дискретизации
        formants1, formants2: Форманты для двух сигналов
        jitter_shimmer1, jitter_shimmer2: Микровариации голоса

    Returns:
        HTML-код с интерактивными графиками
    """
    # Создаем 4x2 подграфиков с увеличенным размером
    fig, ax = plt.subplots(4, 2, figsize=(16, 20))

    # 1. Спектрограммы с акцентом на речевой диапазон
    D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1)), ref=np.max)
    D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2)), ref=np.max)

    # Выделяем голосовой диапазон 80-8000 Hz
    freqs = librosa.fft_frequencies(sr=sr1, n_fft=2048)
    voiced_range = (freqs >= 80) & (freqs <= 8000)

    librosa.display.specshow(D1[voiced_range], sr=sr1,
                             hop_length=512, x_axis='time',
                             y_axis='log', ax=ax[0, 0], cmap='viridis')
    ax[0, 0].set_title('Спектрограмма №1 (голосовой диапазон)')

    librosa.display.specshow(D2[voiced_range], sr=sr2,
                             hop_length=512, x_axis='time',
                             y_axis='log', ax=ax[0, 1], cmap='viridis')
    ax[0, 1].set_title('Спектрограмма №2 (голосовой диапазон)')

    # 2. Динамика формант (F1-F4) - криминалистически важная характеристика
    if formants1 and formants2:
        times1 = np.linspace(0, len(y1) / sr1, len(formants1["F1"]))
        times2 = np.linspace(0, len(y2) / sr2, len(formants2["F1"]))

        for i, label in enumerate(["F1", "F2", "F3", "F4"]):
            if formants1[label].size > 0:
                ax[1, 0].plot(times1, formants1[label], label=label)
            if formants2[label].size > 0:
                ax[1, 1].plot(times2, formants2[label], label=label)

        ax[1, 0].set_ylabel('Частота (Hz)')
        ax[1, 0].set_xlabel('Время (с)')
        ax[1, 0].set_title('Динамика формант №1')
        ax[1, 0].legend()

        ax[1, 1].set_ylabel('Частота (Hz)')
        ax[1, 1].set_xlabel('Время (с)')
        ax[1, 1].set_title('Динамика формант №2')
        ax[1, 1].legend()
    else:
        ax[1, 0].text(0.5, 0.5, 'Нет данных о формантах',
                      ha='center', va='center', transform=ax[1, 0].transAxes)
        ax[1, 1].text(0.5, 0.5, 'Нет данных о формантах',
                      ha='center', va='center', transform=ax[1, 1].transAxes)

    # 3. Джиттер/шиммер с доверительным интервалом
    if jitter_shimmer1 is not None and jitter_shimmer2 is not None:
        # Определение категорий для графика
        categories = ['Local\nJitter', 'Abs\nJitter', 'PPQ5', 'Jitter\nVar',
                      'Local\nShimmer', 'Abs\nShimmer', 'APQ5', 'Shimmer\nVar']

        x = np.arange(len(categories))
        width = 0.35

        ax[2, 0].bar(x - width / 2, jitter_shimmer1, width, label='Голос 1')
        ax[2, 0].bar(x + width / 2, jitter_shimmer2, width, label='Голос 2')
        ax[2, 0].set_title('Сравнение джиттера и шиммера')
        ax[2, 0].set_xticks(x)
        ax[2, 0].set_xticklabels(categories)
        ax[2, 0].legend()

        # Нормированные различия (для судебного заключения)
        diffs = np.abs(jitter_shimmer1 - jitter_shimmer2)
        normalized_diffs = np.minimum(diffs / np.array([2.0, 5.0, 2.0, 3.0, 5.0, 3.0, 5.0, 5.0]), 1.0)

        ax[2, 1].bar(x, normalized_diffs)
        ax[2, 1].axhline(y=0.3, color='g', linestyle='-', label='Высокое сходство')
        ax[2, 1].axhline(y=0.7, color='r', linestyle='-', label='Низкое сходство')
        ax[2, 1].set_title('Нормализованные различия (меньше = более похоже)')
        ax[2, 1].set_xticks(x)
        ax[2, 1].set_xticklabels(categories)
        ax[2, 1].set_ylim(0, 1)
        ax[2, 1].legend()
    else:
        ax[2, 0].text(0.5, 0.5, 'Нет данных о джиттере/шиммере',
                      ha='center', va='center', transform=ax[2, 0].transAxes)
        ax[2, 1].text(0.5, 0.5, 'Нет данных о джиттере/шиммере',
                      ha='center', va='center', transform=ax[2, 1].transAxes)

    # 4. MFCC + сравнение
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=20)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=20)

    librosa.display.specshow(mfcc1, x_axis='time', ax=ax[3, 0])
    ax[3, 0].set_title('MFCC №1')

    librosa.display.specshow(mfcc2, x_axis='time', ax=ax[3, 1])
    ax[3, 1].set_title('MFCC №2')

    plt.tight_layout()

    # Сохраняем в буфер для возврата HTML
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Кодируем в base64 для вставки в HTML
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    html = f"""
    <div style="text-align:center">
        <h3>Расширенная визуализация для судебного сравнения</h3>
        <img src="data:image/png;base64,{data}" style="max-width:100%">
        <p><i>Совпадения в формантах и микровариациях голоса - ключевые биометрические характеристики для идентификации</i></p>
    </div>
    """

    return html
