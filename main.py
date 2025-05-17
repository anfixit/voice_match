import os
import sys
import importlib.util
from app.log import setup_logger

log = setup_logger("main")

def check_environment():
    try:
        required = ["librosa", "gradio", "pydub", "matplotlib", "numpy", "scipy"]
        neural = ["torch", "speechbrain", "tensorflow", "tensorflow_hub", "pyannote.audio"]
        missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
        missing_neural = [pkg for pkg in neural if importlib.util.find_spec(pkg) is None]

        if missing:
            log.error(f"Не установлены основные пакеты: {missing}")
            return False

        if missing_neural:
            log.warning(f"Не установлены нейросетевые пакеты: {missing_neural}")

        # Проверка ffmpeg
        try:
            import subprocess
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        except Exception:
            log.warning("ffmpeg не найден в PATH")

        # Проверка прав на /tmp
        import tempfile
        try:
            temp_fd, temp_path = tempfile.mkstemp()
            os.close(temp_fd)
            os.unlink(temp_path)
        except Exception as e:
            log.error(f"Нет доступа к временной директории: {e}")
            return False

        os.makedirs("pretrained_models/ecapa", exist_ok=True)
        return True

    except Exception as e:
        log.error(f"Ошибка при проверке окружения: {e}")
        return False


def preload_models():
    log.info("Загрузка всех моделей...")
    try:
        from app.voice_compare_dual import (
            lazy_ecapa, lazy_xvector, lazy_yamnet, lazy_res
        )
        from models.antispoofing import get_antispoofing_detector
        from models.formant_tracker import get_formant_tracker
        from models.voice_features import get_voice_feature_extractor

        lazy_ecapa()
        lazy_xvector()
        lazy_yamnet()
        lazy_res()
        get_antispoofing_detector()
        get_formant_tracker()
        get_voice_feature_extractor()

        log.info("✅ Все модели успешно загружены и готовы к работе.")
    except Exception as e:
        log.error(f"Ошибка при загрузке моделей: {e}")


if __name__ == "__main__":
    if check_environment():
        if importlib.util.find_spec("torch"):
            import torch
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            torch.set_num_threads(1)
            log.info("PyTorch настроен на CPU")

        preload_models()

        from app.interface import launch_ui
        launch_ui()
    else:
        log.critical("Приложение не запущено из-за ошибок окружения.")
        sys.exit(1)
