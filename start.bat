@echo off
REM Скрипт запуска voice_match для Windows

echo =========================================
echo     Voice Match - Forensic Voice Comparison
echo =========================================
echo.

REM Проверка наличия Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python не установлен!
    pause
    exit /b 1
)

echo ✅ Python обнаружен

REM Создание виртуального окружения
if not exist "venv" (
    echo 📦 Создание виртуального окружения...
    python -m venv venv
)

REM Активация виртуального окружения
echo 🔧 Активация виртуального окружения...
call venv\Scripts\activate.bat

REM Установка зависимостей
echo 📥 Установка зависимостей...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Создание необходимых директорий
if not exist "logs" mkdir logs
if not exist "pretrained_models" mkdir pretrained_models
if not exist "uploads" mkdir uploads

REM Запуск приложения
echo.
echo 🚀 Запуск приложения...
echo 📍 Интерфейс будет доступен по адресу: http://localhost:7860
echo.
python main.py

pause
