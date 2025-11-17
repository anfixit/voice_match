#!/bin/bash

# 🚀 Автоматическая настройка сервера для voice_match
# Этот скрипт должен быть запущен на сервере Timeweb

set -e

echo "=================================================="
echo "🚀 voice_match Server Setup Script"
echo "=================================================="
echo ""

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Функции для цветного вывода
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Проверка что скрипт запущен не от root
if [ "$EUID" -eq 0 ]; then
    error "Пожалуйста, не запускайте этот скрипт от root. Используйте обычного пользователя."
fi

# 1. Обновление системы
info "Обновление списка пакетов..."
sudo apt update

# 2. Установка необходимых пакетов
info "Установка необходимых зависимостей..."
sudo apt install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common \
    git \
    gnupg \
    lsb-release

# 3. Установка Docker
if ! command -v docker &> /dev/null; then
    info "Установка Docker..."

    # Удаление старых версий
    sudo apt remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

    # Добавление официального GPG ключа Docker
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    # Добавление репозитория Docker
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Установка Docker Engine
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # Добавление текущего пользователя в группу docker
    sudo usermod -aG docker $USER

    info "Docker успешно установлен!"
else
    info "Docker уже установлен: $(docker --version)"
fi

# 4. Установка Docker Compose (standalone)
if ! command -v docker-compose &> /dev/null; then
    info "Установка Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    info "Docker Compose успешно установлен!"
else
    info "Docker Compose уже установлен: $(docker-compose --version)"
fi

# 5. Проверка Git
if ! command -v git &> /dev/null; then
    error "Git не установлен!"
else
    info "Git установлен: $(git --version)"
fi

# 6. Создание директории проекта
info "Создание директории проекта..."
mkdir -p ~/voice_match
cd ~/voice_match

# 7. Установка Certbot для SSL
if ! command -v certbot &> /dev/null; then
    info "Установка Certbot для SSL сертификатов..."
    sudo apt install -y certbot
    info "Certbot успешно установлен!"
else
    info "Certbot уже установлен: $(certbot --version)"
fi

# 8. Настройка файрвола (UFW)
info "Настройка файрвола..."
if command -v ufw &> /dev/null; then
    sudo ufw allow 22/tcp  # SSH
    sudo ufw allow 80/tcp  # HTTP
    sudo ufw allow 443/tcp # HTTPS
    info "Порты 22, 80, 443 открыты в файрволе"
else
    warn "UFW не установлен. Убедитесь, что порты 80 и 443 открыты."
fi

# 9. Проверка прав доступа к Docker
info "Проверка прав доступа к Docker..."
if groups $USER | grep -q docker; then
    info "Пользователь $USER уже в группе docker"
else
    warn "Пользователь $USER добавлен в группу docker. Требуется перезайти в систему!"
    warn "Выполните: exit, затем снова подключитесь по SSH"
fi

echo ""
echo "=================================================="
echo -e "${GREEN}✅ Настройка сервера завершена!${NC}"
echo "=================================================="
echo ""
echo "Следующие шаги:"
echo ""
echo "1. Выйдите из системы и снова подключитесь по SSH:"
echo "   exit"
echo "   ssh $USER@$(hostname -I | awk '{print $1}')"
echo ""
echo "2. Настройте DNS для домена voice-match.ru:"
echo "   A-запись: voice-match.ru → $(hostname -I | awk '{print $1}')"
echo "   A-запись: www.voice-match.ru → $(hostname -I | awk '{print $1}')"
echo ""
echo "3. Дождитесь распространения DNS (15-30 минут)"
echo ""
echo "4. Получите SSL сертификат:"
echo "   sudo certbot certonly --standalone -d voice-match.ru -d www.voice-match.ru"
echo ""
echo "5. Скопируйте SSL сертификаты в проект:"
echo "   mkdir -p ~/voice_match/nginx/ssl"
echo "   sudo cp /etc/letsencrypt/live/voice-match.ru/fullchain.pem ~/voice_match/nginx/ssl/"
echo "   sudo cp /etc/letsencrypt/live/voice-match.ru/privkey.pem ~/voice_match/nginx/ssl/"
echo "   sudo chown -R \$USER:\$USER ~/voice_match/nginx/ssl"
echo ""
echo "6. Клонируйте репозиторий:"
echo "   cd ~/voice_match"
echo "   git clone https://github.com/anfixit/voice_match.git ."
echo ""
echo "7. Запустите приложение:"
echo "   cp docker-compose.prod.yml docker-compose.yml"
echo "   docker-compose up -d --build"
echo ""
echo "Подробные инструкции в файле DEPLOYMENT.md"
echo ""
