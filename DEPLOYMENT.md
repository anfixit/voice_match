# 🚀 Руководство по развертыванию voice_match

Это руководство описывает полный процесс развертывания проекта на сервере Timeweb через GitHub Actions.

---

## 📋 Предварительные требования

### На сервере должны быть установлены:
- Docker
- Docker Compose
- Git

---

## 🔐 Шаг 1: Настройка GitHub Secrets

Перейдите в настройки репозитория на GitHub:
**Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Добавьте следующие секреты:

| Имя секрета | Значение | Описание |
|------------|----------|----------|
| `SERVER_HOST` | `185.114.245.123` или `vh438.timeweb.ru` | IP или хост сервера |
| `SERVER_USER` | `cg82264` | Логин пользователя |
| `SERVER_PASSWORD` | `tu3cRzk2?38o` | Пароль от сервера |

---

## 🖥️ Шаг 2: Подготовка сервера

### 2.1. Подключитесь к серверу по SSH:

```bash
ssh cg82264@185.114.245.123
```

### 2.2. Установите Docker (если еще не установлен):

```bash
# Обновите пакеты
sudo apt update

# Установите зависимости
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

# Добавьте Docker GPG ключ
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Добавьте Docker репозиторий
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Установите Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Установите Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Добавьте пользователя в группу docker
sudo usermod -aG docker $USER

# Перезайдите в систему для применения изменений
exit
```

### 2.3. Снова подключитесь к серверу:

```bash
ssh cg82264@185.114.245.123
```

### 2.4. Проверьте установку:

```bash
docker --version
docker-compose --version
```

---

## 🌐 Шаг 3: Настройка домена

### 3.1. Настройте DNS-записи для домена voice-match.ru:

В панели управления доменом добавьте A-записи:

```
@ (или voice-match.ru)  →  185.114.245.123
www                     →  185.114.245.123
```

### 3.2. Подождите распространения DNS (может занять до 24 часов, обычно 15-30 минут)

Проверить можно командой:
```bash
nslookup voice-match.ru
```

---

## 🔒 Шаг 4: Получение SSL сертификата (Let's Encrypt)

### 4.1. На сервере установите Certbot:

```bash
sudo apt update
sudo apt install -y certbot
```

### 4.2. Остановите Nginx если он запущен:

```bash
sudo systemctl stop nginx || true
```

### 4.3. Получите сертификат:

```bash
sudo certbot certonly --standalone -d voice-match.ru -d www.voice-match.ru
```

Следуйте инструкциям (введите email, согласитесь с условиями).

### 4.4. Создайте директорию для SSL и скопируйте сертификаты:

```bash
mkdir -p ~/voice_match/nginx/ssl
sudo cp /etc/letsencrypt/live/voice-match.ru/fullchain.pem ~/voice_match/nginx/ssl/
sudo cp /etc/letsencrypt/live/voice-match.ru/privkey.pem ~/voice_match/nginx/ssl/
sudo chown -R $USER:$USER ~/voice_match/nginx/ssl
```

### 4.5. Настройте автообновление сертификата:

```bash
# Добавьте cronjob для автообновления
sudo crontab -e
```

Добавьте в конец файла:
```
0 0 1 * * certbot renew --quiet && cp /etc/letsencrypt/live/voice-match.ru/fullchain.pem ~/voice_match/nginx/ssl/ && cp /etc/letsencrypt/live/voice-match.ru/privkey.pem ~/voice_match/nginx/ssl/ && cd ~/voice_match && docker-compose restart nginx
```

---

## ⚙️ Шаг 5: Первый деплой

### 5.1. На сервере создайте директорию проекта:

```bash
mkdir -p ~/voice_match
cd ~/voice_match
```

### 5.2. Клонируйте репозиторий:

```bash
git clone https://github.com/anfixit/voice_match.git .
```

### 5.3. Используйте production конфигурацию:

```bash
cp docker-compose.prod.yml docker-compose.yml
```

### 5.4. Запустите приложение:

```bash
docker-compose up -d --build
```

### 5.5. Проверьте статус:

```bash
docker-compose ps
docker-compose logs -f
```

---

## 🤖 Шаг 6: Автоматический деплой через GitHub Actions

После настройки секретов в GitHub, каждый push в ветку `main` будет автоматически:

1. Подключаться к серверу по SSH
2. Обновлять код из репозитория
3. Пересобирать Docker контейнеры
4. Перезапускать приложение

### Как запустить деплой:

```bash
git add .
git commit -m "Deploy to production"
git push origin main
```

Или вручную через GitHub:
**Actions** → **Deploy to Timeweb** → **Run workflow**

---

## 🔍 Проверка работы

### Проверьте доступность сайта:

- HTTP: http://voice-match.ru (должен редиректить на HTTPS)
- HTTPS: https://voice-match.ru

### Проверьте логи приложения:

```bash
cd ~/voice_match
docker-compose logs -f voice_match
```

### Проверьте логи Nginx:

```bash
docker-compose logs -f nginx
```

---

## 🛠️ Полезные команды

### Перезапуск приложения:
```bash
cd ~/voice_match
docker-compose restart
```

### Остановка:
```bash
docker-compose down
```

### Запуск:
```bash
docker-compose up -d
```

### Просмотр логов:
```bash
docker-compose logs -f
```

### Обновление вручную:
```bash
git pull origin main
docker-compose up -d --build
```

### Очистка старых образов:
```bash
docker system prune -a
```

---

## 🔧 Решение проблем

### Проблема: SSL сертификат не найден

**Решение:**
```bash
# Проверьте наличие сертификатов
ls -la /etc/letsencrypt/live/voice-match.ru/

# Если их нет, повторите шаг 4
```

### Проблема: Контейнер не запускается

**Решение:**
```bash
# Посмотрите логи
docker-compose logs

# Пересоберите контейнеры
docker-compose down
docker-compose up -d --build --force-recreate
```

### Проблема: Домен не открывается

**Решение:**
```bash
# Проверьте DNS
nslookup voice-match.ru

# Проверьте открыты ли порты
sudo netstat -tulpn | grep :80
sudo netstat -tulpn | grep :443

# Проверьте файрвол
sudo ufw status
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
```

### Проблема: GitHub Actions не может подключиться

**Решение:**
1. Проверьте правильность секретов в GitHub
2. Убедитесь что SSH доступен на порту 22
3. Проверьте что пользователь имеет права на docker команды:
   ```bash
   sudo usermod -aG docker cg82264
   ```

---

## 📞 Поддержка

Если возникли проблемы:
- Проверьте логи: `docker-compose logs -f`
- Проверьте статус контейнеров: `docker-compose ps`
- Проверьте ресурсы сервера: `htop` или `docker stats`

---

## ✅ Чеклист развертывания

- [ ] GitHub Secrets настроены
- [ ] Docker установлен на сервере
- [ ] DNS-записи настроены
- [ ] SSL сертификат получен
- [ ] Первый деплой выполнен вручную
- [ ] Сайт доступен по HTTPS
- [ ] Автоматический деплой работает
- [ ] Автообновление SSL настроено

---

Готово! Ваш сайт развернут и доступен по адресу https://voice-match.ru 🎉
