# 🌐 Настройка foodgram-gateway-1 для voice-match.ru

Этот файл содержит инструкции по настройке существующего Docker Nginx контейнера `foodgram-gateway-1` для проксирования voice-match.ru на наше приложение.

---

## 📋 Архитектура

```
Internet (voice-match.ru:443)
    ↓
foodgram-gateway-1 (Docker Nginx, порты 80/443)
    ↓ proxy_pass на host.docker.internal:8081
voice_match_nginx (Docker Nginx, localhost:8081)
    ↓ proxy_pass на voice_match:7860
voice_match_app (Gradio приложение)
```

**Преимущества:**
- ✅ Не конфликтует с существующими проектами
- ✅ Использует общий SSL сертификат от foodgram-gateway-1
- ✅ Централизованное управление через один Nginx
- ✅ Изоляция: каждый проект в своей Docker сети

---

## 🔐 Шаг 1: Получение SSL сертификата

### 1.1. Настройка DNS на reg.ru

Добавьте A-записи для voice-match.ru:
```
@    → 109.73.194.190
www  → 109.73.194.190
```

Проверка (через 15-60 минут):
```bash
nslookup voice-match.ru
# Должен вернуть: 109.73.194.190
```

### 1.2. Получение SSL через Certbot

```bash
# Остановите foodgram-gateway-1 временно
docker stop foodgram-gateway-1

# Получите сертификат
certbot certonly --standalone -d voice-match.ru -d www.voice-match.ru

# Запустите foodgram-gateway-1 обратно
docker start foodgram-gateway-1
```

Сертификаты будут в:
- `/etc/letsencrypt/live/voice-match.ru/fullchain.pem`
- `/etc/letsencrypt/live/voice-match.ru/privkey.pem`

---

## 🐋 Шаг 2: Найдите конфигурацию foodgram-gateway-1

### 2.1. Найдите docker-compose файл foodgram

```bash
cd /opt/foodgram
cat docker-compose.production.yml
```

Найдите раздел `gateway` или `nginx` и посмотрите где монтируется конфигурация.

**Пример из вашего вывода:**
```yaml
# Скорее всего что-то вроде:
gateway:
  image: nginx:1.25-alpine
  volumes:
    - ./infra/nginx.conf:/etc/nginx/nginx.conf:ro
    # или
    - ./nginx:/etc/nginx/conf.d:ro
```

### 2.2. Посмотрите текущую конфигурацию

```bash
# Вариант 1: Если конфиг в файле на хосте
cat /opt/foodgram/infra/nginx.conf

# Вариант 2: Если конфиг внутри контейнера
docker exec foodgram-gateway-1 cat /etc/nginx/nginx.conf
```

---

## 📝 Шаг 3: Добавьте конфигурацию для voice-match.ru

### Метод 1: Если есть include для дополнительных конфигов

**Проверьте есть ли в nginx.conf строка:**
```nginx
include /etc/nginx/conf.d/*.conf;
```

Если есть:

```bash
# Создайте конфигурацию на хосте
cat > /opt/foodgram/infra/conf.d/voice-match.conf << 'EOF'
# HTTP - редирект на HTTPS
server {
    listen 80;
    server_name voice-match.ru www.voice-match.ru;

    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    location / {
        return 301 https://$server_name$request_uri;
    }
}

# HTTPS
server {
    listen 443 ssl http2;
    server_name voice-match.ru www.voice-match.ru;

    ssl_certificate /etc/letsencrypt/live/voice-match.ru/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/voice-match.ru/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    access_log /var/log/nginx/voice-match-access.log;
    error_log /var/log/nginx/voice-match-error.log;

    client_max_body_size 100M;
    client_body_timeout 300s;

    location / {
        # Используем IP хоста вместо host.docker.internal
        proxy_pass http://172.17.0.1:8081;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
EOF

# Обновите docker-compose.yml чтобы монтировать эту директорию
# Добавьте в volumes секцию gateway:
# - ./infra/conf.d:/etc/nginx/conf.d:ro

# Перезапустите
cd /opt/foodgram
docker-compose -f docker-compose.production.yml restart gateway
```

---

### Метод 2: Если нет include, редактируйте основной файл

```bash
# Сделайте бэкап
cp /opt/foodgram/infra/nginx.conf /opt/foodgram/infra/nginx.conf.backup

# Отредактируйте файл
nano /opt/foodgram/infra/nginx.conf
```

Добавьте в конец `http { }` блока (перед закрывающей скобкой) конфигурацию из `foodgram-gateway-voice-match.conf`.

**Затем:**
```bash
# Проверьте синтаксис
docker exec foodgram-gateway-1 nginx -t

# Если OK - перезагрузите
docker exec foodgram-gateway-1 nginx -s reload
```

---

## 🔌 Шаг 4: Настройте доступ к хосту из Docker

### 4.1. Проверьте IP хоста в Docker сети

```bash
# IP шлюза Docker (обычно 172.17.0.1)
docker network inspect bridge | grep Gateway
```

Используйте этот IP вместо `host.docker.internal` в конфигурации Nginx.

### 4.2. Альтернатива: Используйте host network mode

Если не работает через IP, можно добавить `--add-host` в docker-compose:

```yaml
gateway:
  image: nginx:1.25-alpine
  extra_hosts:
    - "host.docker.internal:host-gateway"
  # ...остальное
```

---

## ✅ Шаг 5: Проверка

### 5.1. Проверьте что voice-match запущен

```bash
cd /opt/voice-match
docker-compose ps

# Должны увидеть:
# voice_match_app    running
# voice_match_nginx  running
```

### 5.2. Проверьте доступность на localhost

```bash
# Тест через Docker Nginx voice-match
curl -I http://localhost:8081

# Должен вернуть 200 OK
```

### 5.3. Проверьте foodgram-gateway-1

```bash
# Проверьте конфигурацию
docker exec foodgram-gateway-1 nginx -t

# Проверьте что слушает на 443
docker exec foodgram-gateway-1 netstat -tulpn | grep 443
```

### 5.4. Проверьте с внешнего адреса

```bash
# С сервера
curl -I https://voice-match.ru

# Или откройте в браузере
# https://voice-match.ru
```

---

## 🔧 Решение проблем

### Проблема: 502 Bad Gateway

```bash
# Проверьте что voice-match запущен
cd /opt/voice-match
docker-compose ps

# Проверьте логи
docker logs voice_match_nginx
docker logs voice_match_app

# Проверьте логи foodgram-gateway
docker logs foodgram-gateway-1
```

### Проблема: Не может достучаться до localhost:8081

```bash
# Проверьте что порт 8081 слушает
netstat -tulpn | grep 8081

# Попробуйте с хоста
curl http://127.0.0.1:8081

# Проверьте firewall
ufw status
```

### Проблема: SSL сертификат не найден

```bash
# Проверьте наличие
ls -la /etc/letsencrypt/live/voice-match.ru/

# Проверьте что foodgram-gateway-1 монтирует /etc/letsencrypt
docker inspect foodgram-gateway-1 | grep letsencrypt

# Если не монтирует - добавьте в docker-compose.yml:
# volumes:
#   - /etc/letsencrypt:/etc/letsencrypt:ro
```

---

## 📊 Полная архитектура после настройки

```
┌─────────────────────────────────────────────────────┐
│                    Internet                          │
└──────────────────┬──────────────────────────────────┘
                   │
                   │ HTTPS (443)
                   │
┌──────────────────▼──────────────────────────────────┐
│  foodgram-gateway-1 (Docker Nginx)                   │
│  - Порты 80, 443                                     │
│  - SSL сертификаты                                   │
│  - Проксирует на разные проекты                      │
└────┬──────────────────┬──────────────────────────────┘
     │                  │
     │ foodgram         │ voice-match.ru
     │ → :8000          │ → localhost:8081
     │                  │
     │           ┌──────▼────────────────────┐
     │           │ voice_match_nginx (Docker)│
     │           │ localhost:8081 → :80      │
     │           └──────┬────────────────────┘
     │                  │
     │                  │ Docker network
     │                  │
     │           ┌──────▼────────────────────┐
     │           │ voice_match_app (Docker)  │
     │           │ Gradio :7860              │
     │           └───────────────────────────┘
     │
┌────▼──────────────────────┐
│ Django/foodgram backend   │
│ localhost:8000            │
└───────────────────────────┘
```

---

## ✅ Чеклист

- [ ] DNS настроен на reg.ru
- [ ] DNS распространился (проверено через nslookup)
- [ ] SSL сертификат получен для voice-match.ru
- [ ] voice-match приложение запущено (docker-compose ps)
- [ ] Порт 8081 доступен (curl http://localhost:8081)
- [ ] Конфигурация добавлена в foodgram-gateway-1
- [ ] SSL сертификаты доступны в foodgram-gateway-1
- [ ] foodgram-gateway-1 перезапущен
- [ ] Сайт доступен по https://voice-match.ru
- [ ] Другие проекты работают нормально

---

Готово! После выполнения всех шагов ваш сайт будет доступен на https://voice-match.ru 🎉
