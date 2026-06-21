import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
MODEL_PATH = "models/yolo26s.engine" # Путь к скомпилированной модели TensorRT
DB_PATH = "vidmon_sys.db"

# Создаем директорию для хранения скриншотов нарушений, если её нет
os.makedirs("static/violations", exist_ok=True)

# Базовый конфиг для самого первого запуска (если база данных пуста)
CAMERAS_CONFIG = {
    "cam1": {"name": "Камера 1", "url": "...", "classes": [2, 3, 5, 7], "alert_seconds": 10, "send_tg": True},
}

# Глобальный словарь оперативной памяти. Хранит активные камеры, зоны и текущие состояния трекинга.
# Заполняется из базы данных при старте приложения.
CAMERAS = {}