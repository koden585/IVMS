import sqlite3
import json
from config import DB_PATH, CAMERAS_CONFIG

# Создает таблицы при первом запуске и выполняет миграции структуры БД
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 1. Основной журнал фиксации инцидентов
    c.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cam_id TEXT, label TEXT, track_id INTEGER,
            duration REAL, timestamp INTEGER, filename TEXT
        )
    ''')

    # 2. Таблица настроек видеопотоков (камер)
    c.execute('''
        CREATE TABLE IF NOT EXISTS cameras (
            cam_id TEXT PRIMARY KEY, name TEXT, url TEXT,
            classes TEXT, alert_seconds INTEGER, send_tg BOOLEAN
        )
    ''')

    # 3. Таблица зон контроля
    c.execute('''
            CREATE TABLE IF NOT EXISTS zones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cam_id TEXT,
                name TEXT,
                coordinates TEXT,
                classes TEXT,
                color TEXT
            )
        ''')

    # Блок миграций (добавление новых колонок в старые БД без потери данных)
    try:
        c.execute("ALTER TABLE zones ADD COLUMN color TEXT DEFAULT '#00a5ff'")
        print("База данных обновлена: добавлена колонка color в таблицу zones")
    except sqlite3.OperationalError:
        # Если колонка уже есть, SQLite выдаст ошибку, то её просто игнорируем
        pass

    # Добавление лимита времени для зон
    try:
        c.execute("ALTER TABLE zones ADD COLUMN alert_seconds INTEGER DEFAULT 5")
        print("БД обновлена: добавлена колонка alert_seconds в zones")
    except:
        pass

    # Перенос дефолтных настроек из config.py при "чистом" запуске
    c.execute('SELECT COUNT(*) FROM cameras')
    if c.fetchone()[0] == 0:
        for cid, cfg in CAMERAS_CONFIG.items():
            classes_str = json.dumps(cfg['classes'])
            c.execute('''
                INSERT INTO cameras (cam_id, name, url, classes, alert_seconds, send_tg)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (cid, cfg['name'], cfg['url'], classes_str, cfg.get('alert_seconds', 5), cfg.get('send_tg', True)))

    conn.commit()
    conn.close()
    print("✅ База данных SQLite проверена и инициализирована!")


# Первичная запись факта нарушения (при превышении лимита времени)
def add_violation_to_db(cam_id, label, track_id, duration, timestamp, filename):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO violations (cam_id, label, track_id, duration, timestamp, filename)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (cam_id, label, track_id, duration, timestamp, filename))
    conn.commit()
    conn.close()


# Обновление итогового времени нахождения в зоне после ухода объекта
def update_violation_duration(filename, max_duration):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        UPDATE violations SET duration = ? WHERE filename = ?
    ''', (max_duration, filename))
    conn.commit()
    conn.close()


# Сохранение новой камеры через веб-интерфейс
def add_camera_to_db(cam_id, name, url, send_tg):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO cameras (cam_id, name, url, classes, alert_seconds, send_tg)
        VALUES (?, ?, ?, '[]', 0, ?)
    ''', (cam_id, name, url, send_tg))
    conn.commit()
    conn.close()


# Удаление камеры и всех привязанных к ней геозон
def delete_camera_from_db(cam_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM cameras WHERE cam_id=?', (cam_id,))
    c.execute('DELETE FROM zones WHERE cam_id=?', (cam_id,))
    conn.commit()
    conn.close()


# Запись координат новой нарисованной зоны (сохраняются как JSON)
def add_zone_to_db(cam_id, name, coordinates, classes, color, alert_seconds):
    import json
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO zones (cam_id, name, coordinates, classes, color, alert_seconds)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (cam_id, name, json.dumps(coordinates), json.dumps(classes), color, alert_seconds))
    zone_id = c.lastrowid
    conn.commit()
    conn.close()
    return zone_id


# Удаление геозоны
def delete_zone_from_db(zone_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM zones WHERE id=?', (zone_id,))
    conn.commit()
    conn.close()

# Переименование камеры
def rename_camera_in_db(cam_id, new_name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE cameras SET name=? WHERE cam_id=?', (new_name, cam_id))
    conn.commit()
    conn.close()