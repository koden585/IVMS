import sqlite3
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify
import requests
import matplotlib
matplotlib.use('Agg') # Позволяет рисовать графики без GUI
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import threading
import numpy as np
import queue
import json
import cv2
from flask_socketio import SocketIO

# Создаем глобальную очередь
telegram_queue = queue.Queue()

from config import CAMERAS, CHAT_ID, BOT_TOKEN, DB_PATH
from database import init_db

app = Flask(__name__)
# Инициализация WebSockets для высоконагруженной трансляции видео
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

# Инициализация структуры БД
init_db()

# Загрузка конфигурации камер и зон из БД в RAM
def load_cameras_to_memory():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Загружаем камеры
    c.execute('SELECT * FROM cameras')
    cam_rows = c.fetchall()

    for row in cam_rows:
        CAMERAS[row['cam_id']] = {
            'id': row['cam_id'],
            'name': row['name'],
            'url': row['url'],
            'zones': [],
            'violators': {},
            'alerted_violators': set(),
            'alert_seconds': row['alert_seconds'],
            'send_tg': bool(row['send_tg']),
            'last_frame': None,
            'fps': 0,
            'lock': threading.Lock()
        }

    # Загружаем зоны и раскладываем их по камерам
    c.execute('SELECT * FROM zones')
    zone_rows = c.fetchall()
    for row in zone_rows:
        cid = row['cam_id']
        if cid in CAMERAS:
            CAMERAS[cid]['zones'].append({
                'id': row['id'],
                'name': row['name'],
                'polygon': np.array(json.loads(row['coordinates']), np.int32),
                'classes': json.loads(row['classes']),
                'color': row['color'] if row['color'] else '#00a5ff',
                'alert_seconds': row['alert_seconds'] if row['alert_seconds'] is not None else 5
            })
    conn.close()

load_cameras_to_memory()  # Заполняем словарь CAMERAS!

# Импорт движка (строго после загрузки камер)
from video_engine import load_polygons, process_camera, generate_frames
from telegram_bot import send_report_to_queue



# ВЕБ-ИНТЕРФЕЙС

@app.route('/')
def index():
    # Главная - дашборд со всеми камерами (Dashboard)
    return render_template('dashboard.html', cameras=CAMERAS)

# Страница настройки конкретной камеры (геозоны)
@app.route('/camera/<cam_id>')
def camera_view(cam_id):
    if cam_id not in CAMERAS:
        return "Камера не найдена", 404
    return render_template('camera.html', cameras=CAMERAS, camera=CAMERAS[cam_id], cam_id=cam_id)

# Раздел архива инцидентов с фильтрацией
@app.route('/screenshots')
def screenshots_view():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM violations ORDER BY timestamp DESC')
    rows = c.fetchall()

    # Сбор всех ID камер для фильтра, включая удаленные
    c.execute('SELECT DISTINCT cam_id FROM violations')
    hist_cam_ids = [row['cam_id'] for row in c.fetchall()]
    conn.close()

    files_data = []
    unique_dates = set()
    unique_classes = set()

    # Собираем словарь имен для удаленных камер
    hist_cameras = {}
    for cid in hist_cam_ids:
        hist_cameras[cid] = CAMERAS.get(cid, {}).get('name', f"Удаленная камера ({cid})")

    for row in rows:
        dt = datetime.fromtimestamp(row['timestamp'])
        date_str = dt.strftime('%Y-%m-%d')
        unique_dates.add(date_str)
        unique_classes.add(row['label'])

        files_data.append({
            'filename': row['filename'],
            'cam_id': row['cam_id'],
            'cam_name': hist_cameras[row['cam_id']],
            'label': row['label'],
            'date': date_str,
            'time': dt.strftime('%H:%M:%S'),
            'duration': round(row['duration'], 1)
        })

    return render_template('screenshots.html',
                           cameras=CAMERAS, # для левого меню
                           hist_cameras=hist_cameras, # для фильтров
                           files=files_data,
                           dates=sorted(list(unique_dates), reverse=True),
                           classes=sorted(list(unique_classes)))

# Классический MJPEG поток (используется только на странице камеры)
@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):
    if cam_id not in CAMERAS:
        return "Not found", 404
    return Response(generate_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/snapshot/<cam_id>')
def snapshot(cam_id):
    """Возвращает один кадр (решает проблему лимита соединений браузера)"""
    if cam_id not in CAMERAS:
        return "Not found", 404

    cam = CAMERAS[cam_id]
    with cam['lock']:
        if cam['last_frame'] is not None:
            # Сжимаем кадр до 85% качества для легкости
            _, buffer = cv2.imencode('.jpg', cam['last_frame'], [cv2.IMWRITE_JPEG_QUALITY, 85])
            return Response(buffer.tobytes(), mimetype='image/jpeg')

    # Если кадра еще нет (камера только запускается)
    return "", 204

# Сохранение новой нарисованной зоны
@app.route('/api/add_zone/<cam_id>', methods=['POST'])
def api_add_zone(cam_id):
    if cam_id not in CAMERAS: return jsonify({"status": "error"})
    data = request.json
    points = data.get('points', [])
    name = data.get('name', 'Новая зона')
    classes = data.get('classes', [0])
    color = data.get('color', '#00a5ff')
    alert_seconds = int(data.get('alert_seconds', 5)) # <--- НОВОЕ

    if len(points) >= 3:
        # Пересчет координат из веб-разрешения в реальное разрешение видео
        scale_x = CAMERAS[cam_id]['width'] / data.get('width')
        scale_y = CAMERAS[cam_id]['height'] / data.get('height')
        real_points = [[int(p[0] * scale_x), int(p[1] * scale_y)] for p in points]

        from database import add_zone_to_db
        zone_id = add_zone_to_db(cam_id, name, real_points, classes, color, alert_seconds)

        with CAMERAS[cam_id]['lock']:
            CAMERAS[cam_id]['zones'].append({
                'id': zone_id, 'name': name, 'polygon': np.array(real_points, np.int32),
                'classes': classes, 'color': color, 'alert_seconds': alert_seconds
            })
        return jsonify({"status": "ok"})
    return jsonify({"status": "error", "message": "Нужно минимум 3 точки"})


@app.route('/api/delete_zone/<cam_id>/<int:zone_id>', methods=['POST'])
def api_delete_zone(cam_id, zone_id):
    if cam_id in CAMERAS:
        # Удаляем из БД
        from database import delete_zone_from_db
        delete_zone_from_db(zone_id)
        # Удаляем из памяти
        with CAMERAS[cam_id]['lock']:
            CAMERAS[cam_id]['zones'] = [z for z in CAMERAS[cam_id]['zones'] if z['id'] != zone_id]
    return jsonify({"status": "ok"})


@app.route('/reset/<cam_id>', methods=['POST'])
def reset(cam_id):
    if cam_id in CAMERAS:
        with CAMERAS[cam_id]['lock']:
            CAMERAS[cam_id]['violators'].clear()
            CAMERAS[cam_id]['alerted_violators'].clear()
    return jsonify({"status": "ok"})

# API для обновления статистики в реальном времени
@app.route('/stats/<cam_id>')
def stats(cam_id):
    if cam_id not in CAMERAS:
        return jsonify({"status": "error"}), 404
    cam = CAMERAS[cam_id]
    with cam['lock']:
        return jsonify({
            "zones_count": len(cam.get('zones', [])),
            "objects": len(cam['violators']),
            "violations": len(cam['alerted_violators'])
        })

# БЛОК НАСТРОЕК КАМЕР

@app.route('/settings')
def settings_view():
    return render_template('settings.html', cameras=CAMERAS)

# Динамическое добавление новой камеры
@app.route('/api/add_camera', methods=['POST'])
def api_add_camera():
    data = request.json
    cam_id = data['cam_id']
    from database import add_camera_to_db
    add_camera_to_db(cam_id, data['name'], data['url'], bool(data['send_tg']))

    CAMERAS[cam_id] = {
        'id': cam_id, 'name': data['name'], 'url': data['url'], 'zones': [],
        'violators': {}, 'alerted_violators': set(), 'send_tg': bool(data['send_tg']),
        'width': 1920, 'height': 1080, 'last_frame': None, 'fps': 0, 'lock': threading.Lock()
    }

    from video_engine import process_camera
    threading.Thread(target=process_camera, args=(cam_id,), daemon=True).start()
    return jsonify({"status": "ok"})


@app.route('/api/rename_camera/<cam_id>', methods=['POST'])
def api_rename_camera(cam_id):
    if cam_id in CAMERAS:
        new_name = request.json.get('name')
        if new_name:
            # Обновляем в БД
            from database import rename_camera_in_db
            rename_camera_in_db(cam_id, new_name)
            # Мгновенно обновляем в оперативной памяти
            with CAMERAS[cam_id]['lock']:
                CAMERAS[cam_id]['name'] = new_name
    return jsonify({"status": "ok"})

# Остановка потока и удаление камеры из системы
@app.route('/api/delete_camera/<cam_id>', methods=['POST'])
def api_delete_camera(cam_id):
    print(f"Поступил запрос на удаление камеры: {cam_id}")
    # Удаляем из базы данных в любом случае
    from database import delete_camera_from_db
    delete_camera_from_db(cam_id)

    # Если камера активна в памяти, глушим её
    if cam_id in CAMERAS:
        with CAMERAS[cam_id]['lock']:
            CAMERAS[cam_id]['url'] = ""  # Триггер остановки while-цикла в video_engine
        del CAMERAS[cam_id]
        print(f"Камера {cam_id} удалена из оперативной памяти.")
    else:
        print(f"Камера {cam_id} удалена только из БД (в памяти не найдена).")
    return jsonify({"status": "ok"})

# БЛОК АНАЛИТИКИ

@app.route('/analytics')
def analytics_view():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT DISTINCT cam_id FROM violations')
    hist_cam_ids = [row[0] for row in c.fetchall()]
    conn.close()
    hist_cameras = {}
    for cid in hist_cam_ids:
        hist_cameras[cid] = CAMERAS.get(cid, {}).get('name', f"Удаленная камера ({cid})")
    return render_template('analytics.html', cameras=CAMERAS, hist_cameras=hist_cameras)

# API подготовки JSON-данных для библиотеки Chart.js
@app.route('/api/analytics_data')
def api_analytics_data():
    cam_id = request.args.get('cam_id', 'all')
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    query = 'SELECT label, timestamp FROM violations'
    if cam_id == 'all':
        c.execute(query)
    else:
        c.execute(query + ' WHERE cam_id=?', (cam_id,))

    rows = c.fetchall()
    conn.close()

    # Обрабатываем данные
    dates_count, classes_count = {}, {}
    for row in rows:
        # Группировка по датам
        dt = datetime.fromtimestamp(row['timestamp']).strftime('%d.%m.%Y')
        dates_count[dt] = dates_count.get(dt, 0) + 1
        # Группировка по классам
        classes_count[row['label']] = classes_count.get(row['label'], 0) + 1

    # Сортируем даты
    sorted_dates = sorted(dates_count.keys())
    trend_data = [dates_count[d] for d in sorted_dates]

    return jsonify({
        "dates": sorted_dates,
        "trend_data": trend_data,
        "classes": list(classes_count.keys()),
        "classes_data": list(classes_count.values())
    })


def background_send_report(cam_name, counts, buf_value):
    """Кидаем график в очередь"""
    payload = {
        'chat_id': CHAT_ID,
        'caption': f"Запрошен отчет по объектам.\nИсточник: {cam_name}\nВсего зафиксировано: {sum(counts)}"
    }
    files = {'photo': ('report.png', buf_value, 'image/png')}

    # Кладем в почтовый ящик
    telegram_queue.put({'payload': payload, 'files': files})

# Генерация графиков (Matplotlib) и помещение отчета в очередь Telegram
@app.route('/api/send_report', methods=['POST'])
def send_report():
    cam_id = request.json.get('cam_id', 'all')
    cam_name = CAMERAS.get(cam_id, {}).get('name', 'Все камеры') if cam_id != 'all' else 'Все камеры'

    # Получаем данные
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if cam_id == 'all':
        c.execute('SELECT label, COUNT(*) FROM violations GROUP BY label')
    else:
        c.execute('SELECT label, COUNT(*) FROM violations WHERE cam_id=? GROUP BY label', (cam_id,))

    data = c.fetchall()
    conn.close()

    if not data:
        return jsonify({"status": "error", "message": "Нет данных для отчета"})

    labels = [row[0] for row in data]
    counts = [row[1] for row in data]

    # Потокобезопасная генерация графика (без GUI)
    fig = Figure(figsize=(8, 6))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    colors = ['#4fc3f7', '#ff5252', '#4caf50', '#ffa726', '#ba68c8'] * 5
    ax.bar(labels, counts, color=colors[:len(labels)])
    ax.set_title(f'Аналитика объектов: {cam_name}')
    ax.set_ylabel('Количество фиксаций')

    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)

    # Отправляем в очередь tg
    send_report_to_queue(cam_name, counts, buf.getvalue())
    return jsonify({"status": "ok", "message": "Отчет формируется и отправлен в очередь Telegram."})

# Трансляция видео по WebSockets
def websocket_video_stream():
    import time
    while True:
        time.sleep(0.2)  # Снижение FPS для дашборда (экономия CPU)
        for cam_id, cam in list(CAMERAS.items()):
            if not cam.get('url'):
                continue

            with cam['lock']:
                frame = cam.get('last_frame')

            if frame is not None:
                # Уменьшаем картинку специально для сетки дашборда
                small_frame = cv2.resize(frame, (640, 360))
                # Сжимаем с качеством 50% (для маленьких карточек этого не видно)
                ret, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                if ret:
                    socketio.emit(f'stream_{cam_id}', {'image': buffer.tobytes()})

# ЗАПУСК
if __name__ == '__main__':
    load_polygons()
    init_db()

    # Старт транслятора сокетов
    threading.Thread(target=websocket_video_stream, daemon=True).start()

    # Запускаем по одному потоку на каждую камеру
    for cam_id in CAMERAS:
        threading.Thread(target=process_camera, args=(cam_id,), daemon=True).start()

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)