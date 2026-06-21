import cv2
import time
import json
import numpy as np
from collections import deque
from ultralytics import YOLO
import torch

from database import add_violation_to_db, update_violation_duration
from telegram_bot import send_telegram_alert_bg
from config import CAMERAS, MODEL_PATH

# Оптимизация нагрузки на CPU: запрет на захват всех ядер библиотекой PyTorch.
# Расчеты переносятся на GPU (TensorRT).
torch.set_num_threads(1)

# Конвертирует цвет из HTML (#FF0000) в формат OpenCV BGR
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)
    return (0, 165, 255)

# Переводит русский текст в латиницу, так как cv2 не поддерживает кириллицу
def translit(text):
    symbols = {'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo', 'ж': 'zh',
               'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o',
               'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'ts',
               'ч': 'ch', 'ш': 'sh', 'щ': 'sch', 'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya'}
    return "".join(symbols.get(c.lower(), c.lower()) if c.islower() else symbols.get(c.lower(), c.lower()).upper() for c in text)

# Проверка пересечения рамки объекта с полигоном (5 контрольных точек: 4 угла + центр)
def is_in_zone(box, polygon):
    x1, y1, x2, y2 = box
    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), ((x1 + x2) // 2, (y1 + y2) // 2)]
    inside = sum(1 for p in points if cv2.pointPolygonTest(polygon, p, False) >= 0)
    return inside >= 1


def save_polygons():
    data = {}
    for cid, cdata in CAMERAS.items():
        if cdata['has_polygon']:
            data[cid] = cdata['polygon'].tolist()
    with open('polygons.json', 'w') as f:
        json.dump(data, f)


def load_polygons():
    try:
        with open('polygons.json', 'r') as f:
            data = json.load(f)
            for cid, points in data.items():
                if cid in CAMERAS and len(points) >= 3:
                    CAMERAS[cid]['polygon'] = np.array(points, np.int32)
                    CAMERAS[cid]['has_polygon'] = True
            print("Полигоны загружены!")
    except Exception as e:
        print("Полигоны не найдены, создаем новые.")

# Основной рабочий цикл (Background Thread) для каждой камеры
def process_camera(cam_id):
    cam = CAMERAS[cam_id]

    # Инициализация нейросети с явным указанием задачи для TensorRT
    model = YOLO(MODEL_PATH, task='detect')

    if not cam.get('url'):
        return

    # Попытка инициализации аппаратного декодера видео (HW Acceleration)
    cap = cv2.VideoCapture(cam['url'], cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

    # Жесткое ограничение буфера для предотвращения отставания (задержки) видео
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Принудительно устанавливаем сниженное разрешение для экономии CPU
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960) # 1280
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540) # 720

    cam['width'] = 960
    cam['height'] = 540

    #cam['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #cam['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_delay = 1 / fps
    fps_deque = deque(maxlen=30)
    prev_time = time.time()

    # Определяем реальный FPS видеофайла
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 60:
        fps = 25.0
    frame_delay = 1.0 / fps

    while True:
        # Корректное завершение потока при удалении камеры
        if cam_id not in CAMERAS or not cam.get('url'):
            break

        loop_start = time.time()  # Засекаем начало обработки кадра
        ret, frame = cap.read()

        # Эмуляция бесконечной работы при использовании видеофайлов (.mp4)
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Ресайз кадра перед анализом
        frame = cv2.resize(frame, (960, 540))
        display = frame.copy()
        current_ids = []

        # Сбор активных классов (на что нужно реагировать) из всех зон
        active_classes = set()
        with cam['lock']:
            zones = list(cam.get('zones', []))

        for z in zones:
            active_classes.update(z['classes'])

        active_classes_list = list(active_classes)
        if not active_classes_list:
            active_classes_list = [0] # Дефолт - поиск людей

        # Инференс нейросети. Параметр half=True использует 16-битные вычисления GPU.
        results = model.track(frame, conf=0.25, classes=active_classes_list,
                              device='cuda:0', half=True, verbose=False, persist=True, max_det=15)

        # Отрисовка всех геозон на экране
        for z in zones:
            pts = z['polygon']
            color_bgr = hex_to_bgr(z.get('color', '#00a5ff'))
            safe_name = translit(z['name'])
            cv2.polylines(display, [pts], True, color_bgr, 2)
            cv2.putText(display, safe_name, (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        # Проверка объектов
        for r in results:
            for box in r.boxes:
                if box.id is None: continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = box.id.int().item()
                cls_id = int(box.cls)
                label = r.names[cls_id]

                # Проверка: находится ли объект в одной из зон, реагирующих на его класс
                active_zones_for_object = []
                for z in zones:
                    if cls_id in z['classes'] and is_in_zone((x1, y1, x2, y2), z['polygon']):
                        active_zones_for_object.append(z)

                if not active_zones_for_object:
                    continue  # Игнорируем объекты вне нужных зон

                current_ids.append(track_id)

                with cam['lock']:
                    # Логика трекинга времени
                    if track_id not in cam['violators']:
                        cam['violators'][track_id] = {'label': label, 'zones_entered': {}}

                    violator = cam['violators'][track_id]
                    is_alerted_anywhere = False

                    # Проверка таймера для каждой зоны индивидуально
                    for z in active_zones_for_object:
                        zid = z['id']
                        if zid not in violator['zones_entered']:
                            violator['zones_entered'][zid] = {'time': time.time()}

                        elapsed = time.time() - violator['zones_entered'][zid]['time']
                        alert_key = f"{track_id}_{zid}"  # (Объект + Зона)

                        # Первичная фиксация (превышение лимита времени)
                        if elapsed >= z['alert_seconds'] and alert_key not in cam['alerted_violators']:
                            cam['alerted_violators'].add(alert_key)
                            timestamp = int(time.time())
                            safe_label = label.replace(' ', '-')
                            filename = f"{cam_id}_z{zid}_{safe_label}_{track_id}_{timestamp}.jpg"

                            violator['zones_entered'][zid]['filename'] = filename

                            add_violation_to_db(cam_id, label, track_id, elapsed, timestamp, filename)

                            # Передаем данные о зоне в tg
                            send_telegram_alert_bg(cam_id, track_id, label, elapsed, frame, (x1, y1, x2, y2),
                                                    filename, z)

                        if alert_key in cam['alerted_violators']:
                            is_alerted_anywhere = True

                    color = (0, 0, 255) if is_alerted_anywhere else (0, 255, 0)

                # Визуальная подсветка нарушителя
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"{label} ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,
                            2)

        # Очистка объектов, покинувших кадр, и обновление БД
        with cam['lock']:
            for tid in list(cam['violators'].keys()):
                if tid not in current_ids:
                    violator = cam['violators'][tid]
                    # Проверяем все зоны, где он был
                    for zid, zdata in violator['zones_entered'].items():
                        alert_key = f"{tid}_{zid}"
                        if alert_key in cam['alerted_violators']:
                            total_duration = time.time() - zdata['time']
                            filename = zdata.get('filename')
                            if filename:
                                update_violation_duration(filename, total_duration)
                            cam['alerted_violators'].discard(alert_key)
                    del cam['violators'][tid]

        with cam['lock']:
            cam['last_frame'] = display

        # Жесткая синхронизация скорости (Throttling) для снижения нагрузки на CPU
        processing_time = time.time() - loop_start
        wait_time = frame_delay - processing_time

        if wait_time > 0:
            time.sleep(wait_time)

        """# Считаем реальный FPS
        curr_time = time.time()
        fps_deque.append(1 / (curr_time - loop_start))
        cam['fps'] = sum(fps_deque) / len(fps_deque)

        cv2.putText(display, f"{cam['name']} FPS:{cam['fps']:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)"""

    cap.release()


def generate_frames(cam_id):
    cam = CAMERAS[cam_id]
    while True:
        with cam['lock']:
            if cam['last_frame'] is not None:
                _, buffer = cv2.imencode('.jpg', cam['last_frame'], [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.033)