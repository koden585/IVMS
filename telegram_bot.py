import queue
import threading
import requests
import cv2
import os
import time
from config import BOT_TOKEN, CHAT_ID, CAMERAS
import urllib3

# Отключаем предупреждения о проверке SSL сертификатов
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Глобальная очередь задач (Паттерн Producer-Consumer)
telegram_queue = queue.Queue()

# Фоновый почтальон. Асинхронно отправляет фото в Telegram. Защищает основной видеопоток от зависаний при отсутствии интернета.
def telegram_worker():
    while True:
        try:
            task = telegram_queue.get()
            url = f"https://testwork.danilkakoc.workers.dev/bot{BOT_TOKEN}/sendPhoto"
            retries = task.get('retries', 0)

            try:
                # Отправка с таймаутом, чтобы не зависать бесконечно
                requests.post(url, data=task['payload'], files=task['files'], verify=False, timeout=20)

            except requests.exceptions.ReadTimeout:
                # Запрос ушел, но ТГ долго отвечает. Считаем успешным, чтобы избежать дубликатов.
                print("Долгий ответ от ТГ. Считаем фото доставленным (без повтора).")

            except Exception as e:
                # Реализация повторов при обрыве сети
                if retries < 2:  # Делаем максимум 2 повтора, чтобы не копить мусор
                    print(f"Сеть недоступна. Попытка {retries + 1}/2. Возврат в очередь...")
                    task['retries'] = retries + 1
                    time.sleep(5)
                    telegram_queue.put(task)
                else:
                    print("Лимит попыток исчерпан. Фото удалено из очереди.")

            telegram_queue.task_done()
        except Exception as e:
            print(f"Ошибка в потоке почтальона: {e}")


# Инициализация воркера при загрузке модуля
threading.Thread(target=telegram_worker, daemon=True).start()


# Формирует изображение с графическими маркерами и помещает задачу в очередь
def send_telegram_alert_bg(cam_id, track_id, label, duration, frame, box, filename, zone_data=None):
    cam_name = CAMERAS.get(cam_id, {}).get('name', cam_id)
    try:
        x1, y1, x2, y2 = box
        screenshot = frame.copy()

        # Отрисовка ограничивающей рамки объекта
        cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Отрисовка сработавшей геозоны и её названия
        if zone_data:
            # Конвертация HEX-цвета (#RRGGBB) в формат BGR для OpenCV
            h = zone_data.get('color', '#00a5ff').lstrip('#')
            bgr = (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)) if len(h) == 6 else (0, 165, 255)

            pts = zone_data['polygon']
            cv2.polylines(screenshot, [pts], True, bgr, 2)

            # Пишем название зоны
            from video_engine import translit
            cv2.putText(screenshot, translit(zone_data['name']), (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)

        # Текстовые метки объекта
        cv2.putText(screenshot, f"ID:{track_id} {label}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Сохранение скриншота на диск
        filepath = os.path.join("static", "violations", filename)
        cv2.imwrite(filepath, screenshot)

        # Подготовка данных для HTTP POST запроса
        _, buffer = cv2.imencode('.jpg', screenshot, [cv2.IMWRITE_JPEG_QUALITY, 90])
        photo_bytes = buffer.tobytes()

        # Если передали зону, добавляем её в текст сообщения
        zone_text = f" в зоне «{zone_data['name']}»" if zone_data else " в зоне."

        payload = {
            'chat_id': CHAT_ID,
            'caption': f"Камера: {cam_name}\nНайден нарушитель! {label} (ID:{track_id})\nЗафиксирован{zone_text}"
        }
        files = {'photo': ('photo.jpg', photo_bytes, 'image/jpeg')}

        # Помещение задачи в асинхронную очередь
        telegram_queue.put({'payload': payload, 'files': files})

    except Exception as e:
        print(f"Ошибка подготовки фото: {e}")


# Помещение сгенерированного графика аналитики в очередь отправки
def send_report_to_queue(cam_name, counts, buf_value):
    payload = {
        'chat_id': CHAT_ID,
        'caption': f"Запрошен отчет по объектам.\nИсточник: {cam_name}\nВсего зафиксировано: {sum(counts)}"
    }
    files = {'photo': ('report.png', buf_value, 'image/png')}
    telegram_queue.put({'payload': payload, 'files': files})