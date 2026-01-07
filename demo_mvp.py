# demo_mvp.py
import cv2
import torch
import time
import csv
import threading
from collections import deque

from src.model import CNNLSTM
from src.utils import prepare_input, draw_interface

# --- КОНФІГУРАЦІЯ ---
MODEL_PATH = "weights/CNNLSTM_epoch2.pt"
VIDEO_PATH = "test_videos/V_1.mp4"
STACK_SIZE = 16
WINDOW_SIZE = 75
STRIDE = 10
LOG_FILE = "profiling_log.csv"

latest_prob = 0.0
latest_inf_time = 0.0
latest_prep_time = 0.0
latest_latency = 0.0
is_processing = False  # Прапор: чи зайнята модель зараз


def run_inference(model, frames_list, device, buffer_id, oldest_frame_time):
    global latest_prob, latest_inf_time, latest_prep_time, latest_latency, is_processing

    try:
        # 1. Препроцесинг (включаємо в потік, щоб не гальмувати відео)
        start_prep = time.perf_counter()
        input_tensor = prepare_input(frames_list, stack_size=STACK_SIZE)
        input_tensor = input_tensor.to(device).float()
        prep_time = (time.perf_counter() - start_prep) * 1000

        # 2. Інференс
        start_inf = time.perf_counter()
        with torch.no_grad():
            logits = model(input_tensor)
            prob = torch.sigmoid(logits).item()
        inf_time = (time.perf_counter() - start_inf) * 1000

        # 3. Розрахунок повної затримки
        total_latency = time.time() - oldest_frame_time

        # Оновлюємо глобальні результати
        latest_prob = prob
        latest_inf_time = inf_time
        latest_prep_time = prep_time
        latest_latency = total_latency

        # Запис у лог
        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [buffer_id, "N/A", round(prep_time, 2), round(inf_time, 2), round(total_latency, 4), round(prob, 4)])

        print(f"[{buffer_id}] Result Ready: {prob:.2f} | Inf: {inf_time:.1f}ms | Latency: {total_latency:.2f}s")

    finally:
        is_processing = False  # Звільняємо модель для наступного кроку


def main():
    global is_processing, latest_prob, latest_inf_time

    # 0. Підготовка файлу для логування
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['buffer_id', 'data_age_s', 'prep_ms', 'inf_ms', 'total_latency_s', 'prob'])

    device = torch.device('cpu')
    print(f"Використовується пристрій: {device}")

    # 2. Завантаження моделі
    try:
        model = CNNLSTM()
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device).float().eval()
        print("Модель успішно завантажена!")
    except Exception as e:
        print(f"Помилка завантаження моделі: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    buffer = deque(maxlen=WINDOW_SIZE)
    time_buffer = deque(maxlen=WINDOW_SIZE)

    frame_count = 0
    buffer_id = 0

    print(f"Починаємо асинхронний моніторинг...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_timestamp = time.time()

        # Наповнюємо буфер
        buffer.append(frame.copy())
        time_buffer.append(current_timestamp)

        # 4. АСИНХРОННИЙ ЗАПУСК МОДЕЛІ
        # Запускаємо модель тільки якщо вона вільна (not is_processing)
        if len(buffer) >= STACK_SIZE and frame_count % STRIDE == 0 and not is_processing:
            is_processing = True
            buffer_id += 1

            # Робимо "зліпок" (snapshot) поточного буфера для передачі в потік
            snapshot = list(buffer)
            oldest_frame_time = time_buffer[0]

            # Створюємо та запускаємо фоновий потік
            thread = threading.Thread(
                target=run_inference,
                args=(model, snapshot, device, buffer_id, oldest_frame_time)
            )
            thread.daemon = True  # Потік помре разом з основною програмою
            thread.start()

        current_label = "VIOLENCE" if latest_prob > 0.5 else "NORMAL"

        output_frame = draw_interface(frame, latest_prob, current_label, latest_inf_time)
        cv2.imshow('Violence Detection MVP - AI System', output_frame)

        # Регулювання швидкості відображення (блязько 30 fps)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Систему зупинено.")


if __name__ == "__main__":
    cv2.namedWindow('Violence Detection MVP - AI System', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Violence Detection MVP - AI System', 1300, 800)
    main()
