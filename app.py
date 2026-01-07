# app.py
import streamlit as st
import cv2
import torch
import time
import tempfile
import threading
from collections import deque
from src.model import CNNLSTM
from src.utils import prepare_input, draw_interface


class AIState:
    latest_prob = 0.0
    latest_inf_time = 0.0
    is_processing = False
    buffer_id = 0


state = AIState()

st.set_page_config(page_title="Violence Detection AI", layout="wide")
st.title("🛡️ AI Система моніторингу безпеки")


@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = CNNLSTM()
    checkpoint = torch.load("weights/CNNLSTM_epoch2.pt", map_location=device)
    sd = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(sd)
    model.eval().float()
    return model, device


model, device = load_model()


def run_inference_engine(model_in, frames, dev, s_size):
    try:
        state.is_processing = True
        start = time.perf_counter()

        input_tensor = prepare_input(frames, stack_size=s_size).to(dev).float()
        with torch.no_grad():
            logits = model_in(input_tensor)
            prob = torch.sigmoid(logits).item()

        state.latest_prob = prob
        state.latest_inf_time = (time.perf_counter() - start) * 1000
        state.buffer_id += 1
    finally:
        state.is_processing = False


st.sidebar.header("⚙️ Керування")
source_type = st.sidebar.selectbox("Джерело:", ("Веб-камера", "Завантажити файл"))
uploaded_video = st.sidebar.file_uploader("Завантажте відео",
                                          type=["mp4"]) if source_type == "Завантажити файл" else None

threshold = st.sidebar.slider("Поріг (Threshold)", 0.3, 0.9, 0.6, 0.05)
stack_size_opt = st.sidebar.radio("Stack Size:", (16, 32), index=0)

if 'run' not in st.session_state: st.session_state.run = False
c1, c2 = st.sidebar.columns(2)
if c1.button("▶️ START"): st.session_state.run = True
if c2.button("⏹️ STOP"):
    st.session_state.run = False
    st.rerun()

col_vid, col_stat = st.columns([3, 1])
with col_vid:
    video_placeholder = st.empty()
with col_stat:
    st.subheader("📊 Аналітика")
    status_box = st.empty()
    p_metric = st.metric("Ймовірність", "0.0%")
    inf_metric = st.metric("Inference Time", "0 ms")
    fps_metric = st.metric("Display FPS", "0")
    total_lat_metric = st.metric("Total Latency", "0.0s")

if st.session_state.run:
    v_source = 0 if source_type == "Веб-камера" else None
    if uploaded_video:
        t_file = tempfile.NamedTemporaryFile(delete=False)
        t_file.write(uploaded_video.read())
        v_source = t_file.name

    cap = cv2.VideoCapture(v_source)
    buffer = deque(maxlen=75)
    time_buffer = deque(maxlen=75)

    frame_idx = 0
    prev_time = time.time()  # Для розрахунку FPS

    try:
        while cap.isOpened() and st.session_state.run:
            ret, frame = cap.read()
            if not ret: break

            current_time = time.time()
            frame_idx += 1
            buffer.append(frame.copy())
            time_buffer.append(current_time)

            # Розрахунок Display FPS
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # Запуск
            if len(buffer) >= stack_size_opt and frame_idx % 10 == 0:
                if not state.is_processing:
                    frames_copy = list(buffer)
                    threading.Thread(target=run_inference_engine,
                                     args=(model, frames_copy, device, stack_size_opt)).start()

            # Дані для виводу
            prob = state.latest_prob
            inf_ms = state.latest_inf_time
            label = "🚨 VIOLENCE" if prob > threshold else "✅ NORMAL"

            # Розрахунок загальної затримки (від найстарішого кадру в буфері до зараз)
            if len(time_buffer) > 0:
                total_lat = time.time() - time_buffer[0]
            else:
                total_lat = 0

            # Оновлення інтерфейсу
            status_box.subheader(f"Статус: {label}")
            p_metric.metric("Ймовірність", f"{prob * 100:.1f}%")
            inf_metric.metric("Inference Time", f"{int(inf_ms)} ms")
            fps_metric.metric("Display FPS", f"{int(fps)}")
            total_lat_metric.metric("Total Latency", f"{total_lat:.1f}s")

            # Рендеринг
            out = draw_interface(frame, prob, label.replace("🚨 ", "").replace("✅ ", ""), inf_ms)
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            video_placeholder.image(out, channels="RGB", use_container_width=True)

            # Невелика пауза (для стабільності UI)
            time.sleep(0.01)

    finally:
        cap.release()
        st.session_state.run = False
        st.rerun()