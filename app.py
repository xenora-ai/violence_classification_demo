# app.py
import os
import streamlit as st
import cv2
import torch
import time
import tempfile
import threading
import psutil
from collections import deque
from src.model import CNNLSTM
from src.utils import preprocess_frame, draw_interface
import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore")


class AIState:
    latest_prob = 0.0
    latest_inf_time = 0.0
    latest_reaction_time = 0.0
    buffer_duration = 0.0
    is_processing = False
    buffer_id = 0
    latest_cpu = 0.0
    latest_ram = 0.0
    prob_history = deque(maxlen=50)
    cpu_history = deque(maxlen=50)

    log_df = pd.DataFrame(columns=['Buffer_ID', 'Reaction_ms', 'Memory_Window_s', 'Prob', 'CPU_%'])
    feature_buffer = deque(maxlen=32)


state = AIState()

st.set_page_config(page_title="Violence Classification System", layout="wide")
st.title("🛡️ AI Система моніторингу безпеки")


@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = CNNLSTM()
    checkpoint = torch.load("weights/model.pt", map_location=device)
    sd = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(sd)
    model.eval().float()
    return model, device


model, device = load_model()


def extract_cnn_features(model_cnn, frame):
    """Проганяє кадр через CNN і повертає feature vector [1280]"""
    frame_tensor = torch.from_numpy(preprocess_frame(frame)).unsqueeze(0).float()  # [1,C,H,W]
    with torch.no_grad():
        feat_map = model_cnn.cnn(frame_tensor)
        feat_map = model_cnn.pool(feat_map)
        feat_vec = feat_map.view(-1)  # [1280]
    return feat_vec.cpu().numpy()


def run_lstm_inference(model_lstm, dev, stack_size, cnn_time_ms):
    """Бере останні stack_size feature vector з buffer та робить inference тільки через LSTM"""
    if len(state.feature_buffer) < stack_size:
        return

    try:
        state.is_processing = True
        start_inf = time.perf_counter()
        feats = list(state.feature_buffer)[-stack_size:]
        x = np.stack(feats, axis=0)
        x = torch.from_numpy(x).unsqueeze(0).to(dev).float()

        with torch.no_grad():
            lstm_out, (h_n, _) = model_lstm.lstm(x)

            if getattr(model_lstm, 'temporal_pooling', False):
                feat = lstm_out.mean(dim=1)
            else:
                feat = h_n[-1]

            logits = model_lstm.classifier(feat)
            prob = torch.sigmoid(logits).item()

        inf_time = (time.perf_counter() - start_inf) * 1000  # ms

        reaction_time = cnn_time_ms + inf_time

        state.latest_prob = prob
        state.latest_inf_time = inf_time
        state.latest_reaction_time = reaction_time
        state.buffer_id += 1

        new_row = {
            'Buffer_ID': state.buffer_id,
            'Reaction_ms': round(reaction_time, 2),
            'Memory_Window_s': round(state.buffer_duration, 2),
            'Prob': round(prob, 3),
            'CPU_%': state.latest_cpu
        }
        state.log_df = pd.concat([state.log_df, pd.DataFrame([new_row])], ignore_index=True)

    finally:
        state.is_processing = False


st.sidebar.header("⚙️ Керування")
source_type = st.sidebar.selectbox("Джерело:", ("Веб-камера", "Завантажити файл"))
uploaded_video = st.sidebar.file_uploader("Завантажте відео",
                                          type=["mp4"]) if source_type == "Завантажити файл" else None
threshold = st.sidebar.slider("Поріг (Threshold)", 0.3, 0.9, 0.6, 0.05)
stack_size_opt = st.sidebar.radio("Stack Size:", (8, 16, 32), index=0)

if 'run' not in st.session_state: st.session_state.run = False
c1, c2 = st.sidebar.columns(2)
if c1.button("▶️ START"): st.session_state.run = True
if c2.button("⏹️ STOP"):
    st.session_state.run = False
    st.rerun()

col_vid, col_stat = st.columns([3, 2])

with col_vid:
    video_placeholder = st.empty()
    st.markdown("---")
    st.subheader("📈 Динаміка ймовірності загрози")
    chart_prob_placeholder = st.empty()

with col_stat:
    st.subheader("🛡️ Статус системи")
    status_box = st.empty()
    prob_bar = st.empty()

    st.markdown("### ⚡ Продуктивність AI")
    c3, c4, c5 = st.columns(3)
    inf_metric = c3.metric("Час реакції", "0 ms")
    total_lat_metric = c4.metric("Вікно пам'яті", "0.0 s")
    fps_metric = c5.metric("FPS", "0")

    st.markdown("### 💻 Ресурси системи")
    c6, c7 = st.columns(2)
    cpu_metric = c6.metric("CPU Load", "0%")
    ram_metric = c7.metric("RAM Used", "0 MB")

    st.markdown("---")
    st.subheader("📊 Навантаження CPU")
    chart_cpu_placeholder = st.empty()

    st.markdown("---")
    st.subheader("Лог (останні 5 записів)")
    log_table = st.empty()

if st.session_state.run:
    v_source = 0 if source_type == "Веб-камера" else None
    if uploaded_video:
        t_file = tempfile.NamedTemporaryFile(delete=False)
        t_file.write(uploaded_video.read())
        v_source = t_file.name

    cap = cv2.VideoCapture(v_source)
    frame_idx = 0
    prev_time = time.time()

    FRAME_STRIDE = 3

    try:
        while cap.isOpened() and st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            is_key_frame = (frame_idx % FRAME_STRIDE == 0)

            if is_key_frame:
                current_cpu = psutil.cpu_percent()
                current_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                state.latest_cpu = current_cpu
                state.latest_ram = current_ram
            else:
                current_cpu = state.latest_cpu
                current_ram = state.latest_ram

            if is_key_frame:
                start_cnn = time.perf_counter()
                feat_vec = extract_cnn_features(model, frame)

                if len(state.feature_buffer) == 0:
                    for _ in range(stack_size_opt):
                        state.feature_buffer.append(feat_vec)
                else:
                    state.feature_buffer.append(feat_vec)

                cnn_time_ms = (time.perf_counter() - start_cnn) * 1000

                actual_stack = min(len(state.feature_buffer), stack_size_opt)
                if fps > 0:
                    state.buffer_duration = (actual_stack * FRAME_STRIDE / fps)
                else:
                    state.buffer_duration = 0

                if not state.is_processing and len(state.feature_buffer) >= stack_size_opt:
                    threading.Thread(target=run_lstm_inference,
                                     args=(model, device, stack_size_opt, cnn_time_ms),
                                     daemon=True
                                     ).start()

            prob = state.latest_prob
            reaction_ms = state.latest_reaction_time
            buf_dur = state.buffer_duration

            state.prob_history.append(prob)
            state.cpu_history.append(current_cpu)

            label = "🚨 НАЯВНА ЗАГРОЗА" if prob > threshold else "✅ СИТУАЦІЯ В НОРМІ"
            out = draw_interface(frame, prob, "VIOLENCE" if prob > threshold else "NORMAL", reaction_ms)
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            video_placeholder.image(out, channels="RGB", width='stretch')

            if is_key_frame:
                status_color = "#FF4B4B" if prob > threshold else "#00CC96"

                status_box.markdown(f"<h3 style='text-align: center; color: {status_color};'>{label}</h3>",
                                    unsafe_allow_html=True)
                prob_bar.progress(prob, text=f"Впевненість моделі: {prob * 100:.1f}%")

                inf_metric.metric("Час реакції", f"{int(reaction_ms)} ms")
                total_lat_metric.metric("Вікно пам'яті", f"{buf_dur:.1f} s")
                fps_metric.metric("Display FPS", f"{int(fps)}")

                cpu_metric.metric("CPU Load", f"{current_cpu:.1f}%")
                ram_metric.metric("RAM Used", f"{int(current_ram)} MB")

                chart_prob_placeholder.line_chart(list(state.prob_history), height=150)
                chart_cpu_placeholder.area_chart(list(state.cpu_history), height=150)
                log_table.dataframe(state.log_df.tail(5), width='stretch')

            time.sleep(0.01)
    finally:
        cap.release()
        st.session_state.run = False
        st.rerun()
