# src/utils.py
import cv2
import numpy as np
import torch


MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def preprocess_frame(frame, resize_size=(224, 224)):
    frame = cv2.resize(frame, resize_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = frame.astype(np.float32) / 255.0
    frame = (frame - MEAN) / STD

    frame = np.transpose(frame, (2, 0, 1))
    return frame


def prepare_input(buffer, stack_size=32):
    """
    Бере список кадрів (буфер), робить рівномірний семплінг
    та перетворює на тензор [1, T, C, H, W]
    """
    indices = np.linspace(0, len(buffer) - 1, stack_size, dtype=int)
    sampled_frames = [buffer[i] for i in indices]

    processed_stack = [preprocess_frame(f) for f in sampled_frames]

    tensor = np.array(processed_stack)
    tensor = torch.from_numpy(tensor).unsqueeze(0)

    return tensor


def draw_interface(frame, prob, label, inference_time):
    """Малювання HUD на кадрі"""
    h, w, _ = frame.shape
    color = (0, 0, 255) if label == "VIOLENCE" else (0, 255, 0)

    cv2.rectangle(frame, (0, 0), (w, h), color, 15)

    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (450, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"STATUS: {label}", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"CONFIDENCE: {prob * 100:.1f}%", (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    cv2.putText(frame, f"INF. TIME: {int(inference_time)}ms", (40, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return frame
