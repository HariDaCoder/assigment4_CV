import os
import cv2

from .config import (
    BOX_COLOR,
    BOX_THICKNESS,
    TEXT_COLOR,
    TEXT_FONT_SCALE,
    TEXT_THICKNESS,
    EMOTION_MIN_CONF,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def apply_emotion_threshold(label: str, conf: float) -> str:
    if conf < EMOTION_MIN_CONF:
        return "unknown"
    return label


def draw_box_and_label(img_bgr, box, label: str, conf: float):
    x1, y1, x2, y2 = box
    percent = conf * 100.0
    text = f"{label} {percent:.1f}%"

    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

    y_text = max(0, y1 - 10)
    cv2.putText(
        img_bgr, text, (x1, y_text),
        cv2.FONT_HERSHEY_SIMPLEX, TEXT_FONT_SCALE,
        TEXT_COLOR, TEXT_THICKNESS, lineType=cv2.LINE_AA,
    )

    return img_bgr


def print_emotion_vector(label: str, conf: float, preds_vector):
    print(f"🎯 Emotion: {label} ({conf:.4f})")
    print("🔢 Softmax vector:", preds_vector)
