import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.applications.resnet50 import preprocess_input

from .config import EMOTION_MODEL_PATH, IMG_SIZE, CLASS_NAMES
from .utils import apply_emotion_threshold

IMG_SIZE_LR = (48, 48)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def load_emotion_model():
    import os
    print("Model path:", EMOTION_MODEL_PATH)
    print("Exists?", os.path.exists(EMOTION_MODEL_PATH))
    model = tf.keras.models.load_model(EMOTION_MODEL_PATH)
    return model


def _to_hr_infer(gray48_uint8):
    hr_uint8 = cv2.resize(gray48_uint8, IMG_SIZE)
    return hr_uint8


def _preprocess_face(face_bgr):
    h, w = face_bgr.shape[:2]
    if h < 20 or w < 20:
        return None

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray48 = cv2.resize(gray, IMG_SIZE_LR, interpolation=cv2.INTER_AREA)
    gray48_clahe = clahe.apply(gray48)
    hr_uint8 = cv2.resize(gray48_clahe, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    rgb_uint8 = cv2.cvtColor(hr_uint8, cv2.COLOR_GRAY2RGB)
    x = rgb_uint8.astype("float32")
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def predict_emotion(model, face_img):
    x1 = _preprocess_face(face_img)
    if x1 is None:
        return "unknown", 0.0, np.zeros(len(CLASS_NAMES), dtype="float32")

    preds1 = model.predict(x1, verbose=0)[0]

    face_flip = cv2.flip(face_img, 1)
    x2 = _preprocess_face(face_flip)
    if x2 is not None:
        preds2 = model.predict(x2, verbose=0)[0]
        preds = (preds1 + preds2) / 2.0
    else:
        preds = preds1

    conf = float(np.max(preds))
    label_idx = int(np.argmax(preds))
    label = CLASS_NAMES[label_idx]
    label = apply_emotion_threshold(label, conf)

    return label, conf, preds
