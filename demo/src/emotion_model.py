import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.applications.resnet50 import preprocess_input

from .config import EMOTION_MODEL_PATH, IMG_SIZE, CLASS_NAMES
from .utils import apply_emotion_threshold

# Ảnh gốc FER dùng 48x48
IMG_SIZE_LR = (48, 48)

# CLAHE giống trong data_preprocessing.py
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def load_emotion_model():
    """Tải mô hình ResNet cảm xúc đã train."""
    import os
    print("Model path:", EMOTION_MODEL_PATH)
    print("Exists?", os.path.exists(EMOTION_MODEL_PATH))
    model = tf.keras.models.load_model(EMOTION_MODEL_PATH)
    return model


def _to_hr_infer(gray48_uint8):
    """
    Tăng từ 48x48 lên 224x224 giống bước to_hr (nhánh không dùng ESPCN).
    Input: uint8 [0..255], shape (48,48)
    Output: uint8 [0..255], shape IMG_SIZE (224,224)
    """
    hr_uint8 = cv2.resize(gray48_uint8, IMG_SIZE)  # giữ nguyên 0..255
    return hr_uint8


def _preprocess_face(face_bgr):
    """
    Pipeline infer BÁM SÁT train:

    BGR (webcam) ->
        GRAY -> resize (48,48) -> CLAHE (uint8 0..255) ->
        resize lên IMG_SIZE (224,224, uint8) ->
        chuyển sang RGB 3 kênh ->
        float32 -> preprocess_input -> (1,224,224,3)
    """
    # 0) Nếu mặt quá nhỏ / rỗng thì trả None để caller tự bỏ qua
    h, w = face_bgr.shape[:2]
    if h < 20 or w < 20:
        return None

    # 1) BGR -> GRAY
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    # 2) Resize về 48x48 (giống kích thước FER gốc)
    gray48 = cv2.resize(
        gray,
        IMG_SIZE_LR,
        interpolation=cv2.INTER_AREA  # INTER_AREA cho downscale mượt hơn
    )

    # 3) CLAHE giống lúc chuẩn hóa FER2013
    gray48_clahe = clahe.apply(gray48)  # uint8, 0..255, (48,48)

    # 4) Tăng độ phân giải lên 224x224 (nhánh không dùng ESPCN trong to_hr)
    hr_uint8 = cv2.resize(
        gray48_clahe,
        IMG_SIZE,
        interpolation=cv2.INTER_CUBIC  # CUBIC cho upscale mượt hơn
    )  # (224,224), uint8 0..255

    # 5) Chuyển sang RGB 3 kênh
    rgb_uint8 = cv2.cvtColor(hr_uint8, cv2.COLOR_GRAY2RGB)  # (224,224,3), uint8

    # 6) float32 giữ nguyên range 0..255
    x = rgb_uint8.astype("float32")  # (224,224,3)

    # 7) Thêm batch dim
    x = np.expand_dims(x, axis=0)  # (1,224,224,3)

    # 8) Tiền xử lý theo ResNet (chuẩn hoá mean/std, scale về [-123..] v.v.)
    x = preprocess_input(x)

    return x


def predict_emotion(model, face_img):
    """
    Dự đoán cảm xúc với TTA đơn giản:
    - Ảnh gốc
    - Ảnh lật ngang (flip)
    Trung bình hai vector softmax -> giảm nhiễu, đỡ bị lệch 1 lớp.
    """
    # --- Ảnh gốc ---
    x1 = _preprocess_face(face_img)
    if x1 is None:
        # Nếu crop quá nhỏ/ lỗi, trả về unknown
        return "unknown", 0.0, np.zeros(len(CLASS_NAMES), dtype="float32")

    preds1 = model.predict(x1, verbose=0)[0]

    # --- Ảnh lật ngang (horizontal flip) ---
    face_flip = cv2.flip(face_img, 1)
    x2 = _preprocess_face(face_flip)
    if x2 is not None:
        preds2 = model.predict(x2, verbose=0)[0]
        preds = (preds1 + preds2) / 2.0
    else:
        preds = preds1

    # Lấy kết quả cuối
    conf = float(np.max(preds))
    label_idx = int(np.argmax(preds))
    label = CLASS_NAMES[label_idx]

    label = apply_emotion_threshold(label, conf)

    return label, conf, preds
