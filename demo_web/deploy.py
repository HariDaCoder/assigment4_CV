from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
import io
from PIL import Image
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
IMG_SIZE_LR = (48, 48)

# CLAHE giống trong data_preprocessing.py
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
app = Flask(__name__)

# 1. Load model đúng của m
MODEL_PATH = "../demo/models/fer_resnet_best_finetune.h5"  # <-- tên model của m
model = load_model(MODEL_PATH)

# 2. Nhãn cảm xúc giống code webcam
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 3. (OPTIONAL) Haar cascade phát hiện mặt, giống code cũ
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def read_base64_image(data_url):
    """Decode ảnh base64 (từ browser) thành numpy array RGB."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert("RGB")  # Đảm bảo RGB
    img = np.array(img)
    return img


# def preprocess_face(img):
#     """
#     img: numpy array RGB (H, W, 3)
#     1. detect face bằng HaarCascade
#     2. crop mặt (nếu có) -> resize 224x224
#     3. preprocess_input như code webcam
#     """
#     # 1. detect mặt trên ảnh xám (RGB -> GRAY)
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#     if len(faces) == 0:
#         # Không tìm được mặt -> dùng trung tâm ảnh
#         h, w, _ = img.shape
#         size = min(h, w)
#         y0 = (h - size) // 2
#         x0 = (w - size) // 2
#         face = img[y0:y0+size, x0:x0+size]
#     else:
#         # Lấy cái mặt lớn nhất
#         x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
#         face = img[y:y+h, x:x+w]

#     # 2. resize về 224x224 giống code webcam
#     face_resized = cv2.resize(face, (224, 224))

#     # 3. chuyển thành batch + preprocess_input
#     face_array = np.array(face_resized, dtype="float32")
#     face_array = np.expand_dims(face_array, axis=0)  # (1, 224, 224, 3)
#     face_array = preprocess_input(face_array)

#     return face_array
def preprocess_face(img):
    """
    img: numpy array RGB (H, W, 3)

    1) Detect face bằng HaarCascade
    2) Crop mặt
    3) PREPROCESS GIỐNG TRAIN (từ _preprocess_face):
        - RGB → GRAY
        - resize 48x48
        - CLAHE
        - resize 224x224
        - GRAY->RGB 3 channel
        - float32
        - preprocess_input
    """

    # ============== 1. Detect face ==============
    gray_detect = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray_detect, 1.1, 5)

    if len(faces) == 0:
        # fallback: crop center
        h, w, _ = img.shape
        size = min(h, w)
        y0 = (h - size) // 2
        x0 = (w - size) // 2
        face_rgb = img[y0:y0+size, x0:x0+size]
    else:
        # lấy mặt lớn nhất
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        face_rgb = img[y:y+h, x:x+w]

    # ============== 2. ÁP DỤNG PIPELINE TRAIN ==============
    # Input hiện tại face_rgb: RGB

    # 1) RGB -> BGR để pipeline chuẩn
    face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)

    # 0) Check nếu mặt quá nhỏ
    hh, ww = face_bgr.shape[:2]
    if hh < 20 or ww < 20:
        return None

    # 2) BGR -> GRAY
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    # 3) Resize về 48x48 (giống FER2013)
    gray48 = cv2.resize(
        gray,
        IMG_SIZE_LR,
        interpolation=cv2.INTER_AREA
    )

    # 4) CLAHE
    gray48_clahe = clahe.apply(gray48)  # uint8

    # 5) Resize lên 224x224
    hr_uint8 = cv2.resize(
        gray48_clahe,
        (224, 224),
        interpolation=cv2.INTER_CUBIC
    )

    # 6) Convert GRAY → RGB 3 kênh
    rgb_uint8 = cv2.cvtColor(hr_uint8, cv2.COLOR_GRAY2RGB)

    # 7) float32
    x = rgb_uint8.astype("float32")

    # 8) batch dim
    x = np.expand_dims(x, axis=0)

    # 9) Chuẩn hóa theo preprocess_input (ResNet)
    x = preprocess_input(x)

    return x


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_data = data.get("image")

    if image_data is None:
        return jsonify({"error": "No image data"}), 400

    # 1. decode base64 -> RGB
    img = read_base64_image(image_data)

    # 2. preprocess giống code webcam (crop face + preprocess_input)
    x = preprocess_face(img)

    # 3. predict
    preds = model.predict(x)[0]  # (7,)
    idx = int(np.argmax(preds))
    emotion = emotion_labels[idx]

    probs = {emotion_labels[i]: float(preds[i]) for i in range(len(emotion_labels))}

    return jsonify({
        "emotion": emotion,
        "probs": probs
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
