import torch
import cv2

from .config import FACE_CONF_THRES, FACE_IOU_THRES, MAX_FACES
from .utils import draw_box_and_label


def load_yolo_model():
    model = torch.hub.load("ultralytics/yolov5", "yolov5n")
    model.eval()
    return model


def detect_faces(model, img_bgr):
    results = model(img_bgr)
    faces = results.xyxy[0].cpu().numpy()
    filtered_faces = []

    for face in faces:
        x1, y1, x2, y2, conf, cls = face
        if conf >= FACE_CONF_THRES and cls == 0:
            filtered_faces.append([int(x1), int(y1), int(x2), int(y2), float(conf)])

        if len(filtered_faces) >= MAX_FACES:
            break

    return filtered_faces


def draw_faces(img_bgr, faces, labels=None, confs=None):
    for i, face in enumerate(faces):
        x1, y1, x2, y2, det_conf = face

        label = labels[i] if labels is not None and i < len(labels) else "unknown"
        conf = confs[i] if confs is not None and i < len(confs) else det_conf

        img_bgr = draw_box_and_label(img_bgr, (x1, y1, x2, y2), label, conf)

    return img_bgr
