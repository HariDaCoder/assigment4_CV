import cv2
import numpy as np


def detect_edges(image, low_thresh=50, high_thresh=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_bgr


def detect_edges_on_faces(image, faces, low_thresh=50, high_thresh=150):
    result = image.copy()
    for face in faces:
        x1, y1, x2, y2 = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        face_roi = image[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue
        face_edges = detect_edges(face_roi, low_thresh, high_thresh)
        result[y1:y2, x1:x2] = face_edges
    return result


def create_side_by_side(original, edges, label="Edge Detection"):
    h, w = original.shape[:2]
    edges_resized = cv2.resize(edges, (w, h))
    combined = np.hstack([original, edges_resized])
    cv2.putText(combined, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined, label, (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return combined
