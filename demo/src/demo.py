import cv2
import os

from .yolo_face import load_yolo_model, detect_faces, draw_faces
from .emotion_model import load_emotion_model, predict_emotion
from .edge_detection import detect_edges, create_side_by_side
from .config import (
    IMAGE_DIR,
    VIDEO_DIR,
    OUTPUT_IMAGE_DIR,
    OUTPUT_VIDEO_DIR,
    SHOW_VIDEO_WINDOW,
    SAVE_VIDEO_RESULT,
    SAVE_IMAGE_RESULT,
    DEFAULT_FPS,
)
from .utils import ensure_dir

yolo_model = load_yolo_model()
emotion_model = load_emotion_model()


def demo_image(image_path):
    """Demo nhận dạng cảm xúc từ 1 ảnh."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("Error: Không thể đọc ảnh '%s'" % image_path)
        return

    faces = detect_faces(yolo_model, img_bgr)

    labels = []
    confs = []
    for face in faces:
        x1, y1, x2, y2, _ = face
        face_img = img_bgr[y1:y2, x1:x2]
        if face_img.size == 0:
            continue

        label, conf, _ = predict_emotion(emotion_model, face_img)
        labels.append(label)
        confs.append(conf)

    img_bgr = draw_faces(img_bgr, faces, labels, confs)

    edges = detect_edges(img_bgr)
    combined = create_side_by_side(img_bgr, edges)

    if SAVE_IMAGE_RESULT:
        ensure_dir(OUTPUT_IMAGE_DIR)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(OUTPUT_IMAGE_DIR, "%s_result.jpg" % base_name)
        edge_path = os.path.join(OUTPUT_IMAGE_DIR, "%s_edges.jpg" % base_name)
        cv2.imwrite(output_path, img_bgr)
        cv2.imwrite(edge_path, combined)
        print("✅ Đã lưu ảnh kết quả:", output_path)
        print("✅ Đã lưu ảnh phát hiện biên:", edge_path)

    cv2.imshow("Emotion Detection", img_bgr)
    cv2.imshow("Edge Detection", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _init_video_writer(cap, output_path):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = DEFAULT_FPS

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def _process_stream(cap, save_path=None):
    out = None
    if save_path is not None and SAVE_VIDEO_RESULT:
        ensure_dir(os.path.dirname(save_path))
        out = _init_video_writer(cap, save_path)
        print("🎥 Đang lưu video kết quả:", save_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(yolo_model, frame)

        labels = []
        confs = []
        for face in faces:
            x1, y1, x2, y2, _ = face
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            label, conf, _ = predict_emotion(emotion_model, face_img)
            labels.append(label)
            confs.append(conf)

        frame = draw_faces(frame, faces, labels, confs)

        edges = detect_edges(frame)
        combined = create_side_by_side(frame, edges)

        if SHOW_VIDEO_WINDOW:
            cv2.imshow("Emotion Detection", frame)
            cv2.imshow("Edge Detection", combined)

        if out is not None:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


def demo_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Không thể mở video '%s'" % video_path)
        return

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(OUTPUT_VIDEO_DIR, "%s_result.mp4" % base_name)
    _process_stream(cap, save_path)


def run_demo(input_path=None, demo_type="image"):
    demo_type = demo_type.lower()

    if demo_type == "image":
        if not input_path:
            print("Error: Cần truyền đường dẫn ảnh.")
            return
        demo_image(input_path)

    elif demo_type == "video":
        if not input_path:
            print("Error: Cần truyền đường dẫn video.")
            return
        demo_video(input_path)

    else:
        print("Error: demo_type phải là 'image' hoặc 'video'.")
