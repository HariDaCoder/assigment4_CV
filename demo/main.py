import sys
import os

from src.demo import run_demo


def main():
    # Cách dùng:
    #   python main.py <input_path> <image|video>
    #   python main.py webcam [camera_index]
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py <input_path> <image|video>")
        print("  python main.py webcam [camera_index]")
        sys.exit(1)

    first_arg = sys.argv[1].lower()

    # Chế độ webcam
    if first_arg == "webcam":
        camera_index = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
        run_demo(None, "webcam", camera_index)
        return

    # Chế độ ảnh / video
    if len(sys.argv) < 3:
        print("Error: Thiếu demo_type (image|video)")
        sys.exit(1)

    input_path = sys.argv[1]
    demo_type = sys.argv[2].lower()

    if demo_type not in ["image", "video"]:
        print("Error: demo_type must be 'image' or 'video'")
        sys.exit(1)

    if not os.path.exists(input_path):
        print(f"Error: The input path '{input_path}' does not exist.")
        sys.exit(1)

    run_demo(input_path, demo_type)


if __name__ == "__main__":
    main()
