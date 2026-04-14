import sys
import os

from src.demo import run_demo


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python main.py <input_path> <image|video>")
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
