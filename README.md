# Facial Emotion Recognition (FER) — Assignment 4

A deep learning pipeline for **facial emotion recognition** using a fine-tuned ResNet model and YOLOv5 face detection. Supports inference on images and videos.

## Features

- Face detection with YOLOv5
- Emotion classification into 7 classes: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`
- Two-stage ResNet training (frozen backbone → fine-tuned)
- Demo script for image and video inference
- Edge detection visualization

## Project Structure

```
├── archive/                  # FER2013 raw dataset (train/test)
├── FER2013_tien_xu_li/       # Preprocessed dataset (train/val/test)
├── demo/
│   ├── main.py               # Entry point for demo
│   ├── yolov5n.pt            # YOLOv5 face detection weights
│   ├── models/               # Emotion model weights
│   ├── data/                 # Input images/videos
│   ├── output/               # Inference results
│   └── src/
│       ├── config.py         # Configuration
│       ├── demo.py           # Core demo logic
│       ├── emotion_model.py  # ResNet emotion classifier
│       ├── yolo_face.py      # YOLOv5 face detector wrapper
│       ├── edge_detection.py
│       └── utils.py
├── data_preprocessing.py     # Dataset preprocessing
├── download_dataset.py       # Dataset download script
├── training.ipynb            # Model training notebook
├── fer_resnet_best_stage1.h5 # Stage 1 model checkpoint
├── fer_resnet_best_finetune.h5 # Fine-tuned model checkpoint
└── requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Training

Open and run `training.ipynb` in Jupyter. The notebook covers:
1. Data loading and augmentation
2. Stage 1: training with frozen ResNet backbone
3. Stage 2: fine-tuning all layers

## Demo

Place your input files under `demo/data/images/` or `demo/data/videos/`, then run:

```bash
cd demo

# Image inference
python main.py data/images/sample.jpg image

# Video inference
python main.py data/videos/sample.mp4 video
```

Results are saved to `demo/output/`.

## Requirements

- Python 3.9
- TensorFlow, PyTorch, OpenCV, NumPy, scikit-learn

See `requirements.txt` for full list.

## Dataset

[FER2013](https://www.kaggle.com/datasets/msambare/fer2013) — 35,887 grayscale 48×48 facial images across 7 emotion classes.
