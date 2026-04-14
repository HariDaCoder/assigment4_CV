import os
import shutil
import kagglehub

DATASET_SLUG = "msambare/fer2013"
TARGET_DIR = "archive"

print("📥 Downloading FER2013 dataset from Kaggle...")
path = kagglehub.dataset_download(DATASET_SLUG)
print(f"✔ Downloaded to: {path}")

if os.path.exists(TARGET_DIR):
    print(f"⚠ '{TARGET_DIR}' already exists, skipping copy.")
else:
    shutil.copytree(path, TARGET_DIR)
    print(f"✔ Copied to: {TARGET_DIR}")

print("🎉 Done! Dataset is ready at:", TARGET_DIR)
