import os
import numpy as np
from tqdm import tqdm
import cv2
from cv2 import dnn_superres
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================
# 0. CONFIG
# ==========================
ORIG_ROOT = "archive"                    # Thư mục chứa dữ liệu gốc: archive/train, archive/test
OUT_ROOT  = "FER2013_tien_xu_li"         # Thư mục mới: chỉ chứa ảnh 224x224 (hoặc kích thước HR)

IMG_SIZE_LR = (48, 48)                   # Kích thước ảnh low-res (FER2013 gốc)
IMG_SIZE_HR = (224, 224)                 # Kích thước ảnh high-res đầu ra cho model (VD: ResNet)

N_AUG = 3    # 2                       # Số ảnh augment thêm mỗi ảnh gốc (chỉ áp dụng cho TRAIN)
VAL_RATIO = 0.15   #0.2                      # Tỉ lệ tách validation từ TRAIN (20%, bạn có thể đổi)

USE_ESPCN = True
ESPCN_PATH = "ESPCN_x4.pb"               # File model cho OpenCV dnn_superres

print("🔧 CONFIG:")
print(f"  ORIG_ROOT       = {ORIG_ROOT}")
print(f"  OUT_ROOT        = {OUT_ROOT}")
print(f"  USE_ESPCN       = {USE_ESPCN}")
print(f"  ESPCN_PATH      = {ESPCN_PATH}")
print(f"  IMG_SIZE_LR     = {IMG_SIZE_LR}")
print(f"  IMG_SIZE_HR     = {IMG_SIZE_HR}")
print(f"  AUG_PER_IMAGE   = {N_AUG}")
print(f"  VAL_RATIO       = {VAL_RATIO}")
print("-" * 60)

# ==========================
# CLAHE & RANDOM ERASING CONFIG
# ==========================

# CLAHE: tăng tương phản ảnh xám 48x48
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) #1.0

def apply_clahe_48(img_gray_48):
    """
    img_gray_48: ảnh xám uint8 size 48x48
    return: ảnh xám uint8 sau CLAHE, size 48x48
    """
    return clahe.apply(img_gray_48)

def random_erasing(img_norm_2d, p=0.5, s_l=0.02, s_h=0.2, r1=0.3):   # 0.12, 0.02, 0.05, 0.3
    """
    Random Erasing / Cutout trên ảnh 2D [0,1], shape (H,W).
    - p: xác suất thực hiện erasing
    - s_l, s_h: tỉ lệ diện tích patch so với toàn ảnh
    - r1: giới hạn tỉ lệ khung hình (aspect ratio)
    Trả về ảnh sau khi có (hoặc không) xóa patch ngẫu nhiên.
    """
    if np.random.rand() > p:
        return img_norm_2d

    h, w = img_norm_2d.shape
    area = h * w

    for _ in range(10):  # thử tối đa 10 lần để tìm patch hợp lệ
        target_area = np.random.uniform(s_l, s_h) * area
        aspect_ratio = np.random.uniform(r1, 1.0 / r1)

        h_e = int(round(np.sqrt(target_area * aspect_ratio)))
        w_e = int(round(np.sqrt(target_area / aspect_ratio)))

        if h_e < h and w_e < w:
            top = np.random.randint(0, h - h_e)
            left = np.random.randint(0, w - w_e)
            erased = img_norm_2d.copy()

            # Có thể đặt patch = 0 (đen) hoặc = mean, ở đây đặt = 0 cho đơn giản
            erased[top:top + h_e, left:left + w_e] = 0.0
            return erased

    # Nếu không tìm được patch phù hợp thì trả lại ảnh gốc
    return img_norm_2d

# ==========================
# 1. TẠO THƯ MỤC ĐÍCH
# ==========================
print("📁 Tạo thư mục đích...")

train_src_dir = os.path.join(ORIG_ROOT, "train")
test_src_dir  = os.path.join(ORIG_ROOT, "test")

# --- Tạo thư mục cho TRAIN & VAL (cùng class) ---
if os.path.isdir(train_src_dir):
    for cls in os.listdir(train_src_dir):
        cls_src_dir = os.path.join(train_src_dir, cls)
        if not os.path.isdir(cls_src_dir):
            continue

        # Thư mục output cho train và val
        out_train_dir = os.path.join(OUT_ROOT, "train", cls)
        out_val_dir   = os.path.join(OUT_ROOT, "val", cls)
        os.makedirs(out_train_dir, exist_ok=True)
        os.makedirs(out_val_dir, exist_ok=True)
        print(f"  ✔ Created (train): {out_train_dir}")
        print(f"  ✔ Created (val)  : {out_val_dir}")
else:
    print(f"  ⚠ Không tìm thấy thư mục train trong {ORIG_ROOT}")

# --- Tạo thư mục cho TEST ---
if os.path.isdir(test_src_dir):
    for cls in os.listdir(test_src_dir):
        cls_src_dir = os.path.join(test_src_dir, cls)
        if not os.path.isdir(cls_src_dir):
            continue

        out_test_dir = os.path.join(OUT_ROOT, "test", cls)
        os.makedirs(out_test_dir, exist_ok=True)
        print(f"  ✔ Created (test) : {out_test_dir}")
else:
    print(f"  ⚠ Không tìm thấy thư mục test trong {ORIG_ROOT}")

print("-" * 60)

# ==========================
# 2. AUGMENTATION SETTING (CHỈ DÙNG CHO TRAIN)
# ==========================
print("🎛 Đang cấu hình Augmentation (train only)...")
aug = ImageDataGenerator(
    rescale=1./255,          # đưa về [0,1] trước khi to_hr & random_erasing
    rotation_range=30, # 10
    width_shift_range=0.1, #0.05
    height_shift_range=0.1,#0.05
    shear_range=0.2, #0.05
    zoom_range=0.2, # 0.05
    horizontal_flip=True
)
print("  ✔ Augmentation OK")

# ==========================
# 2.5. LOAD ESPCN (OpenCV dnn_superres)
# ==========================
if USE_ESPCN:
    print("📦 Đang load model ESPCN (OpenCV dnn_superres)...")
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(ESPCN_PATH)     # ví dụ: "ESPCN_x4.pb"
    sr.setModel("espcn", 4)      # model "espcn", scale = 4 (x4)
    print("  ✔ ESPCN (x4) loaded OK")
else:
    sr = None
    print("ℹ Không dùng ESPCN, sẽ resize đơn giản bằng OpenCV.")

# ==========================
# 3. HÀM TĂNG ĐỘ PHÂN GIẢI
# ==========================
def to_hr(img48_norm):
    """
    Tăng độ phân giải ảnh từ 48×48 lên HR (IMG_SIZE_HR).
    - Nếu USE_ESPCN = True: dùng OpenCV dnn_superres (ESPCN x4) rồi resize về IMG_SIZE_HR.
    - Nếu USE_ESPCN = False: resize bilinear bằng cv2.resize.
    img48_norm: ảnh [0,1], dạng (48,48) hoặc (48,48,1)
    """

    # Trường hợp không dùng ESPCN: resize trực tiếp
    if not USE_ESPCN or sr is None:
        return cv2.resize((img48_norm * 255).astype("uint8"), IMG_SIZE_HR)

    # 1) Đưa về uint8 [0,255]
    lr_uint8 = (img48_norm * 255).astype("uint8")

    # 2) Đảm bảo có 3 kênh cho dnn_superres
    if lr_uint8.ndim == 2:             # (48,48) GRAY
        lr_color = cv2.cvtColor(lr_uint8, cv2.COLOR_GRAY2BGR)   # (48,48,3)
    else:
        # (48,48,1) hoặc (48,48,3)
        if lr_uint8.shape[-1] == 1:
            lr_color = cv2.cvtColor(lr_uint8, cv2.COLOR_GRAY2BGR)
        else:
            lr_color = lr_uint8

    # 3) Upsample bằng ESPCN (x4: 48→192)
    hr_color = sr.upsample(lr_color)   # (192,192,3) nếu scale=4

    # 4) Chuyển về GRAY (vì FER2013 là ảnh xám, CNN thường dùng 1 kênh)
    hr_gray = cv2.cvtColor(hr_color, cv2.COLOR_BGR2GRAY)

    # 5) Đảm bảo đúng kích thước IMG_SIZE_HR (ví dụ 224×224)
    if hr_gray.shape[:2] != IMG_SIZE_HR:
        hr_gray = cv2.resize(hr_gray, IMG_SIZE_HR)

    return hr_gray


# ==========================
# 4. XỬ LÝ ẢNH: TRAIN (có tách VAL) + TEST
# ==========================
print("\n🚀 BẮT ĐẦU XỬ LÝ ẢNH...\n")

rng = np.random.default_rng(42)  # Để tách train/val reproducible

# --------- XỬ LÝ SPLIT = TRAIN (tách thành train + val) ---------
if os.path.isdir(train_src_dir):
    print("\n==============================")
    print("📌 Split nguồn: train  →  train + val (output)")
    print("==============================")

    for cls in os.listdir(train_src_dir):
        cls_dir = os.path.join(train_src_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        img_names = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
        img_names = np.array(img_names)
        n_total = len(img_names)
        if n_total == 0:
            print(f"\n📂 Class: {cls} — 0 ảnh, bỏ qua.")
            continue

        # Shuffle để chia train/val ngẫu nhiên nhưng cố định seed
        rng.shuffle(img_names)

        n_val = int(n_total * VAL_RATIO)
        val_names = set(img_names[:n_val])
        train_names = set(img_names[n_val:])

        print(f"\n📂 Class: {cls} — {n_total} ảnh (train: {len(train_names)}, val: {len(val_names)})")

        for img_name in tqdm(img_names, desc=f"train/val/{cls}", ncols=80):
            path = os.path.join(cls_dir, img_name)

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  ⚠ LỖI: Không load được ảnh {path}")
                continue

            # Đảm bảo đúng size LR (48x48)
            img = cv2.resize(img, IMG_SIZE_LR)

            # ÁP DỤNG CLAHE TRƯỚC khi chuẩn hóa
            img = apply_clahe_48(img)          # uint8, 48x48

            img_norm = img.astype("float32") / 255.0   # [0,1]

            # Quyết định ảnh này thuộc train hay val (output)
            if img_name in val_names:
                out_split = "val"
            else:
                out_split = "train"

            # ----- 1) Lưu bản gốc HR (đã qua CLAHE) -----
            hr = to_hr(img_norm)
            out_path = os.path.join(OUT_ROOT, out_split, cls, img_name)
            cv2.imwrite(out_path, hr)
            # print(f"  ✔ [{out_split}] Đã lưu bản gốc HR: {out_path}")

            # ----- 2) Chỉ AUGMENT nếu là TRAIN -----
            if out_split == "train" and N_AUG > 0:
                # Dùng img (uint8, đã CLAHE) cho ImageDataGenerator, nó sẽ tự rescale=1/255
                x = np.expand_dims(img, (0, -1)).astype("float32")  # (1,48,48,1)
                it = aug.flow(x, batch_size=1)

                for i in range(N_AUG):
                    aug_img = next(it)[0]           # (48,48,1), [0,1] vì rescale
                    aug_img_2d = np.squeeze(aug_img, -1)  # (48,48), [0,1]

                    # RANDOM ERASING trên ảnh augment [0,1]
                    aug_img_2d = random_erasing(aug_img_2d, p=0.12)

                    hr_aug = to_hr(aug_img_2d)

                    out_aug = os.path.join(OUT_ROOT, "train", cls, f"aug_{i}_{img_name}")
                    cv2.imwrite(out_aug, hr_aug)
                    # print(f"  ✔ AUG {i}: {out_aug}")

# --------- XỬ LÝ SPLIT = TEST (KHÔNG AUGMENT, KHÔNG CHIA) ---------
if os.path.isdir(test_src_dir):
    print("\n==============================")
    print("📌 Split nguồn: test  →  test (output, không augment)")
    print("==============================")

    for cls in os.listdir(test_src_dir):
        cls_dir = os.path.join(test_src_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        img_names = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
        print(f"\n📂 Class: {cls} — {len(img_names)} ảnh")

        for img_name in tqdm(img_names, desc=f"test/{cls}", ncols=80):
            path = os.path.join(cls_dir, img_name)

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  ⚠ LỖI: Không load được ảnh {path}")
                continue

            img = cv2.resize(img, IMG_SIZE_LR)

            # TEST cũng áp dụng CLAHE cho đồng nhất với train/val
            img = apply_clahe_48(img)            # uint8, 48x48
            img_norm = img.astype("float32") / 255.0   # [0,1]

            hr = to_hr(img_norm)
            out_path = os.path.join(OUT_ROOT, "test", cls, img_name)
            cv2.imwrite(out_path, hr)
            # print(f"  ✔ [test] Đã lưu HR: {out_path}")

print("\n🎉 DONE! Tất cả ảnh đã được lưu dạng HR tại thư mục:", OUT_ROOT)
