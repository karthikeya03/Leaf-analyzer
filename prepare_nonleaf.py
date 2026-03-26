import os
import shutil
import random

# Source - intel image classification seg_train folder
SOURCE_DIR = r"non_leaf_raw\seg_train\seg_train"
# Destination - add as new class inside plantvillage color folder
DEST_DIR = r"data\plantvillage dataset\color\non_leaf"

os.makedirs(DEST_DIR, exist_ok=True)

# Collect all images from all subfolders
all_images = []
for category in os.listdir(SOURCE_DIR):
    category_path = os.path.join(SOURCE_DIR, category)
    if os.path.isdir(category_path):
        for img_file in os.listdir(category_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(category_path, img_file))

print(f"Total non-leaf images found: {len(all_images)}")

# Pick 1000 random images
random.seed(42)
selected = random.sample(all_images, min(1000, len(all_images)))

# Copy to destination
for i, src in enumerate(selected):
    ext = os.path.splitext(src)[1]
    dst = os.path.join(DEST_DIR, f"non_leaf_{i:04d}{ext}")
    shutil.copy2(src, dst)

print(f"Copied {len(selected)} images to {DEST_DIR}")
print("non_leaf class is ready for training!")
