import os
import shutil
import random

SOURCE_DIR = r"non_leaf_raw\seg_train\seg_train\forest"
DEST_DIR = r"data\plantvillage dataset\color\other_leaves"

os.makedirs(DEST_DIR, exist_ok=True)

all_images = [os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR) if f.endswith('.jpg')]
print(f"Found {len(all_images)} images in {SOURCE_DIR}")

random.seed(42)
selected = random.sample(all_images, min(500, len(all_images)))

for i, src in enumerate(selected):
    dst = os.path.join(DEST_DIR, f"other_leaf_{i:04d}.jpg")
    shutil.copy2(src, dst)

print(f"Copied {len(selected)} forest/plant images to {DEST_DIR} for the Other Leaves class.")
