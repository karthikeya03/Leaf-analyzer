
# LeafScanner AI — Plant Disease Detection
A full-stack web application that diagnoses plant diseases from a single leaf photo using a fine-tuned MobileNetV2 model trained on 54,306 images across 39 classes. Works entirely offline after setup. Supports five Indian languages with real-time translation of every result, symptom, treatment step, and prevention tip on the screen.

---
## What It Does
Upload or capture a photo of any plant leaf. The app resizes it, normalizes it, runs it through the model, and returns the disease name, confidence score, symptoms, treatment steps, and prevention tips — all translated into your chosen language, instantly, without reloading the page.

---
## Features
- Photo upload with drag-and-drop or live camera capture
- MobileNetV2 model — 94.8% validation accuracy across 39 classes
- 39 classes: 38 plant diseases (Apple, Tomato, Potato, Corn, Grape, Peach, and more) + 1 custom "Not a Leaf" rejection class
- Real-time translation to English, Telugu, Hindi, Tamil, Kannada using deep-translator
- Every text on the result screen translates — disease name, main message, all 5 symptoms, all 4 treatment steps, all 6 prevention tips
- Confidence score with animated gauge and severity badge
- PDF report export of the diagnosis
- Text-to-speech readout of the result
- Share button
- Statistics modal showing model accuracy and dataset info
- Rejects non-leaf photos via the custom "Not a Leaf" class with a confidence threshold of 0.35
- Glassmorphism dark theme with floating particle background
- Fully responsive — works on phones and laptops
- Runs 100% locally, no internet needed after first setup
- Single-file frontend (all HTML, CSS, JS in one index.html) for easy maintenance
- Uploaded files deleted immediately after prediction in a finally block

## Major Performance Update (March 2026)
### Pre-Translation Optimization — Predictions Now Instant

**Before:**  
Every prediction took **many minutes** because the backend was calling Google Translate ~85 times per image (5 languages × 17 strings).

**Now:**  
All 39 classes (38 diseases + "Not a Leaf") + treatment + prevention tips are **pre-translated once at startup** and stored in memory (`PRE_TRANSLATED` dictionary).

**Result:**
- Prediction time reduced from **many minutes → under 3 seconds**
- First server startup takes 30–90 seconds (only once)
- All future predictions are instant
- All 5 languages still work perfectly
- No features or functionality lost — same JSON output, same UI behavior

This is now the default behavior in `app.py`.

---
## Project Structure
```
leaf-disease-app/
├── app.py # Flask backend — prediction + translation
├── run.bat # One-click launcher (Windows)
├── requirements.txt # All Python dependencies
├── train.py # Complete training script
├── prepare_nonleaf.py # Script to prepare Not a Leaf class data
├── scrape_leaves.py # Script to scrape additional leaf images
├── model/
│ ├── best_model.h5 # Best checkpoint (saved by ModelCheckpoint)
│ ├── leaf_disease_model.h5 # Final model saved after all epochs
│ └── training_plot.png # Accuracy and loss curves graph
├── templates/
│ └── index.html # Complete frontend (HTML + CSS + JS)
├── static/ # Static assets
├── data/
│ └── plantvillage dataset/color/ # 54,306 training images across 38 classes
├── non_leaf_raw/ # Raw images for the Not a Leaf class
└── venv/ # Python virtual environment
```

---
## How to Run
**Windows — one double-click:**
```
run.bat
```

**Manual — in VS Code terminal or PowerShell:**
```bash
# Step 1: Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Step 2: Start Flask
python app.py
```

You will see:
```
Loading model...
Model loaded!
🔄 Pre-translating ALL diseases into 5 languages... (only once at startup)
```
✅** Pre-translation completed successfully for 38 diseases!**
```
* Running on http://127.0.0.1:5000
```

Then open `http://127.0.0.1:5000` in any browser. Keep the terminal open the entire time — closing it stops the server.

**Check the model loaded correctly:**
```
http://127.0.0.1:5000/health
```

Returns `{"model_loaded": true, "classes": 39, "status": "ok"}` if everything is working.

---
## Dataset
- **Source**: PlantVillage dataset (color variant), downloaded from Kaggle
- **Total images**: 54,306
- **Original classes**: 38 disease classes covering Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
- **Custom class added**: "Not a Leaf" — images of random non-leaf objects collected separately so the model can reject irrelevant photos instead of confidently misclassifying them
- **Final class count**: 39
- **Train/validation split**: 80% training (~43,445 images), 20% validation (~10,861 images), split automatically by ImageDataGenerator using `validation_split=0.2`
- **Image format**: Color JPG, variable original sizes, all resized to 224x224 during loading

---
## Model Architecture
The model uses transfer learning on top of MobileNetV2 pretrained on ImageNet.
**Why MobileNetV2**: It is fast, lightweight, and runs well on a laptop CPU without needing a GPU. It was designed for mobile and edge deployment, which matches the goal of running this locally on any machine.
**Why transfer learning**: MobileNetV2 already learned strong low-level and mid-level visual features (edges, textures, shapes) from ImageNet's 1.2 million images. Freezing its convolutional base and only training a new classification head means the model converges faster and needs far fewer images to reach high accuracy.

```
Input: 224 x 224 x 3 (RGB image, normalized to [0, 1])
    |
MobileNetV2 convolutional base
    - 53 convolutional layers using depthwise separable convolutions
    - Pretrained on ImageNet (1.2M images, 1000 classes)
    - Frozen: base_model.trainable = False
    - Output shape: (7, 7, 1280) feature map
    |
GlobalAveragePooling2D
    - Reduces 7x7x1280 feature map to a flat 1280-dimensional vector
    - Less prone to overfitting than Flatten because it has no trainable parameters
    |
Dropout(0.3)
    - Randomly zeros 30% of neurons during each training step
    - Forces the network to not rely on any single feature, reduces overfitting
    |
Dense(128, activation='relu')
    - Learns disease-specific combinations of the 1280-d feature vector
    - ReLU introduces non-linearity (outputs zero for negative values)
    |
Dense(39, activation='softmax')
    - One output neuron per class
    - Softmax converts raw scores to probabilities that sum to exactly 1.0
    - The class with the highest probability is the prediction
```

---
## Preprocessing and Data Augmentation
All preprocessing is handled by Keras ImageDataGenerator. Normalization applies to both training and validation. Augmentation applies to training images only — validation images are only normalized, never augmented, so validation accurately reflects real-world inference.

**Normalization:**
```python
rescale=1./255
```

**Augmentation applied during training:**
| Setting | Value | Reason |
|---|---|---|
| `rotation_range` | 20 degrees | Leaves are photographed at any angle. The model should not fail on a slightly tilted leaf. |
| `zoom_range` | 0.2 | Handles different camera distances — a zoomed-in vs zoomed-out leaf of the same disease should give the same prediction. |
| `horizontal_flip` | True | Disease patterns appear on both sides of a leaf equally. This effectively doubles variety. |
| `brightness_range` | [0.8, 1.2] | Farmers photograph leaves in harsh sunlight and deep shade. The model must handle both. |
| `validation_split` | 0.2 | Reserves 20% of images from every class for validation. These never receive augmentation. |

**Code:**
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)
```

## Training Configuration

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**Optimizer — Adam**: Adaptive Moment Estimation. Adjusts the learning rate for each parameter individually using running estimates of first and second moments of gradients. Works well without manual learning rate tuning.

**Loss — Categorical Crossentropy**: Standard loss for multi-class classification with softmax output. Penalizes confident wrong predictions heavily. Formula: `-sum(y_true * log(y_pred))`.

**Batch size — 32**: Large enough for stable gradient estimates, small enough to fit in RAM comfortably.

**Max epochs — 20**: An upper bound. EarlyStopping typically stops training well before this is reached.

### Callbacks

**ModelCheckpoint:**
```python
ModelCheckpoint(
    'model/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
```
Saves model weights to `best_model.h5` only when `val_accuracy` improves compared to all previous epochs. This guarantees the saved file always has the best-performing weights, not the last epoch's weights which may be slightly worse due to overfitting.

**EarlyStopping:**
```python
EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```
Stops training automatically if `val_loss` does not improve for 5 consecutive epochs. `restore_best_weights=True` rolls the model back to the epoch with the lowest `val_loss`. This prevents wasting compute on epochs that are actively making the model worse.

### Result

- Best validation accuracy: **94.8%**
- Training and validation accuracy/loss curves saved to `model/training_plot.png`

---

## Complete Training Script

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os

# ── Paths ──────────────────────────────────────────────
DATA_DIR   = r"data\plantvillage dataset\color"
MODEL_DIR  = "model"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 20

# ── Data Augmentation ──────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training'
)
val_gen = train_datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation'
)

NUM_CLASSES = len(train_gen.class_indices)
print(f"\nFound {NUM_CLASSES} disease classes")

# ── Build Model ────────────────────────────────────────
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x      = base_model.output
x      = GlobalAveragePooling2D()(x)
x      = Dropout(0.3)(x)
x      = Dense(128, activation='relu')(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)
model  = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ── Callbacks ──────────────────────────────────────────
checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, 'best_model.h5'),
    monitor='val_accuracy', save_best_only=True, verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# ── Train ──────────────────────────────────────────────
print("\nStarting training...\n")
history = model.fit(
    train_gen, validation_data=val_gen,
    epochs=EPOCHS, callbacks=[checkpoint, early_stop]
)

# ── Save final model ───────────────────────────────────
model.save(os.path.join(MODEL_DIR, 'leaf_disease_model.h5'))
print("\nModel saved!")

# ── Plot accuracy & loss ───────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss'); plt.legend()

plt.savefig(os.path.join(MODEL_DIR, 'training_plot.png'))
plt.show()
print("Plot saved to model/training_plot.png")
```

---

## Backend — app.py

Flask serves the frontend at `/` and handles prediction at `/predict` (POST).

### Prediction Pipeline (step by step)

1. Receive uploaded image file via `request.files['file']`
2. Validate file type — reject anything that is not a recognised image format
3. Load image with PIL, convert to RGB (handles PNG with alpha channel correctly)
4. Resize to 224x224 using LANCZOS resampling
5. Convert to NumPy float32 array
6. Normalize: divide all pixel values by 255.0 so range is [0.0, 1.0]
7. Expand dimensions: shape becomes `(1, 224, 224, 3)` — model expects a batch dimension
8. Call `model.predict()` — returns softmax probability array of shape `(1, 39)`
9. Find the index with the highest probability using `np.argmax`
10. Map index to class name using the `CLASS_NAMES` list
11. If top confidence < 0.35 OR predicted class is "Not a Leaf" — return a rejection message
12. Otherwise call `translate_text()` for all five languages on all strings
13. Return a single JSON object with every translated field
14. Delete the uploaded temp file in a `finally` block whether prediction succeeded or failed

### Translation Function

```python
from deep_translator import GoogleTranslator

def translate_text(text, target_lang):
    if target_lang == 'en':
        return text          # English is the source, no API call needed
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except:
        return text          # If translation fails, fall back to English silently
```

Language codes: `te` for Telugu, `hi` for Hindi, `ta` for Tamil, `kn` for Kannada.

Per prediction, the backend translates 17 strings per language (1 disease name + 1 message + 5 symptoms + 4 treatment steps + 6 prevention tips) across 4 non-English languages. All of it is packed into the single JSON response. When the user switches language on the frontend, no new network request is made — the frontend just re-reads the already-received JSON with the new language key.

### Class Names (39 classes)

```python
CLASS_NAMES = [
    'Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy',
    'Blueberry - Healthy',
    'Cherry - Powdery Mildew', 'Cherry - Healthy',
    'Corn - Cercospora Leaf Spot', 'Corn - Common Rust', 'Corn - Northern Leaf Blight', 'Corn - Healthy',
    'Grape - Black Rot', 'Grape - Esca', 'Grape - Leaf Blight', 'Grape - Healthy',
    'Orange - Citrus Greening',
    'Peach - Bacterial Spot', 'Peach - Healthy',
    'Pepper - Bacterial Spot', 'Pepper - Healthy',
    'Potato - Early Blight', 'Potato - Late Blight', 'Potato - Healthy',
    'Raspberry - Healthy',
    'Soybean - Healthy',
    'Squash - Powdery Mildew',
    'Strawberry - Leaf Scorch', 'Strawberry - Healthy',
    'Tomato - Bacterial Spot', 'Tomato - Early Blight', 'Tomato - Late Blight',
    'Tomato - Leaf Mold', 'Tomato - Septoria Leaf Spot', 'Tomato - Spider Mites',
    'Tomato - Target Spot', 'Tomato - Yellow Leaf Curl Virus', 'Tomato - Mosaic Virus',
    'Tomato - Healthy',
    'Not a Leaf',
]
```

---

## Frontend — index.html

The entire frontend is a single HTML file with embedded CSS and JavaScript. No build step, no npm, no bundler needed.

**What the frontend handles:**

- Drag-and-drop zone that accepts JPG, PNG, WEBP and shows a preview immediately
- Camera tab that uses `navigator.mediaDevices.getUserMedia` to stream live video, with a capture button that draws the frame onto a hidden canvas and converts it to a Blob
- Language toggle buttons (EN / తె / हि / த / ಕ) that update the `lang` variable
- Analyze button sends the file as FormData via `fetch('/predict', { method: 'POST', body: fd })`
- Loading animation while waiting for the response
- Result rendering: disease name, animated confidence bar, severity badge, symptoms list, treatment steps, prevention tips
- If result is healthy or not a leaf, a different panel is shown
- Switching language after a result instantly re-renders everything from the cached JSON — no second request
- PDF export uses the browser's print dialog with a print-specific CSS layout
- Text-to-speech uses the Web Speech API (`window.speechSynthesis`)
- Share uses the Web Share API (`navigator.share`) on mobile, falls back to clipboard copy on desktop

---

## Error Handling

| Situation | What Happens |
|---|---|
| Non-leaf photo uploaded | Model predicts "Not a Leaf" or confidence < 0.35, polite message shown |
| Translation API call fails | except block silently returns original English text |
| Model file not found at startup | Python prints error, server does not start |
| No file attached to POST request | Returns `{"error": "No file uploaded"}` with HTTP 400 |
| File is not a valid image | Returns `{"error": "Invalid file type"}` with HTTP 400 |
| Image processing fails (corrupt file) | Returns `{"error": "Image processing failed"}` with HTTP 422 |
| Flask server not running | Frontend fetch catch block shows "Cannot connect to server" message |
| Temp file cleanup | finally block runs after every prediction, success or exception |

---

## Tech Stack

| Component | Technology |
|---|---|
| ML framework | TensorFlow / Keras |
| Base model | MobileNetV2, ImageNet pretrained |
| Backend | Flask (Python) |
| Translation | deep-translator (Google Translate) |
| Image processing | Pillow (PIL) |
| Numerical operations | NumPy |
| Frontend | HTML + CSS + Vanilla JavaScript, single file |
| Dataset | PlantVillage, Kaggle, color variant |
| Runtime | Python 3.10, Windows |

---

## Requirements

```
flask
tensorflow
numpy
pillow
deep-translator
```

```bash
pip install -r requirements.txt
```

---

## Complete Technical Reference

| # | Detail | Value |
|---|---|---|
| 1 | Dataset source | PlantVillage, Kaggle, color variant |
| 2 | Total training images | 54,306 |
| 3 | Training images (80%) | ~43,445 |
| 4 | Validation images (20%) | ~10,861 |
| 5 | Original disease classes | 38 |
| 6 | Custom class added | Not a Leaf |
| 7 | Final class count | 39 |
| 8 | Input image size | 224 x 224 x 3 |
| 9 | Pixel normalization | Divided by 255, range becomes [0.0, 1.0] |
| 10 | Augmentation: rotation | +-20 degrees |
| 11 | Augmentation: zoom | 20% |
| 12 | Augmentation: flip | Horizontal only |
| 13 | Augmentation: brightness | Range [0.8, 1.2] |
| 14 | Validation augmentation | None — only normalization applied |
| 15 | Base model | MobileNetV2, frozen, ImageNet weights |
| 16 | Base model output shape | (7, 7, 1280) |
| 17 | Pooling layer | GlobalAveragePooling2D — output: (1280,) |
| 18 | Dropout rate | 0.3 (30% of neurons zeroed per step) |
| 19 | Hidden dense layer | 128 neurons, ReLU activation |
| 20 | Output layer | 39 neurons, Softmax activation |
| 21 | Optimizer | Adam |
| 22 | Loss function | Categorical Crossentropy |
| 23 | Batch size | 32 |
| 24 | Max epochs | 20 |
| 25 | Early stopping monitor | val_loss |
| 26 | Early stopping patience | 5 epochs |
| 27 | Best weights restored | Yes (restore_best_weights=True) |
| 28 | Checkpoint monitor | val_accuracy |
| 29 | Checkpoint saves | Best only (save_best_only=True) |
| 30 | Best validation accuracy | 94.8% |
| 31 | Confidence rejection threshold | 0.35 |
| 32 | Translation library | deep-translator + GoogleTranslator |
| 33 | Languages supported | English, Telugu, Hindi, Tamil, Kannada |
| 34 | Language codes | en, te, hi, ta, kn |
| 35 | Strings translated per prediction | 17 per language (name + message + 5 + 4 + 6) |
| 36 | Language switch requests | 0 — all languages in one response, re-rendered client-side |
| 37 | Camera API | navigator.mediaDevices.getUserMedia |
| 38 | TTS API | window.speechSynthesis (Web Speech API) |
| 39 | Share API | navigator.share with clipboard fallback |
| 40 | File cleanup method | finally block, runs on success and exception |
| 41 | Server port | 5000 |
| 42 | Frontend architecture | Single HTML file, no build step, no npm |

---

## Future Improvements

- Unfreeze the last few MobileNetV2 blocks and fine-tune at a lower learning rate for potentially higher accuracy
- Switch to LibreTranslate for fully offline translation with no API dependency
- Add more crops and disease classes beyond PlantVillage
- Deploy on Render, Railway, or Hugging Face Spaces for public access
- Add offline PWA support so the app works without a local server

---

## Author

**Sai Satya Karthikeya**
GitHub: [github.com/karthikeya03/Leaf-analyzer](https://github.com/karthikeya03/Leaf-analyzer)
