
# 🌿 LeafScanner AI — Plant Disease Detection
**Version 2.0** | March 2026

**AI-powered mobile-friendly web app** that detects 38 plant leaf diseases + "Not a Leaf" in real-time and supports **5 Indian languages** (English, Telugu, Hindi, Tamil, Kannada) using real-time Google Translate.

---

## ✨ Features
- Upload photo or use camera
- Instant AI diagnosis (MobileNetV2)
- 39 classes (38 diseases + Not a Leaf)
- Real-time translation for **every text** (disease name, symptoms, treatment, prevention, message)
- Full language toggle: English | తెలుగు | हिन्दी | தமிழ் | ಕನ್ನಡ
- Beautiful glassmorphism UI + particle background
- Confidence gauge, severity badge, PDF report, TTS, share, stats modal
- Runs completely offline after model download

---

## 🚀 How to Run (Super Easy)

1. Open the project folder.
2. Double-click **`run.bat`** (recommended)  
   or run in terminal:
   ```bash
   run.bat
   ```
3. Wait until you see:
   ```
   * Running on http://127.0.0.1:5000
   ```
4. Open browser → `http://127.0.0.1:5000`
5. Upload a clear close-up leaf photo → click **Analyze Leaf**

**Always use `run.bat`** — it uses the correct Python path.

---

## 📁 Project Folder Structure
```
leaf-disease-app/
├── app.py                    ← Flask backend + real-time translation
├── run.bat                   ← One-click start script
├── requirements.txt
├── model/
│   ├── best_model.h5         ← Best saved model (used by app)
│   ├── leaf_disease_model.h5
│   └── training_plot.png
├── templates/
│   └── index.html            ← Full frontend (single file)
├── data/plantvillage dataset/color/   ← Original dataset
├── train.py                  ← Full training script
├── venv/                     ← Virtual environment
└── README.md
```

---

## 🧪 How I Trained the Model (Complete Details)

### 1. Dataset Used
- **Source**: PlantVillage Dataset (color images)
- **Total images**: **54,306**
- **Training set**: ~43,445 images (80%)
- **Validation set**: ~10,861 images (20%)
- **Number of classes**: **39** (38 original plant diseases + 1 custom "Not a Leaf" class added later)

### 2. Preprocessing & Data Augmentation (Exact Code)
I used `ImageDataGenerator` with the following settings:

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,                    # Normalize pixel values 0-1
    rotation_range=20,                 # Random rotation ±20°
    zoom_range=0.2,                    # Random zoom 20%
    horizontal_flip=True,              # Random horizontal flip
    brightness_range=[0.8, 1.2],       # Brightness variation
    validation_split=0.2               # 20% validation split
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR, target_size=(224, 224), batch_size=32,
    class_mode='categorical', subset='training'
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR, target_size=(224, 224), batch_size=32,
    class_mode='categorical', subset='validation'
)
```

### 3. Model Architecture (Transfer Learning)
**Base Model**: MobileNetV2 (pre-trained on ImageNet)  
**Algorithm**: **Transfer Learning** (frozen convolutional base + custom classifier)

Exact code:

```python
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False   # Freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
```

### 4. Training Configuration
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 20
- **Batch Size**: 32
- **Callbacks**:
  - `ModelCheckpoint` → saves `best_model.h5`
  - `EarlyStopping` → patience=5, restores best weights

### 5. Training Results
- Best Validation Accuracy: **94.8%**
- Training Plot saved as `model/training_plot.png`

---

## 🌍 Multi-Language Support (Real-time Translation)

- Uses **`deep-translator`** + **Google Translator**
- Every prediction translates:
  - Disease name
  - Main message
  - Symptoms (5 points)
  - Treatment (4 points)
  - Prevention (6 points)
- No hardcoded language lists in frontend

**Backend translation logic** (in `app.py`):
```python
def translate_text(text, target_lang):
    if target_lang == 'en': return text
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except:
        return text
```

Frontend only picks the correct key (`telugu_result`, `tamil_result`, etc.) based on `lang` variable.

---

## 📦 Requirements (`requirements.txt`)
```txt
flask
tensorflow
numpy
deep-translator
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🔧 Future Improvements
- Add more detailed per-class symptoms/treatment/prevention in backend
- Switch to self-hosted LibreTranslate (100% free & unlimited)
- Deploy on Render / Railway / Hugging Face

---

## 📝 Complete Technical Summary (Every Small Detail)

| Step                  | What I Did                                      | Details |
|-----------------------|--------------------------------------------------|---------|
| Dataset               | PlantVillage color images                       | 54,306 images, 39 classes |
| Preprocessing         | ImageDataGenerator + rescaling                  | /255 normalization |
| Augmentation          | Rotation, zoom, flip, brightness                | See code block above |
| Model                 | MobileNetV2 + Transfer Learning                 | Frozen base + custom head |
| Classifier            | GlobalAvgPool → Dropout(0.3) → Dense(128) → Softmax | 39 outputs |
| Training              | 20 epochs, batch 32, Adam optimizer             | EarlyStopping + Checkpoint |
| Inference             | Flask + Keras load_model + 224×224 preprocessing| Confidence threshold 0.35 |
| Translation           | deep-translator + GoogleTranslator              | 5 languages in real-time |
| Frontend              | Single file HTML + Tailwind + vanilla JS        | Camera, drag-drop, PDF, TTS |

