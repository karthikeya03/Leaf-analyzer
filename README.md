# 🌿 LeafScanner AI — Plant Disease Detection

**AI-powered mobile-friendly web app** that detects plant leaf diseases in real-time and supports **5 Indian languages** (English, Telugu, Hindi, Tamil, Kannada).

---

## ✨ Features

- Upload or capture leaf photo
- Instant diagnosis using deep learning
- 39 classes (38 diseases + "Not a Leaf")
- **Real-time translation** for disease name + messages
- Full language toggle: English | తెలుగు | हिन्दी | தமிழ் | ಕನ್ನಡ
- Beautiful responsive UI with glassmorphism design
- Confidence gauge + severity info
- PDF report generation, TTS, share, stats

---

## 🚀 How to Run (Super Easy)

1. Open the project folder in VS Code / terminal.
2. Double-click **`run.bat`** (or run this command):

```bash
run.bat
```

3. Wait until you see:
   ```
   * Running on http://127.0.0.1:5000
   ```
4. Open browser → go to **`http://127.0.0.1:5000`**
5. Upload a clear close-up leaf photo → Analyze!

**Note**: Always use `run.bat` — it starts Flask with the correct Python path.

---

## 📁 Project Folder Structure

```
leaf-disease-app/
├── app.py                    ← Main Flask app (with Google Translate)
├── run.bat                   ← One-click start script
├── requirements.txt
├── model/
│   ├── best_model.h5         ← Best trained model
│   ├── leaf_disease_model.h5
│   └── training_plot.png
├── templates/
│   └── index.html            ← Full frontend (UI + JS)
├── data/plantvillage dataset/color/   ← Dataset
├── train.py                  ← Model training script
└── venv/                     ← Virtual environment
```

---

## 🧪 How I Trained the Model (train.py)

**Dataset Used**:  
**PlantVillage Dataset** (color images)  
- Total images used: **54,306**  
- Training set: ~80% (~43,445 images)  
- Validation set: ~20% (~10,861 images)  
- **39 classes** (38 plant diseases + 1 "Not a Leaf" class added later)

**Model Architecture**:
- **Base**: MobileNetV2 (pre-trained on ImageNet)
- Transfer learning (frozen base layers)
- Added: GlobalAveragePooling2D → Dropout(0.3) → Dense(128, relu) → Softmax output
- Input size: **224×224×3**

**Preprocessing & Augmentation**:
- Rescale: `/255`
- Rotation: ±20°
- Zoom: 0.2
- Horizontal flip
- Brightness: 80% – 120%
- Validation split: 20%

**Training Details**:
- Epochs: **20**
- Batch size: **32**
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Callbacks: ModelCheckpoint (best_model.h5) + EarlyStopping (patience=5)
- Final best accuracy: **94.8%** (shown in app stats)

---

## 🌍 Multi-Language Support (Real-time Translation)

- Uses **`deep-translator`** + **Google Translate**
- Automatically translates:
  - Disease name
  - Messages
  - Title (via frontend logic)
- Supported languages: **English, Telugu (te), Hindi (hi), Tamil (ta), Kannada (kn)**
- Toggle works instantly without page refresh

**How translation works in `app.py`**:
- Every prediction calls `GoogleTranslator` for all 4 Indian languages
- Frontend `showDisease()` picks the correct language based on `lang` variable

---

## 📦 Requirements (requirements.txt)

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

## 🔧 How to Add More Languages (Future)

Just add the language code in `app.py` translate function and in frontend `showDisease()`.

---

## 📈 Model Performance (from training)

- Training Accuracy: ~96%
- Validation Accuracy: **94.8%**
- Training Plot saved in `model/training_plot.png`

---

## 📝 Notes from Developer (You)

- Model saved as `best_model.h5` (used by app)
- Added "Not a Leaf" class with confidence threshold 0.35
- Full single-file frontend (`index.html`) with camera, drag-drop, PDF export, TTS, etc.
- Everything runs locally — no internet needed after model download

---

**Made with ❤️ by sai satya**  
**Date**: March 2026  
**Version**: 2.0 (with real-time multi-language translation)
