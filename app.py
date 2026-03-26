from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from deep_translator import GoogleTranslator

app = Flask(__name__)

print("Loading model...")
model = load_model('model/best_model.h5')
print("Model loaded!")

CLASS_NAMES = [
    'Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy',
    'Blueberry - Healthy', 'Cherry - Powdery Mildew', 'Cherry - Healthy',
    'Corn - Cercospora Leaf Spot', 'Corn - Common Rust', 'Corn - Northern Leaf Blight', 'Corn - Healthy',
    'Grape - Black Rot', 'Grape - Esca', 'Grape - Leaf Blight', 'Grape - Healthy',
    'Orange - Citrus Greening', 'Peach - Bacterial Spot', 'Peach - Healthy',
    'Pepper - Bacterial Spot', 'Pepper - Healthy',
    'Potato - Early Blight', 'Potato - Late Blight', 'Potato - Healthy',
    'Raspberry - Healthy', 'Soybean - Healthy', 'Squash - Powdery Mildew',
    'Strawberry - Leaf Scorch', 'Strawberry - Healthy',
    'Tomato - Bacterial Spot', 'Tomato - Early Blight', 'Tomato - Late Blight',
    'Tomato - Leaf Mold', 'Tomato - Septoria Leaf Spot',
    'Tomato - Spider Mites', 'Tomato - Target Spot',
    'Tomato - Yellow Leaf Curl Virus', 'Tomato - Mosaic Virus', 'Tomato - Healthy',
    'Not a Leaf'
]

def translate_text(text, target_lang):
    if target_lang == 'en':
        return text
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except:
        return text  # fallback to English if translation fails

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    temp_path = 'temp_upload.jpg'
    file.save(temp_path)

    try:
        img = image.load_img(temp_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        confidence = float(np.max(predictions))
        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_index]

        # === AUTO TRANSLATE TO ALL LANGUAGES ===
        telugu_result  = translate_text(predicted_class, 'te')
        hindi_result   = translate_text(predicted_class, 'hi')
        tamil_result   = translate_text(predicted_class, 'ta')
        kannada_result = translate_text(predicted_class, 'kn')

        # Translate message too
        base_message = 'Healthy leaf detected!' if 'Healthy' in predicted_class else f'Disease detected: {predicted_class}'
        telugu_msg  = translate_text(base_message, 'te')
        hindi_msg   = translate_text(base_message, 'hi')
        tamil_msg   = translate_text(base_message, 'ta')
        kannada_msg = translate_text(base_message, 'kn')

        # Not a Leaf handling
        if confidence < 0.35 or predicted_class == 'Not a Leaf':
            return jsonify({
                'result': 'Not a Leaf',
                'telugu_result': 'ఆకు కాదు',
                'hindi_result': 'पत्ता नहीं है',
                'tamil_result': 'இலை இல்லை',
                'kannada_result': 'ಎಲೆ ಅಲ್ಲ',
                'confidence': round(confidence * 100, 2),
                'is_healthy': False,
                'message': 'Please upload a clear close-up photo of a leaf.'
            })

        is_healthy = 'Healthy' in predicted_class

        return jsonify({
            'result': predicted_class,
            'telugu_result': telugu_result,
            'hindi_result': hindi_result,
            'tamil_result': tamil_result,
            'kannada_result': kannada_result,
            'confidence': round(confidence * 100, 2),
            'is_healthy': is_healthy,
            'message': telugu_msg if request.args.get('lang') == 'te' else \
                       hindi_msg if request.args.get('lang') == 'hi' else \
                       tamil_msg if request.args.get('lang') == 'ta' else \
                       kannada_msg if request.args.get('lang') == 'kn' else base_message
        })

    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(debug=True)