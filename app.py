from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

model = load_model('model.h5')

CLASS_NAMES = ['Coccidiosis', 'Healthy', 'Newcastle', 'Salmonella']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    image = Image.open(file).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    prediction = model.predict(image_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    return render_template('predict.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
