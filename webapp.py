from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import joblib
from PIL import Image
from io import BytesIO
import os
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the model and label encoders
model = load_model("prototype_model.keras")
label_encoder_artwork = joblib.load("label_encoder_artwork.pkl")
label_encoder_artist = joblib.load("label_encoder_artist.pkl")
label_encoder_date = joblib.load("label_encoder_date.pkl")
label_encoder_style = joblib.load("label_encoder_style.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Open image using PIL
    pil_image = Image.open(file.stream).convert('RGB')  # Ensure image is in RGB format
    # Resize image to match the model input size
    pil_image = pil_image.resize((224, 224))
    image_array = preprocess_input(img_to_array(pil_image))
    predictions = model.predict(np.expand_dims(image_array, axis=0))

    predicted_artwork_index = np.argmax(predictions[0])
    predicted_artist_index = np.argmax(predictions[1])
    predicted_date_index = np.argmax(predictions[2])
    predicted_style_index = np.argmax(predictions[3])

    predicted_artwork = label_encoder_artwork.inverse_transform([predicted_artwork_index])[0]
    predicted_artist = label_encoder_artist.inverse_transform([predicted_artist_index])[0]
    predicted_date = label_encoder_date.inverse_transform([predicted_date_index])[0]
    predicted_style = label_encoder_style.inverse_transform([predicted_style_index])[0]

    return jsonify({
        'artwork': predicted_artwork,
        'artist': predicted_artist,
        'date': predicted_date,
        'style': predicted_style
    })

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)
