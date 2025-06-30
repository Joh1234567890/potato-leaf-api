import os
from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model with error handling
try:
    model = tf.keras.models.load_model('your_model_path.h5')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Potato Leaf API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Your prediction logic here
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        
        # Preprocess image
        image = image.resize((224, 224))  # Adjust size as needed
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Make prediction
        predictions = model.predict(image_array)
        
        return jsonify({
            "prediction": predictions.tolist(),
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)