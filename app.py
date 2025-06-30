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

# Load TensorFlow Lite model
try:
    interpreter = tf.lite.Interpreter(model_path='model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("TFLite model loaded successfully")
    model_loaded = True
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    interpreter = None
    model_loaded = False

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Potato Leaf API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Get image from request
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        
        # Preprocess image for TFLite
        # Get input shape from model
        input_shape = input_details[0]['shape']
        target_size = (input_shape[1], input_shape[2])  # Height, Width
        
        # Resize and preprocess
        image = image.resize(target_size)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Run inference with TFLite
        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        
        # Get prediction
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
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