from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Create the Flask app instance
app = Flask(__name__)

# Path to your TFLite model
TFLITE_MODEL_PATH = "model.tflite"

# Define your class labels (make sure they match your training output!)
CLASS_LABELS = ["early_blight", "healthy", "late_blight"]

# Load the TFLite model once at startup
tflite_interpreter = None

def load_model():
    """
    Loads the TFLite model and allocates tensors.
    """
    global tflite_interpreter
    try:
        tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        tflite_interpreter.allocate_tensors()
        print("✅ TFLite model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load TFLite model. Make sure '{TFLITE_MODEL_PATH}' is correct. Error: {e}")
        tflite_interpreter = None

# Call it once at startup
load_model()

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image upload, preprocesses it, runs inference, and returns prediction.
    """
    # Check if the request has a file
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Preprocess the uploaded image
        image_data = preprocess_image(file)

        # Run inference
        raw_prediction = run_inference(image_data)

        if raw_prediction is None:
            return jsonify({"error": "Inference failed or model not loaded"}), 500

        # Get the predicted class and confidence
        predicted_index = np.argmax(raw_prediction)
        confidence = float(raw_prediction[0][predicted_index])
        predicted_label = CLASS_LABELS[predicted_index]

        return jsonify({"prediction": predicted_label, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def preprocess_image(image_file):
    """
    Preprocesses an uploaded image to match the model input.

    - Converts to RGB
    - Resizes to 256x256 (adjust if your model expects a different size)
    - Normalizes to [0,1]
    - Adds batch dimension
    """
    image = Image.open(image_file).convert('RGB')
    image = image.resize((256, 256))  # Adjust if your model needs a different size
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def run_inference(image_data):
    """
    Runs the TFLite model on the preprocessed image.

    Returns:
        The raw output tensor or None if failed.
    """
    global tflite_interpreter
    if tflite_interpreter is None:
        print("❌ Error: TFLite model not loaded.")
        return None

    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    tflite_interpreter.set_tensor(input_details[0]['index'], image_data)
    tflite_interpreter.invoke()
    output_data = tflite_interpreter.get_tensor(output_details[0]['index'])

    return output_data


if __name__ == '__main__':
    # Use the port provided by the cloud provider, or 5000 by default
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
