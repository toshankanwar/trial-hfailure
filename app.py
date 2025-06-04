from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import gc  # For garbage collection
from tensorflow import lite  # Much lighter than full TensorFlow

app = Flask(__name__)
CORS(app, resources={r"/*": {
    "origins": ["https://heart-failure-prediction-final.vercel.app","https://heart-failure.toshankanwar.website"]
}})

# Memory tracking
import psutil
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Load models with memory optimization
try:
    # Load TFLite model (uses much less memory)
    interpreter = lite.Interpreter(model_path="model_optimized.tflite")
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load scaler
    scaler = joblib.load("scaler.pkl")
    
    print(f"Initial memory usage: {get_memory_usage():.2f} MB")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    interpreter = None
    scaler = None

# Mappings
sex_map = {'M': 1, 'F': 0}
chest_pain_map = {'ATA': 1, 'NAP': 2, 'ASY': 0, 'TA': 3}
resting_ecg_map = {'Normal': 1, 'ST': 2, 'LVH': 0}
exercise_angina_map = {'N': 0, 'Y': 1}
st_slope_map = {'Up': 2, 'Flat': 1, 'Down': 0}

@app.route('/health', methods=['GET'])
def health_check():
    mem_usage = get_memory_usage()
    return jsonify({
        "status": "healthy",
        "memory_usage_mb": round(mem_usage, 2),
        "timestamp": "2025-05-19 08:54:16"
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not interpreter or not scaler:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Force garbage collection before prediction
        gc.collect()
        
        data = request.get_json()
        
        # Extract and encode features with memory optimization
        features = np.array([[
            float(data['Age']),
            sex_map.get(data['Sex'], -1),
            chest_pain_map.get(data['ChestPainType'], -1),
            float(data['RestingBP']),
            float(data['Cholesterol']),
            int(data['FastingBS']),
            resting_ecg_map.get(data['RestingECG'], -1),
            float(data['MaxHR']),
            exercise_angina_map.get(data['ExerciseAngina'], -1),
            float(data['Oldpeak']),
            st_slope_map.get(data['ST_Slope'], -1)
        ]], dtype=np.float32)  # Use float32 instead of float64 for memory efficiency

        # Scale features
        features_scaled = scaler.transform(features)
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], features_scaled)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output tensor
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        # Clean up memory
        del features
        del features_scaled
        gc.collect()
        
        return jsonify({
            "prediction": "Heart Disease Risk" if prediction > 0.5 else "No Risk",
            "probability": float(prediction),
            "memory_usage_mb": round(get_memory_usage(), 2)
        })

    except Exception as e:
        gc.collect()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
