from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Enable CORS for the app
CORS(app)  # This will allow all origins. You can customize it as needed.

# Load model and scaler
model = load_model("heart_disease_model.h5")
scaler = joblib.load("scaler.pkl")

# Label encoding mappings (same as training)
sex_map = {'M': 1, 'F': 0}
chest_pain_map = {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3}
resting_ecg_map = {'Normal': 1, 'ST': 2, 'LVH': 0}
exercise_angina_map = {'N': 0, 'Y': 1}
st_slope_map = {'Up': 2, 'Flat': 1, 'Down': 0}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # Extract and encode input features
        features = [
            data['Age'],
            sex_map[data['Sex']],
            chest_pain_map[data['ChestPainType']],
            data['RestingBP'],
            data['Cholesterol'],
            data['FastingBS'],
            resting_ecg_map[data['RestingECG']],
            data['MaxHR'],
            exercise_angina_map[data['ExerciseAngina']],
            data['Oldpeak'],
            st_slope_map[data['ST_Slope']]
        ]

        # Preprocess input
        features = np.array([features])
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0][0]
        result = "Heart Disease Risk" if prediction > 0.5 else "No Risk"

        return jsonify({
            "prediction": result,
            "probability": float(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
