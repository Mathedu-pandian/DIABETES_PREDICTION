from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

# Create Flask app
app = Flask(__name__)

# Load model and scaler (adjust file paths as needed)
# Make sure these .pkl files are in the same folder or give full path
model = joblib.load("logistic_regression_model.joblib")
scaler = joblib.load("standard_scaler.joblib")

# Define the feature columns in the same order as during training
FEATURE_COLUMNS = [
    'Pregnancies', 
    'Glucose', 
    'BloodPressure', 
    'SkinThickness', 
    'Insulin', 
    'BMI', 
    'DiabetesPedigreeFunction', 
    'Age'
]

@app.route("/predict", methods=["POST"])
def predict():
    # Check JSON
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json(force=True)

    # If you expect a single record like:
    # {
    #   "Pregnancies": 2,
    #   "Glucose": 120,
    #   ...
    # }
    try:
        input_df = pd.DataFrame([data])
    except Exception as e:
        return jsonify({"error": f"Invalid input data format: {e}"}), 400

    # Check if all required columns present
    missing_cols = [col for col in FEATURE_COLUMNS if col not in input_df.columns]
    if missing_cols:
        return jsonify({
            "error": f"Missing features in input data: {', '.join(missing_cols)}"
        }), 400

    # Reorder columns
    input_df = input_df[FEATURE_COLUMNS]

    # Scale input
    scaled_data = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_data)

    # Convert numpy type to Python int for JSON serialization
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    # For local testing
    # For cloud deployment (Render/Railway/etc.), they often set PORT in env
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
