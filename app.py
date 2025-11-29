from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model & scaler - MAKE SURE THESE FILES ARE IN GITHUB
model = joblib.load("logistic_regression_model.joblib")
scaler = joblib.load("standard_scaler.joblib")

FEATURE_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

@app.route("/")
def home():
    return "Diabetes Prediction API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    df = pd.DataFrame([data])

    df = df[FEATURE_COLUMNS]

    scaled = scaler.transform(df)

    prediction = model.predict(scaled)

    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
