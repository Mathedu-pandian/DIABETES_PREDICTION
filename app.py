from flask import request, jsonify
import pandas as pd
import joblib

flask = (__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json(force=True)

    # Ensure data is in the correct format (list of lists or dict of lists for DataFrame)
    try:
        # Convert the dictionary data to a DataFrame. This assumes data is a single record.
        # If 'data' can contain multiple records, it should be a list of dictionaries.
        input_df = pd.DataFrame([data])
    except ValueError as e:
        return jsonify({"error": f"Invalid input data format: {e}"}), 400

    # Ensure the column order matches the training data features (X)
    # Assuming 'X' (or X.columns) from earlier preprocessing is still available or can be recreated.
    # For this example, let's assume the columns used in 'X' are known.
    # Replace with actual feature names if X is not globally available.
    feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    # Check if all required features are present
    if not all(col in input_df.columns for col in feature_columns):
        missing_cols = [col for col in feature_columns if col not in input_df.columns]
        return jsonify({"error": f"Missing features in input data: {', '.join(missing_cols)}"}), 400
    
    input_df = input_df[feature_columns]

    # Preprocess the input data using the loaded scaler
    scaled_data = scaler.transform(input_df)

    # Make a prediction using the loaded model
    prediction = model.predict(scaled_data)

    # Return the prediction as a JSON response
    return jsonify({"prediction": int(prediction[0])})

print("Flask /predict endpoint defined.")

if__name__ = __main__:
app.run(debug=TRUE)
