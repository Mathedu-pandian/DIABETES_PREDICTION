 🩺 Diabetes Prediction API

A simple **machine learning + Flask** web API that predicts whether a patient is likely to have diabetes based on medical features (e.g., glucose level, BMI, age).  
The model is trained offline and served via a REST endpoint deployed on **Render**.

🚀 Features

- Trained ML model (e.g., Logistic Regression) for diabetes prediction  
- Preprocessing with a scaler (same as used during training)  
- REST API endpoint: `POST /predict`  
- JSON input → JSON output  
- Ready to deploy on **Render** or run locally

🧠 Tech Stack

- Python
- Flask
- scikit-learn
- pandas
- joblib
- Deployed on Render

---

📁 Project Structure

DIABETES_PREDICTION/
├── app.py                # Flask app with /predict endpoint
├── model.pkl             # Trained ML model (joblib/pickle)
├── scaler.pkl            # Fitted scaler used for preprocessing
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

🔧 Setup (Local)
1️⃣ Clone the repository
git clone https://github.com/Mathedu-pandian/DIABETES_PREDICTION.git
cd DIABETES_PREDICTION

2️⃣ Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # On Linux/Mac
# or
venv\Scripts\activate           # On Windows

3️⃣ Install dependencies
pip install -r requirements.txt

Make sure requirements.txt includes at least:

flask
pandas
numpy
scikit-learn
joblib
gunicorn

4️⃣ Ensure model files are present

Place your trained model and scaler in the project root:

model.pkl

scaler.pkl

Update the paths in app.py if your filenames are different.

Example in app.py:

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

5️⃣ Run the app locally
python app.py

The app will start (by default) at:

http://0.0.0.0:5000/

You can change the port in app.py if needed.

🌐 Deployment on Render

Push this project to a GitHub repository.

Go to Render
, create a New → Web Service.

Connect your GitHub repo.

Set:

Build Command:

pip install -r requirements.txt

Start Command (recommended):
gunicorn app:app

Deploy. Render will assign a public URL like:

https://your-service-name.onrender.com

📡 API Usage
Endpoint
POST /predict
Content-Type: application/json

Expected JSON Input

The API expects these 8 features (same order as used in training):

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

Example Request (JSON)
{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 85,
  "BMI": 28.5,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 32
}

Example curl Command
curl -X POST "https://your-service-name.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 85,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 32
      }'

Example Response
{
  "prediction": 0
}


Where typically:

0 → No diabetes predicted

1 → Diabetes predicted

(Interpretation may be adjusted based on your training.)

✅ Notes

Make sure the feature order and preprocessing (scaling) at inference time matches training.

If you retrain the model, regenerate and replace:

model.pkl
scaler.pkl
