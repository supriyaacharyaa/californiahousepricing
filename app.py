import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load Model & Scaler safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "regmodel.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

regmodel = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# JSON API endpoint
@app.route("/predict_api", methods=["POST"])
def predict_api():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.json.get("data")
    if not data:
        return jsonify({"error": "No data provided"}), 400

    features = np.array(list(data.values())).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = regmodel.predict(features_scaled)[0]

    return jsonify({"prediction": float(prediction)})

# HTML Form endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    features = np.array(data).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = regmodel.predict(features_scaled)[0]

    prediction_dollars = round(prediction * 100000, 2)

    return render_template(
        "index.html",
        prediction_text=f"Predicted House Price is ${prediction_dollars}"
    )


