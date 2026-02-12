import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from markupsafe import escape

# Initialize Flask app
app = Flask(__name__)


app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "dev-secret")
# Load Model & Scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "regmodel.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

with open(model_path, "rb") as model_file:
    regmodel = pickle.load(model_file)

with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# JSON API endpoint
@app.route('/predict_api', methods=['POST'])
def predict_api():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.json.get('data')
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Convert JSON values to numpy array
    features = np.array(list(data.values())).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = regmodel.predict(features_scaled)[0]

    return jsonify({"prediction": float(prediction)})

# HTML Form endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Grab data from HTML form
    try:
        data = [float(x) for x in request.form.values()]
    except ValueError:
        return render_template('index.html', prediction_text="Invalid input. Please enter numeric values.")

    features = np.array(data).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = regmodel.predict(features_scaled)[0]

    # Convert to dollars if your target is in 100k units
    prediction_dollars = round(prediction * 100000, 2)

    return render_template('index.html', prediction_text=f'Predicted House Price is ${prediction_dollars}')

# Run Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render sets the PORT environment variable
    app.run(host="0.0.0.0", port=port, debug=False)  # debug=False for production
