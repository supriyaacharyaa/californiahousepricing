import pickle
from flask import Flask, request, jsonify, render_template
from markupsafe import escape
import numpy as np

app = Flask(__name__)

# Load Model & Scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

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
    data = [float(x) for x in request.form.values()]
    features = np.array(data).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = regmodel.predict(features_scaled)[0]

    # Convert to dollars if your target is in 100k units
    prediction_dollars = round(prediction * 100000, 2)

    return render_template('index.html', prediction_text=f'Predicted House Price is ${prediction_dollars}')

if __name__ == "__main__":
    app.run(debug=True)
