from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("stress_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return "AI Stress Detection API is Live!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        temp = float(data["temp"])
        hr   = float(data["hr"])
        spo2 = float(data["spo2"])
        gsr  = float(data["gsr"])
    except Exception as e:
        return jsonify({"error": "Invalid input. Provide JSON with keys temp, hr, spo2, gsr"}), 400

    features = np.array([[temp, hr, spo2, gsr]])
    pred = model.predict(features)[0]
    return jsonify({"result": str(pred)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
