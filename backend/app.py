"""
Flask API backend for fingerprint-based blood group prediction.
"""

import os
import traceback
import cv2
import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from utils.preprocess import preprocess_fingerprint
from utils.feature_extraction import extract_hog_features

app = Flask(__name__)

# Allow requests from frontend
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*")
CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}})

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

_model = None
_scaler = None
_load_error = None

try:
    _model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    _scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
except Exception as e:
    _load_error = str(e)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Fingerprint Blood Group Prediction API",
        "status": "running"
    })


@app.route("/health", methods=["GET"])
def health():
    if _load_error:
        return jsonify({
            "status": "degraded",
            "model_loaded": False,
            "error": _load_error
        }), 503

    return jsonify({
        "status": "ok",
        "model_loaded": True
    })


@app.route("/predict", methods=["POST"])
def predict():

    if _load_error:
        return jsonify({
            "error": f"Model artifacts failed to load: {_load_error}"
        }), 503

    if "image" not in request.files:
        return jsonify({
            "error": "Please upload an image."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "No image selected."
        }), 400

    try:

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({
                "error": "Invalid image."
            }), 400

        processed = preprocess_fingerprint(img)

        features = extract_hog_features(processed)

        features = _scaler.transform(features.reshape(1, -1))

        prediction = _model.predict(features)[0]

        response = {
            "prediction": str(prediction)
        }

        if hasattr(_model, "predict_proba"):
            probs = _model.predict_proba(features)[0]

            response["probabilities"] = {
                str(cls): float(prob)
                for cls, prob in zip(_model.classes_, probs)
            }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
