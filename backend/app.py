"""
Flask API backend for fingerprint-based blood group prediction.

Deployed separately on Render. This app exposes JSON endpoints only
(no HTML templates) since the frontend now lives on Vercel and talks
to this API over HTTP.

Endpoints:
    GET  /health   -> simple liveness check for Render
    POST /predict   -> accepts an image file, returns predicted blood group
"""

import os
import traceback

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from utils.preprocess import preprocess_fingerprint
from utils.feature_extraction import extract_hog_features

app = Flask(__name__)

# Allow the Vercel frontend (and local dev) to call this API.
# Set FRONTEND_ORIGIN as an env var on Render once you know your Vercel URL,
# e.g. https://your-app.vercel.app
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*")
CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}})

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

# Load model artifacts once at startup, not per-request
_model = None
_scaler = None
_label_encoder = None
_load_error = None

try:
    _model = joblib.load(os.path.join(MODEL_DIR, "voting_classifier.pkl"))
    _scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    _label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
except Exception as e:  # noqa: BLE001 - we want to surface any load failure via /health
    _load_error = str(e)


@app.route("/health", methods=["GET"])
def health():
    """Render uses this to check the service is alive."""
    if _load_error:
        return jsonify({"status": "degraded", "model_loaded": False, "error": _load_error}), 503
    return jsonify({"status": "ok", "model_loaded": True})


@app.route("/predict", methods=["POST"])
def predict():
    if _load_error:
        return jsonify({"error": f"Model artifacts failed to load: {_load_error}"}), 503

    if "image" not in request.files:
        return jsonify({"error": "No file uploaded. Send it as multipart/form-data under key 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        # Read the uploaded file into a numpy array OpenCV can use
        file_bytes = np.frombuffer(file.read(), np.uint8)
        import cv2
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        if img is None:
            return jsonify({"error": "Could not decode image. Please upload a valid image file."}), 400

        processed = preprocess_fingerprint(img)
        features = extract_hog_features(processed)
        features_scaled = _scaler.transform(features.reshape(1, -1))

        prediction_idx = _model.predict(features_scaled)[0]
        prediction_label = _label_encoder.inverse_transform([prediction_idx])[0]

        response = {"prediction": str(prediction_label)}

        # Include class probabilities if the model supports it
        if hasattr(_model, "predict_proba"):
            probs = _model.predict_proba(features_scaled)[0]
            classes = _label_encoder.inverse_transform(np.arange(len(probs)))
            response["probabilities"] = {
                str(cls): float(p) for cls, p in zip(classes, probs)
            }

        return jsonify(response)

    except Exception as e:  # noqa: BLE001 - return the error instead of a bare 500
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
