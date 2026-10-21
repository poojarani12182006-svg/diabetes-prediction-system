"""
app.py
------
Flask backend for the Diabetes Prediction System.

Responsibilities:
1. Serve the website pages (Home, About, Predict, Model Performance, Contact).
2. Receive user input from the prediction form.
3. Preprocess the input the same way the training data was preprocessed.
4. Load the saved ML model + scaler and generate a prediction.
5. Return the result to the frontend (rendered page or JSON, depending on route).
"""

import os
import json

import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the trained model, scaler, and metrics once at startup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

model = None
scaler = None
metrics = None

FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def load_artifacts():
    """Load model, scaler and metrics from disk. Called once at startup."""
    global model, scaler, metrics

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        # The website can still run without a trained model — the predict
        # page will show a friendly message telling the user to train first.
        model = None
        scaler = None

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    else:
        metrics = None


load_artifacts()


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------
# (field name, min allowed, max allowed, friendly label)
VALIDATION_RULES = {
    "Pregnancies": (0, 20, "Pregnancies"),
    "Glucose": (0, 300, "Glucose"),
    "BloodPressure": (0, 200, "Blood Pressure"),
    "SkinThickness": (0, 100, "Skin Thickness"),
    "Insulin": (0, 900, "Insulin"),
    "BMI": (0, 80, "BMI"),
    "DiabetesPedigreeFunction": (0, 3, "Diabetes Pedigree Function"),
    "Age": (1, 120, "Age"),
}


def validate_and_parse(form):
    """
    Validates raw form input.
    Returns (values_dict, errors_list).
    """
    values = {}
    errors = []

    for field, (min_val, max_val, label) in VALIDATION_RULES.items():
        raw_value = form.get(field, "")

        # Form posts arrive as strings; JSON posts may already be int/float.
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()

        if raw_value == "" or raw_value is None:
            errors.append(f"{label} is required.")
            continue

        try:
            value = float(raw_value)
        except (ValueError, TypeError):
            errors.append(f"{label} must be a number.")
            continue

        if value < min_val or value > max_val:
            errors.append(f"{label} must be between {min_val} and {max_val}.")
            continue

        values[field] = value

    return values, errors


def preprocess_for_prediction(values):
    """
    Applies the SAME preprocessing logic used during training:
    - Build a feature vector in the correct column order.
    - Apply the fitted StandardScaler.
    Note: we do not zero-impute here because the user provides real values
    directly through the form; the imputation step in training only existed
    to fix invalid zero-placeholders in the historical dataset.
    """
    feature_vector = np.array([[values[col] for col in FEATURE_ORDER]])
    scaled_vector = scaler.transform(feature_vector)
    return scaled_vector


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html", active_page="home")


@app.route("/about")
def about():
    return render_template("about.html", active_page="about")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    errors = []
    submitted_values = {}

    if request.method == "POST":
        submitted_values, errors = validate_and_parse(request.form)

        if not errors:
            if model is None or scaler is None:
                errors.append(
                    "The prediction model is not available yet. "
                    "Please train the model first (see README)."
                )
            else:
                scaled_input = preprocess_for_prediction(submitted_values)
                prediction = int(model.predict(scaled_input)[0])
                probabilities = model.predict_proba(scaled_input)[0]
                confidence = float(probabilities[prediction]) * 100

                result = {
                    "prediction": "Diabetic" if prediction == 1 else "Not Diabetic",
                    "is_diabetic": bool(prediction == 1),
                    "confidence": round(confidence, 2),
                }

    return render_template(
        "predict.html",
        active_page="predict",
        result=result,
        errors=errors,
        submitted_values=submitted_values,
    )


@app.route("/performance")
def performance():
    return render_template("performance.html", active_page="performance", metrics=metrics)


@app.route("/contact")
def contact():
    return render_template("contact.html", active_page="contact")


# ---------------------------------------------------------------------------
# JSON API endpoint (useful for AJAX calls or external clients)
# ---------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if request.is_json:
        form_data = request.get_json()
    else:
        form_data = request.form

    values, errors = validate_and_parse(form_data)

    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    if model is None or scaler is None:
        return jsonify({"success": False, "errors": ["Model not available. Train it first."]}), 503

    scaled_input = preprocess_for_prediction(values)
    prediction = int(model.predict(scaled_input)[0])
    probabilities = model.predict_proba(scaled_input)[0]
    confidence = float(probabilities[prediction]) * 100

    return jsonify(
        {
            "success": True,
            "prediction": "Diabetic" if prediction == 1 else "Not Diabetic",
            "is_diabetic": bool(prediction == 1),
            "confidence": round(confidence, 2),
            "disclaimer": (
                "This is an educational ML prediction, not a medical diagnosis. "
                "Please consult a qualified healthcare professional."
            ),
        }
    )


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
