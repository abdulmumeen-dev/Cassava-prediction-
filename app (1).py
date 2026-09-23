"""
app.py
------
Flask web app for predictive analysis of cassava yield using a trained
RandomForestRegressor (scikit-learn).

Run:
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "cassava_yield_model.pkl")

app = Flask(__name__)

_bundle = None


def get_bundle():
    """Lazy-load the bundled model; train only for local development."""
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            if os.environ.get("VERCEL"):
                raise RuntimeError(
                    "The trained model file is missing from the deployment. "
                    "Include model/cassava_yield_model.pkl in the repository."
                )
            from model.train_model import train
            train()
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def build_feature_row(form, feature_columns):
    """Turn raw form input into a one-row DataFrame matching training columns."""
    numeric = {
        "rainfall_mm": float(form.get("rainfall_mm", 0)),
        "avg_temp_c": float(form.get("avg_temp_c", 0)),
        "soil_ph": float(form.get("soil_ph", 0)),
        "humidity_pct": float(form.get("humidity_pct", 0)),
        "sunlight_hours": float(form.get("sunlight_hours", 0)),
        "fertilizer_kg_ha": float(form.get("fertilizer_kg_ha", 0)),
        "planting_density_k_ha": float(form.get("planting_density_k_ha", 0)),
        "pest_incidence_pct": float(form.get("pest_incidence_pct", 0)),
    }
    variety = form.get("variety", "Local")
    soil_type = form.get("soil_type", "Loam")

    row = {col: 0 for col in feature_columns}
    for k, v in numeric.items():
        if k in row:
            row[k] = v

    variety_col = f"variety_{variety}"
    soil_col = f"soil_type_{soil_type}"
    if variety_col in row:
        row[variety_col] = 1
    if soil_col in row:
        row[soil_col] = 1

    return pd.DataFrame([row], columns=feature_columns)


@app.route("/")
def index():
    bundle = get_bundle()
    return render_template(
        "index.html",
        varieties=bundle["varieties"],
        soil_types=bundle["soil_types"],
        metrics=bundle["metrics"],
    )


@app.route("/health")
def health():
    """Simple readiness endpoint for local tools and future deployment."""
    bundle = get_bundle()
    return jsonify({
        "status": "ok",
        "model_ready": bundle["model"] is not None,
        "metrics": bundle["metrics"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    bundle = get_bundle()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    try:
        X = build_feature_row(request.form, feature_columns)
        prediction = float(model.predict(X)[0])
        prediction = max(0.0, round(prediction, 2))

        importances = sorted(
            zip(feature_columns, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )[:5]
        top_factors = [{"feature": f, "importance": round(float(i), 3)} for f, i in importances]

        return jsonify({
            "success": True,
            "predicted_yield_tonnes_ha": prediction,
            "estimated_total_tonnes": None,
            "top_factors": top_factors,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API version, e.g. for curl / mobile app integration."""
    bundle = get_bundle()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    data = request.get_json(force=True, silent=True) or {}
    try:
        X = build_feature_row(data, feature_columns)
        prediction = float(model.predict(X)[0])
        prediction = max(0.0, round(prediction, 2))
        return jsonify({"success": True, "predicted_yield_tonnes_ha": prediction})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    get_bundle()  # ensure model exists / is trained before serving
    app.run(debug=True, host="0.0.0.0", port=5000)
