# 🌱 Cassava Yield Predictor

A Flask + scikit-learn web app for agricultural predictive analysis of
cassava yield (tonnes/hectare) based on climate, soil, and farming inputs.

## Features
- RandomForestRegressor trained on a generated agronomic dataset
  (rainfall, temperature, soil pH, humidity, sunlight, fertilizer,
  planting density, pest incidence, variety, soil type)
- Clean web UI with a prediction form
- JSON API endpoint (`/api/predict`) for programmatic access
- Feature importance breakdown shown with each prediction
- Model auto-trains on first run if no saved model is found

## Project structure
```
cassava_yield_app/
├── app.py                  # Flask app (routes + prediction logic)
├── requirements.txt
├── model/
│   ├── train_model.py      # Dataset generation + model training
│   └── cassava_yield_model.pkl   # Trained model (generated)
├── data/
│   └── cassava_data.csv    # Training dataset (generated)
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Setup

1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Train the model explicitly — otherwise it trains
   automatically the first time you run the app:
   ```bash
   python model/train_model.py
   ```

4. Run the app:
   ```bash
   python app.py
   ```

5. Open your browser at **http://127.0.0.1:5000**

## API usage

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
        "rainfall_mm": 1300,
        "avg_temp_c": 27,
        "soil_ph": 6.1,
        "humidity_pct": 72,
        "sunlight_hours": 6.8,
        "fertilizer_kg_ha": 140,
        "planting_density_k_ha": 10.5,
        "pest_incidence_pct": 10,
        "variety": "TMS 30572",
        "soil_type": "Loam"
      }'
```

## Using real data

The bundled dataset is **synthetically generated** so the app works
out of the box. To get real-world accuracy:

1. Replace `data/cassava_data.csv` with your own field records, keeping
   the same column names (or update `train_model.py` to match your columns).
2. Re-run `python model/train_model.py` to retrain and overwrite
   `model/cassava_yield_model.pkl`.
3. Restart the Flask app.

## Tech stack
- Python 3.10+
- Flask
- scikit-learn (RandomForestRegressor)
- pandas / numpy
- joblib (model persistence)

## Prototype terminal interface

The repository includes `predict.py`, a command-line interface for field
estimates. After activating the virtual environment, run:

```shell
python predict.py \
  --rainfall 1200 --temperature 27 --soil-ph 6.0 \
  --humidity 70 --sunlight 6.5 --fertilizer 120 \
  --density 10 --pests 15 \
  --variety "TMS 30572" --soil-type Loam
```

Add `--json` for machine-readable output. The local readiness endpoint is
available at `GET /health` and returns the model metrics and readiness status.

## Uploading and deploying

Upload the contents of this repository with the folder structure unchanged.
The included `vercel.json` and `api/index.py` provide a Vercel serverless
entrypoint. For local development, continue to run `python app.py`.
