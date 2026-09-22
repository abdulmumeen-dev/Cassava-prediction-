"""
train_model.py
---------------
Generates a synthetic-but-realistic cassava agronomy dataset and trains a
RandomForestRegressor to predict cassava yield (tonnes/hectare).

Run this once to (re)create data/cassava_data.csv and model/cassava_yield_model.pkl.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "cassava_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "cassava_yield_model.pkl")

np.random.seed(42)


def generate_dataset(n_samples=2000):
    """Create a synthetic dataset with agronomically plausible relationships."""

    rainfall_mm = np.random.normal(1200, 300, n_samples).clip(300, 2500)
    avg_temp_c = np.random.normal(27, 3, n_samples).clip(15, 38)
    soil_ph = np.random.normal(6.0, 0.7, n_samples).clip(3.5, 8.5)
    humidity_pct = np.random.normal(70, 12, n_samples).clip(20, 100)
    sunlight_hours = np.random.normal(6.5, 1.2, n_samples).clip(2, 12)
    fertilizer_kg_ha = np.random.normal(120, 60, n_samples).clip(0, 400)
    planting_density_k_ha = np.random.normal(10, 2.5, n_samples).clip(4, 20)  # thousand plants/ha
    pest_incidence_pct = np.random.normal(15, 10, n_samples).clip(0, 80)
    variety = np.random.choice(["Local", "TME 419", "TMS 30572", "TMS 98/0505"], n_samples,
                                p=[0.3, 0.3, 0.25, 0.15])
    soil_type = np.random.choice(["Sandy", "Loam", "Clay", "Sandy-Loam"], n_samples,
                                  p=[0.25, 0.35, 0.2, 0.2])

    variety_bonus = pd.Series(variety).map({
        "Local": 0.0, "TME 419": 3.5, "TMS 30572": 4.2, "TMS 98/0505": 5.0
    }).values

    soil_bonus = pd.Series(soil_type).map({
        "Sandy": -1.5, "Loam": 2.0, "Clay": -0.5, "Sandy-Loam": 1.0
    }).values

    # Nonlinear, roughly agronomic relationship for yield (tonnes/ha)
    rainfall_effect = -((rainfall_mm - 1300) ** 2) / 250000 + 5
    temp_effect = -((avg_temp_c - 28) ** 2) / 20 + 3
    ph_effect = -((soil_ph - 6.2) ** 2) * 1.5 + 2
    fert_effect = np.log1p(fertilizer_kg_ha) * 0.9
    density_effect = -((planting_density_k_ha - 10) ** 2) * 0.05 + 1
    pest_effect = -pest_incidence_pct * 0.08
    sunlight_effect = sunlight_hours * 0.3
    humidity_effect = -((humidity_pct - 75) ** 2) / 900

    noise = np.random.normal(0, 1.3, n_samples)

    yield_tonnes_ha = (
        8.0
        + rainfall_effect
        + temp_effect
        + ph_effect
        + fert_effect
        + density_effect
        + pest_effect
        + sunlight_effect
        + humidity_effect
        + variety_bonus
        + soil_bonus
        + noise
    ).clip(1.0, 45.0)

    df = pd.DataFrame({
        "rainfall_mm": rainfall_mm.round(1),
        "avg_temp_c": avg_temp_c.round(1),
        "soil_ph": soil_ph.round(2),
        "humidity_pct": humidity_pct.round(1),
        "sunlight_hours": sunlight_hours.round(2),
        "fertilizer_kg_ha": fertilizer_kg_ha.round(1),
        "planting_density_k_ha": planting_density_k_ha.round(2),
        "pest_incidence_pct": pest_incidence_pct.round(1),
        "variety": variety,
        "soil_type": soil_type,
        "yield_tonnes_ha": yield_tonnes_ha.round(2),
    })
    return df


def train():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    df = generate_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved synthetic dataset -> {DATA_PATH} ({len(df)} rows)")

    df_encoded = pd.get_dummies(df, columns=["variety", "soil_type"])

    X = df_encoded.drop(columns=["yield_tonnes_ha"])
    y = df_encoded["yield_tonnes_ha"]

    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"MAE:  {mae:.3f} tonnes/ha")
    print(f"R^2:  {r2:.3f}")

    importances = sorted(
        zip(feature_columns, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("\nTop feature importances:")
    for name, imp in importances[:8]:
        print(f"  {name:<25} {imp:.3f}")

    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "varieties": sorted(df["variety"].unique().tolist()),
        "soil_types": sorted(df["soil_type"].unique().tolist()),
        "metrics": {"mae": mae, "r2": r2},
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved trained model -> {MODEL_PATH}")


if __name__ == "__main__":
    train()
