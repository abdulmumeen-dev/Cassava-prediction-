#!/usr/bin/env python3
"""Run a cassava yield prediction from the terminal.

Examples:
  python predict.py --rainfall 1200 --temperature 27 --soil-ph 6 \
    --humidity 70 --sunlight 6.5 --fertilizer 120 --density 10 \
    --pests 15 --variety "TMS 30572" --soil-type Loam

Add --json for machine-readable output.
"""

import argparse
import json

from app import build_feature_row, get_bundle


def parser_for(bundle):
    parser = argparse.ArgumentParser(
        description="CassavaLab: estimate cassava yield from field conditions."
    )
    parser.add_argument("--rainfall", type=float, required=True, help="Seasonal rainfall in mm (300–2500)")
    parser.add_argument("--temperature", type=float, required=True, help="Average temperature in °C (15–38)")
    parser.add_argument("--soil-ph", type=float, required=True, help="Soil pH (3.5–8.5)")
    parser.add_argument("--humidity", type=float, required=True, help="Average humidity percentage (20–100)")
    parser.add_argument("--sunlight", type=float, required=True, help="Sunlight hours per day (2–12)")
    parser.add_argument("--fertilizer", type=float, required=True, help="Fertilizer applied in kg/ha (0–400)")
    parser.add_argument("--density", type=float, required=True, help="Planting density in thousand plants/ha (4–20)")
    parser.add_argument("--pests", type=float, required=True, help="Pest incidence percentage (0–80)")
    parser.add_argument("--variety", choices=bundle["varieties"], required=True)
    parser.add_argument("--soil-type", choices=bundle["soil_types"], required=True)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the formatted report")
    return parser


def main():
    bundle = get_bundle()
    args = parser_for(bundle).parse_args()
    form = {
        "rainfall_mm": args.rainfall,
        "avg_temp_c": args.temperature,
        "soil_ph": args.soil_ph,
        "humidity_pct": args.humidity,
        "sunlight_hours": args.sunlight,
        "fertilizer_kg_ha": args.fertilizer,
        "planting_density_k_ha": args.density,
        "pest_incidence_pct": args.pests,
        "variety": args.variety,
        "soil_type": args.soil_type,
    }
    row = build_feature_row(form, bundle["feature_columns"])
    prediction = max(0.0, round(float(bundle["model"].predict(row)[0]), 2))
    factors = sorted(
        zip(bundle["feature_columns"], bundle["model"].feature_importances_),
        key=lambda item: item[1], reverse=True
    )[:5]
    result = {
        "predicted_yield_tonnes_ha": prediction,
        "model_metrics": bundle["metrics"],
        "top_factors": [
            {"feature": feature, "importance": round(float(importance), 3)}
            for feature, importance in factors
        ],
    }
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print("\nCassavaLab / FIELD ESTIMATE")
    print("─" * 32)
    print(f"Projected yield : {prediction:.2f} tonnes / hectare")
    print(f"Model R²        : {bundle['metrics']['r2']:.2f}")
    print(f"Model MAE       : {bundle['metrics']['mae']:.2f} tonnes / hectare")
    print("\nTop contributing factors")
    for feature, importance in factors:
        print(f"  • {feature.replace('_', ' ').title():<28} {importance * 100:>5.1f}%")
    print()


if __name__ == "__main__":
    main()
