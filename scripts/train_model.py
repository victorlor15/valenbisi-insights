"""Train the availability-forecasting model for ValenBisi Insights.

Target: number of bikes *available* at a station for a given calendar context
(station id, hour of day, day of week, month, weekend flag, dock capacity, location).

A HistGradientBoostingRegressor is used: fast, handles the categorical station id
natively, and captures the strong hour x day-of-week interactions seen in the data.
The fitted pipeline + metrics are saved to models/forecast_gbr.joblib for the app.

Run:  python scripts/train_model.py   (requires data/processed/training_sample.parquet)
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"

# Station identity is captured by its (unique) lat/lon + capacity, so we avoid the
# HistGradientBoosting 255-category limit while still letting the model learn
# station-specific behaviour through fine splits on coordinates.
FEATURES = ["hour_of_day", "dow", "month", "is_weekend", "total", "lat", "lon"]
TARGET = "available"


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROC / "training_sample.parquet")
    df = df.dropna(subset=FEATURES + [TARGET, "station"])

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    X_tr, y_tr = train_df[FEATURES], train_df[TARGET]
    X_te, y_te = test_df[FEATURES], test_df[TARGET]

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=500,
        learning_rate=0.08,
        max_depth=None,
        max_leaf_nodes=63,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(X_tr, y_tr)

    pred = np.clip(model.predict(X_te), 0, None)
    # naïve baseline: predict each station's own training-set mean availability
    station_means = train_df.groupby("station")[TARGET].mean()
    baseline_pred = test_df["station"].map(station_means).fillna(y_tr.mean())
    metrics = {
        "mae": float(mean_absolute_error(y_te, pred)),
        "rmse": float(root_mean_squared_error(y_te, pred)),
        "r2": float(r2_score(y_te, pred)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "baseline_mae": float(mean_absolute_error(y_te, baseline_pred)),
    }

    payload = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "metrics": metrics,
    }
    joblib.dump(payload, MODELS / "forecast_gbr.joblib")
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("Saved models/forecast_gbr.joblib")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
