import json
import os
from pathlib import Path

import matplotlib

BASE_DIR = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(BASE_DIR / ".matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():
    data_path = BASE_DIR / "processed_weather.csv"
    vis_dir = BASE_DIR / "visualizations"
    metrics_path = BASE_DIR / "metrics.json"
    vis_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)

    time_col = "last_updated" if "last_updated" in df.columns else "lastupdated"
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    location_counts = (
        df.dropna(subset=["location_name", "temperature_celsius"])
        .groupby("location_name")
        .size()
        .sort_values(ascending=False)
    )
    location = location_counts.index[0]

    weather = (
        df.loc[df["location_name"] == location]
        .dropna(subset=[time_col, "temperature_celsius"])
        .sort_values(time_col)
    )

    daily = (
        weather.set_index(time_col)["temperature_celsius"]
        .resample("D")
        .mean()
        .interpolate(limit_direction="both")
        .reset_index()
    )
    daily.columns = ["date", "temperature_celsius"]

    daily["lag_1"] = daily["temperature_celsius"].shift(1)
    daily["lag_3"] = daily["temperature_celsius"].shift(3)
    daily["lag_7"] = daily["temperature_celsius"].shift(7)
    daily["roll_mean_3"] = daily["temperature_celsius"].shift(1).rolling(3).mean()
    daily["roll_mean_7"] = daily["temperature_celsius"].shift(1).rolling(7).mean()
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily = daily.dropna().reset_index(drop=True)

    split_idx = int(len(daily) * 0.8)
    train = daily.iloc[:split_idx].copy()
    test = daily.iloc[split_idx:].copy()

    feature_cols = ["lag_1", "lag_3", "lag_7", "roll_mean_3", "roll_mean_7", "day_of_week", "month"]
    X_train = train[feature_cols]
    y_train = train["temperature_celsius"]
    X_test = test[feature_cols]
    y_test = test["temperature_celsius"]

    baseline_preds = X_test["lag_1"].to_numpy()

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "baseline": {
            "rmse": float(np.sqrt(mean_squared_error(y_test, baseline_preds))),
            "mae": float(mean_absolute_error(y_test, baseline_preds)),
        },
        "random_forest": {
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "mae": float(mean_absolute_error(y_test, preds)),
        },
        "forecast_location": location,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plt.figure(figsize=(12, 6))
    plt.plot(test["date"], test["temperature_celsius"], label="actual", linewidth=2)
    plt.plot(test["date"], baseline_preds, label="baseline", linestyle="--")
    plt.plot(test["date"], preds, label="random_forest", linestyle="-.")
    plt.title(f"Temperature forecast - {location}")
    plt.xlabel("Date")
    plt.ylabel("Temperature (C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(vis_dir / "forecast_actual_vs_predicted.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
