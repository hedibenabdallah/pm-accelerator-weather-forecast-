import os
from pathlib import Path

import matplotlib

BASE_DIR = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(BASE_DIR / ".matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def main():
    data_path = BASE_DIR / "processed_weather.csv"
    vis_dir = BASE_DIR / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)
    sns.set_theme(style="whitegrid")

    corr_cols = [
        "temperature_celsius",
        "humidity",
        "pressure_mb",
        "wind_kph",
        "precip_mm",
        "cloud",
        "visibility_km",
        "uv_index",
        "gust_kph",
        "air_quality_Carbon_Monoxide",
        "air_quality_Ozone",
        "air_quality_Nitrogen_dioxide",
        "air_quality_Sulphur_dioxide",
        "air_quality_PM2.5",
        "air_quality_PM10",
        "air_quality_us-epa-index",
        "air_quality_gb-defra-index",
    ]
    corr_cols = [col for col in corr_cols if col in df.columns]

    if len(corr_cols) >= 2:
        plt.figure(figsize=(14, 10))
        sns.heatmap(df[corr_cols].corr(), cmap="coolwarm", center=0, linewidths=0.3)
        plt.title("Correlation heatmap")
        plt.tight_layout()
        plt.savefig(vis_dir / "correlation_heatmap.png", dpi=200)
        plt.close()

    anomaly_cols = [
        "temperature_celsius",
        "humidity",
        "pressure_mb",
        "wind_kph",
        "precip_mm",
        "air_quality_PM2.5",
        "air_quality_PM10",
    ]
    anomaly_cols = [col for col in anomaly_cols if col in df.columns]

    if len(anomaly_cols) >= 2:
        scaled = pd.DataFrame(
            StandardScaler().fit_transform(df[anomaly_cols]),
            columns=anomaly_cols,
            index=df.index,
        )
        anomaly_model = IsolationForest(contamination=0.03, random_state=42)
        df["anomaly_flag"] = anomaly_model.fit_predict(scaled)

        x_col = "temperature_celsius" if "temperature_celsius" in df.columns else anomaly_cols[0]
        y_col = "humidity" if "humidity" in df.columns else anomaly_cols[1]

        plt.figure(figsize=(11, 7))
        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            hue="anomaly_flag",
            palette={1: "#4c78a8", -1: "#e45756"},
            alpha=0.75,
            s=45,
        )
        plt.title("Anomaly detection")
        plt.tight_layout()
        plt.savefig(vis_dir / "anomaly_detection.png", dpi=200)
        plt.close()

    if {"latitude", "longitude"}.issubset(df.columns):
        color_col = "temperature_celsius" if "temperature_celsius" in df.columns else "humidity"
        if color_col in df.columns:
            plt.figure(figsize=(14, 7))
            scatter = plt.scatter(
                df["longitude"],
                df["latitude"],
                c=df[color_col],
                cmap="viridis",
                alpha=0.8,
                s=55,
                edgecolor="none",
            )
            plt.colorbar(scatter, label=color_col)
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title("Spatial weather pattern")
            plt.tight_layout()
            plt.savefig(vis_dir / "spatial_weather_pattern.png", dpi=200)
            plt.close()

    if {"air_quality_PM2.5", "temperature_celsius"}.issubset(df.columns):
        plt.figure(figsize=(11, 7))
        scatter = plt.scatter(
            df["air_quality_PM2.5"],
            df["temperature_celsius"],
            c=df["humidity"] if "humidity" in df.columns else None,
            cmap="magma",
            alpha=0.7,
            s=50,
            edgecolor="none",
        )
        if "humidity" in df.columns:
            plt.colorbar(scatter, label="humidity")
        plt.xlabel("PM2.5")
        plt.ylabel("Temperature (C)")
        plt.title("Air quality vs weather")
        plt.tight_layout()
        plt.savefig(vis_dir / "air_quality_vs_weather.png", dpi=200)
        plt.close()

    if "temperature_celsius" in df.columns:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        drop_cols = [
            "temperature_celsius",
            "temperature_fahrenheit",
            "feels_like_celsius",
            "feels_like_fahrenheit",
            "anomaly_flag",
        ]
        feature_cols = [col for col in numeric_cols if col not in drop_cols]

        if len(feature_cols) >= 3:
            model_data = df[feature_cols + ["temperature_celsius"]].dropna()
            X = model_data[feature_cols]
            y = model_data["temperature_celsius"]

            model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
            model.fit(X, y)

            importance = (
                pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
                .sort_values("importance", ascending=False)
                .head(12)
            )

            plt.figure(figsize=(11, 7))
            sns.barplot(data=importance, x="importance", y="feature", color="#4c78a8")
            plt.title("Feature importance for temperature")
            plt.tight_layout()
            plt.savefig(vis_dir / "feature_importance.png", dpi=200)
            plt.close()


if __name__ == "__main__":
    main()
