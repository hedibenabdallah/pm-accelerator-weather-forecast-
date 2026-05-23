from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parent
DATASET_CANDIDATES = [
    WORKSPACE / "Global Weather Repository.csv",
    WORKSPACE / "global_weather_repository.csv",
    WORKSPACE / "GlobalWeatherRepository.csv",
]
OUTPUT_PATH = WORKSPACE / "processed_weather.csv"


def find_dataset_path():
    for path in DATASET_CANDIDATES:
        if path.exists():
            return path

    matches = list(WORKSPACE.rglob("*Weather*.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError("Global Weather Repository.csv was not found in the workspace.")


def parse_datetime_column(df):
    for column in ["last_updated", "lastupdated"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
            return column
    return None


def fill_missing_values(df):
    df_clean = df.copy()

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in df_clean.columns if col not in numeric_cols]

    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in categorical_cols:
        mode = df_clean[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "unknown"
        df_clean[col] = df_clean[col].fillna(fill_value)

    return df_clean


def clip_outliers(df):
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {
        "latitude",
        "longitude",
        "last_updated_epoch",
        "temperature_fahrenheit",
        "feels_like_fahrenheit",
        "wind_mph",
        "pressure_in",
        "precip_in",
        "visibility_miles",
        "gust_mph",
    }

    for col in numeric_cols:
        if col in excluded:
            continue
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)

    return df_clean


def main():
    dataset_path = find_dataset_path()
    df = pd.read_csv(dataset_path)

    datetime_col = parse_datetime_column(df)
    df_clean = fill_missing_values(df)
    df_clean = clip_outliers(df_clean)

    if datetime_col:
        df_clean[datetime_col] = df_clean[datetime_col].dt.strftime("%Y-%m-%d %H:%M:%S")

    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
