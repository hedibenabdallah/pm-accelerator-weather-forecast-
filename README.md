# PM Accelerator Weather Assessment

Dataset: `GlobalWeatherRepository.csv`

## Project Summary

This repository contains a weather trend assessment built around three steps:
- preprocessing
- exploratory analysis
- forecasting

The workflow produces a cleaned dataset, six PNG visualizations, and evaluation metrics for a location-level forecasting model.

## Setup

Required files:
- `GlobalWeatherRepository.csv`
- `requirements.txt`

```bash
pip install -r requirements.txt
```

```bash
python 01_data_preprocessing.py
python 02_eda_and_analysis.py
python 03_forecasting_models.py
```

## Methodology

- `01_data_preprocessing.py`
  - parses `last_updated`
  - fills numeric nulls with the median
  - fills categorical nulls with the mode, fallback `unknown`
  - clips outliers on selected numeric columns with an IQR rule
  - writes `processed_weather.csv`

- `02_eda_and_analysis.py`
  - builds a correlation heatmap
  - runs anomaly detection with Isolation Forest
  - creates a spatial weather plot
  - creates an air quality vs weather plot
  - estimates feature importance for `temperature_celsius`

- `03_forecasting_models.py`
  - builds a daily temperature series from `last_updated`
  - selects one location for forecasting
  - creates lag and rolling features
  - compares a previous-day baseline against a Random Forest regressor
  - writes `metrics.json`

## Outputs

- `processed_weather.csv`
- `metrics.json`
- `visualizations/correlation_heatmap.png`
- `visualizations/anomaly_detection.png`
- `visualizations/spatial_weather_pattern.png`
- `visualizations/air_quality_vs_weather.png`
- `visualizations/feature_importance.png`
- `visualizations/forecast_actual_vs_predicted.png`

## Results

- Forecast location: `Accra`
- Baseline RMSE: `1.6606`
- Baseline MAE: `1.1031`
- Random Forest RMSE: `1.3450`
- Random Forest MAE: `0.9755`

The Random Forest outperformed the baseline on both metrics for the current run.
