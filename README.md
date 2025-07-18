# Machine Learning Model Comparison Project

## Overview

This project evaluates and compares various machine learning models for regression tasks. The models are assessed using multiple performance metrics including:

- **MSE** (Mean Squared Error)  
- **RMSE** (Root Mean Squared Error)  
- **MAE** (Mean Absolute Error)  
- **R²** (Coefficient of Determination)

## Model Performance Results

| Model            | MSE          | RMSE        | MAE        | R²        |
|------------------|--------------|-------------|------------|-----------|
| ExtraTrees       | 1.245245e+10 | 111590.54    | 49160.68   | 0.644782  |
| CatBoost         | 1.407062e+10 | 118619.64    | 57013.59   | 0.598622  |
| RandomForest     | 1.472810e+10 | 121359.37    | 55831.86   | 0.579867  |
| LightGBM         | 1.495653e+10 | 122296.89    | 59383.38   | 0.573351  |
| GradientBoosting | 1.529703e+10 | 123681.17    | 64424.09   | 0.563638  |
| KNeighbors       | 1.694824e+10 | 130185.42    | 63694.41   | 0.516535  |
| Ridge            | 1.782756e+10 | 133519.88    | 73271.85   | 0.491452  |
| Lasso            | 1.841482e+10 | 135701.20    | 73994.24   | 0.474700  |
| LinearRegression | 1.845132e+10 | 135835.63    | 74112.85   | 0.473658  |
| MLP              | 1.867224e+10 | 136646.39    | 79175.98   | 0.467357  |
| AdaBoost         | 1.902024e+10 | 137913.88    | 85003.19   | 0.457430  |
| ElasticNet       | 1.915531e+10 | 138402.71    | 80171.57   | 0.453576  |
| XGBoost          | 2.142522e+10 | 146373.55    | 68134.45   | 0.388825  |
| DecisionTree     | 2.421364e+10 | 155607.32    | 68262.31   | 0.309283  |
| SVR              | 3.999321e+10 | 199983.02    | 117891.60  | -0.140845 |


## Key Findings

- **ExtraTrees** achieved the best overall performance with the lowest MSE, RMSE, MAE, and highest R² (0.83).
- **CatBoost** and **LightGBM** also demonstrated strong predictive power.
- **SVR** significantly underperformed, with a negative R², indicating a poor fit to the data.
