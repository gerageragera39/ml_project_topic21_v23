# Machine Learning Model Comparison Project

## Overview

This project evaluates and compares various machine learning models for regression tasks. The models are assessed using multiple performance metrics including:

- **MSE** (Mean Squared Error)  
- **RMSE** (Root Mean Squared Error)  
- **MAE** (Mean Absolute Error)  
- **R²** (Coefficient of Determination)

## Model Performance Results

| Model            | MSE          | RMSE       | MAE       | R²        |
|------------------|--------------|------------|-----------|-----------|
| ExtraTrees       | 2.300680e+09 | 47965.41   | 27334.09  | 0.832969  |
| CatBoost         | 2.332535e+09 | 48296.32   | 28911.44  | 0.830657  |
| LightGBM         | 2.544481e+09 | 50442.85   | 29909.26  | 0.815269  |
| RandomForest     | 2.643218e+09 | 51412.24   | 29510.05  | 0.808101  |
| XGBoost          | 3.004045e+09 | 54809.17   | 32024.21  | 0.781905  |
| GradientBoosting | 3.065971e+09 | 55371.21   | 33496.93  | 0.777409  |
| KNeighbors       | 3.073613e+09 | 55440.17   | 33711.94  | 0.776854  |
| Ridge            | 4.233555e+09 | 65065.78   | 44321.86  | 0.692641  |
| Lasso            | 4.240731e+09 | 65120.90   | 44350.28  | 0.692120  |
| LinearRegression | 4.241846e+09 | 65129.45   | 44358.32  | 0.692040  |
| MLP              | 4.483962e+09 | 66962.39   | 45467.13  | 0.674462  |
| ElasticNet       | 4.663008e+09 | 68286.22   | 45774.96  | 0.661463  |
| AdaBoost         | 5.478387e+09 | 74016.13   | 51134.71  | 0.602266  |
| DecisionTree     | 6.136739e+09 | 78337.34   | 40752.27  | 0.554469  |
| SVR              | 1.557468e+10 | 124798.54  | 84310.21  | -0.130731 |

## Key Findings

- **ExtraTrees** achieved the best overall performance with the lowest MSE, RMSE, MAE, and highest R² (0.83).
- **CatBoost** and **LightGBM** also demonstrated strong predictive power.
- **SVR** significantly underperformed, with a negative R², indicating a poor fit to the data.
