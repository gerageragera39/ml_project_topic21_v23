import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from CreateSkops import FeatureEngineer

from skops.io import load, get_untrusted_types

if __name__ == '__main__':
    loaded_data = pd.read_csv("../data/topic21_v23_train.csv")

    train_set, valid_set = train_test_split(loaded_data, test_size=0.2, random_state=1)

    X_valid = valid_set.drop(columns=['price'])
    y_valid = valid_set['price']

    pipeline_path = "full_pipeline_ExtraTrees.skops"
    # pipeline_path = "full_pipeline_CatBoost.skops"
    # pipeline_path = "full_pipeline_RandomForest.skops"

    trusted_types = get_untrusted_types(file=pipeline_path)

    loaded_pipeline = load(
        pipeline_path, trusted=trusted_types
    )

    y_pred = loaded_pipeline.predict(X_valid)

    mse = mean_squared_error(y_valid, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)

    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

    random_indices = np.random.choice(len(y_valid), 5, replace=False)
    print("\nRandom predictions vs actual values:")
    for idx in random_indices:
        print(f"Predicted: {y_pred[idx]:.2f}, Actual: {y_valid.iloc[idx]:.2f}")
