import logging

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from skops.io import dump


def parse_mid(r):
    if pd.isna(r) or r == 'Unknown': return np.nan
    s = ''.join(c for c in r if c.isdigit() or c == '-')
    if '-' in s:
        lo, hi = map(int, s.split('-'))
        return (lo + hi) / 2
    return float(s)


def add_outlier_column(df, threshold=2.5):
    df_copy = df.copy()

    if 'price' not in df.columns:
        numeric_cols = df.select_dtypes(include=['float64', 'int64'])
    else:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).drop('price', axis=1).columns

    df_copy['is_outlier'] = 0
    for col in numeric_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df_copy['is_outlier'] = (df_copy['is_outlier'] | (z_scores > threshold)).astype(int)

    return df_copy


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, means_dict=None):
        self.means_dict = means_dict if means_dict is not None else {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        for col in ['engine_capacity_cc', 'horsepower']:
            df[col + '_num'] = df[col].apply(parse_mid)
            df[col + '_miss'] = df[col + '_num'].isna().astype(int)
            df.drop(columns=[col], inplace=True)

        df['is_automatic'] = (df['transmission_type'] == 'Automatic Transmission').astype(int)
        df = df.drop('transmission_type', axis=1)

        df['hp_per_cc'] = df['horsepower_num'] / df['engine_capacity_cc_num']
        df = add_outlier_column(df, 2)

        for col in ['model']:
            df[col + '_enc'] = df[col].map(self.means_dict.get(col, {}))

        df.drop(columns=['model'], inplace=True)
        return df


if __name__ == '__main__':
    logging.basicConfig(filename='grid_model_cv_logs.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    mean_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    mean_features = ['1', '2', '3', '4', 'engine_capacity_cc_num', 'engine_capacity_cc_miss', 'horsepower_num',
                     'horsepower_miss', 'is_automatic', 'hp_per_cc', 'is_outlier', 'model_enc']

    median_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    median_features = ['0']

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    categorical_features = ['brand', 'trim', 'body_type', 'fuel_type', 'exterior_color', 'interior_color', 'warranty',
                            'city', 'seller_type']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_mean', mean_transformer, mean_features),
            ('num_median', median_transformer, median_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    models = {
    "ExtraTrees": (ExtraTreesRegressor(random_state=42), {
        'model__n_estimators': [100, 300, 500, 800],
        'model__max_depth': [None, 10, 20, 30, 50],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 'log2', None]
    }),
    "RandomForest": (RandomForestRegressor(random_state=42), {
        'model__n_estimators': [100, 300, 500, 800],
        'model__max_depth': [None, 10, 20, 30, 50],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 'log2', None]
    }),
    "CatBoost": (CatBoostRegressor(verbose=0, random_state=42), {
        'model__iterations': [500, 1000],
        'model__depth': [4, 6, 8, 10],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__l2_leaf_reg': [1, 3, 5, 7, 9],
        'model__bagging_temperature': [0, 0.5, 1]
    })
}

    loaded_data = pd.read_csv("../data/topic21_v23_train.csv")

    train_set = loaded_data.copy()

    X_train = train_set.drop(columns=['price'])
    y_train = train_set['price']

    MEANS = {}
    for col in ['model']:
        MEANS[col] = train_set.groupby(col)['price'].mean().to_dict()

    engineer = FeatureEngineer(MEANS)

    for name, (model, param_grid) in models.items():
        pipeline = Pipeline([
            ('feature_engineer', engineer),
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_

        logging.info(f"{name} | Best CV R2: {best_score:.4f} | Best Params: {grid_search.best_params_}")

        dump(best_model, f"grid_full_pipeline_{name}.skops")