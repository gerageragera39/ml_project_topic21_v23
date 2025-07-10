from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
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
        self.means_dict = means_dict or {}

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
            df[col + '_enc'] = df[col].map(self.means_dict.get(col))

        df.drop(columns=['model'], inplace=True)
        return df

    def count_means(self, X):
        df = X.copy()
        self.means_dict = {}
        for col in ['model']:
            self.means_dict[col] = df.groupby(col)['price'].mean()

if __name__ == '__main__':
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

    categorical_features = ['brand', 'trim', 'body_type', 'fuel_type', 'exterior_color', 'interior_color', 'warranty', 'city', 'seller_type']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_mean', mean_transformer, mean_features),
            ('num_median', median_transformer, median_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    models = {
        "ExtraTrees": ExtraTreesRegressor(random_state=42),
        # "RandomForest": RandomForestRegressor(random_state=42),
        # "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
    }

    param_grids = {
        "ExtraTrees": {
            # 'pca__n_components': [5, 10, 15, 20, 22],
            # 'model__n_estimators': [100, 300, 500, 800],
            # 'model__max_depth': [None, 10, 20, 30, 50],
            # 'model__min_samples_split': [2, 5, 10],
            # 'model__min_samples_leaf': [1, 2, 4],
            # 'model__max_features': ['sqrt', 'log2', None]
        },
        # "RandomForest": {
        #     'pca__n_components': [5, 10, 15, 20, 22],
        #     'model__n_estimators': [100, 300, 500, 800],
        #     'model__max_depth': [None, 10, 20, 30, 50],
        #     'model__min_samples_split': [2, 5, 10],
        #     'model__min_samples_leaf': [1, 2, 4],
        #     'model__max_features': ['sqrt', 'log2', None]
        # },
        # "CatBoost": {
        #     'pca__n_components': [5, 10, 15, 20, 22],
        #     'model__iterations': [500, 1000],
        #     'model__depth': [4, 6, 8, 10],
        #     'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        #     'model__l2_leaf_reg': [1, 3, 5, 7, 9],
        #     'model__bagging_temperature': [0, 0.5, 1]
        # }
    }

    loaded_data = pd.read_csv("../data/topic21_v23_train.csv")

    X_train = loaded_data.drop(columns=['price'])
    y_train = loaded_data['price']

    # engineer = FeatureEngineer()
    # engineer.fit(loaded_data)
    #
    # X_train_transformed = engineer.transform(X_train)
    # X_train_transformed = preprocessor.fit_transform(X_train_transformed)


    # pplines = {}
    # for name, model in models.items():
    #     print(f"\n GridSearch for {name}...")
    #
    #     pipeline = Pipeline([
    #         # ('pca', PCA()),
    #         ('model', model)
    #     ])
    #
    #     grid_search = GridSearchCV(
    #         estimator=pipeline,
    #         param_grid=param_grids[name],
    #         cv=5,
    #         scoring='neg_mean_squared_error',
    #         verbose=2,
    #         n_jobs=-1
    #     )
    #
    #     grid_search.fit(X_train_transformed, y_train)
    #
    #     print(f" Best params for {name}:")
    #     print(grid_search.best_params_)
    #     print(f" Best CV score (neg MSE): {-grid_search.best_score_:.4f}")
    #
    #     best_pipeline = grid_search.best_estimator_
    #     pplines[name] = best_pipeline


    train_set, valid_set = train_test_split(loaded_data, test_size=0.2, random_state=42)

    engineer = FeatureEngineer()
    engineer.count_means(train_set)

    X_train_transformed = engineer.transform(X_train)
    X_train_transformed = preprocessor.fit_transform(X_train_transformed)

    pplines = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_transformed, y_train)
        pplines[name] = model


    for name, best_pipeline in pplines.items():
        full_pipeline = Pipeline([
            ('feature_engineer', engineer),
            ('preprocessor', preprocessor),
            ('model', best_pipeline)
        ])

        full_pipeline.fit(X_train, y_train)
        dump(full_pipeline, f"full_pipeline_{name}.skops")


