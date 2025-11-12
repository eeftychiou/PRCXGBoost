import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import augment_features

class DataFrameTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X

class FeatureAugmenter(DataFrameTransformer):
    def __init__(self, trajectories_folder):
        self.trajectories_folder = trajectories_folder

    def transform(self, X, y=None):
        return augment_features.augment_features(X, self.trajectories_folder)

class ColumnDropper(DataFrameTransformer):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.to_drop = []

    def fit(self, X, y=None):
        missing_ratio = X.isnull().sum() / len(X)
        self.to_drop = missing_ratio[missing_ratio > self.threshold].index.tolist()
        print(f"Dropping {len(self.to_drop)} columns with >{self.threshold:.0%} missing values.")
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.to_drop)

class DataFrameImputer(DataFrameTransformer):
    def __init__(self, strategy='median'):
        self.strategy = strategy
        self.imputer = None
        self.numeric_cols = []

    def fit(self, X, y=None):
        self.numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        if self.numeric_cols:
            self.imputer = SimpleImputer(strategy=self.strategy)
            self.imputer.fit(X[self.numeric_cols])
        return self

    def transform(self, X, y=None):
        if not self.numeric_cols or self.imputer is None:
            return X
        X_transformed = X.copy()
        X_transformed[self.numeric_cols] = self.imputer.transform(X_transformed[self.numeric_cols])
        return X_transformed

class DataFrameScaler(DataFrameTransformer):
    def __init__(self):
        self.scaler = None
        self.numeric_cols = []

    def fit(self, X, y=None):
        self.numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        if self.numeric_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(X[self.numeric_cols])
        return self

    def transform(self, X, y=None):
        if not self.numeric_cols or self.scaler is None:
            return X
        X_transformed = X.copy()
        X_transformed[self.numeric_cols] = self.scaler.transform(X_transformed[self.numeric_cols])
        return X_transformed

class CategoricalEncoder(DataFrameTransformer):
    def __init__(self):
        self.encoders = {}
        self.categorical_cols = []

    def fit(self, X, y=None):
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in self.categorical_cols:
            le = LabelEncoder()
            unique_vals = list(X[col].astype(str).unique())
            le.fit(unique_vals + ['unseen'])
            self.encoders[col] = le
        return self

    def transform(self, X, y=None):
        if not self.categorical_cols:
            return X
        X_transformed = X.copy()
        for col in self.categorical_cols:
            encoder = self.encoders[col]
            known_classes = set(encoder.classes_)
            X_transformed[col] = X[col].astype(str).apply(lambda x: x if x in known_classes else 'unseen')
            X_transformed[col] = encoder.transform(X_transformed[col])
        return X_transformed

def get_data_processing_pipeline(trajectories_folder):
    """
    Creates and returns a scikit-learn pipeline for all data processing tasks.
    """
    processing_pipeline = Pipeline([
        ('augment_features', FeatureAugmenter(trajectories_folder)),
        ('drop_high_missing', ColumnDropper(threshold=0.9)),
        ('datetime_converter', DatetimeConverter()),
        ('categorical_encoder', CategoricalEncoder()),
        ('imputer', DataFrameImputer(strategy='median')),
        ('scaler', DataFrameScaler())
    ])
    return processing_pipeline

class DatetimeConverter(DataFrameTransformer):
    """Converts datetime columns to numeric (Unix timestamps)."""
    def transform(self, X, y=None):
        X_transformed = X.copy()
        for col in X_transformed.select_dtypes(include=['datetime64[ns]', 'datetime64[us]']).columns:
            X_transformed[col] = pd.to_datetime(X_transformed[col]).astype(np.int64) // 10**9
        return X_transformed
