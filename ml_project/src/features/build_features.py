import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer

from src.enities.feature_params import FeatureParams


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:
    categorical_pipeline = build_categorical_pipeline()
    return pd.DataFrame(categorical_pipeline.fit_transform(categorical_df).toarray())


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ('impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
            ('int_to_obj', FunctionTransformer(lambda x: x.astype(object), validate=True)),
            ('ohe', OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df))


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),
            ('scale', MinMaxScaler())
        ]
    )
    return num_pipeline


def build_transformer(feature_params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                'categorical_pipeline',
                build_categorical_pipeline(),
                feature_params.categorical_features,
            ),
            (
                'numerical_pipeline',
                build_numerical_pipeline(),
                feature_params.numerical_features,
            ),
        ]
    )
    return transformer


def drop_features(df: pd.DataFrame, features_to_drop) -> pd.DataFrame:
    if features_to_drop is not None:
        df = df.drop(features_to_drop, axis=1)
    return df
