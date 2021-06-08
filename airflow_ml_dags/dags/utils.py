import pickle

import numpy as np
from typing import Dict, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

from sklearn.model_selection import train_test_split


DATASET_SIZE = 1000
col_dict = {
        'age': (29, 77),
        'sex': (0, 1),
        'cp': (0, 3),
        'trestbps': (94, 200),
        'chol': (126, 564),
        'fbs': (0, 1),
        'restecg': (0, 2),
        'thalach': (71, 202),
        'exang': (0, 1),
        'oldpeak': (0.0, 6.5),
        'slope': (0, 2),
        'ca': (0, 4),
        'thal': (0, 3),
        'target': (0, 1)
    }

CAT_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
NUM_FEATURES = ['age', 'chol', 'ca', 'trestbps', 'thalach', 'oldpeak']


def split_train_val(data: pd.DataFrame, val_size: float=0.2, random_state: int=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, val_data = train_test_split(
        data, test_size=val_size, random_state=random_state
    )
    return train_data, val_data


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ('impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
            ('ohe', OneHotEncoder()),
        ]
    )
    return categorical_pipeline

def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),
            ('scale', MinMaxScaler())
        ]
    )
    return num_pipeline

def build_transformer(categorical_features: list, numerical_features: list) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                'categorical_pipeline',
                build_categorical_pipeline(),
                categorical_features,
            ),
            (
                'numerical_pipeline',
                build_numerical_pipeline(),
                numerical_features,
            ),
        ]
    )
    return transformer


def evaluate_model(model, features, target) -> Dict[str, float]:
    predictions = model.predict(features)
    pred_probas = model.predict_proba(features)[:, 1]
    fpr, tpr, thresholds = roc_curve(target, pred_probas)
    return {
        'roc_auc': auc(fpr, tpr),
        'accuracy': accuracy_score(target, predictions),
        'f1_score': f1_score(target, predictions)
        }


def dump_pickle(file_path, model):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def load_pickle(file_path):
    with open(file_path, 'rb') as fin:
        transformer = pickle.load(fin)
    return transformer