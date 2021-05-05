import os
import pickle
from typing import List, Tuple

import pandas as pd
import pytest
from py._path.local import LocalPath
from sklearn.ensemble import RandomForestClassifier

from src.data.make_dataset import read_data
from src.enities import TrainingParams
from src.enities.feature_params import FeatureParams
from src.features.build_features import build_transformer
from src.models.model_fit_predict import train_model, serialize_model


@pytest.fixture
def features_and_target(
    dataset_path: str, categorical_features: List[str], numerical_features: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=None,
        target_col='target',
    )
    data = read_data(dataset_path)
    transformer = build_transformer(params)
    transformer.fit(data)
    features = transformer.transform(data)
    target = data[params.target_col]
    return features, target


def test_train_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    model = train_model(features, target, training_params=TrainingParams())
    assert isinstance(model, RandomForestClassifier)
    assert model.predict(features).shape[0] == target.shape[0]


def test_serialize_model(tmpdir: LocalPath):
    expected_output = tmpdir.join('model.pkl')
    model = RandomForestClassifier()
    real_output = serialize_model(model, expected_output)
    assert real_output == expected_output
    assert os.path.exists
    with open(real_output, 'rb') as f:
        model = pickle.load(f)
    assert isinstance(model, RandomForestClassifier)
