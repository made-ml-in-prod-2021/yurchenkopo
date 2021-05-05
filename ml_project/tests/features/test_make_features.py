from typing import List
import pytest

from src.data.make_dataset import read_data
from src.enities.feature_params import FeatureParams
from src.features.build_features import (
    build_transformer,
    process_categorical_features
)


@pytest.fixture
def feature_params(
    categorical_features: List[str],
    features_to_drop: List[str],
    numerical_features: List[str],
    target_col: str,
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=features_to_drop,
        target_col=target_col,
    )
    return params


def test_process_categorical_features(categorical_features: list, dataset_path: str):
    categorial_data = read_data(dataset_path)[categorical_features]
    all_unique_values_for_one_hot = 0
    for i in range(len(categorical_features)):
        all_unique_values_for_one_hot += len(categorial_data[categorical_features[i]].unique())

    processed_data = process_categorical_features(categorial_data)
    assert processed_data.shape[1] == all_unique_values_for_one_hot


def test_build_transformer(feature_params: FeatureParams, dataset_path: str):
    data = read_data(dataset_path)
    transformer = build_transformer(feature_params).fit(data)
    transformed_data = transformer.transform(data)
    assert transformed_data is not None


def test_target_is_binary(target_col: str, dataset_path: str):
    data = read_data(dataset_path)
    target = data[target_col]
    assert len(target.unique()) == 2