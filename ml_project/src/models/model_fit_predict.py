import pickle
from typing import Dict, Union, Tuple

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

from src.enities.train_params import TrainingParams

SklearnClassifierModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(features: np.ndarray, target: pd.Series,
                training_params: TrainingParams, logger: logging.Logger=None) -> SklearnClassifierModel:
    if logger:
        if training_params.params:
            param_str = ', '.join([str(k) + '=' + str(v) for k, v in training_params.params.items()])
            logger.info(f'create model {training_params.model_type}({param_str})')
        else:
            logger.info(f'create model {training_params.model_type}()')

    if training_params.model_type == 'RandomForestClassifier':
        if training_params.params:
            model = RandomForestClassifier(**training_params.params, random_state=training_params.random_state)
        else:
            model = RandomForestClassifier(random_state=training_params.random_state)
    elif training_params.model_type == 'LogisticRegression':
        if training_params.params:
            model = LogisticRegression(**training_params.params, random_state=training_params.random_state)
        else:
            model = LogisticRegression(random_state=training_params.random_state)
    else:
        raise NotImplementedError()
    if logger:
        logger.info('training model')
    model.fit(features, target)
    return model


def predict_model(model: SklearnClassifierModel, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    predicts = model.predict(features)
    pred_probas = model.predict_proba(features)[:, 1]
    return predicts, pred_probas


def evaluate_model(predictions: np.ndarray, pred_probas: np.ndarray, target: pd.Series) -> Dict[str, float]:
    fpr, tpr, thresholds = roc_curve(target, pred_probas)
    return {
        'roc_auc': auc(fpr, tpr),
        'accuracy': accuracy_score(target, predictions),
        'f1_score': f1_score(target, predictions)
        }


def serialize_model(model: SklearnClassifierModel, output: str) -> str:
    with open(output, 'wb') as f:
        pickle.dump(model, f)
    return output


def serialize_transformer(transformer: ColumnTransformer, output: str) -> str:
    with open(output, 'wb') as f:
        pickle.dump(transformer, f)
    return output