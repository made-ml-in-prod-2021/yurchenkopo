import json
import logging

import os
import pickle

from omegaconf import DictConfig
import hydra

from src.data import read_data
from src.features import (
    drop_features,
    build_transformer,
)
from src.models import predict_model
from marshmallow_dataclass import class_schema
from src.enities.predict_pipeline_params import PredictPipelineParams


CUR_DIR = os.getcwd()
PredictPipelineParamsSchema = class_schema(PredictPipelineParams)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_pipeline(predict_pipeline_params: DictConfig):
    logger.info(f'start predict pipeline with {predict_pipeline_params}\n')
    logger.info(f'read data from {os.path.join(CUR_DIR, predict_pipeline_params.input_data_path)}')
    df = read_data(os.path.join(CUR_DIR, predict_pipeline_params.input_data_path))
    train_df = read_data(os.path.join(CUR_DIR, predict_pipeline_params.train_data_path))

    df = drop_features(df, predict_pipeline_params.feature_params.features_to_drop)
    train_df = drop_features(train_df, predict_pipeline_params.feature_params.features_to_drop)

    transformer = build_transformer(predict_pipeline_params.feature_params)
    transformer.fit(train_df.drop(predict_pipeline_params.feature_params.target_col, axis=1))
    test_features = transformer.transform(df)

    logger.info(f'load model from {os.path.join(CUR_DIR, predict_pipeline_params.model_path)}')
    with open(os.path.join(CUR_DIR, predict_pipeline_params.model_path), 'rb') as fin:
        model = pickle.load(fin)

    predictions, _ = predict_model(model, test_features)

    with open(os.path.join(CUR_DIR, predict_pipeline_params.predictions_path), 'w') as prediction_file:
        json.dump(predictions.tolist(), prediction_file)

    logger.info(f'save predictions to: {os.path.join(CUR_DIR, predict_pipeline_params.predictions_path)}')
    logger.info('prediction successfully finished')

    return predictions


@hydra.main(config_path='../configs', config_name='predict_config')
def predict_pipeline_command(cfg: DictConfig):
    param_schema = PredictPipelineParamsSchema()
    params = param_schema.load(cfg)
    train_pipeline(params)


if __name__ == '__main__':
    predict_pipeline_command()
