import json
import logging

import os

from omegaconf import DictConfig
import hydra

from src.data import read_data
from src.features import drop_features
from src.models import predict_model, load_object
from marshmallow_dataclass import class_schema
from src.enities.predict_pipeline_params import PredictPipelineParams


CUR_DIR = os.getcwd()
PredictPipelineParamsSchema = class_schema(PredictPipelineParams)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def predict_pipeline(predict_pipeline_params: DictConfig):
    logger.info(f'start predict pipeline with {predict_pipeline_params}\n')
    logger.info(f'read data from {os.path.join(CUR_DIR, predict_pipeline_params.input_data_path)}')
    df = read_data(os.path.join(CUR_DIR, predict_pipeline_params.input_data_path))

    df = drop_features(df, predict_pipeline_params.feature_params.features_to_drop)

    logger.info(f'load model from {os.path.join(CUR_DIR, predict_pipeline_params.model_path)}')
    model = load_object(os.path.join(CUR_DIR, predict_pipeline_params.model_path))
    logger.info(f'load transformer from {os.path.join(CUR_DIR, predict_pipeline_params.transformer_path)}')
    transformer = load_object(os.path.join(CUR_DIR, predict_pipeline_params.transformer_path))

    test_features = transformer.transform(df)
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
    predict_pipeline(params)


if __name__ == '__main__':
    predict_pipeline_command()
