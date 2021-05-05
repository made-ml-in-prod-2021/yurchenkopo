import json
import logging

import os
from omegaconf import DictConfig
import hydra

from src.data import read_data, split_train_val_data
from src.features import (
    drop_features,
    build_transformer,
)
from src.models import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
)
from marshmallow_dataclass import class_schema
from src.enities.train_pipeline_params import TrainingPipelineParams


CUR_DIR = os.getcwd()
TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_pipeline(training_pipeline_params: DictConfig):
    logger.info(f'start train pipeline with {training_pipeline_params}\n')
    logger.info(f'read data from {os.path.join(CUR_DIR, training_pipeline_params.input_data_path)}')
    df = read_data(os.path.join(CUR_DIR, training_pipeline_params.input_data_path))


    df = drop_features(df, training_pipeline_params.feature_params.features_to_drop)
    train_df, val_df = split_train_val_data(df, training_pipeline_params.splitting_params)

    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)

    target_col = training_pipeline_params.feature_params.target_col
    y_train = train_df[target_col]
    y_val = val_df[target_col]
    train_features = transformer.transform(train_df)
    val_features = transformer.transform(val_df)
    logger.info(f'train_features shape = {train_features.shape}, val_features shape = {val_features.shape}')


    model = train_model(train_features, y_train, training_pipeline_params.train_params, logger)


    logger.info(f'save model to {os.path.join(CUR_DIR, training_pipeline_params.output_model_path)}')
    path_to_model = serialize_model(model, os.path.join(CUR_DIR, training_pipeline_params.output_model_path))


    predictions, pred_probas = predict_model(model, val_features)
    metrics = evaluate_model(predictions, pred_probas, y_val)
    with open(os.path.join(CUR_DIR, training_pipeline_params.metric_path), 'w') as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f'metrics: { {k:round(v, 3) for k, v in metrics.items()} }')
    logger.info('training successfully finished')

    return path_to_model, metrics


@hydra.main(config_path='../configs', config_name='train_config')
def train_pipeline_command(cfg: DictConfig):
    param_schema = TrainingPipelineParamsSchema()
    params = param_schema.load(cfg)
    train_pipeline(params)


if __name__ == '__main__':
    train_pipeline_command()
