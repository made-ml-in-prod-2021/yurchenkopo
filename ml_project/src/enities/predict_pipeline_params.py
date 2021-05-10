from dataclasses import dataclass
from .feature_params import FeatureParams


@dataclass()
class PredictPipelineParams:
    input_data_path: str
    train_data_path: str
    model_path: str
    predictions_path: str
    feature_params: FeatureParams
