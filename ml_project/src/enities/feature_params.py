from dataclasses import dataclass, field
from typing import List, Optional


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    target_col: Optional[str]
    features_to_drop: Optional[str] = field(default=None)
