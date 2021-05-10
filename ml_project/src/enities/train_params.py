from dataclasses import dataclass, field
from typing import Dict


@dataclass()
class TrainingParams:
    model_type: str = field(default="RandomForestClassifier")
    random_state: int = field(default=255)
    params: Dict[str, int] = field(default=None)
