import pytest
from typing import List
import pandas as pd
import numpy as np

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

@pytest.fixture()
def dataset_size():
    return 1000

@pytest.fixture()
def dataset_path(tmpdir, dataset_size):
    dataset = tmpdir.join('generated_train_data.csv')
    data = np.zeros((len(col_dict), dataset_size))
    for i, (col, (min_val, max_val)) in enumerate(col_dict.items()):
        if isinstance(max_val, float):
            data[i, :] = np.random.random(dataset_size) * (max_val - min_val) + min_val
        else:
            data[i, :] = np.random.randint(min_val, max_val + 1, size=dataset_size, dtype='int')

    pd.DataFrame(np.array(data).T, columns=col_dict.keys()).to_csv(dataset)
    return dataset


@pytest.fixture()
def target_col():
    return 'target'


@pytest.fixture()
def categorical_features() -> List[str]:
    return ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']


@pytest.fixture
def numerical_features() -> List[str]:
    return ['age', 'chol', 'ca', 'trestbps', 'thalach', 'oldpeak']


@pytest.fixture()
def features_to_drop() -> List[str]:
    return None
