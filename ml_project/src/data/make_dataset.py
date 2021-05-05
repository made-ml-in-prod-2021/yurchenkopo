# -*- coding: utf-8 -*-
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.enities import SplittingParams


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def split_train_val_data(data: pd.DataFrame, splitting_params: SplittingParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, val_data = train_test_split(
        data, test_size=splitting_params.val_size, random_state=splitting_params.random_state
    )
    return train_data, val_data
