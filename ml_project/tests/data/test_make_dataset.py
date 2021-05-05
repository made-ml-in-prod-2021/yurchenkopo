from src.data.make_dataset import read_data, split_train_val_data
from src.enities import SplittingParams


def test_load_dataset(dataset_path: str, target_col: str, dataset_size: int):
    data = read_data(dataset_path)
    assert len(data) == dataset_size
    assert target_col in data.columns


def test_split_dataset(dataset_path: str):
    data = read_data(dataset_path)
    val_size=0.2
    train_data, val_data = split_train_val_data(
        data,
        SplittingParams(val_size=val_size, random_state=42)
    )
    assert round(len(val_data) / len(data), 1) == 0.2
