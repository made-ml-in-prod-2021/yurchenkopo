import sys
import pytest
from airflow.models import DagBag


sys.path.append('dags')


@pytest.fixture()
def dag_bag():
    return DagBag(dag_folder='dags/', include_examples=False)


def test_dag_generate_data(dag_bag):
    assert dag_bag.dags is not None
    assert 'train_validate' in dag_bag.dags
    dag = dag_bag.dags['train_validate']

    assert dag.tasks is not None
    assert len(dag.tasks) == 8
    assert 'data_sensor' in dag.task_dict
    assert 'target_sensor' in dag.task_dict
    assert 'train_val_split' in dag.task_dict
    assert 'data_preprocessing' in dag.task_dict
    assert 'linear_reg_training' in dag.task_dict
    assert 'random_forest_training' in dag.task_dict
    assert 'validate_and_choose_best_model' in dag.task_dict
    assert 'remove_redundant_files' in dag.task_dict


def test_dag_generate_data_structure(dag_bag):
    dag = dag_bag.dags['train_validate']

    structure = {
        'data_sensor': ['train_val_split'],
        'target_sensor': ['train_val_split'],
        'train_val_split': ['data_preprocessing'],
        'data_preprocessing': ['linear_reg_training', 'random_forest_training'],
        'linear_reg_training': ['validate_and_choose_best_model'],
        'random_forest_training': ['validate_and_choose_best_model'],
        'validate_and_choose_best_model': ['remove_redundant_files'],
        'remove_redundant_files': []
    }
    for name, task in dag.task_dict.items():
        downstreams = task.downstream_task_ids
        assert downstreams == set(structure[name])


