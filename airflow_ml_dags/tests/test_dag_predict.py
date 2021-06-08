import sys
import pytest
from airflow.models import DagBag


sys.path.append('dags')


@pytest.fixture()
def dag_bag():
    return DagBag(dag_folder='dags/', include_examples=False)


def test_dag_generate_data(dag_bag):
    assert dag_bag.dags is not None
    assert 'predict' in dag_bag.dags
    dag = dag_bag.dags['predict']

    assert dag.tasks is not None
    assert len(dag.tasks) == 4
    assert 'data_sensor' in dag.task_dict
    assert 'model_sensor' in dag.task_dict
    assert 'transformer_sensor' in dag.task_dict
    assert 'predict' in dag.task_dict


def test_dag_generate_data_structure(dag_bag):
    dag = dag_bag.dags['predict']

    structure = {
        'data_sensor': ['predict'],
        'model_sensor': ['predict'],
        'transformer_sensor': ['predict'],
        'predict': []
    }
    for name, task in dag.task_dict.items():
        downstreams = task.downstream_task_ids
        assert downstreams == set(structure[name])


