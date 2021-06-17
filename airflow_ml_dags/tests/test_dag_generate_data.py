import sys
import pytest
from airflow.models import DagBag


sys.path.append('dags')


@pytest.fixture()
def dag_bag():
    return DagBag(dag_folder='dags/', include_examples=False)


def test_dag_generate_data(dag_bag):
    assert dag_bag.dags is not None
    assert 'generate_data' in dag_bag.dags
    dag = dag_bag.dags['generate_data']

    assert dag.tasks is not None
    assert len(dag.tasks) == 4
    assert 'generate_data' in dag.task_dict
    assert 'generate_target' in dag.task_dict
    assert 'generate_test' in dag.task_dict
    assert 'bash_command' in dag.task_dict


def test_dag_generate_data_structure(dag_bag):
    dag = dag_bag.dags['generate_data']

    structure = {
        'generate_data': ['bash_command'],
        'generate_target': ['bash_command'],
        'generate_test': ['bash_command'],
        'bash_command': []
    }
    for name, task in dag.task_dict.items():
        downstreams = task.downstream_task_ids
        assert downstreams == set(structure[name])


