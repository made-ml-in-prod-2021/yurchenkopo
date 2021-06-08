import os
import pathlib

import numpy as np
import pandas as pd
import airflow.utils.dates
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from utils import DATASET_SIZE, col_dict


def _generate_data(
    execution_date: str,
    output_dir: str,
    dataset_size: int,
    is_test: bool=False,
):
    pathlib.Path(os.path.join(output_dir, execution_date)).mkdir(parents=True, exist_ok=True)

    if is_test:
        output_path = os.path.join(output_dir, execution_date, 'test.csv')
    else:
        output_path = os.path.join(output_dir, execution_date, 'data.csv')
    data = np.zeros((len(col_dict), dataset_size))
    for i, (col, (min_val, max_val)) in enumerate(col_dict.items()):
        if isinstance(max_val, float):
            data[i, :] = np.random.random(dataset_size) * (max_val - min_val) + min_val
        else:
            data[i, :] = np.random.randint(min_val, max_val + 1, size=dataset_size, dtype='int')

    pd.DataFrame(np.array(data).T, columns=col_dict.keys()).to_csv(output_path, index=False)
    print(f'Data was generated and put into {output_path}')


def _generate_target(
    execution_date: str,
    output_dir: str,
    dataset_size: int,
):
    pathlib.Path(os.path.join(output_dir, execution_date)).mkdir(parents=True, exist_ok=True)

    output_path = os.path.join(output_dir, execution_date, 'target.csv')
    data = np.random.randint(0, 2, size=(dataset_size, 1))
    pd.DataFrame(np.array(data), columns=['target']).to_csv(output_path, index=False)
    print(f'Target was generated and put into {output_path}')


with DAG(
    dag_id='generate_data',
    start_date=airflow.utils.dates.days_ago(1),
    schedule_interval='@daily',
    max_active_runs=1,
) as dag:
    generate_data = PythonOperator(
        task_id='generate_data',
        python_callable=_generate_data,
        op_kwargs={
            'execution_date': '{{ ds }}',
            'output_dir': '/opt/airflow/data/raw/',
            'dataset_size': DATASET_SIZE,
        }
    )

    generate_target = PythonOperator(
        task_id='generate_target',
        python_callable=_generate_target,
        op_kwargs={
            'execution_date': '{{ ds }}',
            'output_dir': '/opt/airflow/data/raw/',
            'dataset_size': DATASET_SIZE,
        }
    )

    generate_test = PythonOperator(
        task_id='generate_test',
        python_callable=_generate_data,
        op_kwargs={
            'execution_date': '{{ ds }}',
            'output_dir': '/opt/airflow/data/raw/',
            'dataset_size': int(DATASET_SIZE / 2),
            'is_test': True,
        }
    )

    endpoint = BashOperator(
        task_id='bash_command',
        bash_command='echo "DAG was successfully finished and all data was saved in /data/raw/{{ ds }}/"',
    )

    [generate_data, generate_target, generate_test] >> endpoint
