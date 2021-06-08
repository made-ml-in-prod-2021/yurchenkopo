import os
import pandas as pd
import pathlib


import airflow.utils.dates
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor

from utils import *


def _wait_for_file(path: str):
    return os.path.exists(path)


def _predict(
    test_data_path: str,
    model_path: str,
    transformer_path: str,
    output_dir: str,
):
    output_path = os.path.join(output_dir, 'predictions.csv')
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(test_data_path)
    transformer = load_pickle(transformer_path)
    features = transformer.transform(data)
    model = load_pickle(model_path)
    preds = model.predict(features)
    pd.DataFrame(np.array(preds).T, columns=['target']).to_csv(output_path, index=False)
    print(f'Predict test data and save into {output_path}')


with DAG(
    dag_id='predict',
    start_date=airflow.utils.dates.days_ago(1),
    schedule_interval='@daily',
    max_active_runs=1,
) as dag:
    data_sensor = PythonSensor(
        task_id='data_sensor',
        python_callable=_wait_for_file,
        op_kwargs={'path': '/opt/airflow/data/raw/{{ ds }}/test.csv'},
        timeout=60,
        poke_interval=10,
        retries=100,
        mode='poke',
    )

    model_sensor = PythonSensor(
        task_id='model_sensor',
        python_callable=_wait_for_file,
        op_kwargs={'path': '{{ var.value.model_path }}'}, #Variable.get('model_path')},
        timeout=60,
        poke_interval=10,
        retries=100,
        mode='poke',
    )

    transformer_sensor = PythonSensor(
        task_id='transformer_sensor',
        python_callable=_wait_for_file,
        op_kwargs={'path': '{{ var.value.transformer_path }}'},
        timeout=60,
        poke_interval=10,
        retries=100,
        mode='poke',
    )

    predict = PythonOperator(
        task_id='predict',
        python_callable=_predict,
        op_kwargs={
            'test_data_path': '/opt/airflow/data/raw/{{ ds }}/test.csv',
            'model_path': '{{ var.value.model_path }}',
            'transformer_path': '{{ var.value.transformer_path }}',
            'output_dir': '/opt/airflow/data/predictions/{{ ds }}/',
        }
    )

    [data_sensor, model_sensor, transformer_sensor] >> predict
