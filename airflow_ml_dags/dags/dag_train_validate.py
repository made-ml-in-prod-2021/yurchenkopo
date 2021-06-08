import os
import pandas as pd
import pathlib
import json
from textwrap import dedent

import airflow.utils.dates
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from utils import *


def _wait_for_file(path: str):
    return os.path.exists(path)


def _train_val_split(
    execution_date: str,
    data_dir: str,
):
    output_dir = os.path.join(data_dir, 'processed', execution_date)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(os.path.join(data_dir, 'raw', execution_date, 'data.csv'))
    target = pd.read_csv(os.path.join(data_dir, 'raw', execution_date, 'target.csv'))
    train, val = split_train_val(data)
    train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val.to_csv(os.path.join(output_dir, 'val.csv'), index=False)

    train_target, val_target = split_train_val(target)
    train_target.to_csv(os.path.join(output_dir, 'train_target.csv'), index=False)
    val_target.to_csv(os.path.join(output_dir, 'val_target.csv'), index=False)
    print(f'Split data and target and put into /data/processed/{execution_date}/')


def _data_preprocessing(
    execution_date: str,
    processed_train_path: str,
    output_transformer_dir: str,
):
    output_dir = os.path.join(output_transformer_dir, f'{execution_date}')
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(processed_train_path)
    transformer = build_transformer(CAT_FEATURES, NUM_FEATURES)
    transformer.fit(train)
    dump_pickle(os.path.join(output_dir, 'transformer.pkl'), transformer)
    print(f'Fit transformer and put into ./data/models/{execution_date}/transformer.pkl')


def _linear_reg_training(
    input_data_dir: str,
    input_target_dir: str,
    input_transformer_path: str,
    output_model_path: str,
):
    train = pd.read_csv(input_data_dir)
    target = pd.read_csv(input_target_dir)
    transformer = load_pickle(input_transformer_path)
    features = transformer.transform(train)
    model = LogisticRegression(random_state=111).fit(features, target)
    dump_pickle(output_model_path, model)
    print(f'Save linear regression model into {output_model_path}')


def _random_forest_training(
    input_data_dir: str,
    input_target_dir: str,
    input_transformer_path: str,
    output_model_path: str,
):
    train = pd.read_csv(input_data_dir)
    target = pd.read_csv(input_target_dir)
    transformer = load_pickle(input_transformer_path)
    features = transformer.transform(train)
    model = RandomForestClassifier(max_depth=5, random_state=111).fit(features, target)
    dump_pickle(output_model_path, model)
    print(f'Save random forest model into {output_model_path}')


def _validate_and_choose_best_model(
    val_data_dir: str,
    val_target_dir: str,
    input_transformer_path: str,
    lin_reg_model_path: str,
    random_forest_model_path: str,
    output_model_path:str,
    output_metrics_path:str,
):
    val = pd.read_csv(val_data_dir)
    val_target = pd.read_csv(val_target_dir)
    transformer = load_pickle(input_transformer_path)
    features = transformer.transform(val)
    lin_reg_model = load_pickle(lin_reg_model_path)
    lin_reg_metrics = evaluate_model(lin_reg_model, features, val_target)

    random_forest_model = load_pickle(random_forest_model_path)
    random_forest_metrics = evaluate_model(random_forest_model, features, val_target)

    if lin_reg_metrics['f1_score'] > random_forest_metrics['f1_score']:
        dump_pickle(output_model_path, lin_reg_model)
        with open(output_metrics_path, 'w') as f:
            json.dump(lin_reg_metrics, f)
        print('LinearRegression model showed better results')
    else:
        dump_pickle(output_model_path, random_forest_model)
        with open(output_metrics_path, 'w') as f:
            json.dump(random_forest_metrics, f)
        print('RandomForestClassifier model showed better results')
    print(f'Save model and metrics into {output_model_path}')


with DAG(
    dag_id='train_validate',
    start_date=airflow.utils.dates.days_ago(1),
    schedule_interval='@weekly',
    max_active_runs=1,
) as dag:
    data_sensor = PythonSensor(
        task_id='data_sensor',
        python_callable=_wait_for_file,
        op_kwargs={'path': '/opt/airflow/data/raw/{{ ds }}/data.csv'},
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode='poke',
    )

    target_sensor = PythonSensor(
        task_id='target_sensor',
        python_callable=_wait_for_file,
        op_kwargs={'path': '/opt/airflow/data/raw/{{ ds }}/target.csv'},
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode='poke',
    )

    train_val_split = PythonOperator(
        task_id='train_val_split',
        python_callable=_train_val_split,
        op_kwargs={
            'execution_date': '{{ ds }}',
            'data_dir': '/opt/airflow/data/',
        }
    )

    data_preprocessing = PythonOperator(
        task_id='data_preprocessing',
        python_callable=_data_preprocessing,
        op_kwargs={
            'execution_date': '{{ ds }}',
            'processed_train_path': '/opt/airflow/data/processed/{{ ds }}/train.csv',
            'output_transformer_dir': '/opt/airflow/models/'
        }
    )

    linear_reg_training = PythonOperator(
        task_id='linear_reg_training',
        python_callable=_linear_reg_training,
        op_kwargs={
            'input_data_dir': '/opt/airflow/data/processed/{{ ds }}/train.csv',
            'input_target_dir': '/opt/airflow/data/processed/{{ ds }}/train_target.csv',
            'input_transformer_path': '/opt/airflow/models/{{ ds }}/transformer.pkl',
            'output_model_path': '/opt/airflow/models/{{ ds }}/lin_reg_model.pkl',
        }
    )

    random_forest_training = PythonOperator(
        task_id='random_forest_training',
        python_callable=_random_forest_training,
        op_kwargs={
            'input_data_dir': '/opt/airflow/data/processed/{{ ds }}/train.csv',
            'input_target_dir': '/opt/airflow/data/processed/{{ ds }}/train_target.csv',
            'input_transformer_path': '/opt/airflow/models/{{ ds }}/transformer.pkl',
            'output_model_path': '/opt/airflow/models/{{ ds }}/random_forest_model.pkl',
        }
    )

    validate_and_choose_best_model = PythonOperator(
        task_id='validate_and_choose_best_model',
        python_callable=_validate_and_choose_best_model,
        op_kwargs={
            'val_data_dir': '/opt/airflow/data/processed/{{ ds }}/val.csv',
            'val_target_dir': '/opt/airflow/data/processed/{{ ds }}/val_target.csv',
            'input_transformer_path': '/opt/airflow/models/{{ ds }}/transformer.pkl',
            'lin_reg_model_path': '/opt/airflow/models/{{ ds }}/lin_reg_model.pkl',
            'random_forest_model_path': '/opt/airflow/models/{{ ds }}/random_forest_model.pkl',
            'output_model_path': '/opt/airflow/models/{{ ds }}/model.pkl',
            'output_metrics_path': '/opt/airflow/models/{{ ds }}/metrics.pkl',
        }
    )

    bash_command = dedent(
        """
        rm /opt/airflow/models/{{ ds }}/lin_reg_model.pkl
        rm /opt/airflow/models/{{ ds }}/random_forest_model.pkl
        """
    )
    remove_redundant_files = BashOperator(
        task_id='remove_redundant_files',
        bash_command=bash_command,
    )

    [data_sensor, target_sensor] >> train_val_split >> data_preprocessing >> \
    [linear_reg_training, random_forest_training] >> validate_and_choose_best_model >>\
    remove_redundant_files
