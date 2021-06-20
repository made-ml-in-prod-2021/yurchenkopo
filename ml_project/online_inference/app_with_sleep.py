import logging
import os
import pickle
from typing import List, Union, Optional

import pandas as pd
import time
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, conlist
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def load_object(path: str) -> Pipeline:
    with open(path, 'rb') as f:
        return pickle.load(f)


class HeartDeseasePredictModel(BaseModel):
    data: List[conlist(Union[float, int, None], min_items=13, max_items=13)]
    features: List[str]


model: Optional[Pipeline] = None
transformer: Optional[Pipeline] = None


def make_predict(data: List, features: List[str], model: Pipeline, transformer: Pipeline) -> List[int]:
    df = pd.DataFrame(data, columns=features)
    features = transformer.transform(df)
    predictions = model.predict(features)
    return predictions.tolist()


app = FastAPI()
start = time.time()


@app.get('/')
def main():
    return 'it is entry point of our predictor'


@app.on_event('startup')
def load_model():
    global model, transformer
    time.sleep(30)
    model_path = os.getenv('PATH_TO_MODEL')
    transformer_path = os.getenv('PATH_TO_TRANSFORMER') #'models/transformer.pkl'
    if model_path is None:
        err = f'PATH_TO_MODEL {model_path} is None'
        logger.error(err)
        raise RuntimeError(err)
    if transformer_path is None:
        err = f'PATH_TO_TRANSFORMER {transformer_path} is None'
        logger.error(err)
        raise RuntimeError(err)

    model = load_object(model_path)
    transformer = load_object(transformer_path)


@app.get('/healz')
def health() -> bool:
    if time.time() - start > 120:
        raise OSError('Application stop')
    return not (model is None) and not (transformer is None)


@app.get('/predict/', response_model=List[int])
def predict(request: HeartDeseasePredictModel):
    return make_predict(request.data, request.features, model, transformer)


if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=os.getenv('PORT', 8000))

