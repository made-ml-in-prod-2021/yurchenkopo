from fastapi.testclient import TestClient

from online_inference.app import app


def test_read_entry_point():
    with TestClient(app) as client:
        response = client.get('/')
        assert response.status_code == 200
        assert response.json() == 'it is entry point of our predictor'


def test_healz():
    with TestClient(app) as client:
        response = client.get('/healz')
        assert response.status_code == 200
        assert response.json()


def test_predict(columns):
    request_data = [1] * 13
    with TestClient(app) as client:
        response = client.get('/predict',
                              json={'data': [request_data], 'features': columns}
                              )
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert response.json()[0] in [0, 1]