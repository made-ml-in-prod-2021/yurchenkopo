import numpy as np
import pandas as pd
import requests

if __name__ == '__main__':
    data = pd.read_csv('data/raw/test.csv')
    request_features = data.columns.tolist()
    for i in range(20):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]
        print(request_data)
        response = requests.get(
            'http://localhost:34855/predict/',
            json={'data': [request_data], 'features': request_features},
        )
        print(response.status_code)
        print(response.json())