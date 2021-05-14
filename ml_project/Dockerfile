FROM python:3.8
COPY online_inference/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY models/model.pkl /model.pkl
COPY models/transformer.pkl /transformer.pkl
COPY online_inference/app.py /app.py

WORKDIR .

ENV PATH_TO_MODEL="/model.pkl"
ENV PATH_TO_TRANSFORMER="/transformer.pkl"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]