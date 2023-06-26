# Dockerfile
FROM python:3.8

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

COPY regression_model.pkl /app/regression_model.pkl

CMD [ "python", "app.py" ]

