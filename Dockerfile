FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY gbr_jaundice_model.joblib .
COPY gbr_feature_names.joblib .

EXPOSE 5000

CMD ["python", "app.py"]
