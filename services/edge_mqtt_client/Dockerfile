FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY services/edge_mqtt_client/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/edge_mqtt_client/app.py .

EXPOSE 8010

CMD ["gunicorn", "-b", "0.0.0.0:8010", "-w", "1", "--threads", "2", "app:app"]
