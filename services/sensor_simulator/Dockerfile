FROM python:3.11-slim

WORKDIR /app

COPY services/sensor_simulator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/sensor_simulator/app.py .

CMD ["python", "-u", "app.py"]
