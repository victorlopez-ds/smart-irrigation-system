FROM python:3.11-slim

WORKDIR /app

COPY services/api_bridge/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/api_bridge/app.py .

CMD ["python", "app.py"]
