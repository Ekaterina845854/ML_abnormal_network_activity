FROM python:3.12-slim

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "test.py", "--dataset", "D:/Infotecs/traffic_logs/traffic_data_20250202_105801.csv"]
