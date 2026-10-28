FROM python:3.11-slim

WORKDIR /app

# Ensure Python logs stream immediately without buffering in Docker
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py indicators.py strategy.py risk_manager.py ibkr_execution.py backtest.py optimize.py main.py ./

CMD ["python", "-u", "main.py", "--mode", "paper", "--symbol", "MNQ", "--bar-size", "2 mins", "--contracts", "1", "--host", "127.0.0.1", "--port", "4002"]
