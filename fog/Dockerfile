FROM python:3.10-slim

WORKDIR /app

RUN mkdir -p "/app/models"
RUN mkdir -p "/app/models/outbox"

COPY fog/requirements.txt /app/
COPY *federated_topology* /app/

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    iputils-ping \
    dnsutils \
    iproute2  \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY fog /app/fog
COPY shared /app/shared

ENV PYTHONPATH=/app:/app/fog:/app/shared

EXPOSE 8081

CMD ["uvicorn", "fog.communication.app_factory:app", "--host", "0.0.0.0", "--port", "8081"]
