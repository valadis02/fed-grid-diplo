FROM python:3.10-slim

WORKDIR /app

# Requirements
COPY edge/requirements.txt /app/

# Folders
RUN mkdir -p /app/models \
    && mkdir -p /app/data \
    && mkdir -p /app/data/filtered_data

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    iputils-ping \
    dnsutils \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# 🔴 COPY DATA FIRST (CRITICAL)
COPY edge/data /app/data

# Application code
COPY edge /app/edge
COPY shared /app/shared

# Topology
COPY federated_topology.json /app/

# Python path
ENV PYTHONPATH=/app:/app/edge:/app/shared

EXPOSE 8081

CMD ["uvicorn", "edge.communication.app_factory:app", "--host", "0.0.0.0", "--port", "8081"]

