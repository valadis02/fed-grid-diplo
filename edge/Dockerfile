FROM python:3.10-slim

WORKDIR /app

# 1. Αντιγραφή του requirements
COPY edge/requirements.txt /app/

# 2. System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    iputils-ping \
    dnsutils \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

# 3. Python deps (Εγκατάσταση requirements + pika για το RabbitMQ)
RUN pip install --no-cache-dir --default-timeout=1000 -r /app/requirements.txt pika

# 4. Αντιγραφή των αρχείων
# Πρώτα αντιγράφουμε το data folder
COPY edge/data /app/data
# Μετά τα υπόλοιπα αρχεία
COPY edge /app/edge
COPY shared /app/shared
COPY edge/federated_topology.json /app/

# 5. Δημιουργία φακέλων (Μετά το COPY για να μην υπάρχει conflict)
RUN mkdir -p /app/models /app/data/filtered_data

# 6. Διόρθωση PYTHONPATH
ENV PYTHONPATH=/app

EXPOSE 8081

# CMD για τον server
CMD ["uvicorn", "edge.communication.app_factory:app", "--host", "0.0.0.0", "--port", "8081"]

