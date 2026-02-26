FROM python:3.10-slim

WORKDIR /app

# Δημιουργία απαραίτητων φακέλων
RUN mkdir -p "/app/models" && \
    mkdir -p "/app/models/outbox"

# 1. Requirements (Αφαιρέθηκε το fog/)
COPY requirements.txt /app/

# 2. Topology (Συγκεκριμένο όνομα αρχείου)
COPY federated_topology.json /app/

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    iputils-ping \
    dnsutils \
    iproute2  \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# 3. Application code (Χρήση τελείας '.' αντί για 'fog')
COPY . /app/fog
COPY shared /app/shared

# Python path
ENV PYTHONPATH=/app:/app/fog:/app/shared

# Ο Fog συνήθως τρέχει στην 8081 ή 8080 - βεβαιώσου ότι δεν συμπίπτει με τον Edge 
# αν τα τρέχεις στο ίδιο μηχάνημα (host port)
EXPOSE 8081

CMD ["uvicorn", "fog.communication.app_factory:app", "--host", "0.0.0.0", "--port", "8081"]
