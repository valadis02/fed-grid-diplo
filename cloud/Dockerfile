FROM python:3.10-slim

WORKDIR /app

# Εγκατάσταση απαραίτητων εργαλείων
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Αντιγραφή requirements και εγκατάσταση
COPY cloud/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Αντιγραφή των φακέλων - ΠΡΟΣΟΧΗ ΣΤΑ PATHS
COPY shared /app/shared
COPY cloud /app/cloud

# Ορισμός του PYTHONPATH ώστε η Python να ψάχνει στο /app
ENV PYTHONPATH="/app"

# Εκκίνηση με το module notation
CMD ["python3", "-m", "cloud.communication.cloud_messaging"]
