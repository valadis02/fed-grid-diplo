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

# Δημιούργησε template model για CKKS (χρησιμοποιεί την ίδια έκδοση Keras με το container)
RUN python3 -c "\
import sys; sys.path.insert(0, '/app'); \
from shared.model_architectures import create_model; \
model = create_model(); \
model.save('/app/shared/ckks_keys/model_template.keras'); \
print('CKKS template created:', model.count_params(), 'params')"

# Εκκίνηση με το module notation
CMD ["python3", "-m", "cloud.communication.cloud_messaging"]
