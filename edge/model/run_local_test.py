import os
import sys
import pandas as pd
import numpy as np
import json
import base64
import pika
from datetime import datetime

# Ρύθμιση Paths
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from edge.communication.edge_resources_paths import EdgeResourcesPaths
from edge.model.model_architectures import create_model
from edge.model.model_training_service import train_local_edge_model
from shared.logging_config import logger

def generate_dummy_data(file_path, num_days=10, start_date_str="2024-01-01"):
    logger.info("Generating dummy data...")
    # Μετατροπή σε absolute path για τα Windows
    abs_path = os.path.abspath(file_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    
    start_date = pd.to_datetime(start_date_str)
    periods = num_days * 24 * 4 
    date_range = pd.date_range(start=start_date, periods=periods, freq='15min')
    
    values = np.sin(np.linspace(0, 10 * np.pi, periods)) * 10 + 20
    values += np.random.normal(0, 2, periods)
    
    df = pd.DataFrame({
        'datetime': date_range,
        'value': values,
        'apparent power (kWh)': values 
    })
    
    df.to_csv(abs_path, index=False)
    logger.info(f"Dummy data generated at {abs_path}")

def create_base_model():
    logger.info("Creating base (untrained) model...")
    model_path = EdgeResourcesPaths.NON_TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
    model = create_model('simple_lstm_two_gates')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    logger.info(f"Base model saved at {model_path}")

def send_results_to_fog(results):
    logger.info("Προετοιμασία αποστολής μοντέλου στον Fog...")
    
    # Το μονοπάτι του εκπαιδευμένου μοντέλου
    model_path = EdgeResourcesPaths.TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
    
    try:
        if not os.path.exists(model_path):
            logger.error(f"Το αρχείο μοντέλου δεν βρέθηκε στο: {model_path}")
            return

        with open(model_path, "rb") as f:
            model_bytes = f.read()
            model_base64 = base64.b64encode(model_bytes).decode('utf-8')
            
        payload = {
            "edge_mac": "00:00:00:00:00:00", 
            "edge_name": "edge_node_1",
            "model": model_base64,
            "metrics": results
        }

        # Ρυθμίσεις Σύνδεσης για Windows Local Testing
        fog_host = os.getenv('FOG_RABBITMQ_HOST', 'localhost') 
        fog_port = int(os.getenv('FOG_RABBITMQ_PORT', 5673)) # Default στην 5673 όπως το Cloud

        logger.info(f"Connecting to Fog RabbitMQ at {fog_host}:{fog_port}...")
        
        connection = pika.BlockingConnection(pika.ConnectionParameters(
            host=fog_host,
            port=fog_port,
            heartbeat=60
        ))
        channel = connection.channel()
        channel.queue_declare(queue='edge_to_fog_models', durable=True)

        channel.basic_publish(
            exchange='',
            routing_key='edge_to_fog_models',
            body=json.dumps(payload),
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type='application/json'
            )
        )
        logger.info("✅ Το πραγματικό εκπαιδευμένο μοντέλο στάλθηκε επιτυχώς στον Fog!")
        connection.close()
    except Exception as e:
        logger.error(f"Αποτυχία επικοινωνίας με Fog: {e}")

if __name__ == "__main__":
    test_date = "2024-01-01"
    
    # Διορθωμένα paths για να παίζουν παντού
    data_path = os.path.join(root_path, "edge", "data", "input_data.csv")
    
    # Παραγωγή δεδομένων
    generate_dummy_data(data_path, num_days=10, start_date_str=test_date)
    
    # Δημιουργία αρχικού μοντέλου
    create_base_model()
    
    print("\n--- Starting Training Cycle ---")
    try:
        # Εκπαίδευση (Χρησιμοποιεί το input_data.csv που μόλις φτιάξαμε)
        results = train_local_edge_model(training_date=test_date)
        
        if results:
            print("\n--- Training Finished. Sending to Fog ---")
            send_results_to_fog(results)
        else:
            print("Training failed, no results returned.")
    except Exception as e:
        print(f"❌ Training crashed with error: {e}")
        import traceback
        traceback.print_exc()