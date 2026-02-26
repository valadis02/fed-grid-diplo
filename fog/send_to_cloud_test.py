import pika
import json
import base64
import os
import hashlib

def send_model():
    # Ρυθμίσεις
    RABBIT_HOST = 'localhost' # Αφού τρέχεις από Windows
    QUEUE_NAME = 'fog_to_cloud_models'
    
    # Βρες ένα υπάρχον μοντέλο στον φάκελο του Fog
    # Αντικατάστησε το 'your_model.keras' με το πραγματικό όνομα αρχείου
    model_path = "aggregated_fog_model.keras" 
    
    if not os.path.exists(model_path):
        print(f"Σφάλμα: Το αρχείο {model_path} δεν βρέθηκε!")
        return

    # 1. Κωδικοποίηση μοντέλου
    with open(model_path, "rb") as f:
        model_bytes = f.read()
        model_b64 = base64.b64encode(model_bytes).decode('utf-8')
        model_hash = hashlib.sha256(model_bytes).hexdigest()

    # 2. Δημιουργία Payload (Πρέπει να ταιριάζει με τον CloudMessaging listener)
    payload = {
        "fog_name": "FOG_NODE_1",
        "fog_device_mac": "00:11:22:33:44:55",
        "model": model_b64,
        "round_id": None, # Ο Cloud σου αυτή τη στιγμή μάλλον έχει None
        "hash": model_hash
    }

    # 3. Αποστολή με επιβεβαίωση
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', port=5673))
    channel = connection.channel()
    
    # Σιγουρευόμαστε ότι η ουρά υπάρχει ΠΡΙΝ στείλουμε
    channel.queue_declare(queue=QUEUE_NAME, durable=True)

    channel.basic_publish(
        exchange='',
        routing_key=QUEUE_NAME,
        body=json.dumps(payload),
        properties=pika.BasicProperties(
            delivery_mode=2,  # Κάνει το μήνυμα επίμονο
            content_type='application/json'
        )
    )
    print(f" [v] ΠΡΑΓΜΑΤΙΚΗ ΑΠΟΣΤΟΛΗ: Το μήνυμα έφυγε για την ουρά {QUEUE_NAME}")
    connection.close()

if __name__ == "__main__":
    send_model()