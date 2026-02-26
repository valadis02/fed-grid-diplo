import os
import tensorflow as tf
import numpy as np

# Προσομοίωση των Paths που χρησιμοποιεί το service σου
class MockPaths:
    CLOUD_MODEL_FILE_PATH = type('obj', (object,), {'value': 'global_model.keras'})
    MODELS_FOLDER_PATH = type('obj', (object,), {'value': 'received_models/'})

# Εδώ κάνουμε επικόλληση τη συνάρτηση που μου έστειλες (απλοποιημένη για το τεστ)
def aggregate_test(fog_models_cache):
    aggregated_weights = None
    model_count = 0
    cloud_model = None

    # 1. Φόρτωση υπάρχοντος Cloud Model
    if os.path.exists(MockPaths.CLOUD_MODEL_FILE_PATH.value):
        print("Found existing cloud model. Including in average...")
        cloud_model = tf.keras.models.load_model(MockPaths.CLOUD_MODEL_FILE_PATH.value)
        aggregated_weights = [w.astype(np.float64) for w in cloud_model.get_weights()]
        model_count = 1

    # 2. Φόρτωση των μοντέλων από τον Fog
    for map_id, entry in fog_models_cache.items():
        print(f"Aggregating fog model from: {entry['model_path']}")
        model = tf.keras.models.load_model(model_path := entry["model_path"])
        weights = model.get_weights()
        if aggregated_weights is None:
            aggregated_weights = [w.astype(np.float64) for w in weights]
        else:
            for i in range(len(aggregated_weights)):
                aggregated_weights[i] += weights[i]
        model_count += 1

    if aggregated_weights is None or model_count == 0:
        print("Warning: No models found!")
        return

    # 3. Υπολογισμός Μέσου Όρου (Federated Averaging)
    for i in range(len(aggregated_weights)):
        aggregated_weights[i] = (aggregated_weights[i] / model_count).astype(np.float32)

    cloud_model.set_weights(aggregated_weights)
    cloud_model.save(MockPaths.CLOUD_MODEL_FILE_PATH.value)
    print(f"Success! Aggregated model saved to {MockPaths.CLOUD_MODEL_FILE_PATH.value}")

if __name__ == "__main__":
    # --- Βήμα Α: Δημιουργία Αρχικού Μοντέλου ---
    template = tf.keras.Sequential([tf.keras.layers.Dense(5, input_shape=(10,))])
    template.compile(optimizer='adam', loss='mse')
    template.save('global_model.keras')
    
    # --- Βήμα Β: Δημιουργία ενός Fake Fog Model ---
    # Του δίνουμε ελαφρώς διαφορετικά βάρη για να δούμε την αλλαγή
    fog_model = tf.keras.Sequential([tf.keras.layers.Dense(5, input_shape=(10,))])
    weights = fog_model.get_weights()
    new_weights = [w + 0.5 for w in weights] # Αλλάζουμε τα βάρη
    fog_model.set_weights(new_weights)
    
    os.makedirs('received_models', exist_ok=True)
    fog_model.save('received_models/fog_update_1.keras')

    # --- Βήμα Γ: Τρέχουμε το Aggregation ---
    cache = {
        "fog_1": {"model_path": "received_models/fog_update_1.keras"}
    }
    aggregate_test(cache)