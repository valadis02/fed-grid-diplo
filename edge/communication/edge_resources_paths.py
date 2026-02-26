import os
from enum import Enum

# Δυναμικός προσδιορισμός της βάσης του Edge
# Παίρνει το path του τρέχοντος αρχείου και πάει δύο επίπεδα πάνω
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class EdgeResourcesPaths(str, Enum):
    # Χρησιμοποιούμε os.path.join για να φτιάχνει σωστά paths σε Windows (\) και Linux (/)
    MODELS_FOLDER_PATH = os.path.join(BASE_DIR, "models") + os.sep
    NON_TRAINED_LOCAL_EDGE_MODEL_FILE_PATH = os.path.join(MODELS_FOLDER_PATH, "non_trained_local_edge_model.keras")
    TRAINED_LOCAL_EDGE_MODEL_FILE_PATH = os.path.join(MODELS_FOLDER_PATH, "trained_local_edge_model.keras")

    DATA_FOLDER_PATH = os.path.join(BASE_DIR, "data") + os.sep
    FILTERED_DATA_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, "filtered_data") + os.sep
    INPUT_DATA_PATH = os.path.join(DATA_FOLDER_PATH, "input_data.csv")
    FILTERED_DATA_PATH = os.path.join(FILTERED_DATA_FOLDER_PATH, "filtered_data.csv")

    TRAINING_DAYS_DATA_PATH = os.path.join(FILTERED_DATA_FOLDER_PATH, "training_days_data.csv")
    EVALUATION_DAYS_DATA_PATH = os.path.join(FILTERED_DATA_FOLDER_PATH, "evaluation_days_data.csv")