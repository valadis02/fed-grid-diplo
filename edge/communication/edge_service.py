import base64
import os.path
from pathlib import Path
from typing import TYPE_CHECKING
from shared.logging_config import logger
from edge.communication.edge_resources_paths import EdgeResourcesPaths
from edge.model.model_architectures import create_model
from edge.model.model_training_service import train_local_edge_model

if TYPE_CHECKING:
    from edge.communication.edge_messaging import EdgeMessaging

class EdgeService:

    def __init__(self, messaging: 'EdgeMessaging'):
        self.edge_messaging = messaging

    @staticmethod
    def create_local_edge_model():
        edge_model = create_model(os.getenv('MODEL_ARCHITECTURE', 'base'))

        # ensure dir exists
        Path(EdgeResourcesPaths.MODELS_FOLDER_PATH.value).mkdir(parents=True, exist_ok=True)

        # use the full path constant directly
        local_edge_model_path = EdgeResourcesPaths.NON_TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value

        edge_model.save(local_edge_model_path)
        logger.info("Successfully created and saved local edge model.")

    def train_edge_local_model(self, payload):
        # placeholder date
        date = payload.get('data', {}).get('date')
        metrics = train_local_edge_model(date)
        # complete with fog host
        local_edge_model_path = EdgeResourcesPaths.TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
        self.edge_messaging.send_trained_model(local_edge_model_path, metrics)

    def retrain_fog_model(self, msg):
        local_edge_model_path = EdgeResourcesPaths.NON_TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value

        model_bytes = base64.b64decode(msg['model'])
        with open(local_edge_model_path, "wb") as f:
            f.write(model_bytes)

        date = msg.get('data', {}).get('date')
        metrics = train_local_edge_model(date)
        # complete with fog host
        trained_edge_model_file_path = EdgeResourcesPaths.TRAINED_LOCAL_EDGE_MODEL_FILE_PATH.value
        self.edge_messaging.send_trained_model(trained_edge_model_file_path, metrics)
