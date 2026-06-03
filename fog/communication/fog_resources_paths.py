from enum import Enum
class FogResourcesPaths(str, Enum):
    MODELS_FOLDER_PATH      = "/app/models/"
    FOG_MODEL_FILE_PATH     = MODELS_FOLDER_PATH + "fog_model.keras"
    FLTRUST_REF_PATH        = MODELS_FOLDER_PATH + "fltrust_ref.keras"
    OUTBOX_FOLDER_PATH      = MODELS_FOLDER_PATH + "outbox/"