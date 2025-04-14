from pathlib import Path

# Dataset configuration
DATA_DIR = Path("/kaggle/working/memotion/data")
MAX_TEXT_LENGTH = 128  # Maximum length for text input
IMAGE_SIZE = (224, 224)  # Input image size for the model

# Model configuration
EMOTION_CLASSES = [
    "amusement",
    "sarcasm",
    "offense",
    "motivation",
    "neutral"
]
NUM_CLASSES = len(EMOTION_CLASSES)

# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
NUM_EPOCHS = 10
NUM_WORKERS = 2  # Number of workers for data loading

# Paths
MODELS_DIR = Path("/kaggle/working/outputs/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "meme_emotion_model"

# Ensure DATA_DIR exists
DATA_DIR.mkdir(parents=True, exist_ok=True)
