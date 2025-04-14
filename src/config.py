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

# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
EPOCHS = 10

# Paths
MODEL_SAVE_DIR = Path("/kaggle/working/outputs/models")
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Ensure DATA_DIR exists
DATA_DIR.mkdir(parents=True, exist_ok=True)
