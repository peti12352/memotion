from pathlib import Path

# Dataset configuration
DATA_DIR = Path("/kaggle/working/memotion/data")
MAX_TEXT_LENGTH = 128  # Maximum length for text input
IMAGE_SIZE = (224, 224)  # Input image size for the model

# Task C Configuration - Emotion Scales
EMOTION_CLASSES = {
    "humour": ["not funny", "funny", "very funny", "hilarious"],  # 0-3
    "sarcasm": ["not sarcastic", "general", "twisted meaning", "very twisted"],  # 0-3
    "offensive": ["not offensive", "slight", "very offensive", "hateful offensive"],  # 0-3
    "motivational": ["not motivational", "motivational"]  # 0-1
}

# Number of levels per emotion class
EMOTION_SCALES = {
    "humour": 4,
    "sarcasm": 4,
    "offensive": 4,
    "motivational": 2
}

# Output dimension for each emotion - used in model architecture
EMOTION_DIMS = list(EMOTION_SCALES.values())
NUM_CLASSES = sum(EMOTION_SCALES.values())  # Total number of outputs

# Emotion names for reporting
EMOTION_NAMES = list(EMOTION_CLASSES.keys())

# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 1e-5  # Lower learning rate for ordinal classification
NUM_EPOCHS = 15
NUM_WORKERS = 2  # Number of workers for data loading

# Paths
MODELS_DIR = Path("/kaggle/working/outputs/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "meme_emotion_model"

# Ensure DATA_DIR exists
DATA_DIR.mkdir(parents=True, exist_ok=True)
