import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Model settings
MODEL_NAME = "meme_emotion_model"
MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.pth"

# Training settings
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 4

# Model architecture
EMBEDDING_DIM = 512
NUM_CLASSES = 5  # amusement, sarcasm, offense, motivation, neutral
DROPOUT = 0.1

# Data settings
IMAGE_SIZE = (224, 224)
MAX_TEXT_LENGTH = 128

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
