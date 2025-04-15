import argparse
import torch
from PIL import Image
import os
from pathlib import Path

from .model import MemeEmotionModel
from .dataset import MemeDataset  # For preprocessing utilities
from .config import EMOTION_CLASSES


def predict_emotion(image_path, model_path):
    """Predict emotions for a single meme image"""
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MemeEmotionModel()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Create dataset instance to use its preprocessing
    dataset = MemeDataset(split="val")  # We only need this for preprocessing

    # Load and process image
    img = Image.open(image_path).convert("RGB")
    image_inputs = dataset.clip_processor(
        images=img,
        return_tensors="pt",
        padding=True
    )

    # We need to extract text from the image (simplified version)
    # For a real implementation, you'd use OCR here
    text = input("Enter text from the meme (or press Enter if none): ")

    # Process text
    text_inputs = dataset.tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Prepare inputs
    images = image_inputs["pixel_values"].to(device)
    text_data = {
        "input_ids": text_inputs["input_ids"].to(device),
        "attention_mask": text_inputs["attention_mask"].to(device)
    }

    # Make prediction
    with torch.no_grad():
        outputs = model(images, text_data)
        probabilities = torch.sigmoid(outputs)

    # Map probabilities to emotion classes
    results = {}
    for i, emotion in enumerate(EMOTION_CLASSES):
        results[emotion] = float(probabilities[0, i])

    return results


def main():
    parser = argparse.ArgumentParser(description="Predict emotions in a meme")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to meme image")
    parser.add_argument("--model", type=str, default="models/meme_emotion_model.pt",
                        help="Path to model checkpoint")
    args = parser.parse_args()

    results = predict_emotion(args.image, args.model)

    print("\nEmotion Predictions:")
    print("-" * 30)
    for emotion, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{emotion}: {score:.4f}")

    # Print the dominant emotion
    dominant_emotion = max(results.items(), key=lambda x: x[1])[0]
    print(f"\nDominant emotion: {dominant_emotion}")


if __name__ == "__main__":
    main()
