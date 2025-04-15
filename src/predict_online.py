"""
Meme Emotion Prediction - Online Version
This script uses the trained model to predict emotions in memes
with full model loading from Hugging Face.
"""
import argparse
import torch
from PIL import Image
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Import the required components
from .model import MemeEmotionModel
from .config import EMOTION_CLASSES
from transformers import CLIPProcessor, RobertaTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predict")


def load_model(model_path):
    """Load the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Initialize model
    model = MemeEmotionModel(num_classes=len(EMOTION_CLASSES))

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model state dict from {model_path}")
    else:
        model = checkpoint
        logger.info(f"Loaded full model from {model_path}")

    model.to(device)
    model.eval()

    return model, device


def predict_emotion(model, image_path, text=None, device=None):
    """
    Predict emotions for a meme image

    Args:
        model: The loaded model
        image_path: Path to the image
        text: Optional text from the meme 
        device: Device to run inference on

    Returns:
        Dictionary with emotion scores
    """
    # Initialize processors
    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Load and process image
    img = Image.open(image_path).convert("RGB")
    image_inputs = clip_processor(
        images=img,
        return_tensors="pt",
        padding=True
    )

    # Process text (use empty string if none provided)
    if text is None:
        text = ""
        logger.info("No text provided, using empty string")

    text_inputs = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Prepare inputs for the model
    images = image_inputs["pixel_values"].to(device)
    text_data = {
        "input_ids": text_inputs["input_ids"].to(device),
        "attention_mask": text_inputs["attention_mask"].to(device)
    }

    # Run inference
    with torch.no_grad():
        outputs = model(images, text_data)
        probabilities = torch.sigmoid(outputs)

    # Convert to dictionary of results
    results = {}
    for i, emotion in enumerate(EMOTION_CLASSES):
        results[emotion] = float(probabilities[0, i])

    return results


def visualize_prediction(image_path, results):
    """
    Visualize image with prediction results
    """
    # Load image
    img = Image.open(image_path).convert("RGB")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Display image
    ax1.imshow(img)
    ax1.set_title("Meme Image")
    ax1.axis("off")

    # Display results as bar chart
    emotions = list(results.keys())
    scores = list(results.values())

    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    emotions = [emotions[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]

    ax2.barh(emotions, scores, color='skyblue')
    ax2.set_xlim(0, 1)
    ax2.set_title("Emotion Predictions")
    ax2.set_xlabel("Score")

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Predict emotions in memes")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to meme image")
    parser.add_argument("--text", type=str, help="Text in the meme (optional)")
    parser.add_argument("--model", type=str, default="models/memotion_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str,
                        help="Path to save visualization")
    args = parser.parse_args()

    # Check if image exists
    if not Path(args.image).exists():
        logger.error(f"Image not found: {args.image}")
        return

    logger.info(f"Loading model from {args.model}")
    model, device = load_model(args.model)

    logger.info(f"Running prediction on {args.image}")
    results = predict_emotion(model, args.image, args.text, device)

    # Print results
    print("\nMeme Emotion Predictions:")
    print("-" * 30)
    for emotion, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{emotion}: {score:.4f}")

    # Get dominant emotion
    dominant_emotion = max(results.items(), key=lambda x: x[1])[0]
    print(f"\nDominant emotion: {dominant_emotion}")

    # Visualize if output path is provided
    if args.output:
        fig = visualize_prediction(args.image, results)
        plt.savefig(args.output)
        print(f"Visualization saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
