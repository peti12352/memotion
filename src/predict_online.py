"""
Meme Emotion Prediction - Online Version (Optimized)
This script uses the trained model to predict emotions in memes with improved performance.
"""
import argparse
import torch
from PIL import Image
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from functools import lru_cache
import time

# Import the required components
from .model import MemeEmotionModel
from .config import EMOTION_CLASSES
from transformers import CLIPProcessor, RobertaTokenizer, logging as transformers_logging

# Reduce verbosity of transformers
transformers_logging.set_verbosity_error()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predict")

# Global caches to avoid reloading models
_model_cache = {}
_processor_cache = None
_tokenizer_cache = None


@lru_cache(maxsize=10)
def load_model(model_path):
    """Load the trained model with caching"""
    if model_path in _model_cache:
        return _model_cache[model_path]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    start_time = time.time()

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

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

    # Store in cache
    _model_cache[model_path] = (model, device)

    elapsed = time.time() - start_time
    logger.info(f"Model loading took {elapsed:.2f} seconds")

    return model, device


def get_processor_and_tokenizer():
    """Get or create the CLIP processor and RoBERTa tokenizer"""
    global _processor_cache, _tokenizer_cache

    if _processor_cache is None:
        start_time = time.time()
        _processor_cache = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", use_fast=True)
        elapsed = time.time() - start_time
        logger.info(f"CLIP processor loading took {elapsed:.2f} seconds")

    if _tokenizer_cache is None:
        start_time = time.time()
        _tokenizer_cache = RobertaTokenizer.from_pretrained("roberta-base")
        elapsed = time.time() - start_time
        logger.info(f"RoBERTa tokenizer loading took {elapsed:.2f} seconds")

    return _processor_cache, _tokenizer_cache


def calibrate_probabilities(probs, method='none', temperature=1.0):
    """
    Calibrate model probabilities to be more interpretable

    Methods:
        - 'none': No calibration
        - 'temperature': Apply temperature scaling (higher temp = more uniform)
        - 'softmax': Convert to mutual exclusive probabilities
        - 'top_only': Zero out all but the top prediction
        - 'normalize': Scale probabilities to sum to 1.0
        - 'threshold': Set values below threshold to 0, then normalize
    """
    if method == 'none':
        return probs

    if method == 'normalize':
        # Simple normalization to make values sum to 1.0
        values = np.array(list(probs.values()))
        total = np.sum(values)
        if total > 0:
            normalized = values / total
        else:
            normalized = values
        return {k: float(normalized[i]) for i, k in enumerate(probs.keys())}

    if method == 'temperature':
        # Temperature scaling
        # Lower temperature (< 1.0) = sharper distribution
        # Higher temperature (> 1.0) = more uniform distribution
        values = np.array(list(probs.values()))

        # Apply temperature scaling
        scaled = np.power(values, 1 / temperature)

        # Also normalize to sum to 1.0 for better interpretability
        total = np.sum(scaled)
        if total > 0:
            scaled = scaled / total

        return {k: float(scaled[i]) for i, k in enumerate(probs.keys())}

    if method == 'softmax':
        # Convert to softmax (mutually exclusive classes)
        values = np.array(list(probs.values()))

        # Apply temperature to control sharpness
        exp_probs = np.exp(values / temperature)
        softmax_probs = exp_probs / np.sum(exp_probs)

        return {k: float(softmax_probs[i]) for i, k in enumerate(probs.keys())}

    if method == 'threshold':
        # Apply threshold then normalize
        threshold = 0.3  # Can be adjusted
        values = np.array(list(probs.values()))
        thresholded = np.where(values > threshold, values, 0)

        # Normalize to sum to 1.0
        total = np.sum(thresholded)
        if total > 0:
            normalized = thresholded / total
        else:
            normalized = thresholded

        return {k: float(normalized[i]) for i, k in enumerate(probs.keys())}

    if method == 'top_only':
        # Only keep the top prediction
        top_class = max(probs.items(), key=lambda x: x[1])[0]
        return {k: float(1.0 if k == top_class else 0.0) for k, v in probs.items()}

    return probs


def preprocess_image_and_text(image_path, text=None):
    """
    Process image and text using the same pipeline as training
    to ensure consistency
    """
    # Get processor and tokenizer
    clip_processor, tokenizer = get_processor_and_tokenizer()

    # Process image - same as in dataset.py
    try:
        img = Image.open(image_path)

        # Handle transparency in the same way as dataset.py
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif 'transparency' in img.info:
            img = img.convert('RGBA')
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        else:
            img = img.convert("RGB")
    except Exception as e:
        logger.warning(f"Error loading image {image_path}: {str(e)}. Using placeholder.")
        img = Image.new('RGB', (224, 224), color='black')

    # Process image with CLIP processor - avoiding the compatibility issue
    try:
        # Use the processor's image preprocessing method directly
        image_inputs = {}
        if hasattr(clip_processor, "preprocess"):
            # For newer versions of transformers
            processed_image = clip_processor.preprocess(img, return_tensors="pt")
            image_inputs["pixel_values"] = processed_image["pixel_values"]
        else:
            # Fallback for older versions
            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711])
            ])
            image_tensor = transform(img).unsqueeze(0)
            image_inputs["pixel_values"] = image_tensor
    except Exception as e:
        logger.warning(f"Error processing image: {str(e)}. Using default tensor.")
        # Create a default tensor if processing fails
        image_inputs = {"pixel_values": torch.zeros((1, 3, 224, 224))}

    # Process text - same as in dataset.py
    if text is None:
        text = ""

    text_inputs = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    return image_inputs, text_inputs


def predict_emotion(image_path, model_path, text=None, fp16=False,
                    calibration='temperature', temperature=2.0):
    """
    Predict emotions for a meme image

    Args:
        image_path: Path to the image
        model_path: Path to the model checkpoint
        text: Optional text from the meme
        fp16: Whether to use half precision
        calibration: Probability calibration method
        temperature: Temperature for scaling probabilities

    Returns:
        Dictionary with emotion scores
    """
    # Load model and processors
    model, device = load_model(model_path)

    start_time = time.time()

    # Process image and text consistently with dataset.py
    image_inputs, text_inputs = preprocess_image_and_text(image_path, text)

    # Prepare inputs for the model
    images = image_inputs["pixel_values"].to(device)
    text_data = {
        "input_ids": text_inputs["input_ids"].to(device),
        "attention_mask": text_inputs["attention_mask"].to(device)
    }

    # Use half precision if requested
    if fp16 and device.type == 'cuda':
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(images, text_data)
    else:
        # Run inference with regular precision
        with torch.no_grad():
            outputs = model(images, text_data)

    # Get raw probabilities
    probabilities = torch.sigmoid(outputs)

    # Convert to dictionary of results
    results = {}
    for i, emotion in enumerate(EMOTION_CLASSES):
        results[emotion] = float(probabilities[0, i])

    # Apply calibration
    calibrated_results = calibrate_probabilities(
        results, method=calibration, temperature=temperature)

    elapsed = time.time() - start_time
    logger.info(f"Prediction took {elapsed:.2f} seconds")

    return calibrated_results


def visualize_prediction(image_path, results):
    """Visualize image with prediction results"""
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

    # Create horizontal bar chart
    bars = ax2.barh(emotions, scores, color='skyblue')
    ax2.set_xlim(0, 1)
    ax2.set_title("Emotion Predictions")
    ax2.set_xlabel("Score")

    # Add value labels to bars
    for bar, score in zip(bars, scores):
        if score > 0.01:  # Only show non-zero scores
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{score:.2f}', va='center')

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
    parser.add_argument("--fp16", action="store_true",
                        help="Use half precision for faster inference")
    parser.add_argument("--calibration", type=str, choices=['none', 'temperature', 'softmax', 'top_only', 'normalize', 'threshold'],
                        default='temperature', help="Probability calibration method")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Temperature for probability scaling")
    args = parser.parse_args()

    # Check if image exists
    if not Path(args.image).exists():
        logger.error(f"Image not found: {args.image}")
        return

    # Run prediction
    results = predict_emotion(
        args.image, args.model, args.text, args.fp16,
        args.calibration, args.temperature
    )

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
        visualize_prediction(args.image, results)
        plt.savefig(args.output)
        print(f"Visualization saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
