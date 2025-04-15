"""
Meme Emotion Prediction (Task C) - Online Version
This script uses the trained model to predict emotion intensities in memes.
"""
import argparse
import torch
from PIL import Image
import numpy as np  # Needed for working with arrays in visualization
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from functools import lru_cache
import time

# Import the required components
from .model import MemeEmotionModel
from .config import EMOTION_CLASSES, EMOTION_NAMES, EMOTION_DIMS
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
    model = MemeEmotionModel()

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


def predict_emotion_intensity(image_path, model_path, text=None, fp16=False):
    """
    Predict emotion intensities for a meme image (Task C)

    Args:
        image_path: Path to the image
        model_path: Path to the model checkpoint
        text: Optional text from the meme
        fp16: Whether to use half precision

    Returns:
        Dictionary with emotion intensities and confidence scores
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
            intensity_preds = model.predict_intensities(images, text_data)
    else:
        # Run inference with regular precision
        with torch.no_grad():
            outputs = model(images, text_data)
            intensity_preds = model.predict_intensities(images, text_data)

    elapsed = time.time() - start_time
    logger.info(f"Prediction took {elapsed:.2f} seconds")

    # Extract intensity predictions (values like 0, 1, 2, 3)
    intensity_values = intensity_preds[0].cpu().numpy()

    # Process the raw logits to get confidence scores for each intensity level
    results = {}
    start_idx = 0
    for i, emotion in enumerate(EMOTION_NAMES):
        # Get logits for this emotion's intensities
        emotion_dim = EMOTION_DIMS[i]
        emotion_logits = outputs[0, start_idx:start_idx + emotion_dim].cpu()

        # Apply softmax to get probabilities for each intensity level
        probs = torch.softmax(emotion_logits, dim=0).numpy()

        # Get the predicted intensity level and its confidence
        intensity_level = int(intensity_values[i])
        confidence = float(probs[intensity_level])

        # Map numeric intensity to text label
        intensity_label = EMOTION_CLASSES[emotion][intensity_level]

        # Store detailed results
        results[emotion] = {
            "intensity": intensity_level,
            "label": intensity_label,
            "confidence": confidence,
            "probabilities": {
                EMOTION_CLASSES[emotion][j]: float(probs[j])
                for j in range(emotion_dim)
            }
        }

        start_idx += emotion_dim

    return results


def visualize_prediction(image_path, results, output_path=None):
    """Visualize image with intensity prediction results"""
    # Load image
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        img = Image.new('RGB', (224, 224), color='black')

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)

    # Main image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img)
    ax_img.set_title("Meme Image")
    ax_img.axis("off")

    # Create subplots for each emotion
    axes = [
        fig.add_subplot(gs[0, 1]),  # humor
        fig.add_subplot(gs[0, 2]),  # sarcasm
        fig.add_subplot(gs[1, 0]),  # offensive
        fig.add_subplot(gs[1, 1]),  # motivational
    ]

    # Plot intensity distributions for each emotion
    for i, emotion in enumerate(EMOTION_NAMES):
        emotion_result = results[emotion]
        emotion_labels = list(emotion_result["probabilities"].keys())
        probabilities = np.array(list(emotion_result["probabilities"].values()))

        # Create bar chart
        bars = axes[i].bar(
            range(len(emotion_labels)),
            probabilities,
            color=['lightgray'] * len(emotion_labels)
        )

        # Highlight the predicted intensity
        bars[emotion_result["intensity"]].set_color('orangered')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:  # Only show significant values
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.01,
                    f'{height:.2f}',
                    ha='center', va='bottom',
                    fontsize=8
                )

        # Set title and labels
        axes[i].set_title(f"{emotion.capitalize()}: {emotion_result['label']}")
        axes[i].set_xticks(range(len(emotion_labels)))
        axes[i].set_xticklabels(emotion_labels, rotation=30, ha='right')
        axes[i].set_ylim(0, 1.1)
        axes[i].grid(axis='y', linestyle='--', alpha=0.3)

    # Add a summary text box
    summary_text = "Emotion Intensity Summary:\n\n"
    for emotion in EMOTION_NAMES:
        result = results[emotion]
        summary_text += f"• {emotion.capitalize()}: {result['label']}\n  (Confidence: {result['confidence']:.2f})\n\n"

    # Add text box for summary
    ax_summary = fig.add_subplot(gs[1, 2])
    ax_summary.text(0.1, 0.5, summary_text, va='center', fontsize=10)
    ax_summary.axis('off')

    plt.tight_layout()

    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Predict emotion intensities in memes (Task C)")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to meme image")
    parser.add_argument("--text", type=str, help="Text in the meme (optional)")
    parser.add_argument("--model", type=str, default="models/memotion_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str,
                        help="Path to save visualization")
    parser.add_argument("--fp16", action="store_true",
                        help="Use half precision for faster inference")
    args = parser.parse_args()

    # Check if image exists
    if not Path(args.image).exists():
        logger.error(f"Image not found: {args.image}")
        return

    # Run prediction
    results = predict_emotion_intensity(
        args.image, args.model, args.text, args.fp16
    )

    # Print results
    print("\nMeme Emotion Intensity Predictions:")
    print("-" * 40)
    for emotion, data in results.items():
        print(f"{emotion.capitalize()}: {data['label']} (Level {data['intensity']}, Confidence: {data['confidence']:.2f})")
        print(f"  All probabilities: {', '.join([f'{k}: {v:.2f}' for k, v in data['probabilities'].items()])}")
        print()

    # Visualize if output path is provided
    if args.output:
        visualize_prediction(args.image, results, args.output)
        print(f"Visualization saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
