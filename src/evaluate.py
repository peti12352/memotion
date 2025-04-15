"""
Model evaluation script that loads a trained model and evaluates it on the test set.
This provides consistent evaluation across different model versions.
"""
import argparse
import logging
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from .dataset import MemeDataset
from .model import MemeEmotionModel
from .config import EMOTION_CLASSES, NUM_WORKERS
from .train import calculate_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluator")


def evaluate_model(model_path, kaggle_dataset_path=None, output_dir=None, batch_size=32):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load the test dataset
    logger.info("Loading test dataset...")
    test_dataset = MemeDataset(
        split="test",
        kaggle_dataset_path=kaggle_dataset_path
    )
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Load model
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = MemeEmotionModel(num_classes=len(EMOTION_CLASSES))

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = checkpoint

    model.to(device)
    model.eval()

    # Evaluation loop
    all_preds = []
    all_targets = []

    logger.info("Running evaluation...")
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            images = batch["image"].to(device)
            text_data = {
                "input_ids": batch["text"]["input_ids"].to(device),
                "attention_mask": batch["text"]["attention_mask"].to(device)
            }
            targets = batch["labels"].to(device)

            # Forward pass
            outputs = model(images, text_data)

            # Apply sigmoid
            preds = torch.sigmoid(outputs)

            # Store predictions and targets
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Combine predictions and targets
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_targets)

    # Generate binary predictions and targets for detailed analysis
    binary_preds = (all_preds.numpy() > 0.5).astype(int)
    binary_targets = (all_targets.numpy() > 0).astype(int)

    # Generate detailed classification report
    report = classification_report(
        binary_targets, binary_preds,
        target_names=EMOTION_CLASSES,
        output_dict=True
    )

    # Create per-class metrics
    class_report = {}
    for i, emotion in enumerate(EMOTION_CLASSES):
        class_report[emotion] = {
            "precision": float(metrics['per_class'][f'class_{i}']['precision']),
            "recall": float(metrics['per_class'][f'class_{i}']['recall']),
            "f1": float(metrics['per_class'][f'class_{i}']['f1']),
        }

    # Log metrics
    logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test precision: {metrics['precision']:.4f}")
    logger.info(f"Test recall: {metrics['recall']:.4f}")
    logger.info(f"Test F1: {metrics['f1']:.4f}")

    logger.info("Per-class metrics:")
    for emotion, scores in class_report.items():
        logger.info(f"  {emotion}: Precision={scores['precision']:.4f}, "
                    f"Recall={scores['recall']:.4f}, F1={scores['f1']:.4f}")

    # Save metrics to file if output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / "test_metrics.json"
        results = {
            "overall": {
                "accuracy": float(metrics['accuracy']),
                "precision": float(metrics['precision']),
                "recall": float(metrics['recall']),
                "f1": float(metrics['f1']),
            },
            "per_class": class_report,
            "detailed_report": report,
            "model_path": str(model_path)
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation results saved to {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--kaggle_dataset_path", type=str, help="Path to Kaggle dataset (optional)")
    parser.add_argument("--output_dir", type=str, help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")

    args = parser.parse_args()
    evaluate_model(args.model, args.kaggle_dataset_path, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
