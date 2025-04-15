"""
Meme Emotion Model Test Script
Evaluates model performance on the test set and generates detailed reports.
"""
import argparse
import logging
import os
import time
from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve
)
from tqdm import tqdm
import seaborn as sns

from .dataset import MemeDataset
from .model import MemeEmotionModel
from .config import (
    NUM_WORKERS, BATCH_SIZE, EMOTION_CLASSES, NUM_CLASSES
)
from .train import calculate_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("meme_emotion_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("meme_emotion_tester")


def evaluate_model(
    model,
    test_loader,
    device,
    threshold=0.5,
    fp16=False
):
    """
    Evaluate model on test set

    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        threshold: Decision threshold for binary predictions
        fp16: Whether to use half precision

    Returns:
        Dictionary containing all predictions, targets and metrics
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []
    all_image_paths = []

    logger.info("Running evaluation on test set...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            text_data = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            targets = batch["labels"].to(device)
            image_paths = batch["image_path"]

            # Run inference with or without mixed precision
            if fp16 and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(images, text_data)
            else:
                outputs = model(images, text_data)

            # Get probabilities with sigmoid
            probs = torch.sigmoid(outputs)

            # Apply threshold for binary predictions
            preds = (probs > threshold).float()

            # Add to lists
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_probs.append(probs.cpu())
            all_image_paths.extend(image_paths)

    # Concatenate tensors
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()

    # Calculate metrics
    metrics = calculate_metrics(all_probs, all_targets, threshold)

    # Create per-class metrics for detailed analysis
    per_class_metrics = {}

    for i, class_name in enumerate(EMOTION_CLASSES):
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets[:, i], all_preds[:, i], average='binary'
        )

        try:
            auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
        except:
            auc = 0.0

        per_class_metrics[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc)
        }

    # Return results
    return {
        "predictions": all_preds,
        "probabilities": all_probs,
        "targets": all_targets,
        "image_paths": all_image_paths,
        "metrics": metrics,
        "per_class_metrics": per_class_metrics,
        "overall_accuracy": float(accuracy_score(
            all_targets.astype(int), all_preds.astype(int)
        ))
    }


def generate_visualizations(results, output_dir):
    """
    Generate and save visualization plots

    Args:
        results: Evaluation results
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Extract data
    preds = results["predictions"]
    probs = results["probabilities"]
    targets = results["targets"]

    # 1. Per-class ROC curves
    plt.figure(figsize=(12, 10))
    for i, class_name in enumerate(EMOTION_CLASSES):
        fpr, tpr, _ = roc_curve(targets[:, i], probs[:, i])
        auc = results["per_class_metrics"][class_name]["auc"]
        plt.plot(
            fpr, tpr, lw=2,
            label=f'{class_name} (AUC = {auc:.3f})'
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Emotion Class')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_dir / "roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Confusion matrices per class
    for i, class_name in enumerate(EMOTION_CLASSES):
        cm = confusion_matrix(targets[:, i], preds[:, i])
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        plt.title(f'Confusion Matrix: {class_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(
            output_dir / f"confusion_matrix_{class_name}.png",
            dpi=300, bbox_inches='tight'
        )
        plt.close()

    # 3. Bar chart for precision, recall, f1
    metrics_df = pd.DataFrame(columns=['Precision', 'Recall', 'F1 Score'])

    for class_name in EMOTION_CLASSES:
        metrics_df.loc[class_name] = [
            results["per_class_metrics"][class_name]["precision"],
            results["per_class_metrics"][class_name]["recall"],
            results["per_class_metrics"][class_name]["f1"]
        ]

    plt.figure(figsize=(12, 8))
    metrics_df.plot(kind='bar', ax=plt.gca())
    plt.title('Precision, Recall and F1 Score by Emotion Class')
    plt.xlabel('Emotion Class')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_by_class.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Error analysis - find most common errors
    logger.info("Generating error analysis...")
    error_examples = []

    for i in range(len(targets)):
        for j, class_name in enumerate(EMOTION_CLASSES):
            # If prediction doesn't match target
            if preds[i, j] != targets[i, j]:
                error_examples.append({
                    "image_path": results["image_paths"][i],
                    "class": class_name,
                    "target": int(targets[i, j]),
                    "prediction": int(preds[i, j]),
                    "probability": float(probs[i, j]),
                    "error_type": "False Positive" if preds[i, j] > targets[i, j] else "False Negative"
                })

    # Save error analysis to CSV
    if error_examples:
        error_df = pd.DataFrame(error_examples)
        error_df.to_csv(output_dir / "error_analysis.csv", index=False)

    logger.info(f"All visualizations saved to {output_dir}")


def save_results(results, output_dir):
    """
    Save evaluation results to files

    Args:
        results: Evaluation results
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save metrics
    metrics_file = output_dir / "test_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            "overall_accuracy": results["overall_accuracy"],
            "metrics": {
                k: float(v) for k, v in results["metrics"].items()
                if k != "per_class"
            },
            "per_class_metrics": results["per_class_metrics"]
        }, f, indent=2)

    # Create and save a detailed report
    report_file = output_dir / "test_report.md"

    with open(report_file, 'w') as f:
        f.write("# Meme Emotion Model Evaluation Report\n\n")

        # Overall metrics
        f.write("## Overall Metrics\n\n")
        f.write(f"- Accuracy: {results['overall_accuracy']:.4f}\n")
        f.write(f"- Precision: {results['metrics']['precision']:.4f}\n")
        f.write(f"- Recall: {results['metrics']['recall']:.4f}\n")
        f.write(f"- F1 Score: {results['metrics']['f1']:.4f}\n\n")

        # Per-class metrics
        f.write("## Per-Class Metrics\n\n")
        f.write("| Emotion Class | Precision | Recall | F1 Score | AUC |\n")
        f.write("|---------------|-----------|--------|----------|-----|\n")

        for class_name in EMOTION_CLASSES:
            metrics = results["per_class_metrics"][class_name]
            f.write(
                f"| {class_name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['auc']:.4f} |\n")

        # Interpretation and conclusion
        f.write("\n## Interpretation\n\n")

        # Identify best performing class
        best_class = max(
            results["per_class_metrics"].items(),
            key=lambda x: x[1]["f1"]
        )[0]

        # Identify worst performing class
        worst_class = min(
            results["per_class_metrics"].items(),
            key=lambda x: x[1]["f1"]
        )[0]

        f.write(
            f"The model performs best on the '{best_class}' class and struggles most with the '{worst_class}' class.\n\n")
        f.write("### Recommendations for Improvement\n\n")
        f.write("1. Collect more training data for underperforming classes\n")
        f.write("2. Explore data augmentation techniques\n")
        f.write("3. Fine-tune model hyperparameters\n")
        f.write("4. Consider ensemble approaches for improved robustness\n")

    logger.info(f"Results saved to {output_dir}")

    return {
        "metrics_file": metrics_file,
        "report_file": report_file
    }


def main(args):
    """Main function to run model evaluation"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    start_time = time.time()

    # Load test dataset
    logger.info("Loading test dataset...")
    test_dataset = MemeDataset(
        split="test",
        kaggle_dataset_path=args.kaggle_dataset_path
    )

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    logger.info(f"Loaded {len(test_dataset)} test samples")

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Initialize model
    model = MemeEmotionModel(num_classes=NUM_CLASSES)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded model state dict")
    else:
        model = checkpoint
        logger.info("Loaded full model")

    model.to(device)
    model.eval()

    # Run evaluation
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        threshold=args.threshold,
        fp16=args.fp16
    )

    # Save results
    file_paths = save_results(results, output_dir)

    # Generate visualizations
    if not args.no_visualizations:
        generate_visualizations(results, output_dir / "visualizations")

    # Log overall results
    logger.info("=== Test Results ===")
    logger.info(f"Accuracy: {results['overall_accuracy']:.4f}")
    logger.info(f"Precision: {results['metrics']['precision']:.4f}")
    logger.info(f"Recall: {results['metrics']['recall']:.4f}")
    logger.info(f"F1 Score: {results['metrics']['f1']:.4f}")

    elapsed_time = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed_time:.2f} seconds")

    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Test report: {file_paths['report_file']}")

    return results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate meme emotion model on test set"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_results",
        help="Directory to save test results"
    )

    parser.add_argument(
        "--kaggle_dataset_path",
        type=str,
        help="Path to Kaggle dataset (if using Kaggle)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for testing"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for binary predictions"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision for faster inference"
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force using CPU even if GPU is available"
    )

    parser.add_argument(
        "--no_visualizations",
        action="store_true",
        help="Skip generating visualizations"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
