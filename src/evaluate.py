"""
Model evaluation script that loads a trained model and evaluates it on the test set.
This provides consistent evaluation across different model versions.
"""
import argparse
import logging
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, multilabel_confusion_matrix

from .dataset import MemeDataset
from .model import MemeEmotionModel
from .config import EMOTION_CLASSES, NUM_WORKERS
from .train import calculate_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluator")


def plot_metrics(metrics, class_report, output_dir):
    """Create visualizations of model performance metrics"""
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Plot per-class metrics
    emotions = list(class_report.keys())
    precision = [class_report[e]["precision"] for e in emotions]
    recall = [class_report[e]["recall"] for e in emotions]
    f1 = [class_report[e]["f1"] for e in emotions]

    # Sort by F1 score
    sorted_indices = np.argsort(f1)[::-1]
    emotions = [emotions[i] for i in sorted_indices]
    precision = [precision[i] for i in sorted_indices]
    recall = [recall[i] for i in sorted_indices]
    f1 = [f1[i] for i in sorted_indices]

    # Create bar chart
    fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
    x = np.arange(len(emotions))
    width = 0.25

    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1, width, label='F1 Score')

    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics by Emotion Class')
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(plots_dir / "per_class_metrics.png")
    plt.close()

    # Create radar chart for a different visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False).tolist()
    # Close the polygon
    f1 = f1 + [f1[0]]
    precision = precision + [precision[0]]
    recall = recall + [recall[0]]
    angles = angles + [angles[0]]
    emotions = emotions + [emotions[0]]

    ax.plot(angles, f1, 'o-', linewidth=2, label='F1 Score')
    ax.plot(angles, precision, 'o-', linewidth=2, label='Precision')
    ax.plot(angles, recall, 'o-', linewidth=2, label='Recall')
    ax.fill(angles, f1, alpha=0.25)

    ax.set_thetagrids(np.degrees(angles[:-1]), emotions[:-1])
    ax.set_ylim(0, 1)
    ax.set_title('Emotion Recognition Performance', size=15)
    ax.grid(True)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(plots_dir / "radar_metrics.png")
    plt.close()

    logger.info(f"Performance visualizations saved to {plots_dir}")
    return plots_dir


def plot_confusion_matrix(binary_preds, binary_targets, output_dir):
    """Create confusion matrices for each emotion class"""
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Generate multilabel confusion matrix
    conf_matrices = multilabel_confusion_matrix(binary_targets, binary_preds)

    # Create a figure with subplots for each emotion
    n_classes = len(EMOTION_CLASSES)
    fig, axes = plt.subplots(2, (n_classes + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()

    for i, (cm, emotion) in enumerate(zip(conf_matrices, EMOTION_CLASSES)):
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix: {emotion}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')

        # Add labels for each quadrant
        axes[i].text(-0.1, -0.1, "TN", fontsize=12)
        axes[i].text(0.9, -0.1, "FP", fontsize=12)
        axes[i].text(-0.1, 0.9, "FN", fontsize=12)
        axes[i].text(0.9, 0.9, "TP", fontsize=12)

    # Remove any unused subplots
    for i in range(len(EMOTION_CLASSES), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrices.png")
    plt.close()

    logger.info(f"Confusion matrices saved to {plots_dir}")
    return plots_dir


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

        # Generate visualizations
        plot_metrics(metrics, class_report, output_dir)
        plot_confusion_matrix(binary_preds, binary_targets, output_dir)

        # Generate PR curve and ROC curve for each class
        try:
            from sklearn.metrics import precision_recall_curve, roc_curve, auc

            plots_dir = Path(output_dir) / "plots"
            plots_dir.mkdir(exist_ok=True, parents=True)

            # Create PR curves
            plt.figure(figsize=(10, 8))
            for i, emotion in enumerate(EMOTION_CLASSES):
                precision, recall, _ = precision_recall_curve(
                    binary_targets[:, i], all_preds[:, i].numpy())
                plt.plot(recall, precision, lw=2, label=f'{emotion}')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc='best')
            plt.title('Precision-Recall Curves')
            plt.grid(True)
            plt.savefig(plots_dir / "pr_curves.png")
            plt.close()

            # Create ROC curves
            plt.figure(figsize=(10, 8))
            for i, emotion in enumerate(EMOTION_CLASSES):
                fpr, tpr, _ = roc_curve(binary_targets[:, i], all_preds[:, i].numpy())
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{emotion} (AUC = {roc_auc:.2f})')

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            plt.title('ROC Curves')
            plt.grid(True)
            plt.savefig(plots_dir / "roc_curves.png")
            plt.close()

            logger.info(f"PR and ROC curves saved to {plots_dir}")
        except Exception as e:
            logger.warning(f"Could not generate PR and ROC curves: {e}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--kaggle_dataset_path", type=str, help="Path to Kaggle dataset (optional)")
    parser.add_argument("--output_dir", type=str, help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations of model performance")

    args = parser.parse_args()
    evaluate_model(args.model, args.kaggle_dataset_path, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
