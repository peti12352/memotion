"""
Model evaluation script that loads a trained model and evaluates it on the test set.
This provides consistent evaluation across different model versions for Task C (emotion intensity).
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
from .dataset import MemeDataset
from .model import MemeEmotionModel
from .config import EMOTION_NAMES, EMOTION_DIMS, EMOTION_CLASSES, NUM_WORKERS
from .train import calculate_ordinal_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluator")


def plot_metrics(metrics, output_dir):
    """Create visualizations of model performance metrics for Task C"""
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # 1. Plot accuracy, F1, and MAE for each emotion
    emotions = EMOTION_NAMES
    accuracy = [metrics['per_emotion'][e]['accuracy'] for e in emotions]
    f1 = [metrics['per_emotion'][e]['f1_weighted'] for e in emotions]
    mae = [metrics['per_emotion'][e]['mae'] for e in emotions]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(emotions))
    width = 0.25

    ax.bar(x - width, accuracy, width, label='Accuracy', color='skyblue')
    ax.bar(x, f1, width, label='F1 Weighted', color='orange')
    ax.bar(x + width, mae, width, label='MAE', color='green')

    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics by Emotion (Task C)')
    ax.set_xticks(x)
    ax.set_xticklabels(emotions)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(plots_dir / "emotion_metrics.png")
    plt.close()

    # 2. Create confusion matrices for each emotion
    for i, emotion in enumerate(emotions):
        cm = np.array(metrics['per_emotion'][emotion]['confusion_matrix'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=EMOTION_CLASSES[emotion],
                    yticklabels=EMOTION_CLASSES[emotion])
        plt.title(f'Confusion Matrix: {emotion}')
        plt.xlabel('Predicted Intensity')
        plt.ylabel('True Intensity')
        plt.tight_layout()
        plt.savefig(plots_dir / f"cm_{emotion}.png")
        plt.close()

    # 3. Create regression plot (predicted vs true intensities)
    if 'true_values' in metrics and 'pred_values' in metrics:
        for i, emotion in enumerate(emotions):
            true_values = np.array(metrics['true_values'][emotion])
            pred_values = np.array(metrics['pred_values'][emotion])

            plt.figure(figsize=(8, 6))
            plt.scatter(true_values, pred_values, alpha=0.5)

            # Add perfect prediction line
            max_val = max(EMOTION_DIMS[i] - 1, np.max(pred_values))
            plt.plot([0, max_val], [0, max_val], 'r--')

            plt.title(f'{emotion} - Predicted vs True Intensity')
            plt.xlabel('True Intensity')
            plt.ylabel('Predicted Intensity')
            plt.xticks(range(EMOTION_DIMS[i]))
            plt.yticks(range(EMOTION_DIMS[i]))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / f"regression_{emotion}.png")
            plt.close()

    logger.info(f"Performance visualizations saved to {plots_dir}")
    return plots_dir


def evaluate_model(model_path, kaggle_dataset_path=None, output_dir=None, batch_size=32):
    """Evaluate model performance on the test set for Task C (emotion intensity)"""
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
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = MemeEmotionModel()

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = checkpoint

    model.to(device)
    model.eval()

    # Evaluation loop
    all_preds = []
    all_targets = []
    all_intensities = []

    # Store actual values for regression plots
    true_values = {emotion: [] for emotion in EMOTION_NAMES}
    pred_values = {emotion: [] for emotion in EMOTION_NAMES}

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
            intensities = batch["intensity"].to(device)

            # Forward pass
            outputs = model(images, text_data)

            # Also get intensity predictions
            intensity_preds = model.predict_intensities(images, text_data)

            # Store predictions and targets
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_intensities.append(intensities.cpu())

            # Store intensity values for each emotion
            for i, emotion in enumerate(EMOTION_NAMES):
                true_values[emotion].extend(intensities[:, i].cpu().numpy().tolist())
                pred_values[emotion].extend(intensity_preds[:, i].cpu().numpy().tolist())

    # Combine all batches
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_intensities = torch.cat(all_intensities)

    # Calculate metrics
    logger.info("Calculating evaluation metrics...")
    metrics = calculate_ordinal_metrics(all_preds, all_targets, all_intensities)

    # Add true/pred values for plotting
    metrics['true_values'] = true_values
    metrics['pred_values'] = pred_values

    # Log metrics
    logger.info(f"Overall Accuracy: {metrics['overall']['accuracy']:.4f}")
    logger.info(f"Overall F1 Macro: {metrics['overall']['f1_macro']:.4f}")
    logger.info(f"Overall F1 Weighted: {metrics['overall']['f1_weighted']:.4f}")
    logger.info(f"Overall MAE: {metrics['overall']['mae']:.4f}")
    logger.info(f"Overall RMSE: {metrics['overall']['rmse']:.4f}")
    logger.info(f"Overall Kappa: {metrics['overall']['kappa']:.4f}")

    logger.info("Per-emotion metrics:")
    for emotion in EMOTION_NAMES:
        logger.info(f"  {emotion}:")
        logger.info(f"    Accuracy: {metrics['per_emotion'][emotion]['accuracy']:.4f}")
        logger.info(f"    F1 Weighted: {metrics['per_emotion'][emotion]['f1_weighted']:.4f}")
        logger.info(f"    MAE: {metrics['per_emotion'][emotion]['mae']:.4f}")
        logger.info(f"    Kappa: {metrics['per_emotion'][emotion]['kappa']:.4f}")

    # Save metrics to file if output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / "test_metrics.json"
        results = {
            "overall": metrics['overall'],
            "per_emotion": metrics['per_emotion'],
            "model_path": str(model_path)
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation results saved to {output_path}")

        # Generate visualizations
        plot_metrics(metrics, output_dir)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set for Task C")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--kaggle_dataset_path", type=str, help="Path to Kaggle dataset (optional)")
    parser.add_argument("--output_dir", type=str, help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")

    args = parser.parse_args()
    evaluate_model(args.model, args.kaggle_dataset_path, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
