"""
Training script for the meme emotion recognition model.
"""
import os
import argparse
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, accuracy_score,
    confusion_matrix, cohen_kappa_score, mean_absolute_error, mean_squared_error
)

from .dataset import MemeDataset
from .model import MemeEmotionModel
from .config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, NUM_WORKERS,
    MODELS_DIR, MODEL_NAME, EMOTION_NAMES, EMOTION_DIMS, EMOTION_CLASSES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{MODEL_NAME}_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("meme_emotion_trainer")


def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed}")


def mixup_data(x, y, alpha=0.2, device=None):
    """Performs mixup on the input data and label"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    if device:
        index = index.to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def label_smoothing(targets, num_classes, smoothing=0.1):
    """Applies label smoothing to the targets"""
    with torch.no_grad():
        targets = targets * (1 - smoothing) + smoothing / num_classes
    return targets


class PerEmotionCrossEntropyLoss(nn.Module):
    """Calculates standard CrossEntropyLoss for each emotion head independently."""

    def __init__(self, class_weights=None):
        super().__init__()
        # If class_weights are provided, they should be a list of tensors, one per emotion
        self.weights = class_weights
        self.ce_losses = nn.ModuleList([
            nn.CrossEntropyLoss(weight=self.weights[i] if self.weights else None)
            for i, dim in enumerate(EMOTION_DIMS)
        ])

    def forward(self, logits, targets):
        start_idx = 0
        total_loss = 0
        for i, dim in enumerate(EMOTION_DIMS):
            emotion_logits = logits[:, start_idx:start_idx + dim]
            emotion_targets_one_hot = targets[:, start_idx:start_idx + dim]
            # Convert one-hot targets to class indices for CrossEntropyLoss
            target_indices = torch.argmax(emotion_targets_one_hot, dim=1)

            loss = self.ce_losses[i](emotion_logits, target_indices)
            total_loss += loss
            start_idx += dim

        return total_loss / len(EMOTION_DIMS)  # Average loss across emotions


def calculate_standard_metrics(predictions, targets, raw_intensities):
    """
    Calculate standard classification metrics per emotion head.
    Args:
        predictions: Model logits [batch_size, sum(EMOTION_DIMS)]
        targets: One-hot encoded groundtruth [batch_size, sum(EMOTION_DIMS)] - NOT USED directly
        raw_intensities: Raw intensity values [batch_size, len(EMOTION_NAMES)]

    Returns:
        Dictionary with per-emotion accuracy, F1, MAE, etc.
    """
    # Convert tensors to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().detach().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().detach().numpy()
    if isinstance(raw_intensities, torch.Tensor):
        raw_intensities = raw_intensities.cpu().detach().numpy()

    pred_intensities = []
    start_idx = 0
    for dim in EMOTION_DIMS:
        emotion_logits = predictions[:, start_idx:start_idx + dim]
        pred_intensity = np.argmax(emotion_logits, axis=1)
        pred_intensities.append(pred_intensity)
        start_idx += dim
    pred_intensities = np.column_stack(pred_intensities)

    metrics = {'per_emotion': {}, 'overall': {}}
    all_y_true = []
    all_y_pred = []

    for i, emotion in enumerate(EMOTION_NAMES):
        y_pred = pred_intensities[:, i]
        y_true = raw_intensities[:, i]
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        cm = confusion_matrix(y_true, y_pred, labels=range(EMOTION_DIMS[i]))

        metrics['per_emotion'][emotion] = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'mae': float(mae),
            'rmse': float(rmse),
            'kappa': float(kappa),
            'confusion_matrix': cm.tolist()
        }

    # Calculate overall metrics (simple average)
    overall_acc = np.mean([m['accuracy'] for m in metrics['per_emotion'].values()])
    overall_f1_macro = np.mean([m['f1_macro'] for m in metrics['per_emotion'].values()])
    overall_f1_weighted = np.mean([m['f1_weighted'] for m in metrics['per_emotion'].values()])
    overall_mae = np.mean([m['mae'] for m in metrics['per_emotion'].values()])
    overall_rmse = np.mean([m['rmse'] for m in metrics['per_emotion'].values()])
    overall_kappa = np.mean([m['kappa'] for m in metrics['per_emotion'].values()])

    metrics['overall'] = {
        'accuracy': overall_acc,
        'f1_macro': overall_f1_macro,
        'f1_weighted': overall_f1_weighted,
        'mae': overall_mae,
        'rmse': overall_rmse,
        'kappa': overall_kappa
    }

    # Add compatibility keys for the old calculate_metrics wrapper if needed
    metrics["f1"] = overall_f1_macro
    metrics["per_class"] = {}
    for i, emotion in enumerate(EMOTION_NAMES):
        metrics["per_class"][f"class_{i}"] = metrics['per_emotion'][emotion]

    return metrics


def calculate_metrics(predictions, targets, raw_intensities=None):
    """
    Calculate metrics for Task C (wrapper for backward compatibility)

    This function maintains compatibility with the original API while 
    delegating to the new ordinal metrics function.
    """
    if raw_intensities is None:
        # If no raw intensities provided, fall back to original behavior
        logger.warning("Raw intensities not provided. Metrics may be incorrect.")
        # Create dummy raw intensities
        raw_intensities = torch.zeros((predictions.shape[0], len(EMOTION_NAMES)))

    # Use the new ordinal metrics function
    metrics = calculate_standard_metrics(predictions, targets, raw_intensities)

    # Reformat for backward compatibility
    compat_metrics = {
        "accuracy": metrics['overall']['accuracy'],
        "precision": metrics['overall']['f1_macro'],  # Using F1 as a proxy
        "recall": metrics['overall']['f1_macro'],     # Using F1 as a proxy
        "f1": metrics['overall']['f1_macro'],
        "per_class": {}
    }

    # Convert per-emotion metrics to per-class format
    for i, emotion in enumerate(EMOTION_NAMES):
        compat_metrics["per_class"][f"class_{i}"] = {
            "precision": metrics['per_emotion'][emotion]['f1_macro'],
            "recall": metrics['per_emotion'][emotion]['f1_macro'],
            "f1": metrics['per_emotion'][emotion]['f1_macro']
        }

    return compat_metrics


def get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """Create a scheduler with linear warmup and cosine decay"""
    from transformers import get_scheduler
    return get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )


def train(
    train_dataset,
    val_dataset,
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    batch_size,
    device,
    model_save_path,
    early_stopping_patience=3,
    gradient_accumulation_steps=1,
    fp16_training=True,
    output_dir=None
):
    """Training loop with validation and early stopping"""
    # Initialize data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if fp16_training else None

    # Initialize early stopping variables
    best_val_metric = 0.0  # Using Accuracy or F1 now (higher is better)
    early_stopping_counter = 0
    best_epoch = -1
    # Define the metric to use for saving the best model
    save_metric = 'f1_macro'  # Options: 'accuracy', 'f1_macro', 'f1_weighted', 'kappa', 'mae', 'rmse'
    lower_is_better = save_metric in ['mae', 'rmse']
    if lower_is_better:
        best_val_metric = float('inf')

    # Create lists to store loss history for visualization
    train_losses = []
    val_losses = []
    train_f1_scores = []
    val_f1_scores = []
    epochs = []

    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        epochs.append(epoch + 1)

        # Training phase
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_targets = []
        all_train_intensities = []

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch["image"].to(device)
            text_data = {
                "input_ids": batch["text"]["input_ids"].to(device),
                "attention_mask": batch["text"]["attention_mask"].to(device)
            }
            targets = batch["labels"].to(device)
            intensities = batch["intensity"].to(device)

            # Forward pass with mixed precision if enabled
            if fp16_training:
                with autocast('cuda'):
                    # Apply mixup during training
                    if model.training:
                        images, targets = mixup_data(images, targets, alpha=0.2, device=device)

                    outputs = model(images, text_data)
                    loss = criterion(outputs, targets)
                    # Normalize loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
            else:
                # Standard training without mixed precision
                # Apply mixup during training
                if model.training:
                    images, targets = mixup_data(images, targets, alpha=0.2, device=device)

                outputs = model(images, text_data)
                loss = criterion(outputs, targets)
                loss = loss / gradient_accumulation_steps

                loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()

            # Track loss and predictions
            train_loss += loss.item() * gradient_accumulation_steps

            # Store predictions, targets, and intensities
            all_train_preds.append(outputs.detach().cpu())
            all_train_targets.append(targets.cpu())
            all_train_intensities.append(intensities.cpu())

            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item() * gradient_accumulation_steps
            })

        # Calculate training metrics
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Carefully combine tensors
        all_train_preds = torch.cat(all_train_preds)
        all_train_targets = torch.cat(all_train_targets)
        all_train_intensities = torch.cat(all_train_intensities)

        # Log shapes before metric calculation for debugging
        logger.info(f"Training shapes - predictions: {all_train_preds.shape}, targets: {all_train_targets.shape}, intensities: {all_train_intensities.shape}")

        # Calculate training metrics using the detailed ordinal function
        train_metrics = calculate_standard_metrics(all_train_preds, all_train_targets, all_train_intensities)
        train_f1_scores.append(train_metrics['overall']['f1_macro'])  # Use overall macro F1

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_targets = []
        all_val_intensities = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
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
                loss = criterion(outputs, targets)

                # Track loss and predictions
                val_loss += loss.item()

                # Store predictions, targets, and intensities
                all_val_preds.append(outputs.cpu())
                all_val_targets.append(targets.cpu())
                all_val_intensities.append(intensities.cpu())

        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Carefully combine tensors
        all_val_preds = torch.cat(all_val_preds)
        all_val_targets = torch.cat(all_val_targets)
        all_val_intensities = torch.cat(all_val_intensities)

        # Log shapes before metric calculation for debugging
        logger.info(f"Validation shapes - predictions: {all_val_preds.shape}, targets: {all_val_targets.shape}, intensities: {all_val_intensities.shape}")

        # Calculate validation metrics using the detailed ordinal function
        val_metrics = calculate_standard_metrics(all_val_preds, all_val_targets, all_val_intensities)
        val_f1_scores.append(val_metrics['overall']['f1_macro'])  # Use overall macro F1

        # Log metrics
        train_f1_overall = train_metrics['overall'].get('f1_macro', float('nan'))
        val_f1_overall = val_metrics['overall'].get('f1_macro', float('nan'))
        val_mae_overall = val_metrics['overall'].get('mae', float('nan'))
        val_acc_overall = val_metrics['overall'].get('accuracy', float('nan'))
        logger.info(
            f"Train Loss: {train_loss:.4f}, Train F1: {train_f1_overall:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val F1: {val_f1_overall:.4f}, "
            f"Val Acc: {val_acc_overall:.4f}, Val MAE: {val_mae_overall:.4f}"
        )

        # Get the current validation metric for comparison
        current_val_metric = val_metrics['overall'].get(save_metric, float('inf') if lower_is_better else 0.0)

        # Save model based on the chosen metric
        improved = (current_val_metric < best_val_metric) if lower_is_better else (current_val_metric > best_val_metric)

        if improved:
            best_val_metric = current_val_metric
            best_epoch = epoch + 1
            logger.info(f"New best model found! Best Val {save_metric.upper()}: {best_val_metric:.4f}")
            logger.info(f"Saving model to: {model_save_path}")

            # Save checkpoint with detailed info
            # Convert metrics to basic types for safe loading
            safe_train_metrics = {k: float(v) if isinstance(v, (torch.Tensor, np.number)) else v for k, v in train_metrics['overall'].items()}
            safe_val_metrics = {k: float(v) if isinstance(v, (torch.Tensor, np.number)) else v for k, v in val_metrics['overall'].items()}

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'best_val_metric_name': save_metric,
                'best_val_metric_value': float(best_val_metric),
                'train_metrics': safe_train_metrics,
                'val_metrics': safe_val_metrics,
                'save_time': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            torch.save(checkpoint, model_save_path)
            logger.info(f"Checkpoint saved successfully at epoch {best_epoch}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            logger.info(f"No improvement in validation {save_metric.upper()}. Best: {best_val_metric:.4f}, Current: {current_val_metric:.4f}")
            logger.info(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
            if early_stopping_counter >= early_stopping_patience:
                logger.info("Early stopping triggered")
                break

    # At the end of training, load the best model for final evaluation
    logger.info(f"Loading best model from epoch {best_epoch} (based on {save_metric}) for final evaluation...")
    checkpoint = torch.load(model_save_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    loaded_epoch = checkpoint.get('epoch', -1) + 1
    loaded_best_val_metric = checkpoint.get('best_val_metric_value', float('nan'))
    loaded_metric_name = checkpoint.get('best_val_metric_name', 'unknown')
    save_time = checkpoint.get('save_time', 'unknown')
    logger.info(f"Loaded checkpoint from epoch {loaded_epoch} with best val {loaded_metric_name}: {loaded_best_val_metric:.4f}")
    logger.info(f"Checkpoint was saved at: {save_time}")

    # Plot loss evolution
    if output_dir:
        try:
            import matplotlib.pyplot as plt

            # Set up the figure with two subplots - one for loss, one for F1
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Plot training and validation loss
            ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
            ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
            ax1.set_title('Loss Evolution')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)

            # Plot training and validation F1 scores
            ax2.plot(epochs, train_f1_scores, 'b-', label='Training F1')
            ax2.plot(epochs, val_f1_scores, 'r-', label='Validation F1')
            ax2.set_title('F1 Score Evolution')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('F1 Score')
            ax2.legend()
            ax2.grid(True)

            # Save the figure
            plots_dir = Path(output_dir) / "plots"
            plots_dir.mkdir(exist_ok=True, parents=True)
            fig.savefig(plots_dir / "loss_evolution.png")
            plt.close(fig)

            logger.info(f"Loss evolution plot saved to {plots_dir / 'loss_evolution.png'}")
        except Exception as e:
            logger.error(f"Error creating loss evolution plot: {e}")

    return best_val_metric


def main(args):
    """Main function to run training"""
    # Set seed for reproducibility
    set_seed(args.seed)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model save directory
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = model_dir / f"{MODEL_NAME}.pt"

    # Dataset split configuration
    splits_dir = Path(args.output_dir) / "splits" if args.output_dir else None
    if splits_dir:
        splits_dir.mkdir(exist_ok=True, parents=True)
        splits_file = splits_dir / "dataset_splits.json"
    else:
        splits_file = None

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = MemeDataset(
        split="train",
        kaggle_dataset_path=args.kaggle_dataset_path,
        fixed_split_file=args.fixed_split_file,
        save_splits_to=str(splits_file) if splits_file else None
    )
    val_dataset = MemeDataset(
        split="val",
        kaggle_dataset_path=args.kaggle_dataset_path,
        fixed_split_file=args.fixed_split_file
    )
    test_dataset = MemeDataset(
        split="test",
        kaggle_dataset_path=args.kaggle_dataset_path,
        fixed_split_file=args.fixed_split_file
    )
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # Initialize model
    logger.info("Initializing model...")
    model = MemeEmotionModel()
    model = model.to(device)

    # Initialize loss function with weights moved to the correct device
    # Placeholder weights for now, replace with actual calculation if needed
    # TODO: Implement actual class weight calculation if desired
    class_weights_list = [torch.tensor([1.0] * dim, device=device) for dim in EMOTION_DIMS]
    criterion = PerEmotionCrossEntropyLoss(class_weights=class_weights_list)

    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Initialize learning rate scheduler
    num_training_steps = len(train_dataset) // args.batch_size * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_lr_scheduler(
        optimizer, num_warmup_steps, num_training_steps
    )

    # Print training config
    logger.info("Training configuration:")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Warmup ratio: {args.warmup_ratio}")
    logger.info(
        f"  Gradient accumulation steps: {args.gradient_accumulation_steps}"
    )
    logger.info(f"  FP16 training: {args.fp16}")

    # Start training
    logger.info("Starting training...")
    best_val_metric = train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        model_save_path=model_save_path,
        early_stopping_patience=args.patience,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16_training=args.fp16,
        output_dir=args.output_dir
    )

    logger.info(
        f"Training completed. Best validation metric achieved: {best_val_metric:.4f}")

    # Evaluate on test set
    logger.info("Evaluating final model on test set...")

    # Load best model from checkpoint
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Run evaluation on test set
    test_loss = 0.0
    all_test_preds = []
    all_test_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Evaluation"):
            # Move data to device
            images = batch["image"].to(device)
            text_data = {
                "input_ids": batch["text"]["input_ids"].to(device),
                "attention_mask": batch["text"]["attention_mask"].to(device)
            }
            targets = batch["labels"].to(device)

            # Forward pass
            outputs = model(images, text_data)
            loss = criterion(outputs, targets)

            # Track loss and predictions
            test_loss += loss.item()

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(outputs).detach()

            # Store predictions and targets
            all_test_preds.append(preds.cpu())
            all_test_targets.append(targets.cpu())

    # Calculate test metrics
    test_loss /= len(test_loader)

    # Combine predictions and targets
    all_test_preds = torch.cat(all_test_preds)
    all_test_targets = torch.cat(all_test_targets)

    # Log shapes before metric calculation for debugging
    logger.info(f"Test shapes - predictions: {all_test_preds.shape}, targets: {all_test_targets.shape}")

    # Calculate test metrics
    test_metrics = calculate_metrics(all_test_preds, all_test_targets)

    # Log test metrics
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")

    # Log per-class metrics
    logger.info("Per-class test metrics:")
    for class_idx, metrics in test_metrics['per_class'].items():
        logger.info(f"  {class_idx}: Precision={metrics['precision']:.4f}, "
                    f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

    # Save test metrics in the model checkpoint
    checkpoint['test_loss'] = test_loss
    checkpoint['test_metrics'] = test_metrics

    # Save updated checkpoint
    torch.save(checkpoint, model_save_path)

    # Print the full absolute path of saved model
    print("\n" + "=" * 50)
    print(f"Model saved at: {os.path.abspath(model_save_path)}")
    print("=" * 50 + "\n")

    # Also save a separate JSON file with metrics
    import json
    metrics_path = Path(args.model_dir) / f"{MODEL_NAME}_metrics.json"

    # Convert metrics for JSON serialization
    serializable_metrics = {
        'train': {k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                  for k, v in checkpoint['train_metrics'].items() if k != 'per_class'},
        'validation': {k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                       for k, v in checkpoint['val_metrics'].items() if k != 'per_class'},
        'test': {k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                 for k, v in test_metrics.items() if k != 'per_class'},
    }

    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)

    logger.info(f"Test metrics saved to {metrics_path}")

    # Create a comprehensive model card with all metrics
    from datetime import datetime
    model_card = {
        "model_name": MODEL_NAME,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_parameters": vars(args),
        "metrics": {
            "training": {
                "loss": float(checkpoint['train_loss']),
                "accuracy": float(checkpoint['train_metrics']["accuracy"]),
                "precision": float(checkpoint['train_metrics']["precision"]),
                "recall": float(checkpoint['train_metrics']["recall"]),
                "f1": float(checkpoint['train_metrics']["f1"])
            },
            "validation": {
                "loss": float(checkpoint['val_loss']),
                "accuracy": float(checkpoint['val_metrics']["accuracy"]),
                "precision": float(checkpoint['val_metrics']["precision"]),
                "recall": float(checkpoint['val_metrics']["recall"]),
                "f1": float(checkpoint['val_metrics']["f1"])
            },
            "test": {
                "loss": float(test_loss),
                "accuracy": float(test_metrics["accuracy"]),
                "precision": float(test_metrics["precision"]),
                "recall": float(test_metrics["recall"]),
                "f1": float(test_metrics["f1"])
            }
        },
        "emotion_classes": EMOTION_CLASSES,
        "per_class_test_metrics": {
            emotion: {
                "precision": float(test_metrics["per_class"][f"class_{i}"]["precision"]),
                "recall": float(test_metrics["per_class"][f"class_{i}"]["recall"]),
                "f1": float(test_metrics["per_class"][f"class_{i}"]["f1"])
            }
            for i, emotion in enumerate(EMOTION_CLASSES)
        }
    }

    # Save the model card
    model_card_path = Path(args.output_dir) / f"{MODEL_NAME}_card.json" if args.output_dir else Path(f"{MODEL_NAME}_card.json")
    with open(model_card_path, 'w') as f:
        json.dump(model_card, f, indent=2)

    logger.info(f"Model card saved to {model_card_path}")

    return best_val_metric


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the meme emotion recognition model")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--model_dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kaggle_dataset_path", type=str,
                        help="Path to Kaggle dataset")
    parser.add_argument("--output_dir", type=str,
                        help="Directory to save outputs")
    parser.add_argument("--fixed_split_file", type=str,
                        help="Path to fixed split file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
