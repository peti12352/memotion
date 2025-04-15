import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score
)

from .dataset import MemeDataset
from .model import MemeEmotionModel
from .config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, NUM_WORKERS,
    MODELS_DIR, MODEL_NAME, NUM_CLASSES, EMOTION_CLASSES
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


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in multi-label classification

    Args:
        alpha: Controls the balance between classes (default: 0.25)
        gamma: Controls the focus on hard examples (default: 2.0)
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        return focal_loss.mean()


def calculate_metrics(predictions, targets, threshold=0.5):
    """Calculate metrics for multi-label classification"""
    # Convert tensors to numpy arrays if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().detach().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().detach().numpy()

    # Ensure 2D arrays with shape (n_samples, n_classes)
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(1, -1)
    if len(targets.shape) == 1:
        targets = targets.reshape(1, -1)
    if len(targets.shape) > 2:
        targets = targets.reshape(targets.shape[0], -1)
    if len(predictions.shape) > 2:
        predictions = predictions.reshape(predictions.shape[0], -1)

    # Apply threshold to get binary predictions
    binary_preds = (predictions > threshold).astype(int)

    # Ensure targets are binary (0 or 1)
    binary_targets = (targets > 0).astype(int)

    try:
        # Calculate metrics
        precision = precision_score(
            binary_targets, binary_preds, average='samples', zero_division=0)
        recall = recall_score(
            binary_targets, binary_preds, average='samples', zero_division=0)
        f1 = f1_score(
            binary_targets, binary_preds, average='samples', zero_division=0)
        accuracy = accuracy_score(binary_targets, binary_preds)

        # Calculate per-class metrics
        per_class_metrics = {}
        for i in range(targets.shape[1]):
            per_class_metrics[f"class_{i}"] = {
                "precision": precision_score(
                    binary_targets[:, i], binary_preds[:, i], zero_division=0),
                "recall": recall_score(
                    binary_targets[:, i], binary_preds[:, i], zero_division=0),
                "f1": f1_score(
                    binary_targets[:, i], binary_preds[:, i], zero_division=0)
            }

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "per_class": per_class_metrics
        }

    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        # Return default metrics to avoid breaking the training loop
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "per_class": {}
        }


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
    scaler = GradScaler('cuda') if fp16_training else None

    # Initialize early stopping variables
    best_val_f1 = 0.0
    early_stopping_counter = 0

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

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch["image"].to(device)
            text_data = {
                "input_ids": batch["text"]["input_ids"].to(device),
                "attention_mask": batch["text"]["attention_mask"].to(device)
            }
            targets = batch["labels"].to(device)

            # Forward pass with mixed precision if enabled
            if fp16_training:
                with autocast('cuda'):
                    outputs = model(images, text_data)
                    loss = criterion(outputs, targets)
                    # Normalize loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
            else:
                # Standard training without mixed precision
                outputs = model(images, text_data)
                loss = criterion(outputs, targets)
                loss = loss / gradient_accumulation_steps

                loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()

            # Track loss and predictions
            train_loss += loss.item() * gradient_accumulation_steps

            # Apply sigmoid to get probabilities and ensure correct shape
            preds = torch.sigmoid(outputs).detach()
            targets_detached = targets.detach()

            # Store predictions and targets with consistent shapes
            all_train_preds.append(preds.cpu())
            all_train_targets.append(targets_detached.cpu())

            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item() * gradient_accumulation_steps
            })

        # Calculate training metrics
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Carefully combine and reshape predictions and targets
        all_train_preds = torch.cat(all_train_preds)
        all_train_targets = torch.cat(all_train_targets)

        # Calculate training metrics without excessive logging
        train_metrics = calculate_metrics(all_train_preds, all_train_targets)
        train_f1_scores.append(train_metrics['f1'])

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
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
                val_loss += loss.item()

                # Apply sigmoid to get probabilities and ensure correct shape
                preds = torch.sigmoid(outputs).detach()
                targets_detached = targets.detach()

                # Store predictions and targets with consistent shapes
                all_val_preds.append(preds.cpu())
                all_val_targets.append(targets_detached.cpu())

        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Carefully combine and reshape predictions and targets
        all_val_preds = torch.cat(all_val_preds)
        all_val_targets = torch.cat(all_val_targets)

        # Calculate validation metrics
        val_metrics = calculate_metrics(all_val_preds, all_val_targets)
        val_f1_scores.append(val_metrics['f1'])

        # Log metrics
        logger.info(
            f"Train Loss: {train_loss:.4f}, "
            f"Train F1: {train_metrics['f1']:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        # Save model if validation F1 improves
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            logger.info(f"Saving best model with val F1: {best_val_f1:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_f1': val_metrics['f1'],
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_f1_scores': train_f1_scores,
                'val_f1_scores': val_f1_scores
            }, model_save_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            logger.info(
                f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}"
            )
            if early_stopping_counter >= early_stopping_patience:
                logger.info("Early stopping triggered")
                break

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

    return best_val_f1


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
    model = MemeEmotionModel(num_classes=NUM_CLASSES)
    model = model.to(device)

    # Initialize loss function
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

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
    logger.info(f"  Focal loss alpha: {args.focal_alpha}")
    logger.info(f"  Focal loss gamma: {args.focal_gamma}")

    # Start training
    logger.info("Starting training...")
    best_val_f1 = train(
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
        f"Training completed with best validation F1: {best_val_f1:.4f}")

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
            all_test_targets.append(targets.detach().cpu())

    # Calculate test metrics
    test_loss /= len(test_loader)

    # Combine predictions and targets
    all_test_preds = torch.cat(all_test_preds)
    all_test_targets = torch.cat(all_test_targets)

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

    return best_val_f1


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
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
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
