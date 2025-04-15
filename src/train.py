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
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score
)

from .dataset import MemeDataset
from .model import MemeEmotionModel
from .config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, NUM_WORKERS,
    MODELS_DIR, MODEL_NAME, NUM_CLASSES
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
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


def calculate_metrics(predictions, targets, threshold=0.5):
    """Calculate various metrics for multi-label classification"""
    # Convert predictions to binary using threshold
    predictions = predictions > threshold

    # Convert tensors to numpy arrays if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # Ensure both arrays are of type int and have the same shape
    predictions = predictions.astype(int)
    targets = targets.astype(int)

    # Calculate metrics for each class
    precision = precision_score(
        targets, predictions, average='samples', zero_division=0)
    recall = recall_score(targets, predictions,
                          average='samples', zero_division=0)
    f1 = f1_score(targets, predictions, average='samples', zero_division=0)

    # Calculate accuracy (exact match ratio)
    accuracy = accuracy_score(targets, predictions)

    # Calculate per-class metrics
    per_class_metrics = {}
    for i in range(targets.shape[1]):
        per_class_metrics[f"class_{i}"] = {
            "precision": precision_score(targets[:, i], predictions[:, i], zero_division=0),
            "recall": recall_score(targets[:, i], predictions[:, i], zero_division=0),
            "f1": f1_score(targets[:, i], predictions[:, i], zero_division=0)
        }

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class": per_class_metrics
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
    best_val_f1 = 0.0
    early_stopping_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")

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
                with autocast():
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
            all_train_preds.append(preds.cpu())
            all_train_targets.append(targets.cpu())

            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item() * gradient_accumulation_steps
            })

        # Calculate training metrics
        train_loss /= len(train_loader)
        all_train_preds = torch.cat(all_train_preds)
        all_train_targets = torch.cat(all_train_targets)

        # Ensure predictions and targets have shape (num_samples, num_classes)
        if len(all_train_preds.shape) == 1:
            # Add dimension at the end
            all_train_preds = all_train_preds.unsqueeze(-1)
        if len(all_train_targets.shape) == 1:
            # Add dimension at the end
            all_train_targets = all_train_targets.unsqueeze(-1)

        # Convert to numpy arrays
        all_train_preds = all_train_preds.numpy()
        all_train_targets = all_train_targets.numpy()

        train_metrics = calculate_metrics(all_train_preds, all_train_targets)

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
                preds = torch.sigmoid(outputs)
                all_val_preds.append(preds.cpu())
                all_val_targets.append(targets.cpu())

        # Calculate validation metrics
        val_loss /= len(val_loader)
        all_val_preds = torch.cat(all_val_preds)
        all_val_targets = torch.cat(all_val_targets)

        # Ensure predictions and targets have shape (num_samples, num_classes)
        if len(all_val_preds.shape) == 1:
            # Add dimension at the end
            all_val_preds = all_val_preds.unsqueeze(-1)
        if len(all_val_targets.shape) == 1:
            # Add dimension at the end
            all_val_targets = all_val_targets.unsqueeze(-1)

        # Convert to numpy arrays
        all_val_preds = all_val_preds.numpy()
        all_val_targets = all_val_targets.numpy()

        val_metrics = calculate_metrics(all_val_preds, all_val_targets)

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
                'val_metrics': val_metrics
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

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = MemeDataset(
        split="train",
        kaggle_dataset_path=args.kaggle_dataset_path
    )
    val_dataset = MemeDataset(
        split="val",
        kaggle_dataset_path=args.kaggle_dataset_path
    )
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

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
        fp16_training=args.fp16
    )

    logger.info(
        f"Training completed with best validation F1: {best_val_f1:.4f}")

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
