import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import numpy as np
from transformers import AutoTokenizer, CLIPProcessor
from .config import DATA_DIR, MAX_TEXT_LENGTH, IMAGE_SIZE
import logging

# Configure logging if not already configured
logger = logging.getLogger(__name__)


class MemeDataset(Dataset):
    def __init__(
        self,
        data_dir=None,
        labels_file=None,
        split="train",
        transform=None,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        kaggle_dataset_path=None
    ):
        """Initialize a MemeDataset instance.

        Args:
            data_dir (str): Directory containing the images
            labels_file (str): Path to the CSV file with labels
            split (str): One of 'train', 'val', or 'test'
            transform: Optional image transforms
            val_ratio (float): Percentage of data to use for validation
            test_ratio (float): Percentage of data to use for testing
            seed (int): Random seed for reproducibility
            kaggle_dataset_path (str, optional): Path to Kaggle dataset
                from kagglehub. If provided, uses this instead of DATA_DIR
        """
        self.split = split
        self.transform = transform

        # Load tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        # Initialize CLIP processor for image processing
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32")

        # Check if kaggle_dataset_path is provided
        if kaggle_dataset_path is not None:
            self.base_dir = kaggle_dataset_path

            # For kagglehub downloads, check for
            # memotion_dataset_7k subdirectory
            memotion_path = os.path.join(self.base_dir, "memotion_dataset_7k")
            if os.path.exists(memotion_path):
                self.base_dir = memotion_path

            # Set path to images directory and labels file
            self.images_dir = os.path.join(self.base_dir, "images")
            self.labels_path = os.path.join(self.base_dir, "labels.csv")

            # Print paths for debugging
            print(f"Base directory: {self.base_dir}")
            print(f"Images directory: {self.images_dir}")
            print(f"Labels path: {self.labels_path}")

            # Verify paths exist
            if not os.path.exists(self.images_dir):
                raise FileNotFoundError(
                    f"Images directory not found at {self.images_dir}"
                )

            if not os.path.exists(self.labels_path):
                raise FileNotFoundError(
                    f"Labels file not found at {self.labels_path}"
                )
        else:
            # Use default paths
            if data_dir is None:
                data_dir = os.path.join(DATA_DIR, "memotion")
            self.base_dir = data_dir
            self.images_dir = os.path.join(self.base_dir, "images")
            self.labels_path = (
                labels_file or os.path.join(self.base_dir, "labels.csv")
            )

            # Verify paths exist
            if not os.path.exists(self.images_dir):
                raise FileNotFoundError(
                    f"Images directory not found at {self.images_dir}"
                )

            if not os.path.exists(self.labels_path):
                raise FileNotFoundError(
                    f"Labels file not found at {self.labels_path}"
                )

        # Load and preprocess labels
        self.full_df = pd.read_csv(self.labels_path)

        # Create train/val/test split if not already done
        np.random.seed(seed)
        self._create_splits(val_ratio, test_ratio)

        # Select the appropriate split
        if split == "train":
            self.labels_df = self.train_df
        elif split == "val":
            self.labels_df = self.val_df
        elif split == "test":
            self.labels_df = self.test_df
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be one of 'train', 'val', "
                f"or 'test'"
            )

        print(f"Loaded {len(self.labels_df)} samples for {split} split")

    def _create_splits(self, val_ratio, test_ratio):
        """Create train/val/test splits from full dataset"""
        # Convert text labels to numerical values
        self._preprocess_labels()

        # Get indices for all samples
        indices = np.arange(len(self.full_df))

        # Shuffle indices
        np.random.shuffle(indices)

        # Calculate split sizes
        test_size = int(len(indices) * test_ratio)
        val_size = int(len(indices) * val_ratio)
        train_size = len(indices) - test_size - val_size

        # Create splits
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Create dataframes for each split
        self.train_df = self.full_df.iloc[train_indices].reset_index(drop=True)
        self.val_df = self.full_df.iloc[val_indices].reset_index(drop=True)
        self.test_df = self.full_df.iloc[test_indices].reset_index(drop=True)

    def _preprocess_labels(self):
        """Convert label text to appropriate numerical values"""
        # Map overall sentiment to -1, 0, 1
        sentiment_map = {
            'negative': -1, 'very negative': -1,
            'neutral': 0,
            'positive': 1, 'very positive': 1
        }

        # Convert text columns to lowercase for consistent mapping
        columns_to_convert = [
            'humour', 'sarcasm', 'offensive',
            'motivational', 'overall_sentiment'
        ]
        for col in columns_to_convert:
            if col in self.full_df.columns:
                self.full_df[col] = self.full_df[col].str.lower()

        # Map overall sentiment
        if 'overall_sentiment' in self.full_df.columns:
            self.full_df['sentiment'] = self.full_df['overall_sentiment'].map(
                sentiment_map)

        # Create binary classification columns for Task B
        self.full_df['amusement'] = self.full_df['humour'].apply(
            lambda x: 0 if x == 'not funny' else 1)

        self.full_df['sarcasm'] = self.full_df['sarcasm'].apply(
            lambda x: 0 if x == 'not sarcastic' else 1)

        self.full_df['offense'] = self.full_df['offensive'].apply(
            lambda x: 0 if x == 'not offensive' else 1)

        self.full_df['motivation'] = self.full_df['motivational'].apply(
            lambda x: 0 if x == 'not motivational' else 1)

        # Add neutral class (inverse of any other emotion)
        has_emotion = (self.full_df['amusement'] |
                       self.full_df['sarcasm'] |
                       self.full_df['offense'] |
                       self.full_df['motivation'])
        self.full_df['neutral'] = (~has_emotion).astype(int)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        """Get a single sample from the dataset"""
        # Get image name and construct full path
        img_name = self.labels_df.iloc[idx]["image_name"]

        # Verify image path formatting and existence
        img_path = (
            os.path.join(self.images_dir, img_name)
            if isinstance(img_name, str)
            else img_name
        )

        if not os.path.exists(img_path):
            logger.warning(
                "Image not found: %s. Possible issues: "
                "1) Run download_data.py first, "
                "2) Check file permissions. "
                "Using placeholder image.",
                img_path
            )
            # Placeholder black image
            img = Image.new('RGB', IMAGE_SIZE, color='black')
        else:
            img = Image.open(img_path).convert("RGB")

        # Apply any custom transforms if specified
        if self.transform:
            img = self.transform(img)

        # Get text - use corrected text if available, else use OCR text
        if ("text_corrected" in self.labels_df.columns and
                pd.notna(self.labels_df.iloc[idx]["text_corrected"])):
            text = self.labels_df.iloc[idx]["text_corrected"]
        elif ("text_ocr" in self.labels_df.columns and
              pd.notna(self.labels_df.iloc[idx]["text_ocr"])):
            text = self.labels_df.iloc[idx]["text_ocr"]
        else:
            text = ""  # Fallback for images without text

        # Process image with CLIP processor
        image_inputs = self.clip_processor(
            images=img,
            return_tensors="pt",
            padding=True
        )

        # Process text with tokenizer
        text_inputs = self.tokenizer(
            text,
            max_length=MAX_TEXT_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Get labels for multi-label classification
        labels = torch.tensor([
            self.labels_df.iloc[idx]["amusement"],
            self.labels_df.iloc[idx]["sarcasm"],
            self.labels_df.iloc[idx]["offense"],
            self.labels_df.iloc[idx]["motivation"],
            self.labels_df.iloc[idx]["neutral"]
        ], dtype=torch.float)

        return {
            "image": image_inputs["pixel_values"].squeeze(0),
            "text": {
                "input_ids": text_inputs["input_ids"].squeeze(0),
                "attention_mask": text_inputs["attention_mask"].squeeze(0)
            },
            "labels": labels
        }
