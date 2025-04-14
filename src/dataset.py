import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, CLIPProcessor
from .config import DATA_DIR, MAX_TEXT_LENGTH, IMAGE_SIZE


class MemeDataset(Dataset):
    def __init__(
        self, split="train", transform=None, val_ratio=0.15,
        test_ratio=0.15, seed=42
    ):
        """
        Dataset for loading and processing Memotion data

        Args:
            split (str): One of 'train', 'val', or 'test'
            transform: Optional image transforms
            val_ratio (float): Percentage of data to use for validation
            test_ratio (float): Percentage of data to use for testing
            seed (int): Random seed for reproducibility
        """
        self.split = split
        self.transform = transform

        # Load tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        # Initialize CLIP processor for image processing
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32")

        # Base path to images directory
        self.images_dir = DATA_DIR / "memotion" / "images"

        # Load all labels from single CSV file
        labels_path = DATA_DIR / "memotion" / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found at {labels_path}")

        # Load and preprocess labels
        self.full_df = pd.read_csv(labels_path)

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
        img_path = self.images_dir / img_name

        # Load and convert image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new("RGB", IMAGE_SIZE, (0, 0, 0))

        # Apply any custom transforms if specified
        if self.transform:
            image = self.transform(image)

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
            images=image,
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
