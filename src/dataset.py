import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import numpy as np
from transformers import AutoTokenizer, CLIPProcessor
from .config import DATA_DIR, MAX_TEXT_LENGTH, IMAGE_SIZE, EMOTION_SCALES, EMOTION_NAMES
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
        kaggle_dataset_path=None,
        fixed_split_file=None,
        save_splits_to=None
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
            fixed_split_file (str, optional): Path to a JSON file with
                predefined train/val/test splits for reproducibility
            save_splits_to (str, optional): Path to save generated splits
                for future reproducibility
        """
        self.split = split
        self.transform = transform
        self.fixed_split_file = fixed_split_file
        self.save_splits_to = save_splits_to

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
        self._create_splits(val_ratio, test_ratio, self.fixed_split_file)

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

    def _create_splits(self, val_ratio, test_ratio, fixed_split_file=None):
        """Create train/val/test splits from full dataset

        Args:
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            fixed_split_file: Optional path to a JSON file containing predefined splits
        """
        # Convert text labels to numerical values
        self._preprocess_labels()

        # If fixed split file is provided, use it
        if fixed_split_file and os.path.exists(fixed_split_file):
            logger.info(f"Using fixed splits from {fixed_split_file}")
            with open(fixed_split_file, 'r') as f:
                import json
                splits = json.load(f)

            train_indices = splits['train']
            val_indices = splits['val']
            test_indices = splits['test']

            logger.info(f"Loaded fixed splits - Train: {len(train_indices)}, "
                        f"Val: {len(val_indices)}, Test: {len(test_indices)}")
        else:
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

            # Save splits for reproducibility if requested
            if hasattr(self, 'save_splits_to') and self.save_splits_to:
                save_path = self.save_splits_to
                logger.info(f"Saving splits to {save_path}")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w') as f:
                    import json
                    json.dump({
                        'train': train_indices.tolist(),
                        'val': val_indices.tolist(),
                        'test': test_indices.tolist()
                    }, f)

        # Create dataframes for each split
        self.train_df = self.full_df.iloc[train_indices].reset_index(drop=True)
        self.val_df = self.full_df.iloc[val_indices].reset_index(drop=True)
        self.test_df = self.full_df.iloc[test_indices].reset_index(drop=True)

    def _preprocess_labels(self):
        """Convert label text to appropriate numerical values for Task C (emotion intensity)"""
        # Convert text columns to lowercase for consistent mapping
        columns_to_convert = ['humour', 'sarcasm', 'offensive', 'motivational', 'overall_sentiment']
        for col in columns_to_convert:
            if col in self.full_df.columns:
                self.full_df[col] = self.full_df[col].str.lower()

        # Task C - Map humour levels (0-3)
        humor_map = {
            'not funny': 0,
            'funny': 1,
            'very funny': 2,
            'hilarious': 3
        }
        self.full_df['humour_intensity'] = self.full_df['humour'].map(
            humor_map).fillna(0).astype(int)

        # Task C - Map sarcasm levels (0-3)
        sarcasm_map = {
            'not sarcastic': 0,
            'general': 1,
            'twisted meaning': 2,
            'very twisted': 3
        }
        self.full_df['sarcasm_intensity'] = self.full_df['sarcasm'].map(
            sarcasm_map).fillna(0).astype(int)

        # Task C - Map offensive levels (0-3)
        offensive_map = {
            'not offensive': 0,
            'slight': 1,
            'very offensive': 2,
            'hateful offensive': 3
        }
        self.full_df['offensive_intensity'] = self.full_df['offensive'].map(
            offensive_map).fillna(0).astype(int)

        # Task C - Map motivational (0-1)
        motivational_map = {
            'not motivational': 0,
            'motivational': 1
        }
        self.full_df['motivational_intensity'] = self.full_df['motivational'].map(
            motivational_map).fillna(0).astype(int)

        # Create one-hot encoded columns for each intensity level
        # This allows for ordinal classification where we predict probabilities for each level
        for emotion in EMOTION_NAMES:
            intensity_col = f"{emotion}_intensity"
            max_scale = EMOTION_SCALES[emotion]

            # Create one-hot column for each level of intensity for this emotion
            for level in range(max_scale):
                col_name = f"{emotion}_{level}"
                self.full_df[col_name] = (self.full_df[intensity_col] == level).astype(int)

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

        try:
            # Try to open and convert the image
            img = Image.open(img_path)

            # Handle transparency properly by ensuring RGBA is converted to RGB
            if img.mode == 'RGBA':
                # Create a white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                # Paste the image with transparency onto the background
                # Use alpha channel as mask
                background.paste(img, mask=img.split()[3])
                img = background
            elif 'transparency' in img.info:
                # For palette images with transparency
                img = img.convert('RGBA')
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            else:
                img = img.convert("RGB")

            # Apply any custom transforms if specified
            if self.transform:
                img = self.transform(img)
        except (OSError, IOError) as e:
            logger.warning(
                f"Error loading image {img_path}: {str(e)}. "
                "Using placeholder image."
            )
            # Create a placeholder black image
            img = Image.new('RGB', IMAGE_SIZE, color='black')
            if self.transform:
                img = self.transform(img)

        # Get text - use corrected text if available, else use OCR text
        if ("text_corrected" in self.labels_df.columns and pd.notna(self.labels_df.iloc[idx]["text_corrected"])):
            text = self.labels_df.iloc[idx]["text_corrected"]
        elif ("text_ocr" in self.labels_df.columns and pd.notna(self.labels_df.iloc[idx]["text_ocr"])):
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

        # For Task C, we create the labels by concatenating each intensity one-hot vector
        # For each emotion, we grab all of its one-hot encoded intensity levels and append them
        label_tensors = []
        for emotion in EMOTION_NAMES:
            for level in range(EMOTION_SCALES[emotion]):
                col_name = f"{emotion}_{level}"
                label_tensors.append(float(self.labels_df.iloc[idx][col_name]))

        # Convert to tensor - shape is (sum of all possible intensity levels)
        labels = torch.tensor(label_tensors, dtype=torch.float32)

        # Also include raw intensity levels for evaluation metrics
        intensity_values = [
            self.labels_df.iloc[idx][f"{emotion}_intensity"]
            for emotion in EMOTION_NAMES
        ]
        intensity = torch.tensor(intensity_values, dtype=torch.long)

        return {
            "image": image_inputs["pixel_values"].squeeze(0),
            "text": {
                "input_ids": text_inputs["input_ids"].squeeze(0),
                "attention_mask": text_inputs["attention_mask"].squeeze(0)
            },
            "labels": labels,
            "intensity": intensity  # Raw intensity values for evaluation
        }
