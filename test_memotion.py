"""
Simple test script to verify the Memotion dataset implementation
"""
import matplotlib.pyplot as plt
import torch
import numpy as np

# Import our modules
from src.dataset import MemeDataset
from src.config import DATA_DIR


def display_sample(sample, idx, title=None):
    """Display a single sample from the dataset"""
    # Get the image tensor and convert for display
    # CLIP returns images in the format [3, height, width]
    image_tensor = sample["image"]

    # Convert to numpy for matplotlib
    if torch.is_tensor(image_tensor):
        # Normalize to [0, 1] range for display
        image_np = image_tensor.numpy().transpose(1, 2, 0)
        # Apply normalization if needed
        image_np = np.clip(image_np, 0, 1)
    else:
        image_np = image_tensor

    # Get labels
    labels = sample["labels"]
    label_names = ["Amusement", "Sarcasm", "Offense", "Motivation", "Neutral"]
    label_str = ", ".join(
        [f"{name}: {val.item():.0f}" for name, val in zip(label_names, labels)]
    )

    # Display
    plt.figure(figsize=(10, 8))
    plt.imshow(image_np)
    plt.title(title or f"Sample {idx}")
    plt.axis('off')
    plt.figtext(0.5, 0.01, f"Labels: {label_str}", ha="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"sample_{idx}.png")
    plt.close()

    print(f"Sample {idx}:")
    print(f"- Labels: {label_str}")
    print(f"- Image shape: {image_tensor.shape}")
    print("-" * 50)


def main():
    """Test the MemeDataset implementation"""
    print(f"Data directory: {DATA_DIR}")

    # Check if dataset is downloaded
    memotion_dir = DATA_DIR / "memotion"
    if not (memotion_dir / "labels.csv").exists():
        print(f"Dataset not found at {memotion_dir / 'labels.csv'}")
        print("Please run the download_data.py script first.")
        return

    # Create dataset
    print("Creating dataset...")
    try:
        dataset = MemeDataset(split="train")

        # Print dataset info
        print(f"Dataset size: {len(dataset)}")

        # Display a few samples
        num_samples = min(5, len(dataset))
        for i in range(num_samples):
            sample = dataset[i]

            # Print some information about the sample
            print(f"Sample {i} image shape: {sample['image'].shape}")
            print(
                f"Sample {i} text shape: {sample['text']['input_ids'].shape}")

            display_sample(sample, i)

        print(f"Successfully visualized {num_samples} samples.")
        print("Sample images saved as sample_0.png, sample_1.png, etc.")

    except Exception as e:
        print(f"Error creating or using dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
