from src.config import DATA_DIR
from src.dataset import MemeDataset
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.transforms import ToPILImage

# Add the parent directory to the path so we can import modules properly
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Now import our modules


def display_sample(sample, idx, title=None):
    """Display a single sample from the dataset"""
    # Convert tensor to image
    to_pil = ToPILImage()
    image = to_pil(sample["image"].permute(2, 0, 1))

    # Get text
    text_ids = sample["text"]["input_ids"]

    # Get labels
    labels = sample["labels"]
    label_names = ["Amusement", "Sarcasm", "Offense", "Motivation", "Neutral"]
    label_str = ", ".join(
        [f"{name}: {val.item()}" for name, val in zip(label_names, labels)])

    # Display
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(title or f"Sample {idx}")
    plt.axis('off')
    plt.figtext(0.5, 0.01, f"Labels: {label_str}", ha="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"sample_{idx}.png")
    plt.close()

    print(f"Sample {idx}:")
    print(f"- Labels: {label_str}")
    print(f"- Image shape: {sample['image'].shape}")
    print(f"- Text input ids shape: {text_ids.shape}")
    print("-" * 50)


def main():
    """Test the MemeDataset implementation"""
    print(f"Data directory: {DATA_DIR}")

    # Check if dataset is downloaded
    memotion_dir = DATA_DIR / "memotion"
    if not (memotion_dir / "labels.csv").exists():
        print("Dataset not found. Please run download_data.py first.")
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
            display_sample(sample, i)

        print(f"Successfully visualized {num_samples} samples.")
        print("Sample images saved as sample_0.png, sample_1.png, etc.")

    except Exception as e:
        print(f"Error creating or using dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
