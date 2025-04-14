import argparse
from pathlib import Path
import logging
import kagglehub
import shutil
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_and_prepare_memotion(output_dir: str):
    """
    Download and organize Memotion dataset using Kaggle Hub.
    Copies the main 'images' folder and 'labels.csv' from the cache
    to the specified output directory.

    Args:
        output_dir: Root directory to save processed data (e.g., 'data')
    """
    try:
        # Define base path for output (e.g., data/memotion)
        base_path = Path(output_dir) / "memotion"
        base_path.mkdir(parents=True, exist_ok=True)

        # Download dataset using Kaggle Hub
        logger.info("Downloading Memotion dataset from Kaggle...")
        # This downloads to a cache directory like ~/.cache/kagglehub/...
        dataset_cache_path = kagglehub.dataset_download(
            "williamscott701/memotion-dataset-7k"
        )
        logger.info(f"Dataset downloaded to cache: {dataset_cache_path}")

        # Verify downloaded path exists
        dataset_cache_path = Path(dataset_cache_path)
        if not dataset_cache_path.exists():
            raise FileNotFoundError(
                f"Kaggle Hub download path does not exist: {dataset_cache_path}")

        # Find the actual data directory within the cache
        # It's usually named after the dataset slug
        src_dir = dataset_cache_path / "memotion_dataset_7k"
        if not src_dir.is_dir():
            # Fallback: Check if cache path itself contains the data directly
            if (dataset_cache_path / "labels.csv").exists() and \
               (dataset_cache_path / "images").is_dir():
                src_dir = dataset_cache_path
            else:
                raise FileNotFoundError(
                    f"Dataset directory 'memotion_dataset_7k' not found in "
                    f"cache path: {dataset_cache_path}. Contents: "
                    f"{os.listdir(dataset_cache_path)}"
                )

        logger.info(f"Using source data from: {src_dir}")
        logger.info(f"Contents of source directory: {os.listdir(src_dir)}")

        # Define specific source files/folders needed
        src_images_dir = src_dir / "images"
        src_labels_csv = src_dir / "labels.csv"

        # Define destination paths
        dst_images_dir = base_path / "images"
        dst_labels_csv = base_path / "labels.csv"

        # Ensure destination image directory exists
        dst_images_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Copying dataset files...")

        # --- Copy images ---
        if not src_images_dir.is_dir():
            logger.error(
                f"Source images directory not found or not a directory: "
                f"{src_images_dir}")
        else:
            logger.info(
                f"Copying images from {src_images_dir} to {dst_images_dir}")
            image_count = 0
            # Use shutil.copytree for potentially faster directory copy if needed,
            # but iterating handles errors per file and might be safer.
            for item in src_images_dir.iterdir():
                # Only copy files, ignore potential subdirectories if any
                if item.is_file():
                    try:
                        shutil.copy2(item, dst_images_dir)
                        image_count += 1
                    except Exception as e:
                        logger.error(
                            f"Error copying image '{item.name}': {str(e)}")
            logger.info(f"Copied {image_count} images.")

        # --- Copy labels CSV file ---
        if not src_labels_csv.is_file():
            logger.error(
                f"Source labels CSV file not found or not a file: "
                f"{src_labels_csv}")
        else:
            try:
                shutil.copy2(src_labels_csv, dst_labels_csv)
                logger.info(f"Copied labels CSV to {dst_labels_csv}")
            except Exception as e:
                logger.error(
                    f"Error copying labels CSV file '{src_labels_csv.name}': "
                    f"{str(e)}")

        # --- Verify final structure ---
        logger.info(
            f"Verifying final directory structure in {base_path}...")
        final_image_count = 0
        if not dst_images_dir.is_dir():
            logger.error(f"MISSING final images directory: {dst_images_dir}")
        else:
            logger.info(f"Images directory exists: {dst_images_dir}")
            # Count files in the destination directory
            final_image_count = len(
                [f for f in dst_images_dir.iterdir() if f.is_file()])
            logger.info(
                f"Number of images found in destination: {final_image_count}")

        if not dst_labels_csv.is_file():
            logger.error(f"MISSING final labels CSV file: {dst_labels_csv}")
        else:
            logger.info(f"Labels CSV file exists: {dst_labels_csv}")

        if final_image_count > 0 and dst_labels_csv.is_file():
            logger.info("Dataset preparation appears complete!")
        else:
            logger.warning(
                "Dataset preparation may be incomplete due to missing files/images.")

    except Exception as e:
        logger.error(f"Failed during dataset preparation: {str(e)}")
        # Re-raise the exception to signal failure
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare Memotion dataset from Kaggle Hub."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Root directory to save processed data (e.g., 'data' -> 'data/memotion')"
    )

    args = parser.parse_args()

    try:
        download_and_prepare_memotion(args.output_dir)
    except Exception as e:
        # Log the error already happened in the function, just exit
        logger.error(f"Script failed with error: {e}. See details above.")
        exit(1)
