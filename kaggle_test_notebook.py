# Meme Emotion Model Test Notebook
# This notebook evaluates the trained model on the test set and generates detailed reports

# Import necessary libraries
import kagglehub
import os
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display, Image, FileLink

# First, check if the repository is already cloned
if not os.path.exists("memotion"):
    # Clone the repository if it doesn't exist
    !git clone https: // github.com/your-username/memotion.git
    %cd memotion
else:
    print("Repository already exists, using the existing one")
    %cd memotion
    # Update the repo to get the latest code
    if os.path.exists(".git"):
        !git pull

# Install required dependencies
!pip install - q matplotlib seaborn scikit-learn tqdm

# Verify GPU availability
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Access the Memotion dataset with KaggleHub
!pip install - q "kagglehub[hf-datasets]"

# Download the dataset directly from Kaggle
print("Downloading Memotion dataset...")
dataset_path = kagglehub.dataset_download(
    "williamscott701/memotion-dataset-7k")
print(f"Dataset downloaded to: {dataset_path}")

# Verify the download
if os.path.exists(dataset_path):
    print("Dataset contents:", os.listdir(dataset_path))

    # Verify images directory
    images_dir = Path(dataset_path) / "images"
    if images_dir.exists():
        print(f"Found {len(list(images_dir.glob('*')))} images")

    # Verify labels file
    labels_file = Path(dataset_path) / "labels.csv"
    if labels_file.exists():
        print("Labels file found")

# Create output directories for test results
output_dir = Path("/kaggle/working/test_results")
output_dir.mkdir(exist_ok=True, parents=True)
vis_dir = output_dir / "visualizations"
vis_dir.mkdir(exist_ok=True, parents=True)

# Find the latest trained model
models_dir = Path("/kaggle/working/outputs/models")
model_files = list(models_dir.glob("*.pt"))
if not model_files:
    raise FileNotFoundError(
        "No model files found. Please run training first.")

latest_model = max(model_files, key=os.path.getctime)
print(f"Using the latest model: {latest_model}")

# Run the test script
print("\n---------- Starting Model Evaluation ----------\n")

# Make sure PYTHONPATH includes the current directory for imports
if "." not in sys.path:
    sys.path.append(".")

# Command to run the test script
!python - m src.test \
    - -model_path "{latest_model}" \
    - -output_dir "{output_dir}" \
    - -kaggle_dataset_path "{dataset_path}" \
    - -batch_size 32 \
    - -fp16

# Display test results


def display_test_results():
    # Load the metrics from json file
    import json
    metrics_file = output_dir / "test_metrics.json"

    if not metrics_file.exists():
        print("Test metrics file not found. Test may have failed.")
        return

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Display overall metrics
    print("\n=== Overall Model Performance ===")
    print(f"Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Precision: {metrics['metrics']['precision']:.4f}")
    print(f"Recall: {metrics['metrics']['recall']:.4f}")
    print(f"F1 Score: {metrics['metrics']['f1']:.4f}")

    # Display per-class metrics as a table
    print("\n=== Per-Class Performance ===")
    metrics_df = pd.DataFrame(
        columns=['Precision', 'Recall', 'F1 Score', 'AUC'])

    for class_name, class_metrics in metrics['per_class_metrics'].items():
        metrics_df.loc[class_name] = [
            class_metrics['precision'],
            class_metrics['recall'],
            class_metrics['f1'],
            class_metrics['auc']
        ]

    display(metrics_df)

    # Find best and worst performing classes
    best_class = metrics_df['F1 Score'].idxmax()
    worst_class = metrics_df['F1 Score'].idxmin()

    print(
        f"\nBest performing class: {best_class} (F1 Score: {metrics_df.loc[best_class, 'F1 Score']:.4f})")
    print(
        f"Worst performing class: {worst_class} (F1 Score: {metrics_df.loc[worst_class, 'F1 Score']:.4f})")


# Display the test results
display_test_results()

# Display visualization images


def display_visualizations():
    # ROC curves
    roc_path = vis_dir / "roc_curves.png"
    if roc_path.exists():
        display(Markdown("### ROC Curves for Each Emotion Class"))
        display(Image(filename=str(roc_path)))

    # Metrics by class
    metrics_path = vis_dir / "metrics_by_class.png"
    if metrics_path.exists():
        display(Markdown("### Precision, Recall and F1 Score by Class"))
        display(Image(filename=str(metrics_path)))

    # Display one confusion matrix as an example
    cm_files = list(vis_dir.glob("confusion_matrix_*.png"))
    if cm_files:
        display(Markdown("### Example Confusion Matrix"))
        display(Image(filename=str(cm_files[0])))

        if len(cm_files) > 1:
            display(Markdown(
                "*Additional confusion matrices are available in the visualizations directory*"))


# Display the visualizations
display_visualizations()

# Analyze the error examples


def analyze_errors():
    error_file = vis_dir / "error_analysis.csv"
    if not error_file.exists():
        print("Error analysis file not found.")
        return

    error_df = pd.read_csv(error_file)

    # Show error distribution
    display(Markdown("### Error Analysis"))

    error_counts = error_df.groupby(['class', 'error_type']).size().unstack()
    display(error_counts)

    # Create a bar chart of error types by class
    plt.figure(figsize=(12, 6))
    error_counts.plot(kind='bar')
    plt.title('Error Distribution by Class and Type')
    plt.ylabel('Number of Errors')
    plt.xlabel('Emotion Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "error_distribution.png")
    plt.close()

    # Show the saved error distribution chart
    display(Image(filename=str(output_dir / "error_distribution.png")))

    # Show a few example errors
    if len(error_df) > 0:
        display(Markdown("### Sample Error Cases"))
        sample_errors = error_df.sample(min(5, len(error_df)))

        for _, row in sample_errors.iterrows():
            print(f"Class: {row['class']}")
            print(f"Error Type: {row['error_type']}")
            print(
                f"Target: {row['target']}, Prediction: {row['prediction']}, Probability: {row['probability']:.4f}")
            print("-" * 50)


# Run error analysis
analyze_errors()

# Downloadable files


def create_downloadable_links():
    display(Markdown("### Download Files"))

    # Test report markdown
    report_file = output_dir / "test_report.md"
    if report_file.exists():
        display(Markdown("**Test Report:**"))
        display(FileLink(str(report_file)))

    # Metrics JSON
    metrics_file = output_dir / "test_metrics.json"
    if metrics_file.exists():
        display(Markdown("**Test Metrics:**"))
        display(FileLink(str(metrics_file)))

    # Create a ZIP archive with all results
    !zip - r / kaggle/working/test_results.zip {output_dir}

    display(Markdown("**All Test Results (ZIP):**"))
    display(FileLink("/kaggle/working/test_results.zip"))


# Create downloadable links
create_downloadable_links()

# Suggestions for next steps
display(Markdown("""
## Conclusions and Next Steps

Based on the model evaluation results, here are some suggested next steps:

1. **Fine-tune the model**: Adjust model parameters to improve performance on underperforming classes.
2. **Data augmentation**: Add more training examples, especially for classes with lower F1 scores.
3. **Error analysis**: Look at the specific error cases to understand where the model struggles.
4. **Model enhancements**: Consider ensemble approaches or larger vision/text backbones for improved performance.
5. **Deployment**: Using the test results, create API endpoints optimized for production.

The test results show that the model achieves good performance overall, but there are specific emotion classes that could benefit from further optimization.
"""))

print("\nTest evaluation complete!")
