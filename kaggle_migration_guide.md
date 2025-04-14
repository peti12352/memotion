# Migrating the Meme Emotion Recognition Project to Kaggle

This guide shows how to leverage Kaggle's GPU resources while continuing development in Cursor IDE. The workflow synchronizes your local development with Kaggle Notebooks, allowing you to use Kaggle's free GPU acceleration for training.

## 1. Set Up Kaggle Credentials

First, you need to set up your Kaggle account and API access:

1. **Create or log in to your Kaggle account** at [kaggle.com](https://www.kaggle.com/)
2. **Get your API credentials**:
   - Go to your account settings (click on your profile icon → "Settings")
   - Scroll down to the "API" section and click "Create New API Token"
   - This downloads a `kaggle.json` file with your credentials
3. **Save your credentials** in the appropriate location:
   - On Windows: `C:\Users\<Windows-username>\.kaggle\kaggle.json`
   - On Linux/Mac: `~/.kaggle/kaggle.json`
   - Set proper permissions: `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac)

## 2. Create a GitHub Repository (Recommended)

Using GitHub as an intermediary makes synchronization easier:

1. Create a GitHub repository for your project
2. Initialize Git in your local project directory if not already done:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/your-username/memotion.git
   git push -u origin main
   ```
3. Create a `.gitignore` file to exclude data and models:

   ```
   # Data and model directories
   data/
   models/

   # Python artifacts
   __pycache__/
   *.py[cod]
   *$py.class
   .ipynb_checkpoints/

   # Distribution / packaging
   dist/
   build/
   *.egg-info/
   ```

## 3. Create a Kaggle Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/notebooks) and click "New Notebook"
2. Configure the notebook settings:
   - Click on the "⚙️" (Settings) in the right sidebar
   - Under "Accelerator", select "GPU: T4 x 1" (free tier)
   - Enable internet access
   - Set session length to suit your training needs (up to 9 hours in free tier)

## 4. Set Up Your Kaggle Environment

Add these cells to your notebook:

```python
# Cell 1: Clone your GitHub repository
# First check if the directory already exists
import os
if os.path.exists('memotion'):
    print("Directory already exists. Using existing directory.")
    %cd memotion
    # Optionally pull latest changes if it's a git repository
    if os.path.exists('.git'):
        !git pull
else:
    # Clone the repository if it doesn't exist
    !git clone https://github.com/your-username/memotion.git
    %cd memotion

# Install required dependencies
!pip install -r requirements.txt
```

### Handling "Directory Already Exists" Error

If you encounter a "fatal: destination path already exists" error when trying to clone your repository, use this comprehensive solution:

```python
import os
import shutil

# Define the target directory
target_dir = "/kaggle/working/memotion"

# Check if directory exists
if os.path.exists(target_dir):
    print(f"Directory {target_dir} already exists.")

    # Check if it's a git repository
    if os.path.exists(os.path.join(target_dir, '.git')):
        print("Directory is a git repository. Pulling latest changes...")
        %cd {target_dir}
        !git pull
    else:
        print("Directory is not a git repository. Creating backup and cloning fresh...")
        # Create backup of existing directory
        backup_dir = f"{target_dir}_backup"
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.move(target_dir, backup_dir)
        print(f"Backed up existing directory to {backup_dir}")

        # Clone fresh
        !git clone https://github.com/your-username/memotion.git {target_dir}
        %cd {target_dir}
else:
    print(f"Directory {target_dir} does not exist. Cloning fresh...")
    !git clone https://github.com/your-username/memotion.git {target_dir}
    %cd {target_dir}

# Verify we're in the correct directory
print(f"Current working directory: {os.getcwd()}")
```

This solution provides three possible outcomes:

1. If the directory exists and is a git repository: Pulls the latest changes
2. If the directory exists but isn't a git repository: Creates a backup and clones fresh
3. If the directory doesn't exist: Clones fresh

The script also:

- Preserves existing work by creating backups when needed
- Provides clear feedback about what actions are being taken
- Verifies the final working directory
- Handles edge cases like backup directory already existing

## 5. Access the Memotion Dataset with KaggleHub

Use KaggleHub to directly access the Memotion dataset without downloading:

```python
# Install additional dependencies if needed
!pip install -q "kagglehub[hf-datasets]"

import kagglehub
import os
from pathlib import Path

# Download the dataset directly from Kaggle
dataset_path = kagglehub.dataset_download("williamscott701/memotion-dataset-7k")
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
```

## 6. Configure the Training Script

Create a cell to run training with GPU optimizations:

```python
import torch
from pathlib import Path

# Verify GPU availability
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Prepare output directories
output_dir = Path("/kaggle/working/outputs")
output_dir.mkdir(exist_ok=True)
models_dir = output_dir / "models"
models_dir.mkdir(exist_ok=True)

# Run training with the Kaggle dataset path
!python -m src.train \
  --batch_size 32 \
  --epochs 10 \
  --learning_rate 2e-4 \
  --fp16 \
  --kaggle_dataset_path "{dataset_path}" \
  --output_dir "{output_dir}"
```

## 7. Optimize Your Code for Kaggle GPU

Your project already has `kaggle_dataset_path` support in the `MemeDataset` class. Here are additional optimizations:

### 1. Enable Mixed Precision Training

Add this to your training script:

```python
# In your training script
from torch.cuda.amp import autocast, GradScaler

# Initialize the scaler
scaler = GradScaler(enabled=args.fp16)

# In the training loop
with autocast(enabled=args.fp16):
    outputs = model(batch)
    loss = criterion(outputs, targets)

# Scale the gradients and optimize
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. Implement Gradient Accumulation

For larger batch sizes with limited GPU memory:

```python
# In your training arguments
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

# In your training loop
loss = loss / args.gradient_accumulation_steps
scaler.scale(loss).backward()

if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 3. Efficient Data Loading

```python
# Set num_workers for DataLoader based on Kaggle environment
num_workers = 2  # Kaggle has 2 CPU cores

# In your DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True  # Speeds up host to device transfers
)
```

## 8. Save and Download Models

Add a cell to save and download your trained models:

```python
# List available models
!ls -la /kaggle/working/outputs/models/

# Create a link to download the final model
from IPython.display import FileLink
FileLink(r'/kaggle/working/outputs/models/memotion_model.pth')

# Optionally compress for easier download
!tar -czvf /kaggle/working/memotion_models.tar.gz /kaggle/working/outputs/models/
FileLink(r'/kaggle/working/memotion_models.tar.gz')
```

## 9. Hugging Face Integration (Optional)

You can push your trained model directly to Hugging Face:

```python
# Install Hugging Face Hub if needed
!pip install -q huggingface_hub

# Log in to Hugging Face
from huggingface_hub import notebook_login
notebook_login()

# Push your model to Hugging Face Hub
from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path="/kaggle/working/outputs/models/",
    repo_id="your-username/memotion-model",
    repo_type="model"
)
```

## 10. Complete Development Workflow

Here's the recommended workflow to develop in Cursor IDE and train on Kaggle:

1. **Local Development** (Cursor IDE):

   - Develop code, test with small samples
   - Commit and push changes to GitHub

2. **Kaggle Training**:

   - Pull latest code from GitHub
   - Run training with GPU acceleration
   - Save models to Kaggle working directory
   - Download models or push to Hugging Face

3. **Continue Development** (Cursor IDE):
   - Pull model from Kaggle/Hugging Face
   - Evaluate, test inference
   - Make improvements, repeat process

## 11. Performance Tips

- **Monitor GPU memory usage**: Add `!nvidia-smi` to notebook cells
- **Checkpoint models regularly**: Save every few epochs to prevent losing progress
- **Use Kaggle's dataset API**: Direct access is faster than downloading
- **Optimize batch size**: Find the largest batch size that fits in memory
- **Limit model size during development**: Use smaller variants of CLIP/RoBERTa when debugging

## 12. Kaggle Time Limits

Be aware of Kaggle's free tier limitations:

- GPU sessions limited to 9 hours
- Total weekly GPU quota of 30 hours
- Notebooks idle for 1 hour are automatically terminated

For longer training sessions, implement checkpointing and resuming from saved states:

```python
# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler': scaler.state_dict(),  # If using mixed precision
}, f"{args.output_dir}/checkpoint_epoch_{epoch}.pt")

# Resume training
if args.resume:
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    if 'scaler' in checkpoint and args.fp16:
        scaler.load_state_dict(checkpoint['scaler'])
```

This workflow lets you use Cursor's excellent development environment while leveraging Kaggle's free GPU resources for the compute-intensive training phase.

[Learn more about using Kaggle's free GPU compute](https://nkoenig06.github.io/GPU-K.html)
