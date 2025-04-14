# Meme Emotion Recognition System

A multimodal deep learning system that recognizes emotions in internet memes by analyzing both visual content and text.

## Overview

This system combines state-of-the-art vision models (CLIP) with text encoders (RoBERTa) to classify memes into emotion categories:

- Amusement
- Sarcasm
- Offense
- Motivation
- Neutral

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/meme_emotion.git
cd meme_emotion
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The system uses the Memotion dataset, which contains approximately 7K annotated memes with emotion labels.

1. Download and prepare the dataset:

```bash
python src/download_data.py --output-dir data
```

2. Test the dataset loading:

```bash
python test_memotion.py
```

This will display example memes with their labels and save them as sample_0.png, sample_1.png, etc.

## Training

To train the model with default parameters:

```bash
python -m src.train
```

### Training Options

You can customize the training with various parameters:

```bash
python -m src.train --batch_size 8 --epochs 5 --learning_rate 5e-4
```

Key parameters:

- `--batch_size`: Number of samples per batch (default: 32)
- `--epochs`: Number of training epochs (default: 10)
- `--learning_rate`: Learning rate for optimization (default: 1e-4)
- `--fp16`: Enable mixed precision training
- `--weight_decay`: Weight decay for regularization (default: 1e-2)
- `--patience`: Early stopping patience (default: 3)

## Model Architecture

The model uses a multimodal architecture:

1. CLIP vision model for processing images
2. RoBERTa for processing text
3. Custom modality fusion layer with attention
4. Classification head for emotion prediction

## Performance

On the Memotion dataset, the model achieves:

- Task A (Sentiment): ~XX% F1 score
- Task B (Emotion): ~XX% F1 score

## Inference

For making predictions on new memes:

```bash
python -m src.predict --image path/to/meme.jpg
```

## Project Structure

- `src/`: Source code
  - `download_data.py`: Data download script
  - `dataset.py`: Dataset handling and preprocessing
  - `model.py`: Model architecture
  - `train.py`: Training pipeline
  - `config.py`: Configuration parameters
- `data/`: Dataset storage
- `models/`: Saved model checkpoints
- `test_memotion.py`: Script to test dataset loading

## Troubleshooting

- **Out of memory errors**: Reduce batch size or use a smaller model
- **Slow training**: Consider enabling FP16 training with `--fp16` flag
- **Poor performance**: Increase training epochs or adjust learning rate

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code or the Memotion dataset, please cite:

```
@inproceedings{sharma2020memotion,
    title={Task Report: Memotion Analysis 1.0 @SemEval 2020: The Visuo-Lingual Metaphor!},
    author={Sharma, Chhavi and Paka, Scott, William and Bhageria, Deepesh and Das, Amitava and Poria, Soujanya and Chakraborty, Tanmoy and Gamb\"ack, Bj\"orn},
    booktitle={Proceedings of the 14th International Workshop on Semantic Evaluation (SemEval-2020)},
    year={2020},
    month={Sep},
    address={Barcelona, Spain},
    publisher={Association for Computational Linguistics}
}
```
