import torch
import torch.nn as nn
from transformers import CLIPVisionModel, RobertaModel
from .config import EMOTION_DIMS


class MemeEmotionModel(nn.Module):
    """Multimodal model for meme emotion intensity classification (Task C)."""

    def __init__(self, freeze_ratio=0.85):
        super().__init__()

        # Vision encoder - only load vision part of CLIP to save memory
        self.vision_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32")

        # Partially freeze the vision model (unfreeze the final layers)
        total_layers = len(list(self.vision_model.parameters()))
        frozen_layers = int(total_layers * freeze_ratio)

        # Set all layers to trainable first
        for param in self.vision_model.parameters():
            param.requires_grad = True

        # Freeze earlier layers according to the ratio
        for i, param in enumerate(self.vision_model.parameters()):
            if i < frozen_layers:
                param.requires_grad = False

        # Text encoder with partial unfreezing
        self.text_model = RobertaModel.from_pretrained("roberta-base")

        # Set last few layers to trainable
        for param in self.text_model.parameters():
            param.requires_grad = False

        # Unfreeze the final encoder layers and pooler
        for param in self.text_model.encoder.layer[-2:].parameters():
            param.requires_grad = True
        for param in self.text_model.pooler.parameters():
            param.requires_grad = True

        # Projection layers to common embedding space
        self.vision_projection = nn.Linear(768, 512)
        self.text_projection = nn.Linear(768, 512)

        # Simple fusion mechanism (concat + attention)
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Attention for weighting modalities
        self.modality_attention = nn.Sequential(
            nn.Linear(512, 2),  # 2 for text and vision modalities
            nn.Softmax(dim=1)
        )

        # Multi-head emotion classifier for Task C
        # Each emotion gets its own head with output size matching its scale
        self.shared_features = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Create separate classifier heads for each emotion with appropriate sizes
        self.classifier_heads = nn.ModuleList([
            nn.Linear(256, dim) for dim in EMOTION_DIMS
        ])

    def forward(self, images, text):
        # Get image features - with partial gradient flow
        with torch.set_grad_enabled(True):
            vision_outputs = self.vision_model(images)
            image_embeds = vision_outputs.pooler_output  # [batch_size, 768]

        # Get text features - with partial gradient flow
        with torch.set_grad_enabled(True):
            text_outputs = self.text_model(**text)
            text_embeds = text_outputs.pooler_output  # [batch_size, 768]

        # Project to common embedding space
        image_features = self.vision_projection(
            image_embeds)  # [batch_size, 512]
        text_features = self.text_projection(text_embeds)  # [batch_size, 512]

        # Concatenate features
        # [batch_size, 1024]
        combined = torch.cat([image_features, text_features], dim=1)

        # Fuse modalities
        fused = self.fusion(combined)  # [batch_size, 512]

        # Calculate modality weights
        weights = self.modality_attention(fused)  # [batch_size, 2]

        # Apply weights to modalities
        weighted_image = image_features * weights[:, 0].unsqueeze(1)
        weighted_text = text_features * weights[:, 1].unsqueeze(1)

        # Combine weighted features
        multimodal_features = weighted_image + \
            weighted_text  # [batch_size, 512]

        # Generate shared features
        shared = self.shared_features(multimodal_features)  # [batch_size, 256]

        # Apply each emotion classifier head and concatenate results
        emotion_outputs = []
        for head in self.classifier_heads:
            # Each head output: [batch_size, emotion_scale]
            emotion_outputs.append(head(shared))

        # Two options for returning outputs:
        # 1. Return list of separate emotion outputs (more structured)
        # 2. Return concatenated outputs (compatible with existing code)

        # Option 2: Concatenate all outputs for compatibility
        # We'll convert to logits for each intensity level
        # [batch_size, sum(EMOTION_DIMS)]
        return torch.cat(emotion_outputs, dim=1)

    def predict_intensities(self, images, text):
        """Predict the intensity level for each emotion"""
        # Forward pass to get logits
        logits = self.forward(images, text)

        # Split logits by emotion
        start_idx = 0
        intensity_predictions = []

        for dim in EMOTION_DIMS:
            # Get logits for this emotion
            emotion_logits = logits[:, start_idx:start_idx + dim]

            # Apply softmax to get probabilities for each intensity level
            probs = torch.softmax(emotion_logits, dim=1)

            # Get most likely intensity class
            intensity = torch.argmax(probs, dim=1)

            intensity_predictions.append(intensity)
            start_idx += dim

        # Stack to get [batch_size, num_emotions] tensor of predicted intensities
        return torch.stack(intensity_predictions, dim=1)
