import torch
import torch.nn as nn
from transformers import CLIPVisionModel, RobertaModel


class MemeEmotionModel(nn.Module):
    """Simplified multimodal model for meme emotion classification."""

    def __init__(self, num_classes=5):
        super().__init__()

        # Vision encoder - only load vision part of CLIP to save memory
        self.vision_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32")
        self.vision_model.eval()  # Set to eval mode to freeze batch norm layers
        for param in self.vision_model.parameters():
            param.requires_grad = False  # Freeze all vision parameters

        # Text encoder
        self.text_model = RobertaModel.from_pretrained("roberta-base")
        self.text_model.eval()  # Set to eval mode
        for param in self.text_model.parameters():
            param.requires_grad = False  # Freeze all text parameters

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

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, text):
        batch_size = images.size(0)

        # Get image features
        with torch.no_grad():
            vision_outputs = self.vision_model(images)
            image_embeds = vision_outputs.pooler_output  # [batch_size, 768]

        # Get text features
        with torch.no_grad():
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

        # Classification
        # [batch_size, num_classes]
        logits = self.classifier(multimodal_features)

        return logits
