import torch
import torch.nn as nn
from transformers import CLIPVisionModel, RobertaModel
from .config import EMOTION_DIMS

# Helper for Multi-Head Attention


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # Ensure inputs are seq_len=1 for attention
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        attn_output, _ = self.multihead_attn(query, key, value)
        output = query + self.dropout(attn_output)  # Residual connection
        output = self.layer_norm(output)
        return output.squeeze(1)  # Remove seq_len dimension


class MemeEmotionModel(nn.Module):
    """Multimodal model for meme emotion intensity classification (Task C)."""

    def __init__(self, freeze_ratio=0.85, embed_dim=512, num_heads=8, fusion_dropout=0.4, classifier_dropout=0.5):
        super().__init__()
        self.embed_dim = embed_dim

        # Vision encoder - only load vision part of CLIP to save memory
        self.vision_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32")

        # Partially freeze the vision model (unfreeze the final layers)
        total_layers_vision = len(list(self.vision_model.parameters()))
        frozen_layers_vision = int(total_layers_vision * freeze_ratio)

        # Set all layers to trainable first
        for param in self.vision_model.parameters():
            param.requires_grad = True

        # Freeze earlier layers according to the ratio
        for i, param in enumerate(self.vision_model.parameters()):
            if i < frozen_layers_vision:
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

        # Projection layers to common embedding space with regularization
        self.vision_projection = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.text_projection = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # --- Cross-Modal Attention ---
        self.img_cross_attn = CrossAttention(embed_dim, num_heads, dropout=fusion_dropout)
        self.txt_cross_attn = CrossAttention(embed_dim, num_heads, dropout=fusion_dropout)

        # --- Fusion Layer (MLP after cross-attention) ---
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),  # Process concatenated cross-attended features
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(embed_dim * 2, embed_dim)  # Fuse down to embed_dim
        )

        # --- Shared Features & Classifier Heads (remain similar but use updated dropout) ---
        self.shared_features = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
        )

        self.classifier_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim // 2, embed_dim // 2),
                nn.LayerNorm(embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(embed_dim // 2, dim)
            ) for dim in EMOTION_DIMS
        ])

    def forward(self, images, text):
        # Get image features
        with torch.set_grad_enabled(any(p.requires_grad for p in self.vision_model.parameters())):
            vision_outputs = self.vision_model(images)
            image_embeds = vision_outputs.pooler_output

        # Get text features
        with torch.set_grad_enabled(any(p.requires_grad for p in self.text_model.parameters())):
            text_outputs = self.text_model(**text)
            text_embeds = text_outputs.pooler_output

        # Project features
        image_proj = self.vision_projection(image_embeds)  # [batch_size, embed_dim]
        text_proj = self.text_projection(text_embeds)   # [batch_size, embed_dim]

        # --- Apply Cross-Modal Attention ---
        # Image features attend to text features
        img_attended = self.img_cross_attn(query=image_proj, key=text_proj, value=text_proj)
        # Text features attend to image features
        txt_attended = self.txt_cross_attn(query=text_proj, key=image_proj, value=image_proj)

        # --- Fusion ---
        # Concatenate the cross-attended features
        combined = torch.cat([img_attended, txt_attended], dim=1)  # [batch_size, embed_dim * 2]
        fused = self.fusion(combined)  # [batch_size, embed_dim]

        # --- Classification ---
        shared = self.shared_features(fused)  # [batch_size, embed_dim // 2]

        emotion_outputs = []
        for head in self.classifier_heads:
            emotion_outputs.append(head(shared))

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
