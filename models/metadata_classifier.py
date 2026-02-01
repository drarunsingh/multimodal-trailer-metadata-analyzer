"""
Definition of the multi-label metadata classifier.
Uses sigmoid outputs for genre, mood, and theme prediction.
"""
"""
Metadata Classifier
Predicts genre / mood / themes from multimodal embeddings
"""

import torch
import torch.nn as nn


class MetadataClassifier(nn.Module):
    def __init__(self, input_dim: int, num_labels: int):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, input_dim)
        returns: logits (before sigmoid)
        """
        return self.classifier(x)

