"""
Train metadata prediction model
(genre / mood / themes)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ------------------------------------------------------------------
# Fix project root import
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.metadata_classifier import MetadataClassifier
from training.dataset import MetadataDataset

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
EMBEDDINGS_DIR = os.path.join(
    PROJECT_ROOT, "data", "embeddings", "multimodal"
)
LABELS_CSV = os.path.join(
    PROJECT_ROOT, "data", "labels", "labels.csv"
)
MODEL_SAVE_PATH = os.path.join(
    PROJECT_ROOT, "models", "metadata_model.pt"
)

# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-3

# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset = MetadataDataset(
        embeddings_dir=EMBEDDINGS_DIR,
        labels_csv=LABELS_CSV
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    input_dim = dataset[0][0].shape[0]
    num_labels = dataset[0][1].shape[0]

    model = MetadataClassifier(
        input_dim=input_dim,
        num_labels=num_labels
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("[INFO] Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[SAVED] Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
