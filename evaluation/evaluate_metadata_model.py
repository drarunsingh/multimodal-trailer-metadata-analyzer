"""
Evaluate metadata prediction model
"""

import os
import sys
import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report

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
MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "metadata_model.pt"
)

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset = MetadataDataset(
        embeddings_dir=EMBEDDINGS_DIR,
        labels_csv=LABELS_CSV
    )

    input_dim = dataset[0][0].shape[0]
    num_labels = dataset[0][1].shape[0]

    model = MetadataClassifier(
        input_dim=input_dim,
        num_labels=num_labels
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in dataset:
            x = x.unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

            y_true.append(y.numpy())
            y_pred.append((probs >= 0.5).astype(int))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")

    print("\n📊 Evaluation Results")
    print(f"Micro F1-score : {f1_micro:.4f}")
    print(f"Macro F1-score : {f1_macro:.4f}")

    print("\n📋 Detailed Report")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()
