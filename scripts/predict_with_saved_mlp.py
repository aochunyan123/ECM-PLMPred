#!/usr/bin/env python3
"""Run inference with the saved ProtT5-mean MLP model.

Input embedding pickle format:
  {"embeddings": {"seq_id": vector}} or {"seq_id": vector}

The script infers labels from IDs when they contain label=0/label=1,
positive/negative pipe-delimited fields, or pos/neg prefixes.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=128, num_classes=2, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def infer_label(seq_id):
    s = str(seq_id).lower()
    if "label=1" in s:
        return 1
    if "label=0" in s:
        return 0
    fields = [field.strip() for field in s.split("|")]
    if "positive" in fields:
        return 1
    if "negative" in fields:
        return 0
    if s.startswith("pos_") or "_pos_" in s or s.startswith("test_pos") or s.startswith("train_pos"):
        return 1
    if s.startswith("neg_") or "_neg_" in s or s.startswith("test_neg") or s.startswith("train_neg"):
        return 0
    return np.nan


def load_embeddings(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    emb = data["embeddings"] if isinstance(data, dict) and "embeddings" in data else data
    ids = list(emb.keys())
    X = np.vstack([np.asarray(emb[i], dtype=np.float32).reshape(-1) for i in ids])
    return ids, X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, help="ProtT5 mean embedding pkl")
    parser.add_argument("--model", default="models/ProtT5_mean_best_overall.pt")
    parser.add_argument("--out_csv", default="results/inference_predictions.csv")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    ids, X = load_embeddings(args.embeddings)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = MLPClassifier(input_dim=X.shape[1]).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    probs = []
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), args.batch_size):
            xb = torch.tensor(X[start : start + args.batch_size], dtype=torch.float32).to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            probs.extend(prob.tolist())
            preds.extend(pred.tolist())

    out = pd.DataFrame({
        "id": ids,
        "true_label": [infer_label(i) for i in ids],
        "pred_label": preds,
        "pred_prob": probs,
    })
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved predictions to {args.out_csv}")


if __name__ == "__main__":
    main()
