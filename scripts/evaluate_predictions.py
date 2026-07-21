#!/usr/bin/env python3
"""Compute binary-classification metrics from a prediction CSV.

The input CSV must contain:
  - true labels: true_label or y_true
  - predicted labels: pred_label or y_pred
  - positive-class probability: pred_prob, prob_1, or score
"""

import argparse
import csv
import math
from pathlib import Path


def auc_from_points(x, y):
    return sum((x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2 for i in range(1, len(x)))


def roc_auc(y_true, score):
    order = sorted(range(len(score)), key=lambda i: score[i], reverse=True)
    y = [y_true[i] for i in order]
    s = [score[i] for i in order]
    pos = sum(y)
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    tps, fps = 0, 0
    fpr, tpr = [0.0], [0.0]
    last = None
    for label, sc in zip(y, s):
        if last is not None and sc != last:
            fpr.append(fps / neg)
            tpr.append(tps / pos)
        if label == 1:
            tps += 1
        else:
            fps += 1
        last = sc
    fpr.append(fps / neg)
    tpr.append(tps / pos)
    return auc_from_points(fpr, tpr)


def pr_auc(y_true, score):
    order = sorted(range(len(score)), key=lambda i: score[i], reverse=True)
    y = [y_true[i] for i in order]
    pos = sum(y)
    if pos == 0:
        return float("nan")
    tp, fp = 0, 0
    recall = [0.0]
    precision = [1.0]
    last = None
    for label, sc in zip(y, [score[i] for i in order]):
        if last is not None and sc != last:
            recall.append(tp / pos)
            precision.append(tp / (tp + fp) if tp + fp else 1.0)
        if label == 1:
            tp += 1
        else:
            fp += 1
        last = sc
    recall.append(tp / pos)
    precision.append(tp / (tp + fp) if tp + fp else 1.0)
    return auc_from_points(recall, precision)


def read_predictions(path):
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    cols = rows[0].keys()
    true_col = "true_label" if "true_label" in cols else "y_true"
    pred_col = "pred_label" if "pred_label" in cols else "y_pred"
    score_col = next((c for c in ["pred_prob", "prob_1", "score"] if c in cols), None)
    if score_col is None:
        raise ValueError("Prediction CSV must contain pred_prob, prob_1, or score")
    y_true = [int(float(r[true_col])) for r in rows]
    y_pred = [int(float(r[pred_col])) for r in rows]
    score = [float(r[score_col]) for r in rows]
    return y_true, y_pred, score


def compute_metrics(y_true, y_pred, score):
    tp = sum(yt == 1 and yp == 1 for yt, yp in zip(y_true, y_pred))
    tn = sum(yt == 0 and yp == 0 for yt, yp in zip(y_true, y_pred))
    fp = sum(yt == 0 and yp == 1 for yt, yp in zip(y_true, y_pred))
    fn = sum(yt == 1 and yp == 0 for yt, yp in zip(y_true, y_pred))
    n = len(y_true)
    acc = (tp + tn) / n
    sn = tp / (tp + fn) if tp + fn else float("nan")
    sp = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * precision * sn / (precision + sn) if precision + sn else float("nan")
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else float("nan")
    return {
        "ACC": acc,
        "BACC": (sn + sp) / 2,
        "Precision": precision,
        "Recall/Sn": sn,
        "Specificity/Sp": sp,
        "F1": f1,
        "MCC": mcc,
        "AUPR": pr_auc(y_true, score),
        "AUC": roc_auc(y_true, score),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()
    y_true, y_pred, score = read_predictions(args.predictions)
    metrics = compute_metrics(y_true, y_pred, score)
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
    if args.out_csv:
        path = Path(args.out_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics))
            writer.writeheader()
            writer.writerow(metrics)


if __name__ == "__main__":
    main()
