"""
MLP baseline for comparing multiple PLM feature files with stratified 5-fold CV.

Default behavior for feature screening:
- MLP classifier only
- class-weighted CrossEntropyLoss enabled by default
- Stratified 5-fold cross validation enabled by default
- independent test set evaluated by each fold-trained model
- CSV metrics/predictions are saved
- best model .pt files are saved by default
- ROC curve data are saved for validation and independent test predictions
"""

import os
import pickle
import argparse
import random
import copy
import shutil
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, roc_curve


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def safe_name(x):
    return str(x).replace('/', '_').replace(' ', '_').replace(':', '_')


def load_pkl_feature(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'] if isinstance(data, dict) and 'embeddings' in data else data


def infer_label_from_id(seq_id):
    s = str(seq_id).lower()
    if 'label=1' in s: return 1
    if 'label=0' in s: return 0
    fields = [field.strip() for field in s.split('|')]
    if 'positive' in fields: return 1
    if 'negative' in fields: return 0
    if s.startswith('pos_') or '_pos_' in s or s.startswith('train_pos') or s.startswith('test_pos'): return 1
    if s.startswith('neg_') or '_neg_' in s or s.startswith('train_neg') or s.startswith('test_neg'): return 0
    raise ValueError(f'Cannot infer label from ID: {seq_id}. Use label=1/0, positive/negative, pos/neg, or provide label CSV.')


def load_label_csv(label_csv):
    df = pd.read_csv(label_csv)
    if 'id' not in df.columns or 'label' not in df.columns:
        raise ValueError('Label CSV must contain columns: id,label')
    return dict(zip(df['id'].astype(str), df['label'].astype(int)))


class PLMFeatureDataset(Dataset):
    def __init__(self, pkl_path, label_csv=None):
        self.embeddings = load_pkl_feature(pkl_path)
        label_dict = load_label_csv(label_csv) if label_csv else None
        self.ids, self.labels = [], []
        for seq_id in self.embeddings.keys():
            seq_id = str(seq_id)
            if label_dict is not None:
                if seq_id not in label_dict:
                    print(f'[Warning] {seq_id} not found in label CSV, skipped.'); continue
                label = label_dict[seq_id]
            else:
                label = infer_label_from_id(seq_id)
            self.ids.append(seq_id); self.labels.append(label)
        print(f'\nLoaded: {pkl_path}')
        print('Total samples:', len(self.ids))
        print('Label distribution:', Counter(self.labels))
        if len(self.ids) == 0: raise ValueError('No valid samples loaded.')

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        seq_id = self.ids[idx]; label = self.labels[idx]
        x = self.embeddings[seq_id]
        if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError(f'This script expects mean pooled feature shape (D,), got {x.shape} for {seq_id}.')
        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.long), seq_id


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=128, num_classes=2, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1), nn.BatchNorm1d(hidden_dim1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2), nn.BatchNorm1d(hidden_dim2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim2, num_classes)
        )
    def forward(self, x): return self.net(x)


def compute_class_weights(labels, num_classes=2, device='cpu'):
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts == 0): raise ValueError(f'Some class has zero samples in training subset: {counts}')
    weights = len(labels) / (num_classes * counts)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print('\nClass counts in training subset:', counts)
    print('Class weights:', weights.detach().cpu().numpy())
    return weights


def evaluate(model, loader, device):
    model.eval(); y_true=[]; y_pred=[]; y_prob=[]; ids_all=[]
    with torch.no_grad():
        for x,y,ids in loader:
            x=x.to(device); y=y.to(device)
            logits=model(x)
            prob=torch.softmax(logits, dim=1)[:,1]
            pred=torch.argmax(logits, dim=1)
            y_true += y.cpu().numpy().tolist(); y_pred += pred.cpu().numpy().tolist(); y_prob += prob.cpu().numpy().tolist(); ids_all += list(ids)
    metrics = {
        'ACC': accuracy_score(y_true, y_pred),
        'BACC': balanced_accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred),
    }
    metrics['AUC'] = roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else np.nan
    metrics['AUPR'] = average_precision_score(y_true, y_prob) if len(set(y_true)) == 2 else np.nan
    pred_df = pd.DataFrame({'id': ids_all, 'true_label': y_true, 'pred_label': y_pred, 'pred_prob': y_prob})
    return metrics, pred_df


def save_roc_curve_data(pred_df, out_csv, feature, fold, split_name):
    """Save fpr/tpr/thresholds for drawing ROC curves later."""
    y_true = pred_df['true_label'].astype(int).to_numpy()
    y_prob = pred_df['pred_prob'].astype(float).to_numpy()
    if len(np.unique(y_true)) < 2:
        roc_df = pd.DataFrame(columns=['feature', 'fold', 'split', 'fpr', 'tpr', 'threshold', 'auc'])
        roc_df.to_csv(out_csv, index=False)
        return out_csv
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    auc_value = roc_auc_score(y_true, y_prob)
    roc_df = pd.DataFrame({
        'feature': feature,
        'fold': fold,
        'split': split_name,
        'fpr': fpr,
        'tpr': tpr,
        'threshold': thresholds,
        'auc': auc_value,
    })
    roc_df.to_csv(out_csv, index=False)
    return out_csv


def print_metrics(title, metrics):
    print('\n' + title)
    for k,v in metrics.items(): print(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}')


def make_paths(args, feature, fold_id=None):
    feature = safe_name(feature)
    root, ext = os.path.splitext(args.save_path)
    if ext == '': ext = '.pt'
    if fold_id is None: return f'{root}_{feature}{ext}', f'{args.output_prefix}_{feature}'
    return f'{root}_{feature}_fold{fold_id}{ext}', f'{args.output_prefix}_{feature}_fold{fold_id}'


def train_one_split(args, feature, train_ds, test_ds, train_idx, val_idx, device, input_dim, fold_id=None):
    fold = 'single_split' if fold_id is None else f'fold{fold_id}'
    set_seed(args.seed if fold_id is None else args.seed + fold_id)
    save_path, out_prefix = make_paths(args, feature, fold_id)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True); os.makedirs(os.path.dirname(out_prefix) or '.', exist_ok=True)
    print('\n' + '='*80); print(f'Feature: {feature} | {fold} | input_dim={input_dim}'); print('='*80)

    g = torch.Generator(); g.manual_seed(args.seed if fold_id is None else args.seed + fold_id)
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=args.drop_last, generator=g)
    val_loader = DataLoader(Subset(train_ds, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MLPClassifier(input_dim, args.mlp_hidden1, args.mlp_hidden2, 2, args.dropout).to(device)
    train_labels = np.array(train_ds.labels)[train_idx]
    if args.use_class_weight:
        criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_labels, 2, device))
    else:
        print('\nClass weight is disabled.'); criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.scheduler_patience) if args.use_scheduler else None

    best, best_epoch, patience_counter, history = -1.0, -1, 0, []
    best_state_dict = None
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss=0.0
        for x,y,_ in train_loader:
            x=x.to(device); y=y.to(device); optimizer.zero_grad()
            loss=criterion(model(x), y); loss.backward()
            if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step(); total_loss += loss.item()
        avg_loss = total_loss / max(1, len(train_loader))
        val_metrics,_ = evaluate(model, val_loader, device)
        score = val_metrics[args.select_metric]
        score = -1.0 if isinstance(score, float) and np.isnan(score) else score
        if scheduler: scheduler.step(score)
        history.append({'feature':feature,'fold':fold,'epoch':epoch,'train_loss':avg_loss,'lr':optimizer.param_groups[0]['lr'], **{f'val_{k}':v for k,v in val_metrics.items()}})
        print(f'\n[{feature} | {fold}] Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f}')
        print_metrics('Validation metrics', val_metrics)
        if score > best:
            best, best_epoch, patience_counter = score, epoch, 0
            best_state_dict = copy.deepcopy(model.state_dict())
            if args.save_model:
                torch.save(best_state_dict, save_path)
                print(f'Saved best model: {save_path} based on val {args.select_metric}={best:.4f}')
            else:
                print(f'Updated best model in memory based on val {args.select_metric}={best:.4f}')
        else:
            patience_counter += 1
        if args.patience > 0 and patience_counter >= args.patience:
            print(f'\nEarly stopping at epoch {epoch}.'); break

    pd.DataFrame(history).to_csv(out_prefix + '_training_history.csv', index=False)
    if best_state_dict is None:
        raise RuntimeError('No best model state was recorded. Please check training/validation data.')
    model.load_state_dict(best_state_dict)
    val_metrics, val_pred = evaluate(model, val_loader, device)
    test_metrics, test_pred = evaluate(model, test_loader, device)
    print_metrics('Best validation metrics', val_metrics); print_metrics('Independent test metrics', test_metrics)

    pd.DataFrame([{**{'feature':feature,'fold':fold}, **val_metrics}]).to_csv(out_prefix + '_val_metrics.csv', index=False)
    val_pred.insert(0,'fold',fold); val_pred.insert(0,'feature',feature); val_pred.to_csv(out_prefix + '_val_predictions.csv', index=False)
    pd.DataFrame([{**{'feature':feature,'fold':fold}, **test_metrics}]).to_csv(out_prefix + '_test_metrics.csv', index=False)
    test_pred.insert(0,'fold',fold); test_pred.insert(0,'feature',feature); test_pred.to_csv(out_prefix + '_test_predictions.csv', index=False)
    save_roc_curve_data(val_pred, out_prefix + '_val_roc_curve.csv', feature, fold, 'val')
    save_roc_curve_data(test_pred, out_prefix + '_test_roc_curve.csv', feature, fold, 'test')
    return {
        'feature':feature,
        'fold':fold,
        'best_epoch':best_epoch,
        'model_path': save_path if args.save_model else '',
        f'best_val_{args.select_metric}':best,
        **{f'val_{k}':v for k,v in val_metrics.items()},
        **{f'test_{k}':v for k,v in test_metrics.items()},
    }


def train_feature(args, feature, train_pkl, test_pkl, device):
    print('\n' + '#'*80); print('Current feature:', feature); print('#'*80)
    train_ds = PLMFeatureDataset(train_pkl, args.train_label_csv)
    test_ds = PLMFeatureDataset(test_pkl, args.test_label_csv)
    indices = np.arange(len(train_ds)); labels = np.array(train_ds.labels)
    input_dim = train_ds[0][0].shape[0]
    rows=[]
    if args.cv_folds > 1:
        counts = np.bincount(labels, minlength=2)
        if args.cv_folds > counts.min(): raise ValueError(f'cv_folds={args.cv_folds} > smallest class count={counts.min()}')
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        for fold_id,(tr,va) in enumerate(skf.split(indices, labels), 1):
            rows.append(train_one_split(args, feature, train_ds, test_ds, tr, va, device, input_dim, fold_id))
    else:
        tr,va = train_test_split(indices, test_size=args.val_ratio, random_state=args.seed, stratify=labels)
        rows.append(train_one_split(args, feature, train_ds, test_ds, tr, va, device, input_dim, None))
    return rows


def summarize_mean_std(df, group_col='feature'):
    """Return mean/std table for numeric metric columns grouped by feature."""
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ['best_epoch']]
    agg = df.groupby(group_col)[numeric_cols].agg(['mean', 'std'])
    agg.columns = [f'{metric}_{stat}' for metric, stat in agg.columns]
    return agg.reset_index()


def save_summary(rows, output_prefix):
    """
    Save both detailed fold-level results and clear mean/std summaries.

    Outputs:
      *_all_features_summary.csv          : val + test metrics, one row per feature per fold
      *_all_features_mean_std.csv         : val + test metrics, mean/std across folds
      *_cv_val_metrics_each_fold.csv      : validation metrics only, one row per feature per fold
      *_cv_val_metrics_mean_std.csv       : validation metrics, mean/std across folds
      *_cv_test_metrics_each_fold.csv     : independent test metrics only, one row per feature per fold
      *_cv_test_metrics_mean_std.csv      : independent test metrics, mean/std across fold-trained models
    """
    os.makedirs(os.path.dirname(output_prefix) or '.', exist_ok=True)
    df = pd.DataFrame(rows)

    summary_path = output_prefix + '_all_features_summary.csv'
    df.to_csv(summary_path, index=False)

    all_mean_std = summarize_mean_std(df, group_col='feature')
    all_mean_std_path = output_prefix + '_all_features_mean_std.csv'
    all_mean_std.to_csv(all_mean_std_path, index=False)

    val_cols = ['feature', 'fold', 'best_epoch'] + [c for c in df.columns if c.startswith('val_')]
    val_df = df[val_cols].copy()
    val_each_path = output_prefix + '_cv_val_metrics_each_fold.csv'
    val_mean_std_path = output_prefix + '_cv_val_metrics_mean_std.csv'
    val_df.to_csv(val_each_path, index=False)
    summarize_mean_std(val_df, group_col='feature').to_csv(val_mean_std_path, index=False)

    test_cols = ['feature', 'fold', 'best_epoch'] + [c for c in df.columns if c.startswith('test_')]
    test_df = df[test_cols].copy()
    test_each_path = output_prefix + '_cv_test_metrics_each_fold.csv'
    test_mean_std_path = output_prefix + '_cv_test_metrics_mean_std.csv'
    test_df.to_csv(test_each_path, index=False)
    summarize_mean_std(test_df, group_col='feature').to_csv(test_mean_std_path, index=False)

    for feature, feature_df in df.groupby('feature'):
        metric_col = 'val_' + 'BACC'
        if metric_col not in feature_df.columns:
            metric_candidates = [c for c in feature_df.columns if c.startswith('val_') and c.endswith('MCC')]
            metric_col = metric_candidates[0] if metric_candidates else None
        if metric_col and 'model_path' in feature_df.columns:
            best_row = feature_df.sort_values(metric_col, ascending=False, na_position='last').iloc[0]
            model_path = str(best_row.get('model_path', ''))
            if model_path and os.path.exists(model_path):
                root, ext = os.path.splitext(model_path)
                best_overall = os.path.join(os.path.dirname(model_path), f'{safe_name(feature)}_best_overall{ext or ".pt"}')
                shutil.copy2(model_path, best_overall)
                pd.DataFrame([{
                    'feature': feature,
                    'best_fold': best_row['fold'],
                    'best_epoch': best_row['best_epoch'],
                    'selection_metric': metric_col,
                    'selection_metric_value': best_row[metric_col],
                    'best_model_path': model_path,
                    'best_overall_model_path': best_overall,
                }]).to_csv(output_prefix + f'_{safe_name(feature)}_best_overall_model.csv', index=False)

    print('\nSaved summary files:')
    for path in [summary_path, all_mean_std_path, val_each_path, val_mean_std_path, test_each_path, test_mean_std_path]:
        print('  -', path)

    print('\nFeature comparison, mean ± std across folds:')
    for _, r in all_mean_std.iterrows():
        print('\nFeature:', r['feature'])
        for m in ['val_MCC', 'val_BACC', 'val_AUPR', 'test_MCC', 'test_BACC', 'test_AUPR']:
            mean_col, std_col = f'{m}_mean', f'{m}_std'
            if mean_col in all_mean_std.columns and std_col in all_mean_std.columns:
                print(f"{m}: {r[mean_col]:.4f} ± {r[std_col]:.4f}")

def main(args):
    set_seed(args.seed)
    if not (len(args.feature_names) == len(args.train_pkls) == len(args.test_pkls)):
        raise ValueError('The numbers of --feature_names, --train_pkls, --test_pkls must be the same.')
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print('Using device:', device); print('Classifier: MLP only'); print('Class weight:', 'enabled' if args.use_class_weight else 'disabled'); print('Save model:', 'enabled' if args.save_model else 'disabled')
    rows=[]
    for feature, train_pkl, test_pkl in zip(args.feature_names, args.train_pkls, args.test_pkls):
        rows.extend(train_feature(args, feature, train_pkl, test_pkl, device))
    save_summary(rows, args.output_prefix)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--feature_names', nargs='+', required=True)
    p.add_argument('--train_pkls', nargs='+', required=True)
    p.add_argument('--test_pkls', nargs='+', required=True)
    p.add_argument('--train_label_csv', type=str, default=None)
    p.add_argument('--test_label_csv', type=str, default=None)
    p.add_argument('--cv_folds', type=int, default=5)
    p.add_argument('--val_ratio', type=float, default=0.2)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--drop_last', action='store_true')
    p.add_argument('--mlp_hidden1', type=int, default=256)
    p.add_argument('--mlp_hidden2', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-3)
    p.add_argument('--grad_clip', type=float, default=5.0)
    p.add_argument('--select_metric', type=str, default='MCC', choices=['ACC','BACC','Precision','Recall','F1','MCC','AUC','AUPR'])
    p.add_argument('--use_class_weight', action='store_true', default=True)
    p.add_argument('--no_class_weight', action='store_false', dest='use_class_weight')
    p.add_argument('--use_scheduler', action='store_true')
    p.add_argument('--scheduler_patience', type=int, default=10)
    p.add_argument('--save_path', type=str, default='models/best_mlp.pt')
    p.add_argument('--save_model', action='store_true', default=True, help='Save best model weights for each feature/fold. Default: enabled.')
    p.add_argument('--no_save_model', action='store_false', dest='save_model', help='Disable saving best model weights.')
    p.add_argument('--output_prefix', type=str, default='results/mlp_feature_compare')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu', action='store_true')
    main(p.parse_args())
