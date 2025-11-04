#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import json
import argparse
import random
from typing import Dict, Tuple, Generator

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, matthews_corrcoef,
    precision_recall_curve, classification_report, precision_recall_fscore_support
)

EVAL_BATCH_SIZE = 8192
STREAM_BATCH_SIZE = 8192

# Reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# HDF5 loaders
def extract_from_hdf5(hdf5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f"HDF5 not found: {hdf5_path}")

    X_list, y_list = [], []
    skipped = 0
    with h5py.File(hdf5_path, 'r') as f:
        if 'interactions' not in f:
            raise KeyError("Missing 'interactions' group in HDF5")
        for interaction_id in f['interactions']:
            g = f['interactions'][interaction_id]
            if 'targets' not in g:
                skipped += 1
                continue
            y = g['targets'][()]
            protein_path = g.attrs.get('protein')
            ligand_path  = g.attrs.get('ligand')
            if protein_path is None or ligand_path is None:
                skipped += 1
                continue
            try:
                p = f[protein_path]['features'][()]
                l = f[ligand_path]['features'][()]
                L = np.repeat(l, p.shape[0], axis=0)
                X = np.concatenate([p, L], axis=-1)
                X_list.append(X)
                y_list.append(y)
            except KeyError:
                skipped += 1
                continue
    if not X_list:
        raise RuntimeError(f"No valid interactions found in {hdf5_path}")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0).reshape(-1, 1)
    print(f"Loaded: {hdf5_path}\n  → X.shape={X.shape}, y.shape={y.shape}, skipped={skipped}")
    return X, y


def iter_batches_from_hdf5(hdf5_path: str, batch_size: int = STREAM_BATCH_SIZE
                           ) -> Generator[Tuple[str, np.ndarray, np.ndarray], None, None]:
    with h5py.File(hdf5_path, 'r') as f:
        if 'interactions' not in f:
            return
        for interaction_id in f['interactions']:
            g = f['interactions'][interaction_id]
            if 'targets' not in g:
                continue
            y = g['targets'][()].ravel().astype(np.int64)

            protein_path = g.attrs.get('protein')
            ligand_path  = g.attrs.get('ligand')
            if protein_path is None or ligand_path is None:
                continue

            try:
                p = f[protein_path]['features'][()]
                l = f[ligand_path]['features'][()]
                L = np.repeat(l, p.shape[0], axis=0)
                Feat = np.concatenate([p, L], axis=-1)

                for s in range(0, len(y), batch_size):
                    e = min(s + batch_size, len(y))
                    yield interaction_id, Feat[s:e], y[s:e]
            except KeyError:
                continue


# Model
class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 activation: str, dropout: float, batch_norm: bool = False):
        super().__init__()
        act_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
        }
        if activation not in act_map:
            raise ValueError(f"Unsupported activation: {activation}")
        act_fn = act_map[activation]

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).view(-1)


# Train / Eval helpers
def train_with_early_stopping(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                              optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str,
                              max_epochs: int = 50, patience: int = 5) -> nn.Module:
    model.to(device)
    best_state = None
    best_f1 = -1.0
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb.view(-1).float())
            loss.backward()
            optimizer.step()
            running += loss.item()

        # validation (batched)
        model.eval()
        preds, labels = [], []
        with torch.inference_mode():
            for xb, yb in val_loader:
                pb = torch.sigmoid(model(xb.to(device))).cpu().numpy().ravel()
                preds.append((pb > 0.5).astype(int))
                labels.append(yb.numpy().ravel())
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(labels)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"Epoch {epoch+1:3d}/{max_epochs} | loss {running/len(train_loader):.4f} | val F1 {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (best F1={best_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_dataset_batched(X: np.ndarray, y: np.ndarray, model: nn.Module, device: str,
                             name: str, results_dir: str) -> None:
    y = np.asarray(y).ravel()
    model.eval()
    probs = np.empty(len(y), dtype=np.float32)

    with torch.inference_mode():
        for s in range(0, len(y), EVAL_BATCH_SIZE):
            e = min(s + EVAL_BATCH_SIZE, len(y))
            xb = torch.from_numpy(X[s:e]).float().to(device, non_blocking=True)
            logits = model(xb)
            probs[s:e] = torch.sigmoid(logits).cpu().numpy().ravel()
            del xb, logits

    preds = (probs > 0.5).astype(int)
    if len(np.unique(y)) > 1:
        auc_roc = roc_auc_score(y, probs)
        auc_pr  = average_precision_score(y, probs)
    else:
        auc_roc = float('nan'); auc_pr = float('nan')

    print(f"\n[{name}]")
    print(f" ACC     : {accuracy_score(y, preds):.4f}")
    print(f" F1      : {f1_score(y, preds, zero_division=0):.4f}")
    print(f" PREC    : {precision_score(y, preds, zero_division=0):.4f}")
    print(f" REC     : {recall_score(y, preds, zero_division=0):.4f}")
    print(f" AUC-ROC : {auc_roc:.4f}")
    print(f" AUC-PR  : {auc_pr:.4f}")
    print(f" MCC     : {matthews_corrcoef(y, preds):.4f}")
    print("\n[Classification Report]")
    print(classification_report(y, preds, digits=4, zero_division=0))

    precision, recall, _ = precision_recall_curve(y, probs)
    os.makedirs(results_dir, exist_ok=True)
    out_png = os.path.join(results_dir, f"{name.lower().replace(' ', '_')}_precision_recall_curve.png")
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', lw=2)
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'Precision-Recall Curve ({name})')
    plt.grid(True); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()


def find_best_interactions_streaming(hdf5_path: str, model: nn.Module, device: str,
                                     dataset_name: str, results_dir: str,
                                     top_k: int = 4):
    model.eval()
    buckets: Dict[str, Dict[str, list]] = {}

    with torch.inference_mode():
        for interaction_id, Xb, yb in iter_batches_from_hdf5(hdf5_path, batch_size=STREAM_BATCH_SIZE):
            xb = torch.from_numpy(Xb).float().to(device, non_blocking=True)
            pb = torch.sigmoid(model(xb)).cpu().numpy().ravel()
            yhat = (pb > 0.5).astype(int)
            if interaction_id not in buckets:
                buckets[interaction_id] = {"y": [], "yhat": []}
            buckets[interaction_id]["y"].append(yb)
            buckets[interaction_id]["yhat"].append(yhat)
            del xb

    best = []
    for iid, d in buckets.items():
        y_true = np.concatenate(d["y"]).astype(int)
        y_pred = np.concatenate(d["yhat"]).astype(int)
        _, _, f1s, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1], zero_division=0
        )
        f1_pos = float(f1s[1]) if len(f1s) == 2 else 0.0
        true_nested = [[int(v)] for v in y_true.tolist()]
        best.append({
            "interaction_id": iid,
            "f1_score_class_1": f1_pos,
            "true_labels": true_nested,
            "predicted_labels": [int(v) for v in y_pred.tolist()]
        })

    best.sort(key=lambda x: -x['f1_score_class_1'])
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"best_interactions_{dataset_name.lower().replace(' ', '_')}.json")
    with open(save_path, "w") as f:
        json.dump(best[:top_k], f, indent=4)
    print(f"\nSaved top {min(top_k, len(best))} for {dataset_name} → {save_path}")

    print(f"\n[Top {min(top_k, len(best))} interactions by F1(class=1) — {dataset_name}]")
    for ex in best[:top_k]:
        print(f"Interaction: {ex['interaction_id']}")
        print(f"F1 Score (Class 1): {ex['f1_score_class_1']:.4f}")
        print(f"True Labels: {ex['true_labels']}")
        print(f"Predicted Labels: {ex['predicted_labels']}\n")

    del buckets; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Params
def load_params(json_path: str) -> Dict:
    with open(json_path, "r") as f:
        cfg = json.load(f)
    if isinstance(cfg, dict) and "parameters" in cfg:
        cfg = cfg["parameters"]
    return cfg


# Main
def main():
    p = argparse.ArgumentParser(description="Train + eval MLP with best hyperparameters (memory-safe).")
    p.add_argument('--train', required=True, help='Path to train HDF5')
    p.add_argument('--val', required=True, help='Path to validation HDF5')
    p.add_argument('--eval', action='append', default=[], help='Extra eval datasets as \"Name=path\" (repeatable)')
    p.add_argument('--params', required=True, help='Path to JSON of best hyperparameters')
    p.add_argument('--results-dir', required=True, help='Directory for outputs (curves, json, model)')
    p.add_argument('--model-name', required=True, help='Base name of saved TorchScript model (.pt)')
    args = p.parse_args()

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load train/val fully (to match original behavior)
    X_train, y_train = extract_from_hdf5(args.train)
    X_val,   y_val   = extract_from_hdf5(args.val)

    # Load best hyperparameters
    params = load_params(args.params)
    hidden_dim   = int(params['hidden_dim'])
    num_layers   = int(params['num_layers'])
    activation   = str(params['activation'])
    dropout      = float(params['dropout'])
    batch_norm   = bool(params.get('batch_norm', False))
    batch_size   = int(params['batch_size'])
    max_epochs   = int(params['epochs'])
    optimizer_nm = str(params['optimizer']).lower()
    learning_rate= float(params['learning_rate'])
    weight_decay = float(params['weight_decay'])
    pos_w_value  = float(params['pos_weight_value'])

    # Model / Optimizer / Loss
    input_dim = X_train.shape[1]
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w_value, device=device))
    optimizer_cls = torch.optim.Adam if optimizer_nm == 'adam' else torch.optim.SGD
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # DataLoaders
    g = torch.Generator(); g.manual_seed(42)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                            torch.tensor(y_train, dtype=torch.float32)),
                              batch_size=batch_size, shuffle=True, generator=g)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                            torch.tensor(y_val, dtype=torch.float32)),
                              batch_size=batch_size, shuffle=False)

    # Train
    model = train_with_early_stopping(
        model, train_loader, val_loader, optimizer, criterion, device,
        max_epochs=max_epochs, patience=5
    )

    # Ensure output dir
    os.makedirs(args.results_dir, exist_ok=True)

    # Evaluate train/val (batched)
    evaluate_dataset_batched(X_train, y_train, model, device, 'Train Set', args.results_dir)
    evaluate_dataset_batched(X_val,   y_val,   model, device, 'Validation Set', args.results_dir)

    # Free host RAM before big evals
    del X_train, y_train, X_val, y_val
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Extra evals (streaming)
    eval_specs: Dict[str, str] = {}
    for spec in args.eval:
        if '=' not in spec:
            raise ValueError("--eval expects Name=path")
        name, path = spec.split('=', 1)
        eval_specs[name.strip()] = path.strip()

    for name, path in eval_specs.items():
        # stream metrics + PR curve
        probs, labels = [], []
        model.eval()
        with torch.inference_mode():
            for _, Xb, yb in iter_batches_from_hdf5(path, batch_size=STREAM_BATCH_SIZE):
                xb = torch.from_numpy(Xb).float().to(device, non_blocking=True)
                pb = torch.sigmoid(model(xb)).cpu().numpy().ravel()
                probs.append(pb); labels.append(yb)
                del xb
        probs = np.concatenate(probs); y = np.concatenate(labels)
        preds = (probs > 0.5).astype(int)

        if len(np.unique(y)) > 1:
            auc_roc = roc_auc_score(y, probs); auc_pr = average_precision_score(y, probs)
        else:
            auc_roc = float('nan'); auc_pr = float('nan')

        print(f"\n[{name}]")
        print(f" ACC     : {accuracy_score(y, preds):.4f}")
        print(f" F1      : {f1_score(y, preds, zero_division=0):.4f}")
        print(f" PREC    : {precision_score(y, preds, zero_division=0):.4f}")
        print(f" REC     : {recall_score(y, preds, zero_division=0):.4f}")
        print(f" AUC-ROC : {auc_roc:.4f}")
        print(f" AUC-PR  : {auc_pr:.4f}")
        print(f" MCC     : {matthews_corrcoef(y, preds):.4f}")
        print("\n[Classification Report]")
        print(classification_report(y, preds, digits=4, zero_division=0))

        precision, recall, _ = precision_recall_curve(y, probs)
        out_png = os.path.join(args.results_dir, f"{name.lower().replace(' ', '_')}_precision_recall_curve.png")
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.', lw=2)
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'Precision-Recall Curve ({name})')
        plt.grid(True); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

        # best interactions streaming
        find_best_interactions_streaming(path, model, device, dataset_name=name,
                                         results_dir=args.results_dir, top_k=4)

        del probs, labels, y, preds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save TorchScript
    model_cpu = model.to('cpu').eval()
    example_input = torch.randn(1, input_dim)
    scripted = torch.jit.trace(model_cpu, example_input)
    pt_path = os.path.join(args.results_dir, f"{args.model_name}.pt")
    torch.jit.save(scripted, pt_path)
    print(f"Saved TorchScript → {pt_path}")

    if eval_specs:
        first_name, first_path = next(iter(eval_specs.items()))
        take = 2048
        buf, cnt = [], 0
        for _, Xb, _ in iter_batches_from_hdf5(first_path, batch_size=1024):
            buf.append(Xb); cnt += len(Xb)
            if cnt >= take: break
        if buf:
            X_ref = torch.tensor(np.concatenate(buf, axis=0)[:take], dtype=torch.float32)
            with torch.no_grad():
                p1 = torch.sigmoid(model_cpu(X_ref)).numpy().ravel()
                ts = torch.jit.load(pt_path, map_location='cpu').eval()
                p2 = torch.sigmoid(ts(X_ref)).numpy().ravel()
            print("Δ(mean) =", float(abs(p1.mean()-p2.mean())))
            print("Δ(std)  =", float(abs(p1.std()-p2.std())))
            print("max|Δ|  =", float(np.max(np.abs(p1-p2))))
            print("allclose?", np.allclose(p1, p2, atol=1e-6, rtol=1e-6))


if __name__ == '__main__':
    main()
