#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import optuna
import numpy as np
import os
import random
import h5py
import json
import pandas as pd
import argparse

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description="Optuna MLP tuner")
parser.add_argument("--train", required=True, help="Path to train HDF5")
parser.add_argument("--val", required=True, help="Path to validation HDF5")
parser.add_argument("--results-dir", required=True, help="Directory to store DB and best params JSON")
parser.add_argument("--study-name", default="mlp_study_esm2_clusters", help="Optuna study name")
parser.add_argument("--db-name", default="optuna_mlp_esm2_clusters.db", help="SQLite DB file name")
parser.add_argument("--best-json-name", default="best_mlp_params_esm2_clusters.json", help="Best params JSON file name")
parser.add_argument("--n-trials", type=int, default=200, help="Number of Optuna trials")
args = parser.parse_args()
os.makedirs(args.results_dir, exist_ok=True)

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
g = torch.Generator()
g.manual_seed(42)

# Load Data from HDF5
def extract_from_hdf5(hdf5_path):
    X_list, y_list = [], []
    with h5py.File(hdf5_path, 'r') as f:
        for interaction_id in f['interactions']:
            group = f['interactions'][interaction_id]
            y = group['targets'][()]
            protein_path = group.attrs['protein']
            ligand_path = group.attrs['ligand']
            try:
                protein_features = f[protein_path]['features'][()]
                ligand_features = f[ligand_path]['features'][()]
                ligand_broadcast = np.repeat(ligand_features, protein_features.shape[0], axis=0)
                features = np.concatenate([protein_features, ligand_broadcast], axis=-1)
                X_list.append(features)
                y_list.append(y)
            except KeyError as e:
                print(f"[Skipping] {interaction_id}: {e}", flush=True)
                continue
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0).reshape(-1, 1)
    print(f"Loaded dataset: X.shape={X.shape}, y.shape={y.shape}")
    return X, y

hdf5_path = args.train
hdf5_val_path = args.val

X_train, y_train = extract_from_hdf5(hdf5_path)
X_val_fixed, y_val_fixed = extract_from_hdf5(hdf5_val_path)

# Make them global for the objective()
X_TR_ALL, Y_TR_ALL = X_train, y_train
X_VAL_ALL, Y_VAL_ALL = X_val_fixed, y_val_fixed

# Custom MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, activation, dropout, batch_norm=False):
        super().__init__()
        act_fn = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU()
        }[activation]

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))  # Output layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).view(-1)

# Training Loop per Fold
def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs, early_stopping_rounds=5):
    model.to(device)
    best_f1 = 0
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb.view(-1).float())
            loss.backward()
            optimizer.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = torch.sigmoid(model(xb)).cpu().numpy()
                preds.append(out)
                labels.append(yb.numpy())

        y_true = np.concatenate(labels)
        y_score = np.concatenate(preds)
        y_pred = (y_score > 0.5).astype(int)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= early_stopping_rounds:
                break

    return best_f1


# Optuna Objective Function
def objective(trial):
    activation = trial.suggest_categorical("activation", ["relu", "gelu", "tanh", "leaky_relu"])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    epochs = trial.suggest_int("epochs", 20, 100)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    hidden_dim = trial.suggest_int("hidden_dim", 128, 1024, step=128)
    num_layers = trial.suggest_int("num_layers", 2, 12)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_norm = trial.suggest_categorical("batch_norm", [True, False])

    # Class imbalance 
    pos_weight_value = trial.suggest_float("pos_weight_value", 1, 25, log=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_weight_tensor = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Build model & optimizer
    model = MLPClassifier(X_TR_ALL.shape[1], hidden_dim, num_layers, activation, dropout, batch_norm=batch_norm)
    optimizer_cls = torch.optim.Adam if optimizer_name == "adam" else torch.optim.SGD
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Tensors & loaders
    X_tr = torch.tensor(X_TR_ALL, dtype=torch.float32)
    y_tr = torch.tensor(Y_TR_ALL, dtype=torch.float32)
    X_val = torch.tensor(X_VAL_ALL, dtype=torch.float32)
    y_val = torch.tensor(Y_VAL_ALL, dtype=torch.float32)

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, generator=g)

    val_pos_rate = float(np.mean(y_val.numpy()))
    print(f"Fixed VAL pos rate: {val_pos_rate:.3f}")

    f1 = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device=device,
        epochs=epochs,
        early_stopping_rounds=5
    )
    return f1

# Run Optuna and Save Hyperparameters
def run_optuna(results_dir, study_name, db_path, best_params_path, n_trials=200):
    os.makedirs(results_dir, exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials)

    print("\nBest Trial:")
    print(study.best_trial)

    with open(best_params_path, "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
    print(f"Best hyperparameters saved to {best_params_path}")
    print(f"Optuna DB stored at: {db_path}")

    return study

if __name__ == "__main__":
    db_path = os.path.join(args.results_dir, args.db_name)
    best_json_path = os.path.join(args.results_dir, args.best_json_name)
    study = run_optuna(
        results_dir=args.results_dir,
        study_name=args.study_name,
        db_path=db_path,
        best_params_path=best_json_path,
        n_trials=args.n_trials
    )
