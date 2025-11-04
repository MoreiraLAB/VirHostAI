#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
import json
import os
import random
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Optuna LGM tuner")
parser.add_argument("--train", required=True, help="Path to train HDF5")
parser.add_argument("--val", required=True, help="Path to validation HDF5")
parser.add_argument("--results-dir", required=True, help="Directory to store DB and best params JSON")
parser.add_argument("--study-name", required=True, help="Optuna study name. Can be any")
parser.add_argument("--db-name", required=True, help="SQLite DB file name. Can be any")
parser.add_argument("--best-json-name", required=True, help="Best params JSON file name")
parser.add_argument("--n-trials", type=int, default=200, help="Number of Optuna trials")
args = parser.parse_args()
os.makedirs(args.results_dir, exist_ok=True)

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

os.environ["LGBM_DETERMINISTIC"] = "1"
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "1" 

set_seed(42)

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

# Define Optuna Objective Function
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0, 5), 
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 25.0),  # Class imbalance
        "objective": "binary",
        "boosting_type": "gbdt",
        "device": "cpu", 
        "deterministic": True
    }

    y_tr = Y_TR_ALL.ravel().astype(int)
    y_va = Y_VAL_ALL.ravel().astype(int)

    clf = LGBMClassifier(**params)

    clf.fit(
        X_TR_ALL, y_tr,
        eval_set=[(X_VAL_ALL, y_va)],
        eval_metric="binary_logloss",
    )

    y_prob = clf.predict_proba(X_VAL_ALL)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return f1_score(y_va, y_pred)

storage_path = f"sqlite:///{os.path.join(args.results_dir, args.db_name)}"
study_name = args.study_name

try:
    study = optuna.load_study(study_name=study_name, storage=storage_path)
    print("Loaded existing study.", flush=True)
except KeyError:
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name=study_name,
        storage=storage_path
    )
    print("Created new study.", flush=True)

# Optimize
study.optimize(objective, n_trials=args.n_trials)

print("\nBest Trial:")
print(study.best_trial)
    
best_json_path = os.path.join(args.results_dir, args.best_json_name)
with open(best_json_path, "w") as f:
    json.dump(study.best_params, f, indent=4)
print(f"\n Best hyperparameters saved to {best_json_path}")
