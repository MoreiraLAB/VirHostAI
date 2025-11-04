#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import h5py
import json
import joblib
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, matthews_corrcoef,
    precision_recall_curve, classification_report, precision_recall_fscore_support
)
from lightgbm import LGBMClassifier

parser = argparse.ArgumentParser(description="General LGBM runner for ViralBindPredict-style HDF5s")
parser.add_argument('--train', required=True, help='Path to train HDF5')
parser.add_argument('--val', required=True, help='Path to validation HDF5')
parser.add_argument('--eval', action='append', default=[], help='Additional eval datasets: name=path (repeatable)')
parser.add_argument('--params', required=True, help='Path to JSON with best hyperparameters')
parser.add_argument('--results-dir', required=True, help='Directory to save model, curves, and summaries')
parser.add_argument('--model-name', default='lgbm_model', help='Base name for saved model (.pkl)')
args = parser.parse_args()
os.makedirs(args.results_dir, exist_ok=True)

# Seed
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
set_seed(42)

# Data Loading
def extract_from_hdf5(hdf5_path):
    X_list, y_list = [], []
    with h5py.File(hdf5_path, 'r') as f:
        for interaction_id in f['interactions']:
            group = f['interactions'][interaction_id]
            y = group['targets'][()].astype(np.int8) 
            protein_path = group.attrs['protein']
            ligand_path = group.attrs['ligand']
            try:
                protein_features = f[protein_path]['features'][()].astype(np.float32)
                ligand_features = f[ligand_path]['features'][()].astype(np.float32)
                ligand_broadcast = np.repeat(ligand_features, protein_features.shape[0], axis=0)
                features = np.concatenate([protein_features, ligand_broadcast], axis=-1)
                X_list.append(features)
                y_list.append(y)
            except KeyError as e:
                print(f"[Skipping] {interaction_id}: {e}", flush=True)
                continue
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0).reshape(-1, 1)
    print(f"Loaded dataset: X.shape={X.shape}, y.shape={y.shape}, dtype={X.dtype}")
    return X, y

# Metrics / Plots
def evaluate_lgbm_dataset(X, y, model, name, results_dir, save_curve=False):
    y = np.asarray(y).ravel()
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # handle single-class AUCs
    if len(np.unique(y)) > 1:
        auc_roc = roc_auc_score(y, y_prob)
        auc_pr  = average_precision_score(y, y_prob)
    else:
        auc_roc = float('nan'); auc_pr = float('nan')

    print(f"\n[{name} Performance]")
    print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y, y_pred, zero_division=0):.4f}")
    print(f"Precision: {precision_score(y, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y, y_pred, zero_division=0):.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    print(f"AUC-PR:    {auc_pr:.4f}")
    print(f"MCC:       {matthews_corrcoef(y, y_pred):.4f}")

    print("\n[Classification Report]")
    print(classification_report(y, y_pred, digits=4, zero_division=0))

    if save_curve:
        precision, recall, _ = precision_recall_curve(y, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.', lw=2)
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve ({name})')
        plt.grid(True); plt.tight_layout()
        out_png = os.path.join(results_dir, f"{name.lower().replace(' ', '_')}_precision_recall_curve.png")
        plt.savefig(out_png, dpi=150)
        plt.close()

# Top 3 interactions
def top_interactions(hdf5_path, model, dataset_name, results_dir, threshold=0.5):
    best_examples = []
    with h5py.File(hdf5_path, 'r') as f:
        if 'interactions' not in f:
            print(f"[{dataset_name}] No 'interactions' group found.")
            return []
        for interaction_id in f['interactions']:
            group = f['interactions'][interaction_id]
            if 'targets' not in group:
                continue
            y_true = group['targets'][()]
            protein_path = group.attrs.get('protein')
            ligand_path  = group.attrs.get('ligand')
            if protein_path is None or ligand_path is None:
                continue
            try:
                protein_features = f[protein_path]['features'][()]
                ligand_features  = f[ligand_path]['features'][()]
                ligand_broadcast = np.repeat(ligand_features, protein_features.shape[0], axis=0)
                features = np.concatenate([protein_features, ligand_broadcast], axis=-1)

                y_prob = model.predict_proba(features)[:, 1]
                y_pred = (y_prob > threshold).astype(int)
                _, _, f1s, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
                f1_pos = float(f1s[1]) if len(f1s) == 2 else 0.0

                best_examples.append({
                    "interaction_id": interaction_id,
                    "f1_score_class_1": f1_pos,
                    "true_labels": y_true.tolist(),
                    "predicted_labels": y_pred.tolist()
                })
            except KeyError as e:
                print(f"[{dataset_name}] Skipping {interaction_id}: {e}")
                continue

    best_examples = sorted(best_examples, key=lambda x: -x['f1_score_class_1'])
    top = best_examples[:3]  # fixed to top 3

    out_json = os.path.join(results_dir, f"best_interactions_{dataset_name.lower().replace(' ', '_')}.json")
    with open(out_json, "w") as f:
        json.dump(top, f, indent=2)

    print(f"\nSaved top 3 interactions for {dataset_name} → {out_json}")
    for ex in top:
        print(f"\nInteraction: {ex['interaction_id']}")
        print(f"F1 Score (Class 1): {ex['f1_score_class_1']:.4f}")
        print(f"True Labels: {ex['true_labels']}")
        print(f"Predicted Labels: {ex['predicted_labels']}")
    return top

# Run
def main():
    # Load train/val
    X_train, y_train = extract_from_hdf5(args.train)
    X_val,   y_val   = extract_from_hdf5(args.val)

    # Load best params
    with open(args.params) as f:
        best_params = json.load(f)

    best_params.update({
    "objective": "binary",
    "boosting_type": "gbdt",
    "device": "cpu",
    "n_jobs": -1,
    "deterministic": True,
    })

    # Train
    clf = LGBMClassifier(**best_params)
    clf.fit(
        X_train, np.asarray(y_train).ravel(),
        eval_set=[(X_val, np.asarray(y_val).ravel())],
        eval_metric="binary_logloss",
    )

    # Save model
    model_path = os.path.join(args.results_dir, f"{args.model_name}.pkl")
    joblib.dump(clf, model_path)
    print(f"\nSaved trained LightGBM model to {model_path}")

    # Evaluate
    evaluate_lgbm_dataset(X_train, y_train, clf, "Train", args.results_dir, save_curve=False)
    evaluate_lgbm_dataset(X_val,   y_val,   clf, "Validation", args.results_dir, save_curve=False)

    # Parse eval datasets
    eval_specs = {}
    for spec in args.eval:
        if '=' not in spec:
            raise ValueError("--eval expects name=path")
        name, path = spec.split('=', 1)
        eval_specs[name.strip()] = path.strip()

    for name, path in eval_specs.items():
        X_e, y_e = extract_from_hdf5(path)
        evaluate_lgbm_dataset(X_e, y_e, clf, name, args.results_dir, save_curve=False)
        top_interactions(path, clf, dataset_name=name, results_dir=args.results_dir)

if __name__ == "__main__":
    main()
