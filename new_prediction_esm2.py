#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, warnings, argparse
import numpy as np
import pandas as pd
import torch
import joblib
from rdkit import Chem
from mordred import Calculator, descriptors

warnings.filterwarnings("ignore", category=UserWarning)


def build_cli():
    p = argparse.ArgumentParser(
        description="Predict per-residue probabilities from an Excel file (PDB, CHAIN, LIG_ID, SEQUENCE, SMILES) using ESM-2 + Mordred + TorchScript MLP."
    )
    p.add_argument("--input-file", required=True, help="Excel (.xlsx) with the required columns")
    p.add_argument("--ref-h5", required=True, help="HDF5 reference file with 'mordred_column_names'")
    p.add_argument("--prot-scaler", required=True, help="PKL with fitted protein scaler")
    p.add_argument("--lig-scaler", required=True, help="PKL with fitted ligand scaler")
    p.add_argument("--model", required=True, help="TorchScript model (.pt)")
    p.add_argument("--outdir", required=True, help="Output directory for per-interaction CSVs")
    return p


def _assert_file(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def _safe_name(s: str) -> str:
    """Make filesystem-safe filename."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def canonical_smiles(s):
    """Convert SMILES to canonical form; if invalid, return the original string."""
    try:
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            return str(s)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        # keep original
        return str(s)


def load_mordred_columns_only(ref_h5):
    import h5py
    with h5py.File(ref_h5, "r") as f:
        if "mordred_column_names" not in f:
            raise KeyError("REFERENCE_HDF5 must contain dataset 'mordred_column_names'")
        mordred_cols = [c.decode("utf-8") for c in f["mordred_column_names"][:]]
    return mordred_cols


# ESM-2
def load_esm(device, esm_name="esm2_t33_650M_UR50D"):
    import esm
    print(f"Loading ESM-2 model ({esm_name})...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(esm_name)
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device).eval()
    layer_idx = getattr(model, "num_layers", 33)
    return model, batch_converter, layer_idx


def esm_embed(seq_id, seq, esm_model, batch_converter, device, layer_idx):
    """Generate ESM-2 embeddings for one protein sequence."""
    data = [(seq_id, seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    with torch.no_grad():
        res = esm_model(tokens, repr_layers=[layer_idx], return_contacts=False)
    arr = res["representations"][layer_idx].cpu().numpy()[0]  # (L+1, D)
    return arr[1:1+len(seq), :].astype(np.float32)            # drop [CLS]


# MORDRED
_MORDRED = Calculator(descriptors, ignore_3D=True)

def mordred_vec(smiles, mordred_columns):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    df = _MORDRED.pandas([mol])
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype("Int8")
    for c in mordred_columns:
        if c not in df.columns:
            df[c] = 0.0
    df = df.reindex(columns=mordred_columns)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vec = df.values.astype(np.float32)
    return vec


# FEATURES
def build_residue_features(prot_feats_std, lig_feats_std):
    if lig_feats_std.ndim == 1:
        lig_feats_std = lig_feats_std.reshape(1, -1)
    L = prot_feats_std.shape[0]
    lig_broadcast = np.repeat(lig_feats_std, L, axis=0)
    return np.concatenate([prot_feats_std, lig_broadcast], axis=-1)


def main():
    args = build_cli().parse_args()

    esm_name = "esm2_t33_650M_UR50D"
    apply_sigmoid = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Validate files
    for f, lbl in [
        (args.input_file, "INPUT_EXCEL"),
        (args.ref_h5, "REFERENCE_HDF5"),
        (args.prot_scaler, "PROT_SCALER_PKL"),
        (args.lig_scaler, "LIG_SCALER_PKL"),
        (args.model, "TORCHSCRIPT_PT"),
    ]:
        _assert_file(f, lbl)

    # Load reference and scalers
    mordred_cols = load_mordred_columns_only(args.ref_h5)
    prot_scaler = joblib.load(args.prot_scaler)
    lig_scaler  = joblib.load(args.lig_scaler)
    print(f"Loaded {len(mordred_cols)} Mordred columns.")
    mlp = torch.jit.load(args.model, map_location=device).eval()
    esm_model, batch_converter, esm_layer = load_esm(device, esm_name)

    # Load Excel
    df = pd.read_excel(args.input_file, dtype=str).fillna("")
    colmap = {c.lower(): c for c in df.columns}
    required = ["pdb", "chain", "lig_id", "sequence", "smiles"]
    missing = [r for r in required if r not in colmap]
    if missing:
        raise ValueError(f"Missing required columns in Excel: {missing}")
    PDB, CHAIN, LIG_ID, SEQ, SMILES = [colmap[k] for k in required]

    # Canonicalize SMILES
    df["SMILES_CANON"] = df[SMILES].apply(canonical_smiles)
    bad = int((df["SMILES_CANON"] == "").sum())
    if bad:
        print(f" Dropping {bad} invalid SMILES")
    df = df[df["SMILES_CANON"] != ""].reset_index(drop=True)

    # Unique proteins & ligands
    prot_keys = df[[PDB, CHAIN, SEQ]].drop_duplicates().reset_index(drop=True)
    lig_keys  = df[[LIG_ID, "SMILES_CANON"]].drop_duplicates().reset_index(drop=True)

    # Compute features
    print("\nGenerating ESM-2 embeddings for proteins...")
    prot_feat_std = {}
    for _, row in prot_keys.iterrows():
        pid = f"{row[PDB]}:{row[CHAIN]}"
        seq = str(row[SEQ]).upper().replace(" ", "")
        feats = esm_embed(pid, seq, esm_model, batch_converter, device, esm_layer)
        prot_feat_std[pid] = prot_scaler.transform(feats)

    print("Generating Mordred descriptors for ligands...")
    lig_feat_std = {}
    for _, row in lig_keys.iterrows():
        lid = str(row[LIG_ID])
        smi = str(row["SMILES_CANON"])
        vec = mordred_vec(smi, mordred_cols)
        lig_feat_std[lid] = lig_scaler.transform(vec)

    # Base output directory under "new_prediction/"
    base_dir = "new_prediction"
    os.makedirs(base_dir, exist_ok=True)

    # Create subdirectory for user-specified outdir
    outdir = os.path.join(base_dir, _safe_name(args.outdir))
    os.makedirs(outdir, exist_ok=True)

    # Per-interaction CSV folder
    per_interaction_dir = os.path.join(outdir, "per_interaction_csv")
    os.makedirs(per_interaction_dir, exist_ok=True)

    # Predict per interaction
    print("\nRunning predictions...")
    for _, r in df.iterrows():
        pid = f"{r[PDB]}:{r[CHAIN]}"
        lid = str(r[LIG_ID])
        seq = str(r[SEQ]).upper().replace(" ", "")
        interaction_id = f"{r[PDB]}:{r[CHAIN]}:{lid}"

        X = build_residue_features(prot_feat_std[pid], lig_feat_std[lid])
        with torch.no_grad():
            out = mlp(torch.tensor(X, dtype=torch.float32, device=device)).squeeze(-1)
            probs = torch.sigmoid(out).cpu().numpy().ravel() if apply_sigmoid else out.cpu().numpy().ravel()
            preds = (probs >= 0.5).astype(int)

        per_rows = []
        for pos, aa in enumerate(seq, start=1):
            per_rows.append({
                "Interaction_ID": interaction_id,
                "PDB": r[PDB],
                "CHAIN": r[CHAIN],
                "LIG_ID": lid,
                "Position": pos,
                "Residue_1L": aa,
                "Probability": float(probs[pos-1]),
                "Pred": int(preds[pos-1]),
            })

        csv_path = os.path.join(per_interaction_dir, _safe_name(f"{r[PDB]}_{r[CHAIN]}_{lid}.csv"))
        pd.DataFrame(per_rows).to_csv(csv_path, index=False)

    print(f"\n Predictions saved in: {per_interaction_dir}")


if __name__ == "__main__":
    main()