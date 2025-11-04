#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import h5py
import torch
import re
import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from tqdm import tqdm
from Bio import SeqIO
from transformers import T5Tokenizer, T5EncoderModel
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Paths
binary_csv_dir = "binary_interactions_csv"
fasta_dir = "fasta_sequences_filtered"
output_hdf5_path = "interaction_dataset_prottrans_mordred.hdf5"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ProtTrans model
print(" Loading ProtTrans T5 model...")
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)
if device.type == "cpu":
    model = model.to(torch.float32)
model.eval()

# Initialize Mordred
print(" Initializing Mordred descriptors...")
mordred_calc = Calculator(descriptors, ignore_3D=True)

# Reference descriptor set (fixed shape)
print(" Precomputing full Mordred descriptor set...")
dummy = Chem.MolFromSmiles("CC")
ref_df = mordred_calc.pandas([dummy])
mordred_columns = ref_df.columns

# Debug tracking
skipped_proteins = []
skipped_ligands = []
saved_proteins = set()
saved_ligands = set()

# Create HDF5 file
with h5py.File(output_hdf5_path, "w") as hdf:

    # Save full reference column names
    mordred_col_bytes = np.array([c.encode("utf-8") for c in mordred_columns], dtype="S")
    hdf.create_dataset("mordred_column_names", data=mordred_col_bytes)

    for csv_file in tqdm(sorted(os.listdir(binary_csv_dir)), desc=" Processing CSVs"):
        if not csv_file.endswith("_binary.csv.gz"):
            continue

        try:
            base = csv_file.replace("_binary.csv.gz", "")
            pdb_id, chain_id, ligand_id, ligand_chain = base.split("_")
        except ValueError:
            print(f" Skipping malformed filename: {csv_file}")
            continue

        prot_chain_key = f"{pdb_id}:{chain_id}"
        lig_key = f"{ligand_id}_{ligand_chain}"
        inter_key = f"{pdb_id}:{chain_id}:{lig_key}"

        # Load interaction CSV
        csv_path = os.path.join(binary_csv_dir, csv_file)
        df = pd.read_csv(csv_path, compression='gzip')
        num_residues = len(df)

        # Load FASTA
        fasta_path = os.path.join(fasta_dir, f"{pdb_id}_chain_{chain_id}.fasta")
        if not os.path.exists(fasta_path):
            print(f" Missing FASTA for {prot_chain_key}, skipping.")
            skipped_proteins.append(prot_chain_key)
            continue

        fasta_record = next(SeqIO.parse(fasta_path, "fasta"))
        sequence = str(fasta_record.seq)

        if len(sequence) != num_residues:
            print(f" Sequence length mismatch for {prot_chain_key}: {len(sequence)} != {num_residues}")
            skipped_proteins.append(prot_chain_key)
            continue

        # Compute ProtTrans features
        clean_sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
        tokens = tokenizer(clean_sequence, return_tensors="pt", add_special_tokens=False)
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        prottrans_embed = output.last_hidden_state[0][:len(sequence)]

        if prottrans_embed.shape[0] != len(sequence):
            print(f" Embedding length mismatch for {prot_chain_key}, skipping.")
            skipped_proteins.append(prot_chain_key)
            continue
        
        csv_sequence = ''.join(df["Residue_1L"].astype(str).tolist())
        if csv_sequence != sequence:
            print(f" Sequence mismatch in order for {prot_chain_key}")
            skipped_proteins.append(prot_chain_key)
            continue

        esm2_features = prottrans_embed.cpu().numpy().astype(np.float32)

        # Compute Mordred features
        smile = df["Canonical_Smile"].iloc[0]
        mol = Chem.MolFromSmiles(smile)
        mordred_df = mordred_calc.pandas([mol])
        mordred_df = mordred_df.reindex(columns=mordred_columns)

        for col in mordred_df.select_dtypes(include="bool").columns:
            mordred_df[col] = mordred_df[col].astype("Int8")

        if mordred_df.shape[1] != len(mordred_columns):
            print(f" Mordred shape mismatch for {lig_key}, skipping.")
            skipped_ligands.append(lig_key)
            continue

        ligand_features = mordred_df.values.astype(np.float32)

        # Extract binary interaction labels
        targets = df["Interacting"].astype(np.int8).values.reshape(-1, 1)

        # HDF5 paths
        prot_path = f"proteins/{pdb_id}/{chain_id}"
        lig_path = f"ligands/{lig_key}"
        inter_path = f"interactions/{inter_key}"

        # Save protein features (once per chain)
        if prot_path not in hdf:
            hdf.create_group(prot_path).create_dataset("features", data=esm2_features)
            saved_proteins.add(prot_path)

        # Save ligand features (once per ID)
        if lig_path not in hdf:
            hdf.create_group(lig_path).create_dataset("features", data=ligand_features)
            saved_ligands.add(lig_path)

        # Save interaction targets and metadata
        int_grp = hdf.require_group(inter_path)
        if "targets" not in int_grp:
            int_grp.create_dataset("targets", data=targets)
            int_grp.attrs["protein"] = prot_path
            int_grp.attrs["ligand"] = lig_path

    # Final stats 
    print(f" Proteins saved: {len(saved_proteins)}")
    print(f" Ligands saved: {len(saved_ligands)}")
    if 'interactions' in hdf:
        print(f" Interactions saved: {sum(1 for _ in hdf['interactions'])}")
    else:
        print(" No interactions saved.")

print(f"\n Skipped proteins: {len(skipped_proteins)}")
if skipped_proteins:
    print("   - Example:", ", ".join(skipped_proteins[:5]), "..." if len(skipped_proteins) > 5 else "")

print(f"\n Skipped ligands: {len(skipped_ligands)}")
if skipped_ligands:
    print("   - Example:", ", ".join(skipped_ligands[:5]), "..." if len(skipped_ligands) > 5 else "")

print(f"\n Final HDF5 file saved to: {output_hdf5_path}")
