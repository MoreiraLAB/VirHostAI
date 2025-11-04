#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import pandas as pd
from Bio import SeqIO
import os

input_hdf5 = "interaction_dataset_esm2_mordred_deduplicated.hdf5"
filtered_fasta_file = "mmseqs2_90seqs.fasta"  # FASTA of representative sequences after 90% redundancy removal
seq_mapping_excel = "deduplicated_esm2_mordred_filtered_df.xlsx"
output_hdf5 = "interaction_dataset_esm2_mordred_deduplicated_no_red.hdf5"

# Load sequence mapping from Excel
df = pd.read_excel(seq_mapping_excel)
df['PDB:Chain'] = df['PDB ID'].astype(str) + ':' + df['Chain ID'].astype(str)
pdb_chain_to_seq = dict(zip(df['PDB:Chain'], df['SEQUENCE_1L']))

print(f"Loaded {len(pdb_chain_to_seq)} PDB:Chain → sequence mappings.")

# Load allowed sequences from FASTA
allowed_sequences = set(str(record.seq) for record in SeqIO.parse(filtered_fasta_file, "fasta"))
print(f"Loaded {len(allowed_sequences)} allowed unique sequences from FASTA.")

# Determine which protein chains to keep
filtered_prot_chains = {pc for pc, seq in pdb_chain_to_seq.items() if seq in allowed_sequences}
print(f"Keeping {len(filtered_prot_chains)} protein chains matching allowed sequences.")

# Identify matching interactions, proteins, and ligands
interactions_to_copy = []
proteins_to_copy = set()
ligands_to_copy = set()

with h5py.File(input_hdf5, 'r') as hdf:
    all_interactions = list(hdf['interactions'].keys())
    print(f"Found {len(all_interactions)} total interactions in {input_hdf5}")

    for interaction_key in all_interactions:
        parts = interaction_key.split(":")
        if len(parts) < 3:
            continue
        prot_c = f"{parts[0]}:{parts[1]}"
        ligand = parts[2]

        if prot_c in filtered_prot_chains:
            interactions_to_copy.append(interaction_key)
            proteins_to_copy.add((parts[0], parts[1]))
            ligands_to_copy.add(ligand)

print(f"{len(interactions_to_copy)} interactions matched allowed sequences.")
print(f"{len(proteins_to_copy)} unique protein chains to copy.")
print(f"{len(ligands_to_copy)} unique ligands to copy.")

# Write filtered dataset
with h5py.File(input_hdf5, 'r') as hdf_in, h5py.File(output_hdf5, 'w') as hdf_out:
    # Copy protein features
    for prot, chain in proteins_to_copy:
        src_path = f"proteins/{prot}/{chain}/features"
        dst_group = hdf_out.require_group(f"proteins/{prot}/{chain}")
        hdf_in.copy(src_path, dst_group)

    # Copy ligand features
    for ligand in ligands_to_copy:
        src_path = f"ligands/{ligand}/features"
        dst_group = hdf_out.require_group(f"ligands/{ligand}")
        hdf_in.copy(src_path, dst_group)

    # Copy interaction groups
    interactions_out = hdf_out.require_group("interactions")
    for key in interactions_to_copy:
        hdf_in.copy(f"interactions/{key}", interactions_out)

    # Copy mordred_column_names if present
    if "mordred_column_names" in hdf_in:
        mordred_names = hdf_in["mordred_column_names"][:]
        hdf_out.create_dataset("mordred_column_names", data=mordred_names)
        print(f"Copied mordred_column_names with {len(mordred_names)} entries.")

print("\nDone!")
print(f"Total interactions copied: {len(interactions_to_copy)}")
print(f"Total protein chains copied: {len(proteins_to_copy)}")
print(f"Total ligands copied: {len(ligands_to_copy)}")

def norm(s: str) -> str:
    return str(s).strip().upper()

# Build normalized chain keys
df['PDB:Chain'] = df['PDB ID'].astype(str) + ':' + df['Chain ID'].astype(str)
df['PDB:Chain_NORM'] = df['PDB:Chain'].str.upper()

filtered_prot_chains_norm = {norm(pc) for pc in filtered_prot_chains}

# Keep ALL original rows for kept chains
df_chains_kept_allrows = df.loc[df['PDB:Chain_NORM'].isin(filtered_prot_chains_norm)].copy()

# Save ONLY this sheet
out_xlsx = os.path.splitext(seq_mapping_excel)[0] + "_nored.xlsx"
df_chains_kept_allrows.to_excel(out_xlsx, index=False)

print(f"\nFiltered Excel saved to: {out_xlsx}")
print(f"  chains_kept_allrows rows: {len(df_chains_kept_allrows)} "
      f"| unique PDB:Chain: {df_chains_kept_allrows['PDB:Chain'].nunique()}")