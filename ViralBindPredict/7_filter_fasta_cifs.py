#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import shutil

merged_data_path = "filtered_df.xlsx"
cif_dir = "split_cif_chains"
fasta_dir = "fasta_sequences"
filtered_cif_dir = "split_cif_chains_filtered"
filtered_fasta_dir = "fasta_sequences_filtered"

os.makedirs(filtered_cif_dir, exist_ok=True)
os.makedirs(filtered_fasta_dir, exist_ok=True)

# Load valid ligand-chain combinations
df = pd.read_excel(merged_data_path, dtype=str, na_filter=False)
df["PDB ID"] = df["PDB ID"].str.strip()
df["Chain ID"] = df["Chain ID"].str.strip()
df["LIGAND_ID"] = df["LIGAND_ID"].str.strip()

valid_ligands = set(df["LIGAND_ID"].unique())
valid_pdb_chains = set(df["PDB ID"] + "_" + df["Chain ID"])

# Helper to extract _atom_site block
def extract_atom_site_lines(cif_lines):
    fields, atom_lines = [], []
    in_loop, reading_fields = False, False
    for i, line in enumerate(cif_lines):
        stripped = line.strip()
        if stripped == "loop_":
            in_loop = True
            continue
        if in_loop and stripped.startswith("_atom_site."):
            fields.append(stripped)
            reading_fields = True
            continue
        if reading_fields and in_loop:
            if stripped.startswith("_"):
                break
            elif stripped:
                atom_lines = cif_lines[i:]
                break
    return fields, atom_lines

# Filtering loop
saved_pdb_chains = set()
for file in os.listdir(cif_dir):
    if not file.endswith(".cif"):
        continue
    cif_path = os.path.join(cif_dir, file)

    # Extract PDB ID and Chain ID
    parts = file.split("_")
    if len(parts) < 3:
        print(f" Unexpected filename format: {file}")
        continue
    pdb_id = parts[0]
    chain_id = parts[-1].split(".")[0]
    pdb_chain_key = f"{pdb_id}_{chain_id}"

    if pdb_chain_key not in valid_pdb_chains:
        continue

    with open(cif_path, "r") as f:
        lines = f.readlines()

    fields, atom_lines = extract_atom_site_lines(lines)
    if not fields:
        print(f" No _atom_site fields in {file}")
        continue

    field_idx = {f: i for i, f in enumerate(fields)}
    try:
        idx_group = field_idx["_atom_site.group_PDB"]
        idx_resname = field_idx["_atom_site.label_comp_id"]
        idx_chain = field_idx["_atom_site.label_asym_id"]
    except KeyError as e:
        print(f" Missing required field in {file}: {e}")
        continue

    # Process and filter atoms
    atom_lines_for_chain = []
    ligand_lines = []

    for line in atom_lines:
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) <= max(idx_group, idx_resname, idx_chain):
            continue

        group = parts[idx_group]
        resname = parts[idx_resname]
        chain = parts[idx_chain]

        full_chain_key = f"{pdb_id}_{chain}"

        if group == "ATOM" and full_chain_key == pdb_chain_key:
            atom_lines_for_chain.append(line)
        elif group == "HETATM" and resname in valid_ligands:
            ligand_lines.append(line)

    if atom_lines_for_chain:
        out_path = os.path.join(filtered_cif_dir, f"{pdb_chain_key}.cif")
        with open(out_path, "w") as fout:
            fout.write("loop_\n")
            for f in fields:
                fout.write(f"{f}\n")
            for line in atom_lines_for_chain + ligand_lines:
                fout.write(line if line.endswith("\n") else line + "\n")
        print(f"Saved CIF: {out_path}")
        saved_pdb_chains.add(pdb_chain_key)

# Copy corresponding FASTA files
existing_fasta_files = {f: f for f in os.listdir(fasta_dir) if f.endswith(".fasta")}
for chain_key in saved_pdb_chains:
    pdb_id, chain_id = chain_key.split("_")
    fasta_filename = f"{pdb_id}_chain_{chain_id}.fasta"
    if fasta_filename in existing_fasta_files:
        src = os.path.join(fasta_dir, fasta_filename)
        dst = os.path.join(filtered_fasta_dir, fasta_filename)
        shutil.copy(src, dst)
        print(f" Copied FASTA: {dst}")
    else:
        print(f"Missing FASTA for {chain_key}")

# Verification 
print("\n Verification Results:")
expected_pdb_chains = set(df["PDB ID"] + "_" + df["Chain ID"])
existing_pdb_files = {f.replace(".cif", "") for f in os.listdir(filtered_cif_dir) if f.endswith(".cif")}
existing_fasta_keys = {
    f.replace(".fasta", "").replace("_chain_", "_")
    for f in os.listdir(filtered_fasta_dir)
    if f.endswith(".fasta")
}

missing_pdb_files = sorted(expected_pdb_chains - existing_pdb_files)
missing_fasta_files = expected_pdb_chains - existing_fasta_keys

print(f"Total expected PDB chains from Excel: {len(expected_pdb_chains)}")
print(f"Total existing CIF files: {len(existing_pdb_files)}")
print(f"Total existing FASTA files in {filtered_fasta_dir}: {len(existing_fasta_keys)}")

if not missing_pdb_files and not missing_fasta_files:
    print("\n All PDB chains from the Excel are present in both CIF and FASTA directories!")
else:
    print("\n Some files are missing:")
    if missing_pdb_files:
        print(f"\nMissing CIF files ({len(missing_pdb_files)}):")
        print("\n".join(missing_pdb_files))
    if missing_fasta_files:
        print(f"\nMissing FASTA files ({len(missing_fasta_files)}):")
        print("\n".join(missing_fasta_files))