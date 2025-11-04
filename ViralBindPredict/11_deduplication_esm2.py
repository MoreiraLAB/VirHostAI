#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import h5py
import pandas as pd
from tqdm import tqdm

# Paths
binary_csv_dir = "binary_interactions_csv"
filtered_df_path = "filtered_df.xlsx"
original_hdf5_path = "interaction_dataset.hdf5"
deduplicated_df_path = "deduplicated_esm2_mordred_filtered_df.xlsx"
output_hdf5_path = "interaction_dataset_esm2_mordred_deduplicated.hdf5"
log_file = "deduplication_esm2_log.txt"

# Load filtered DataFrame
print(" Loading filtered dataframe...")
df = pd.read_excel(filtered_df_path)

# Filter out chains with resolution > 5Å
initial_count = len(df)
df = df[df["Resolution"] <= 5.0].reset_index(drop=True)
filtered_count = len(df)

# Add Ligand Chain from CSV filenames
def get_ligand_chains(pdb_id, chain_id, lig_id):
    """Get all ligand chains from binary CSV filenames."""
    prefix = f"{pdb_id}_{chain_id}_{lig_id}_"
    matches = [
        f for f in os.listdir(binary_csv_dir)
        if f.startswith(prefix) and f.endswith("_binary.csv.gz")
    ]
    #if not matches:
        #print(f" No binary CSV found for {pdb_id}:{chain_id}:{lig_id}")
    return [f.split("_")[3] for f in matches]

print(" Adding ligand chains...")
# Expand rows for multiple ligand chains
expanded_rows = []
for _, row in df.iterrows():
    ligand_chains = get_ligand_chains(row["PDB ID"], row["Chain ID"], row["LIGAND_ID"])
    if not ligand_chains:
        continue  # Skip if no binary CSV found
    for lig_chain in ligand_chains:
        new_row = row.copy()
        new_row["Ligand_Chain"] = lig_chain
        expanded_rows.append(new_row)

# Create expanded DataFrame
df = pd.DataFrame(expanded_rows)
print(f" Expanded DataFrame to {len(df)} rows (was {len(expanded_rows)})")

# Compute Residue Patterns
def load_residue_pattern(pdb_id, chain_id, lig_id, lig_chain):
    """Load interacting residues from binary CSV."""
    csv_name = f"{pdb_id}_{chain_id}_{lig_id}_{lig_chain}_binary.csv.gz"
    csv_path = os.path.join(binary_csv_dir, csv_name)
    if not os.path.exists(csv_path):
        return set()
    csv_df = pd.read_csv(csv_path, compression="gzip")
    residues = set(
        csv_df[csv_df["Interacting"] == 1]["Position"].astype(str) +
        "_" + csv_df[csv_df["Interacting"] == 1]["Residue_1L"]
    )
    return residues

print(" Computing residue patterns...")
df["Residue_Pattern"] = df.apply(
    lambda row: load_residue_pattern(
        row["PDB ID"], row["Chain ID"], row["LIGAND_ID"], row["Ligand_Chain"]
    ),
    axis=1
)
df["Num_Interacting"] = df["Residue_Pattern"].apply(len)

# Filter for ≥2 non-consecutive interacting residues
def has_min_non_consecutive_residues(residue_pattern):
    """
    Check if there are at least 2 non-consecutive interacting residues.
    """
    if len(residue_pattern) < 2:
        return False  # Less than 2 interacting residues
    
    # Extract positions as integers (Position_Residue1L → Position)
    positions = sorted(int(pos.split("_")[0]) for pos in residue_pattern)
    # Check differences between consecutive positions
    consecutive = all(abs(b - a) == 1 for a, b in zip(positions, positions[1:]))
    
    return not consecutive  # Keep only if NOT all consecutive

print("Filtering chains with ≥2 non-consecutive interacting residues...")
initial_count = len(df)
df = df[df["Residue_Pattern"].apply(has_min_non_consecutive_residues)].reset_index(drop=True)
filtered_count = len(df)
removed_count = initial_count - filtered_count
print(f" Filtered dataset: {filtered_count} rows kept ({removed_count} rows removed)")


# Start logging
with open(log_file, "w") as log:
    log.write("Deduplication Log\n")
    log.write("=================\n\n")

    ## Intra-PDB Deduplication
    log.write("INTRA-PDB DEDUPLICATION\n\n")
    print(" Intra-PDB deduplication...")
    dedup_intra = []
    removed_intra_count = 0

    for pdb_id, group in tqdm(df.groupby("PDB ID")):
        for (seq, lig_id), lig_group in group.groupby(["SEQUENCE_1L", "LIGAND_ID"]):

            # Step 1: Deduplicate within the same PDB for same Ligand_ID and Ligand_Chain
            kept_same_chain = []

            # Inside the intra-PDB loop, for each lig_chain:
            for lig_chain, chain_group in lig_group.groupby("Ligand_Chain"):
                chain_group = chain_group.copy()
                chain_group["PatternKey"] = chain_group["Residue_Pattern"] \
                    .apply(lambda s: "_".join(sorted(s)))
                
                is_dup = chain_group.duplicated(subset="PatternKey", keep="first")
                
                for idx, row in chain_group.iterrows():
                    key = row["PatternKey"]
                    if is_dup.loc[idx]:
                        log.write(f"[Same Ligand Chain] DUPLICATE REMOVED: "
                                  f"{row['PDB ID']}:{row['Chain ID']}:{row['LIGAND_ID']}_{row['Ligand_Chain']} "
                                  f"(Pattern: {key})\n\n")
                        removed_intra_count += 1

                    else:
                        kept_same_chain.append(row)
                        log.write(f"[Same Ligand Chain] KEPT: "
                                  f"{row['PDB ID']}:{row['Chain ID']}:{row['LIGAND_ID']}_{row['Ligand_Chain']} "
                                  f"(Pattern: {key})\n")

            # Step 2: Deduplicate across different ligand chains
            final_kept = []
            for row in kept_same_chain:
                duplicate = False
                for k in final_kept:
                    if row["Residue_Pattern"] == k["Residue_Pattern"]:
                        log.write(f"[Cross Ligand Chains] KEPT: {k['PDB ID']}:{k['Chain ID']}:{k['LIGAND_ID']}_{k['Ligand_Chain']}\n")
                        log.write(f"REMOVED: {row['PDB ID']}:{row['Chain ID']}:{row['LIGAND_ID']}_{row['Ligand_Chain']}\n")
                        log.write("  Reason: identical residue pattern across ligand chains\n\n")
                        removed_intra_count += 1
                        duplicate = True
                        break
                if not duplicate:
                    final_kept.append(row)
            # Add deduplicated rows for this group
            dedup_intra.extend(final_kept)

    dedup_intra_df = pd.DataFrame(dedup_intra).reset_index(drop=True)
    log.write(f"Total removed in intra-PDB: {removed_intra_count}\n\n")

    ## Inter-PDB Deduplication
    log.write("INTER-PDB DEDUPLICATION" + "\n\n")
    print(" Inter-PDB deduplication...")
    dedup_inter = []
    kept_inter_count = 0
    removed_inter_count = 0

    for (seq, lig_id), group in tqdm(dedup_intra_df.groupby(["SEQUENCE_1L", "LIGAND_ID"])):
        kept = []
        sorted_group = group.sort_values("Resolution", ascending=True)

        for idx, row in sorted_group.iterrows():
            duplicate = False
            for k in kept:
                if row["PDB ID"] == k["PDB ID"]:
                    continue  # Skip if same PDB ID
                if row["Residue_Pattern"] == k["Residue_Pattern"]:
                    log.write(f"KEPT: {k['PDB ID']}:{k['Chain ID']}:{k['LIGAND_ID']}_{k['Ligand_Chain']} (Resolution: {k['Resolution']})\n")
                    log.write(f"  Residues: {len(k['Residue_Pattern'])}\n")
                    log.write(f"  Pattern: {sorted(list(k['Residue_Pattern']))}\n")
                    log.write(f"REMOVED: {row['PDB ID']}:{row['Chain ID']}:{row['LIGAND_ID']}_{row['Ligand_Chain']} (Resolution: {row['Resolution']})\n")
                    log.write("  Reason: same sequence, ligand & identical residue pattern; lower resolution\n\n")
                    removed_inter_count += 1
                    duplicate = True
                    break
            if not duplicate:
                kept.append(row)
                kept_inter_count += 1
        dedup_inter.extend(kept)

    dedup_df = pd.DataFrame(dedup_inter).reset_index(drop=True)

    log.write(f"Total kept in inter-PDB: {kept_inter_count}\n")
    log.write(f"Total removed in inter-PDB: {removed_inter_count}\n\n")
    log.write(f"Final deduplicated dataset size: {len(dedup_df)} entries\n")

print(f"Deduplication log saved to: {log_file}")

# Save deduplicated DataFrame
dedup_df.to_excel(deduplicated_df_path, index=False)
print(f" Deduplicated dataframe saved to: {deduplicated_df_path}")

# Filter HDF5 file
print(" Filtering and saving deduplicated HDF5...")
proteins_to_copy = set()
referenced_ligands = set()
interactions_to_copy = []

for _, row in dedup_df.iterrows():
    pdb_id, chain_id = row["PDB ID"], row["Chain ID"]
    lig_id, lig_chain = row["LIGAND_ID"], row["Ligand_Chain"]
    proteins_to_copy.add((pdb_id, chain_id))
    referenced_ligands.add(f"{lig_id}_{lig_chain}")
    interactions_to_copy.append(f"{pdb_id}:{chain_id}:{lig_id}_{lig_chain}")

ligands_to_copy = referenced_ligands 

with h5py.File(original_hdf5_path, 'r') as hdf_in, h5py.File(output_hdf5_path, 'w') as hdf_out:
    hdf_out.create_dataset("mordred_column_names", data=hdf_in["mordred_column_names"][:])

    for pdb_id, chain_id in tqdm(proteins_to_copy, desc="Copying proteins"):
        src_path = f"proteins/{pdb_id}/{chain_id}"
        if src_path in hdf_in:
            hdf_in.copy(src_path, hdf_out.require_group(f"proteins/{pdb_id}"))

    for lig in tqdm(ligands_to_copy, desc="Copying ligands"):
        src_path = f"ligands/{lig}"
        if src_path in hdf_in:
            hdf_in.copy(src_path, hdf_out.require_group("ligands"))

    for key in tqdm(interactions_to_copy, desc="Copying interactions"):
        src_path = f"interactions/{key}"
        if src_path in hdf_in:
            hdf_in.copy(src_path, hdf_out.require_group("interactions"))

print(" Verifying consistency of proteins, ligands, and interactions...")

# Extract referenced proteins and ligands from interactions
referenced_proteins = set()
referenced_ligands_check = set()

for key in interactions_to_copy:
    pdb_id, chain_id, lig = key.split(":")
    referenced_proteins.add((pdb_id, chain_id))
    referenced_ligands_check.add(lig)

# Compare proteins
extra_proteins = proteins_to_copy - referenced_proteins
missing_proteins = referenced_proteins - proteins_to_copy

# Compare ligands
extra_ligands = ligands_to_copy - referenced_ligands_check
missing_ligands = referenced_ligands_check - ligands_to_copy

# Report
if not extra_proteins and not missing_proteins:
    print(" All proteins in HDF5 are referenced in interactions.")
else:
    print(" Inconsistent proteins!")
    if extra_proteins:
        print(f"    Extra proteins: {extra_proteins}")
    if missing_proteins:
        print(f"    Missing proteins: {missing_proteins}")

if not extra_ligands and not missing_ligands:
    print(" All ligands in HDF5 are referenced in interactions.")
else:
    print("  Inconsistent ligands!")
    if extra_ligands:
        print(f"    Extra ligands: {extra_ligands}")
    if missing_ligands:
        print(f"    Missing ligands: {missing_ligands}")

# Raise an error if inconsistencies found
if extra_proteins or missing_proteins or extra_ligands or missing_ligands:
    raise ValueError("Inconsistent HDF5: proteins or ligands do not match interactions.")
else:
    print("  HDF5 consistency check passed!")

print(f"    {len(interactions_to_copy)} interactions copied")
print(f"    {len(proteins_to_copy)} protein chains copied")
print(f"    {len(ligands_to_copy)} ligands copied")
print(f" Deduplicated HDF5 saved to: {output_hdf5_path}")