#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd

# Load datasets
ligands_per_chain_path = "ligands_per_chain.csv"
chains_organisms_path = "chain_organisms.csv"
protein_data_path = "protein_data.csv"
organism_output_excel = "chain_organisms_filtered.xlsx"
protein_output_excel = "protein_data_filtered.xlsx"

# Filter ligands_per_chain.csv
lig_df = pd.read_csv(ligands_per_chain_path, dtype=str, na_filter=False)
lig_df = lig_df[(lig_df["LIGAND_ID"] != "NO_LIGAND") & (lig_df["SEQUENCE_1L"].str.len() >= 30)]

# Normalize IDs
lig_df["PDB ID"] = lig_df["PDB ID"].astype(str).str.strip()
lig_df["Chain ID"] = lig_df["Chain ID"].astype(str).str.strip()
lig_keys = lig_df[["PDB ID", "Chain ID"]].drop_duplicates()
lig_pdbs = lig_df["PDB ID"].drop_duplicates()

# chain_organisms.csv
org_raw = pd.read_csv(chains_organisms_path, dtype=str, na_filter=False, header=None)
org_raw.columns = ["PDB_ID", "Chain_ID"] + [f"Org_{i}" for i in range(org_raw.shape[1] - 2)]

# Normalize IDs
org_raw["PDB_ID"] = org_raw["PDB_ID"].astype(str).str.strip()
org_raw["Chain_ID"] = org_raw["Chain_ID"].astype(str).str.strip()

# Filter based on chains with ligands
org_filtered_raw = pd.merge(
    org_raw, lig_keys,
    left_on=["PDB_ID", "Chain_ID"],
    right_on=["PDB ID", "Chain ID"],
    how="inner"
)

# Remove duplicated columns from lig_keys
org_filtered_raw = org_filtered_raw.drop(columns=["PDB ID", "Chain ID"])

# Melt and format
org_melted = org_filtered_raw.melt(
    id_vars=["PDB_ID", "Chain_ID"], value_name="Organism"
).drop("variable", axis=1)

org_melted["Organism"] = org_melted["Organism"].fillna("").astype(str)

def join_nonempty(series):
    items = [str(s).strip() for s in series if str(s).strip()]
    return "; ".join(items) if items else ""

org_df = org_melted.groupby(["PDB_ID", "Chain_ID"])["Organism"].apply(join_nonempty).reset_index()
org_df.to_excel(organism_output_excel, index=False)
print(f" chain_organisms_filtered.xlsx saved with {len(org_df)} chains.")

# CHECK: compare with lig_keys
merged_check = pd.merge(lig_keys, org_df, how="left", left_on=["PDB ID", "Chain ID"], right_on=["PDB_ID", "Chain_ID"])
missing = merged_check[merged_check["Organism"].isna()]
if missing.empty:
    print(" All chains with ligand are present in filtered chain_organisms!")
else:
    print(f" {len(missing)} chains with ligands do not have a valid entry in chain organisms!")
    print(missing[["PDB ID", "Chain ID"]].head())

# protein_data.csv
prot_raw = pd.read_csv(protein_data_path, header=None, dtype=str, na_filter=False)
n_cols = prot_raw.shape[1]

fixed_cols = ["PDB_ID", "Experimental_Method", "Organism", "Resolution"]
org_cols = [f"Org_{i}" for i in range(n_cols - len(fixed_cols))]
prot_raw.columns = fixed_cols + org_cols

# Normalize PDB_IDs
prot_raw["PDB_ID"] = prot_raw["PDB_ID"].astype(str).str.strip()

# Filter only PDB_IDs with ligand
prot_filtered = prot_raw[prot_raw["PDB_ID"].isin(lig_pdbs)]

prot_df = prot_filtered[["PDB_ID", "Experimental_Method", "Resolution", "Organism"]].drop_duplicates()

prot_df.to_excel(protein_output_excel, index=False)
print(f" protein_data_filtered.xlsx saved with {len(prot_df)} entries.")
