#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd 

# Directory
filtered_ligands_path = "ligands_filtered.csv"
ligands_per_chain_path = "ligands_per_chain_viral_only.csv"
protein_data_path = "protein_data_filtered.xlsx"
organisms_path = "chain_organisms_viral_only.xlsx"

df=pd.read_csv(ligands_per_chain_path, dtype=str, na_filter=False)

# Expand multiple ligands per chain
df["LIGAND_ID"]=df["LIGAND_ID"].str.split(",")
df = df.explode("LIGAND_ID")

# Keeping only ligands that are in filtered_ligands
filtered_ligands = pd.read_csv(filtered_ligands_path, dtype=str, na_filter=False)[["LIGAND_ID","TYPE", "CHEM_TYPE" ,"SMILES", "InChIKey"]]
df = df[df["LIGAND_ID"].isin(filtered_ligands["LIGAND_ID"])]

# Joining the datasets filtered_ligands and ligands_per_chain
merged_df = df.merge(filtered_ligands, on="LIGAND_ID", how="left")

# Remove duplicates
merged_df = merged_df.drop_duplicates(subset=["PDB ID", "Chain ID", "LIGAND_ID"])

# Join protein_data with PDB ID
protein_data = pd.read_excel(protein_data_path, dtype=str, na_filter=False)

protein_data_filtered = protein_data[protein_data["PDB_ID"].isin(merged_df["PDB ID"])]

final_df = merged_df.merge(protein_data_filtered[["PDB_ID", "Experimental_Method", "Resolution"]], 
                           left_on="PDB ID", right_on="PDB_ID", how="left")
final_df = final_df.drop(columns=["PDB_ID"])

# Load organisms metadata
organisms = pd.read_excel(organisms_path, dtype=str, na_filter=False)

# Ensure consistent column naming
organisms = organisms.rename(columns={"Chain_ID": "Chain ID"})

# Filter organisms data to match PDB ID and Chain ID in merged_df
organisms_filtered = organisms[
    organisms[["PDB_ID", "Chain ID"]]
    .apply(tuple, axis=1)
    .isin(merged_df[["PDB ID", "Chain ID"]].apply(tuple, axis=1))
]

# Merge organism info into final_df using PDB ID and Chain ID
final_df = final_df.merge(
    organisms_filtered[["PDB_ID", "Chain ID", "Organism"]],
    left_on=["PDB ID", "Chain ID"],
    right_on=["PDB_ID", "Chain ID"],
    how="left"
).drop(columns=["PDB_ID"])

print("\nSummary of filtered data:")
print(f"- Unique PDB IDs: {final_df['PDB ID'].nunique()}")
print(f"- Unique PDB chains: {final_df[['PDB ID', 'Chain ID']].drop_duplicates().shape[0]}")
print(f"- Unique ligand IDs: {final_df['LIGAND_ID'].nunique()}")

final_df.to_excel("filtered_df.xlsx", index=False)