#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd

# Keywords for viral organisms
viral_keywords = [
    "virus", "phage", "sars", "hiv", "ebola", "coronavirus", "viridae", "virinae", "viricetes",
    "h1n1", "h3n2", "h5n1", "h6n1", "h7n9", "h17n10", "htlv", "hbv", "virulent", "prrsv", "viral", "betacov"
]

filtered_path = "chain_organisms_filtered.xlsx"
keep_path = "keep_chain_organisms.xlsx"
ligands_path = "ligands_per_chain.csv"

# Load datasets
df_filtered = pd.read_excel(filtered_path, dtype=str, na_filter=False)
df_keep = pd.read_excel(keep_path, dtype=str, na_filter=False)
lig_df = pd.read_csv(ligands_path, dtype=str, na_filter=False)

# Normalize IDs
for col in ["PDB_ID", "Chain_ID"]:
    df_filtered[col] = df_filtered[col].astype(str).str.strip()
    df_keep[col] = df_keep[col].astype(str).str.strip()

# Correct "unknown" using keep dataset (which contains some of the unknown organisms - maaped mannually to the correct source organism)
df_unknown = df_filtered[df_filtered["Organism"].str.lower() == "unknown"].copy()
df_unknown = df_unknown.merge(df_keep, on=["PDB_ID", "Chain_ID"], how="left", suffixes=("", "_keep"))
df_unknown.loc[df_unknown["Organism_keep"].notna(), "Organism"] = df_unknown["Organism_keep"]
df_unknown = df_unknown[["PDB_ID", "Chain_ID", "Organism"]]

# Remove the rows to be updated from original
df_filtered = df_filtered[~df_filtered.set_index(["PDB_ID", "Chain_ID"]).index.isin(df_unknown.set_index(["PDB_ID", "Chain_ID"]).index)]

# Append the corrected entries
df_filtered = pd.concat([df_filtered, df_unknown], ignore_index=True)

n_corrigidas = df_unknown["Organism"].ne("unknown").sum()
print(f" Chains with 'unknown' corrected: {n_corrigidas} of {len(df_unknown)}")

# Verify if ALL organisms in a pdb chain are viral
def all_organisms_viral(org_str):
    organisms = [o.strip().lower() for o in str(org_str).split(";") if o.strip()]
    return all(any(vk in org for vk in viral_keywords) for org in organisms)

df_filtered["All_Viral"] = df_filtered["Organism"].apply(all_organisms_viral)

df_viral_only = df_filtered[df_filtered["All_Viral"]].drop(columns=["All_Viral"])
df_non_viral = df_filtered[~df_filtered["All_Viral"]].drop(columns=["All_Viral"])

viral_organisms = sorted(df_viral_only["Organism"].dropna().unique())
non_viral_organisms_full = sorted(df_non_viral["Organism"].dropna().unique())

df_viral_only.to_excel("chain_organisms_viral_only.xlsx", index=False)

with open("unique_viral_organisms.txt", "w", encoding="utf-8") as f:
    for org in viral_organisms:
        f.write(f"{org}\n")

with open("excluded_non_viral_organisms.txt", "w", encoding="utf-8") as f:
    for entry in non_viral_organisms_full:
        f.write(f"{entry}\n")

#  Filtrar lig_df to keep only viral chains
lig_df[["PDB ID", "Chain ID"]] = lig_df[["PDB ID", "Chain ID"]].astype(str).apply(lambda x: x.str.strip())
viral_keys = df_viral_only[["PDB_ID", "Chain_ID"]].drop_duplicates()
lig_viral = pd.merge(lig_df, viral_keys, left_on=["PDB ID", "Chain ID"], right_on=["PDB_ID", "Chain_ID"], how="inner")
lig_viral = lig_viral.drop(columns=["PDB_ID", "Chain_ID"])
lig_viral.to_csv("ligands_per_chain_viral_only.csv", index=False)

print(f" - Viral chains kept: {len(df_viral_only)}")
print(f" - Unique viral organisms: {len(viral_organisms)}")
print(f" - Excluded organisms (including chimeric chains with at least 1 non viral organism): {len(non_viral_organisms_full)}")

print(f" - Final entries in ligands_per_chain_viral_only.csv: {len(lig_viral)}")
