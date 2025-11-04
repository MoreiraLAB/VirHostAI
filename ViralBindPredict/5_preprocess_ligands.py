#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')

cif_path = "components_18_06.cif"
ligands_csv_path = "ligands_per_chain_viral_only.csv"
biolip2_path = "ligand_list_excluded_BioLiP2.txt"
excluded_txt_path = "excluded_ligand_ids.txt"
extra_excluded_path = "excluded_ligands_Q_BioLiP.txt"
output_path = "full_excluded_ligand_ids.txt"

# Allowed ligand_types
ALLOWED_CHEM_TYPES = {
    "D-SACCHARIDE", "D-saccharide", "L-SACCHARIDE", "L-saccharide",
    "NON-POLYMER", "non-polymer",
    "PEPTIDE-LIKE", "Peptide-like", "peptide-like",
    "SACCHARIDE", "saccharide"
}

# Allowed types, HETAIN: inhibitor, HETAC: coenzyme, ATOMS: sugar; HETAD:drug
ALLOWED_TYPES = {"HETAIN", "HETAC", "ATOMS", "HETAD", "hetain"}

# Terms that should NOT appear in the ligand name/type
FORBIDDEN_TERMS = {
    "ion", "metal", "solvent", "water", "buffer", "salt",
    "terminus", "linking", "rna", "dna", "cation", "anion"
}

# Allowed atomic symbols in the molecule 
ALLOWED_ATOMS = {'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'}

def contains_forbidden(text):
    return any(re.search(rf"\b{word}\b", str(text).lower()) for word in FORBIDDEN_TERMS)

# Funciton to standardize and give canonical smiles using RDKIT
def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except:
        pass
    return "Invalid SMILES"

def read_ligand_ids(filepath, column=0):
    ids = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) > column:
                    ligand_id = parts[column].strip().upper()
                    if ligand_id and ligand_id != '""':
                        ids.add(ligand_id)
    return ids

def contains_disallowed_atoms(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return True
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in ALLOWED_ATOMS:
                return True
        return False

# Load exclusion sets (from BioLIP2, Q-BioLIP and PLIC)
biolip2_ids = read_ligand_ids(biolip2_path)
excluded_ids = read_ligand_ids(excluded_txt_path)
extra_ids = read_ligand_ids(extra_excluded_path)
predefined_exclusion = biolip2_ids.union(excluded_ids).union(extra_ids)

with open(output_path, "w", encoding="utf-8") as out:
    for ligand in sorted(predefined_exclusion):
        out.write(f"{ligand}\n")

print(f"\nCombined exclusion list saved to: {output_path}")
print(f"Total ligands to exclude from predefined lists: {len(predefined_exclusion)}")

# Load ligand list
df_ligands = pd.read_csv(ligands_csv_path, dtype=str, na_filter=False)
unique_ligand_ids = set()
for val in df_ligands["LIGAND_ID"].dropna():
    for lid in str(val).split(','):
        unique_ligand_ids.add(lid.strip().upper())

# Parse CIF
ligand_entries = []
current = {}
collecting_descriptors = False

with open(cif_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("data_"):
            if current and current.get("ID") in unique_ligand_ids:
                ligand_entries.append(current)
            current = {"ID": line[5:], "TYPE": "", "CHEM_TYPE": "", "NAME": "", "DESCRIPTORS": []}
            collecting_descriptors = False

        elif line.startswith("_chem_comp.id"):
            current["ID"] = line.split()[-1].strip('"')
        elif line.startswith("_chem_comp.name"):
            current["NAME"] = " ".join(line.split()[1:]).strip('"')
        elif line.startswith("_chem_comp.pdbx_type"):
            current["TYPE"] = line.split()[-1].strip('"')
        elif line.startswith("_chem_comp.type"):
            current["CHEM_TYPE"] = line.split()[-1].strip('"')
        elif line.startswith("loop_"):
            collecting_descriptors = False
        elif "_pdbx_chem_comp_descriptor.descriptor" in line:
            collecting_descriptors = True
        elif collecting_descriptors:
            if not line or line.startswith("#") or line.startswith("data_"):
                collecting_descriptors = False
                continue
            parts = re.findall(r'"[^"]*"|\S+', line)
            if len(parts) >= 5:
                comp_id = parts[0].strip('"')
                dtype = parts[1].strip('"')
                program = parts[2].strip('"')
                version = parts[3].strip('"')
                descriptor = parts[4].strip('"')
                if comp_id == current.get("ID"):
                    current.setdefault("DESCRIPTORS", []).append((dtype, program, descriptor))

if current and current.get("ID") in unique_ligand_ids:
    ligand_entries.append(current)

print(f"\nParsed {len(ligand_entries)} ligands from CIF (subset of ligands_per_chain).")

# Filter and collect
included = []
excluded = []

for entry in ligand_entries:
    lig_id = entry["ID"]

    if lig_id in predefined_exclusion:
        excluded.append([lig_id, "", "", "Predefined exclusion list"])
        continue

    lig_type = entry["TYPE"]
    lig_name = entry["NAME"]
    lig_chem_type = entry.get("CHEM_TYPE", "").strip()
    descriptors = entry.get("DESCRIPTORS", [])

    if lig_chem_type.lower() not in {t.lower() for t in ALLOWED_CHEM_TYPES}:
        excluded.append([lig_id, lig_name, lig_type, "Chem type not allowed"])
        continue

    if contains_forbidden(lig_type) or contains_forbidden(lig_name):
        excluded.append([lig_id, lig_name, lig_type, "Forbidden term in name or type"])
        continue

    type_allowed = lig_type in ALLOWED_TYPES
    chem_type_allowed = lig_chem_type.lower() in {t.lower() for t in ALLOWED_CHEM_TYPES}
    is_questionable_but_valid = lig_type == "?" and chem_type_allowed

    if not type_allowed and not is_questionable_but_valid:
        excluded.append([lig_id, lig_name, lig_type, "Unknown or disallowed TYPE"])
        continue

    if is_questionable_but_valid:
        with open("question_type_valid_chemtype_ligands.txt", "a", encoding="utf-8") as logf:
            logf.write(f"{lig_id}\t{lig_name}\t{lig_type}\t{lig_chem_type}\n")

    # Try OpenEye first
    smiles_list = [
        val for dtype, prog, val in descriptors
        if "SMILES_CANONICAL" in dtype.upper() and "OPENEYE" in prog.upper()
    ]
    
    # Fallback to CACTVS if OpenEye not available
    if not smiles_list:
        smiles_list = [
            val for dtype, prog, val in descriptors
            if "SMILES_CANONICAL" in dtype.upper() and "CACTVS" in prog.upper()
        ]
        
    inchi_key = next((desc for dtype, prog, desc in descriptors if "INCHIKEY" in dtype.upper()), "")
    inchi = next((desc for dtype, prog, desc in descriptors if dtype.upper() == "INCHI"), "")

    canonical_smiles = "Invalid SMILES"
    for smiles in smiles_list:
        std = standardize_smiles(smiles)
        if std != "Invalid SMILES":
            canonical_smiles = std
            break

    if canonical_smiles == "Invalid SMILES":
        excluded.append([lig_id, lig_name, lig_type, "No valid SMILES"])
        continue

    if contains_disallowed_atoms(canonical_smiles):
        excluded.append([lig_id, lig_name, lig_type, "Contains disallowed atoms"])
        continue

    # Remove ligands with < 4 heavy atoms
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol and Descriptors.HeavyAtomCount(mol) < 4:
        excluded.append([lig_id, lig_name, lig_type, "Fewer than 4 heavy atoms"])
        continue
    
    included.append([
        lig_id, lig_type, lig_chem_type, lig_name,
        canonical_smiles, inchi, inchi_key
    ])


df_incl = pd.DataFrame(included, columns=[
    "LIGAND_ID", "TYPE", "CHEM_TYPE", "NAME", "SMILES", "InChI", "InChIKey"
])
df_excl = pd.DataFrame(excluded, columns=[
    "LIGAND_ID", "NAME", "TYPE", "REASON"
])

# Rule 1: Same SMILES, different InChIs → remove all
print("\n[Rule 1] Checking for same SMILES with different InChIs...")
to_exclude_smiles_conflict = set()
for smiles, group in df_incl.groupby("SMILES"):
    if smiles == "Invalid SMILES":
        continue
    inchi_set = set(group["InChI"])
    if len(inchi_set) > 1:
        print(f"  - SMILES '{smiles}' has {len(inchi_set)} different InChIs → excluding: {list(group['LIGAND_ID'])}")
        to_exclude_smiles_conflict.update(group["LIGAND_ID"].tolist())

# Rule 2: Same SMILES & InChI, different IDs → keep only one
print("\n[Rule 2] Checking for duplicates with same SMILES and InChI...")

to_exclude_duplicates = set()
for (smiles, inchi), group in df_incl.groupby(["SMILES", "InChI"]):
    if len(group) > 1:
        keep_id = group.iloc[0]["LIGAND_ID"]
        all_ids = set(group["LIGAND_ID"])
        all_ids.discard(keep_id)
        print(f"  - Keeping {keep_id} for SMILES: {smiles}, InChI: {inchi}; removing duplicates: {all_ids}")
        to_exclude_duplicates.update(all_ids)

# RULE 2 did nothing in this case, so no duplicates on InChI and SMILES with different ligand IDs

# Rule 3: Same InChI, different SMILES → exclude all involved ligands
print("\n[Rule 3] Checking for same InChI with different SMILES...")
inconclusive_conflict = []
to_exclude_inchi_conflict = set()

for inchi, group in df_incl.groupby("InChI"):
    if inchi and len(set(group["SMILES"])) > 1:
        print(f"  - InChI '{inchi}' has multiple SMILES:")
        for _, row in group.iterrows():
            print(f"      {row['LIGAND_ID']} → {row['SMILES']}")
        inconclusive_conflict.append(group)
        to_exclude_inchi_conflict.update(group["LIGAND_ID"].tolist())

# Final exclusion set (Rule 1 + Rule 2 + Rule 3)
to_exclude_conflicts = (
    to_exclude_smiles_conflict
    .union(to_exclude_duplicates)
    .union(to_exclude_inchi_conflict)
)

df_conflict_excluded = df_incl[df_incl["LIGAND_ID"].isin(to_exclude_conflicts)]
df_final_incl = df_incl[~df_incl["LIGAND_ID"].isin(to_exclude_conflicts)]

# Save everything
df_final_incl.to_csv("ligands_filtered.csv", index=False)
df_excl.to_csv("ligands_excluded.csv", index=False)
df_excl.groupby("REASON").size().to_csv("excluded_counts_by_reason.csv")
df_conflict_excluded.to_csv("ligands_excluded_conflicts.csv", index=False)

# Print summary
print(f"\n Cleaned filtered ligands saved to 'ligands_filtered.csv' ({len(df_final_incl)} entries)")
print(f" Filter-level exclusions saved to 'ligands_excluded.csv' ({len(df_excl)} entries)")
print(f" Conflict-based exclusions saved to 'ligands_excluded_conflicts.csv' ({len(df_conflict_excluded)} entries)")

print("\n[Final Check] Duplicates in ligands_filtered.csv")
# Clean and filter before checking
df_check = df_final_incl.copy()

# Check duplicates
dup_smiles = df_check[df_check.duplicated(subset="SMILES", keep=False)]
df_check_valid_inchi = df_check[df_check["InChI"].notna() & (df_check["InChI"].str.strip() != "")]
dup_inchi = df_check_valid_inchi[df_check_valid_inchi.duplicated(subset="InChI", keep=False)]
dup_inchikey = df_check[df_check.duplicated(subset="InChIKey", keep=False)]

# Print summary - should be 0 in all
print("\n[Summary of Duplicates (should be 0 in all)]")
print(f"  - SMILES duplicates: {dup_smiles['SMILES'].nunique()} unique SMILES ({len(dup_smiles)} total entries)")
print(f"  - InChI duplicates: {dup_inchi['InChI'].nunique()} unique InChIs ({len(dup_inchi)} total entries)")
print(f"  - InChIKey duplicates: {dup_inchikey['InChIKey'].nunique()} unique InChIKeys ({len(dup_inchikey)} total entries)")
    
# Summary of ligand types
unique_types = df_final_incl["TYPE"].unique()
print(f"\nUnique ligand types in filtered cleaned set ({len(unique_types)}):")
for t in sorted(unique_types):
    print(f"  - {t}")

# Summary of chemical types
if "CHEM_TYPE" in df_final_incl.columns:
    unique_chem_types = set(df_final_incl["CHEM_TYPE"].dropna().unique())
    print("\nUnique _chem_comp.type values in included ligands:")
    for t in sorted(unique_chem_types):
        print(f"  - {t}")
