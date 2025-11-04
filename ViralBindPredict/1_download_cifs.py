#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import requests
import time
import pandas as pd

pdb_id_file = "virus_pdb_ids_18_06.txt"
output_dir = "cif"
os.makedirs(output_dir, exist_ok=True)

# Load PDB IDs
with open(pdb_id_file, "r") as f:
    all_pdb_ids = [line.strip().lower() for line in f if line.strip()]
print(f"\n Loaded {len(all_pdb_ids)} PDB IDs.")

def robust_get(url, retries=5, delay=5):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response
            print(f"Attempt {attempt+1} - status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1} failed: {e}")
        time.sleep(delay)
    return None

# Viral protein keywords
viral_keywords = [
    "virus", "phage", "sars", "hiv", "ebola", "coronavirus", "viridae", "virinae", "viricetes",
    "h1n1", "h3n2", "h5n1", "h6n1", "h7n9", "h17n10", "htlv", "hbv", "virulent", "prrsv", "viral"
]

# Data containers
chain_mapping = {}
protein_data = []
failed_downloads = []
valid_protein_pdb_ids = []

# Main loop 
print("\n Fetching metadata and downloading CIFs...")
for pdb_id in all_pdb_ids:
    # Metadata from RCSB
    entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    entry_response = robust_get(entry_url)
    if not entry_response or entry_response.status_code == 404:
        continue

    entry_data = entry_response.json()
    experimental_method = entry_data.get("rcsb_entry_info", {}).get("experimental_method", "N/A")
    if experimental_method == "N/A" or entry_data["rcsb_entry_info"].get("polymer_entity_count_protein", 0) == 0:
        continue
    valid_protein_pdb_ids.append(pdb_id)


    resolution = entry_data["rcsb_entry_info"].get("resolution_combined", ["N/A"])[0]

    # Download CIF file
    cif_filename = os.path.join(output_dir, f"{pdb_id}.cif")
    if not os.path.exists(cif_filename):
        cif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        cif_response = robust_get(cif_url)
        if cif_response:
            with open(cif_filename, "wb") as f:
                f.write(cif_response.content)
            print(f" Downloaded: {cif_filename}")
        else:
            print(f" Failed to download: {pdb_id}")
            failed_downloads.append(pdb_id)
            continue

    # Organism & chain mapping from PDBe
    ebi_url = f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/entities/{pdb_id}"
    ebi_response = robust_get(ebi_url)
    organisms = set()
    chain_to_organism = {}

    if ebi_response and ebi_response.status_code == 200:
        ebi_data = ebi_response.json()
        if pdb_id in ebi_data:
            for entity in ebi_data[pdb_id]:
                mtype = entity.get("molecule_type", "").lower()
                if not ("polypeptide" in mtype or "peptide" in mtype):
                    continue

                chains = entity.get("in_struct_asyms", [])
                names = entity.get("molecule_name", [])
                synonyms = entity.get("synonym", [])
                if isinstance(synonyms, str):
                    synonyms = [synonyms]
                all_names = names + synonyms

                sources = entity.get("source", [])
                if sources:
                    for src in sources:
                        org = src.get("organism_scientific_name")
                        for chain in chains:
                            chain_to_organism.setdefault(chain, set()).add(org if org else "unknown")
                            organisms.add(org if org else "unknown")
                else:
                    name_str = " ".join(all_names).lower()
                    guess = "possible_viral (check manually)" if any(k in name_str for k in viral_keywords) else "unknown"
                    for chain in chains:
                        chain_to_organism.setdefault(chain, set()).add(guess)
                    organisms.add(guess)

    # Flatten organism sets
    for chain, orgs in chain_to_organism.items():
        chain_to_organism[chain] = "; ".join(sorted(orgs))
    chain_mapping[pdb_id] = chain_to_organism

    organism_summary = "; ".join(sorted(o for o in organisms if o)) if organisms else "Unknown"
    protein_data.append([pdb_id, experimental_method, organism_summary, resolution])

print(f"\n Total valid protein-containing PDB entries: {len(valid_protein_pdb_ids)}")

print("\n Saving outputs...")

# Main metadata
df_main = pd.DataFrame(protein_data, columns=["PDB_ID", "Experimental_Method", "Organism", "Resolution"])
df_main.to_csv("protein_data.csv", index=False)

# Chain-level organism mapping
chain_rows = []
for pdb_id, chains in chain_mapping.items():
    for chain, org in chains.items():
        chain_rows.append([pdb_id, chain, org])
pd.DataFrame(chain_rows, columns=["PDB_ID", "Chain_ID", "Organism"]).to_csv("chain_organisms.csv", index=False)

# Failed downloads
if failed_downloads:
    with open("failed_cif_downloads.txt", "w") as f:
        f.write("\n".join(failed_downloads))
    print(f"\n {len(failed_downloads)} downloads failed. See 'failed_cif_downloads.txt'.")

print("\n All done! Files saved:")
print(" - protein_data.csv")
print(" - chain_organisms.csv")
print(f" - CIFs in: {output_dir}/")
