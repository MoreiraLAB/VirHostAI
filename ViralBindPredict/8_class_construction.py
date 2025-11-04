#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from multiprocessing import Process, Queue
from tqdm import tqdm
from timeout_decorator.timeout_decorator import TimeoutError

timeout_entries = []
input_csv = "filtered_df.xlsx"
cif_dir = "split_cif_chains_filtered"
output_dir = "interaction_details_csv_cif"
os.makedirs(output_dir, exist_ok=True)
log_file = "error_log.txt"

def log_error(msg):
    with open(log_file, "a") as f:
        f.write(msg + "\n")

# Load data
df = pd.read_excel(input_csv, dtype=str, na_filter=False)
records = df.to_dict(orient="records")

# Extract atoms fro Cif
import shlex
def extract_atoms_from_cif(cif_path):
    protein_atoms, ligand_atoms = [], {}
    with open(cif_path, 'r') as f:
        lines = f.readlines()

    atom_site_keys = []
    atom_lines = []
    reading_atoms = False

    for idx, line in enumerate(lines):
        line = line.strip()
        if line == "loop_":
            if idx + 1 < len(lines) and lines[idx + 1].strip().startswith("_atom_site"):
                reading_atoms = True
                continue
        if reading_atoms and line.startswith("_atom_site"):
            atom_site_keys.append(line)
        elif reading_atoms and not line.startswith("_") and line != "":
            atom_lines.append(line)
        elif reading_atoms and line.startswith("loop_"):
            break

    if not atom_site_keys:
        raise ValueError(f"No _atom_site headers found in: {cif_path}")
    if not atom_lines:
        raise ValueError(f"No atom lines found in: {cif_path}")

    # Map header names to their indices
    key_indices = {key: i for i, key in enumerate(atom_site_keys)}

    required_keys = [
        "_atom_site.group_PDB",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z"
    ]
    missing = [k for k in required_keys if k not in key_indices]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    for line in atom_lines:
        try:
            parts = shlex.split(line)
            if len(parts) < len(atom_site_keys):
                print(f"Skipping malformed line: {line}")
                continue

            group = parts[key_indices["_atom_site.group_PDB"]]
            element = parts[key_indices["_atom_site.type_symbol"]].upper()
            if element == "H":
                continue  # skip hydrogens

            atom_name = parts[key_indices["_atom_site.label_atom_id"]]
            residue_name = parts[key_indices["_atom_site.label_comp_id"]]
            chain_id = parts[key_indices["_atom_site.label_asym_id"]]
            residue_id_raw = parts[key_indices["_atom_site.label_seq_id"]]
            try:
                residue_id = int(residue_id_raw)
            except:
                residue_id = -1

            coords = [
                float(parts[key_indices["_atom_site.Cartn_x"]]),
                float(parts[key_indices["_atom_site.Cartn_y"]]),
                float(parts[key_indices["_atom_site.Cartn_z"]])
            ]
        except Exception as e:
            print(f"Failed to parse line: {line}\nReason: {e}")
            continue

        atom_info = {
            'atom_name': atom_name,
            'residue_name': residue_name,
            'chain_id': chain_id,
            'residue_id': residue_id,
            'coords': coords,
            'type_symbol': element
        }

        if group == "ATOM":
            protein_atoms.append(atom_info)
        elif group == "HETATM":
            key = (residue_name, chain_id)
            ligand_atoms.setdefault(key, []).append(atom_info)

    print(f"Extracted {len(protein_atoms)} protein atoms and {sum(len(v) for v in ligand_atoms.values())} ligand atoms from {os.path.basename(cif_path)}")

    return protein_atoms, ligand_atoms

# Interaction computation - Classify as 'Interacting' if distance <= cutoff (Å), else 'Non-Interacting'
def compute_interactions_batch(residue_atoms, ligand_atoms, cutoff=4.5):
    if not residue_atoms or not ligand_atoms:
        return []
    residue_coords = np.array([a['coords'] for a in residue_atoms])
    ligand_coords = np.array([a['coords'] for a in ligand_atoms])
    dists = np.linalg.norm(residue_coords[:, None, :] - ligand_coords[None, :, :], axis=-1)

    interactions = []
    for i, r_atom in enumerate(residue_atoms):
        for j, l_atom in enumerate(ligand_atoms):
            dist = dists[i, j]
            interactions.append({
                'Atom_Residue': r_atom['atom_name'],
                'Atom_Residue_Coordinates': r_atom['coords'],
                'Atom_Ligand': l_atom['atom_name'],
                'Atom_Ligand_Coordinates': l_atom['coords'],
                'Distance (A)': round(float(dist), 3),
                'Classification': "Interacting" if dist <= cutoff else "Non-Interacting"
            })
    return interactions

# Timeout
def process_ligand_wrapper(q, *args):
    try:
        result = process_ligand(*args)
        q.put(result)
    except Exception as e:
        q.put(e)

def run_ligand_with_timeout(*args, timeout=3000):
    q = Queue()
    p = Process(target=process_ligand_wrapper, args=(q, *args))
    p.start()
    try:
        result = q.get(timeout=timeout + 30)
    except:
        p.terminate()
        p.join()
        raise TimeoutError("Ligand processing queue read exceeded timeout.")
    p.join()
    if isinstance(result, Exception):
        raise result
    return result

# Processing
def process_one(row):
    pdb_id = row['PDB ID']
    chain_id = row['Chain ID']
    pdb_chain_key = f"{pdb_id}_{chain_id}"
    cif_filename = f"{pdb_chain_key}.cif"
    cif_path = os.path.join(cif_dir, cif_filename)

    if not os.path.exists(cif_path):
        log_error(f"[MISSING FILE] {cif_filename}")
        return

    try:
        protein_atoms, ligand_atoms_dict = extract_atoms_from_cif(cif_path)
    except Exception as e:
        log_error(f"[EXTRACTION ERROR] {cif_filename}: {e}")
        return

    positions = [int(p) for p in str(row['POSITIONS']).split('-') if p.isdigit()]
    sequence_1L = row['SEQUENCE_1L']
    sequence_3L_list = row['SEQUENCE_3L_STD'].split('-')

    for ligand_key, ligand_atoms in ligand_atoms_dict.items():
        try:
            run_ligand_with_timeout(
                pdb_id, chain_id, ligand_key, ligand_atoms,
                protein_atoms, row, positions, sequence_1L, sequence_3L_list,
                timeout=3000
            )
        except TimeoutError:
            lig_id, lig_chain = ligand_key
            msg = f"[LIGAND TIMEOUT] {pdb_chain_key} ligand {lig_id} ({lig_chain}) took too long."
            print(msg)
            log_error(msg)
            timeout_entries.append(msg)
        except Exception as e:
            lig_id, lig_chain = ligand_key
            msg = f"[LIGAND ERROR] {pdb_chain_key} ligand {lig_id} ({lig_chain}) failed: {e}"
            print(msg)
            log_error(msg)

def process_ligand(pdb_id, chain_id, ligand_key, ligand_atoms, protein_atoms, row, positions, sequence_1L, sequence_3L_list):
    ligand_resname, ligand_chain = ligand_key
    base_filename = f"{pdb_id}_{chain_id}_{ligand_resname}_{ligand_chain}_interactions"
    csv_path = os.path.join(output_dir, f"{base_filename}.csv.gz")

    if os.path.exists(csv_path):
        return

    result = []
    for res_1l, res_3l, pos in zip(sequence_1L, sequence_3L_list, positions):
        residue_atoms = [a for a in protein_atoms if a['residue_id'] == pos]
        if not residue_atoms:
            continue

        interactions = compute_interactions_batch(residue_atoms, ligand_atoms)

        for inter in interactions:
            result.append({
                "PDB ID": pdb_id,
                "PDB Chain": chain_id,
                "LIG ID": ligand_resname,
                "Position": pos,
                "Residue_1L": res_1l,
                "Residue_3L": res_3l,
                "Ligand_Type": row.get("TYPE", ligand_resname),
                "Type": row.get("CHEM_TYPE", ligand_resname),
                "Organism": row.get("Organism", ""),
                "Resolution": row.get("Resolution", ""),
                "Experimental_Method": row.get("Experimental_Method", ""),
                "Canonical_Smile": row.get("SMILES", ""),
                "Inchikey": row.get("InChIKey", ""),
                "Atom_Residue": inter['Atom_Residue'],
                "Atom_Residue_Coordinates": inter['Atom_Residue_Coordinates'],
                "Atom_Ligand": inter['Atom_Ligand'],
                "Atom_Ligand_Coordinates": inter['Atom_Ligand_Coordinates'],
                "Chain_Id_Ligand": ligand_chain,
                "Distance (A)": inter['Distance (A)'],
                "Classification": inter['Classification']
            })

    if result:
        pd.DataFrame(result).to_csv(csv_path, index=False, compression='gzip')
    else:
        log_error(f"[NO INTERACTIONS] {pdb_id}_{chain_id}_{ligand_resname}_{ligand_chain}")

# Main
if __name__ == "__main__":
    for i, row in enumerate(tqdm(records)):
        try:
            print(f">>> PROCESSING {row['PDB ID']}_{row['Chain ID']}", flush=True)
            process_one(row)
        except TimeoutError:
            print(f"[TIMEOUT] {row['PDB ID']}_{row['Chain ID']} took too long.")
        except Exception as e:
            print(f"[ERROR] {row['PDB ID']}_{row['Chain ID']}: {e}")

    if timeout_entries:
        print("\n=== TIMEOUTS ===")
        for entry in timeout_entries:
            print(entry)
        print(f"\nTotal timeouts: {len(timeout_entries)}")
    else:
        print("\n No timeouts!")