#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd

# Ambiguous or uncommon residues
ambiguous_residues = {"SEC", "PYL", "UNK"}

# Standard 3-letter to 1-letter amino acid code mapping
aa_3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

# Split CIF into "loop_" blocks for parsing structured tables
def extract_loop_blocks(cif_lines):
    blocks, current_block, in_loop = [], [], False
    for line in cif_lines:
        if line.strip() == 'loop_':
            if current_block:
                blocks.append(current_block)
            current_block = ['loop_']
            in_loop = True
        elif in_loop:
            # Collect lines inside loop until a blank line or new section
            if line.startswith('_') or line.strip():
                current_block.append(line)
            else:
                in_loop = False
                blocks.append(current_block)
                current_block = []
    if current_block:
        blocks.append(current_block)
    return blocks

# Extract _atom_site data table (atomic coordinates, residue info)
def extract_atom_site_lines(cif_lines):
    blocks = extract_loop_blocks(cif_lines)
    for block in blocks:
        if any(line.startswith('_atom_site.') for line in block):
            fields = [line.strip() for line in block if line.startswith('_atom_site.')]

            # Everything after the field headers are data lines
            data_start = next(i for i, line in enumerate(block) if not line.startswith('_'))
            atom_lines = block[data_start:]

            # Filter only actual ATOM/HETATM lines
            atom_lines = [l if l.endswith('\n') else l + '\n' for l in atom_lines
                          if l.strip().startswith(('ATOM', 'HETATM'))]

            return fields, atom_lines
    return [], []

# Preserve original CIF spacing when modifying HETATM→ATOM
def preserve_spacing(original_line, updated_fields):
    import re
    tokens = re.findall(r'\S+|\s+', original_line.rstrip('\n'))
    parts = tokens[::2]
    spaces = tokens[1::2] + [' ']

    if len(parts) != len(updated_fields):
        print("Warning: preserve_spacing fallback used due to mismatched field count.")
        return " ".join(updated_fields) + "\n"  # fallback

    # Detect if converted HETATM → ATOM
    if parts[0] == "HETATM" and updated_fields[0] == "ATOM":
        # Add 2 spaces after 'ATOM'
        spaces[0] = spaces[0] + "  "

    return ''.join(p + s for p, s in zip(updated_fields, spaces)) + "\n"

# Extract mapping of modified residues (e.g. MSE → MET)
def extract_mod_residue_mapping(cif_lines):
    blocks = extract_loop_blocks(cif_lines)
    mod_residue_map = {}
    for block in blocks:
        if not any(line.startswith('_pdbx_struct_mod_residue.') for line in block):
            continue
        headers = [h.strip() for h in block if h.startswith('_')]
        data = [l.strip().split() for l in block if not l.startswith('_') and l.strip()]
        try:
            idx_chain = headers.index("_pdbx_struct_mod_residue.label_asym_id")
            idx_seq = headers.index("_pdbx_struct_mod_residue.label_seq_id")
            idx_mod = headers.index("_pdbx_struct_mod_residue.label_comp_id")
            idx_parent = headers.index("_pdbx_struct_mod_residue.parent_comp_id")
            for row in data:
                if len(row) > max(idx_chain, idx_seq, idx_mod, idx_parent):
                    key = (row[idx_chain], row[idx_seq], row[idx_mod])
                    mod_residue_map[key] = row[idx_parent]
        except ValueError:
            continue
    return mod_residue_map

# Process a single CIF file (split by chain, generate FASTA)
def process_cif(cif_path, out_dir, fasta_dir, mapping_dir=None, ambiguity_threshold=0.0):
    pdb_id = os.path.basename(cif_path).split('.')[0]
    print(f"\n Processing {pdb_id}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fasta_dir, exist_ok=True)
    if mapping_dir:
        os.makedirs(mapping_dir, exist_ok=True)
        
    # Read entire CIF content
    with open(cif_path, "r") as f:
        lines = f.readlines()
        
    # Extract atom site table and modified residue mappings
    fields, atom_lines = extract_atom_site_lines(lines)
    mod_map = extract_mod_residue_mapping(lines)

    if not fields:
        print(f" No _atom_site fields found in {pdb_id}. Skipping.")
        return []
    # Map field names to column indices for quick access
    field_idx = {f: i for i, f in enumerate(fields)}
    model_idx = field_idx.get("_atom_site.pdbx_PDB_model_num", None)
    
    # Identify the first model (if multiple exist)
    first_model = None
    if model_idx is not None:
        for line in atom_lines:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) > model_idx:
                model = parts[model_idx]
                if model.strip().isdigit():
                    first_model = model
                    break
   
    # Ensure all required atom site columns are present
    required = [
        "_atom_site.group_PDB", "_atom_site.label_comp_id", "_atom_site.label_asym_id",
        "_atom_site.label_seq_id", "_atom_site.auth_seq_id", "_atom_site.auth_asym_id"
    ]
    for r in required:
        if r not in field_idx:
            print(f" Required field {r} missing in {pdb_id}")
            return []

    required_idxs = [field_idx[r] for r in required]
    sequences, sequences1L, positions, mapping = {}, {}, {}, {}
    seen_residues = set()
    ambiguous_counts, total_counts = {}, {}
    kept_lines_by_chain = {}
    ligand_lines = []
    ligand_ids = set()
    
    # Process each atom line
    for line in atom_lines:
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if any(i >= len(parts) for i in required_idxs):
            continue
        if model_idx is not None and model_idx < len(parts):
            if parts[model_idx] != first_model:
                continue

        group = parts[field_idx["_atom_site.group_PDB"]]
        resname = parts[field_idx["_atom_site.label_comp_id"]]
        chain = parts[field_idx["_atom_site.label_asym_id"]]
        resseq = str(parts[field_idx["_atom_site.label_seq_id"]])
        auth_seq = parts[field_idx["_atom_site.auth_seq_id"]]
        auth_chain = parts[field_idx["_atom_site.auth_asym_id"]]

        key_mod = (chain, resseq, resname)
        is_modified = key_mod in mod_map
        
        # HETATM modified residue (convert to standard)
        if group == "HETATM" and is_modified:
            mapped_resname = mod_map[key_mod]
            if mapped_resname in aa_3to1:
                parts[field_idx["_atom_site.group_PDB"]] = "ATOM"
                parts[field_idx["_atom_site.label_comp_id"]] = mapped_resname
                line = preserve_spacing(line, parts)
                print(f" {pdb_id} chain {chain} pos {resseq} — {resname} → {mapped_resname} (corrected HETATM)")
                kept_lines_by_chain.setdefault(chain, []).append(line)
        
                #  Add the corrected residue to the sequence and output structures
                key = (chain, resseq)
                if key not in seen_residues:
                    seen_residues.add(key)
                    sequences.setdefault(chain, []).append(mapped_resname)
                    sequences1L.setdefault(chain, []).append(aa_3to1[mapped_resname])
                    positions.setdefault(chain, []).append(resseq)
                    mapping.setdefault(chain, []).append((resseq, chain, auth_seq, auth_chain))
        
                continue

            else:
                # Modified residue not standard → skip
                print(f" {pdb_id} chain {chain} pos {resseq} — {resname} → {mapped_resname} is not standard. Skipping.")
                ambiguous_counts[chain] = ambiguous_counts.get(chain, 0) + 1
                continue
            
        # Regular HETATM (ligand/water)
        elif group == "HETATM":
            if resname == "HOH":
                continue # Skip waters
            ligand_lines.append(line)
            ligand_ids.add(resname)
            continue
        
        # Normal ATOM residue
        else:
            mapped_resname = resname

        total_counts[chain] = total_counts.get(chain, 0) + 1
        
        # Skip ambiguous or non-standard residues
        if mapped_resname in ambiguous_residues or mapped_resname not in aa_3to1:
            print(f" Ambiguous residue {mapped_resname} at chain {chain} pos {resseq}")
            ambiguous_counts[chain] = ambiguous_counts.get(chain, 0) + 1
            continue
        
        # Add unique residue to sequence and mapping
        key = (chain, resseq)
        if key not in seen_residues:
            seen_residues.add(key)
            sequences.setdefault(chain, []).append(mapped_resname)
            sequences1L.setdefault(chain, []).append(aa_3to1[mapped_resname])
            positions.setdefault(chain, []).append(resseq)
            mapping.setdefault(chain, []).append((resseq, chain, auth_seq, auth_chain))

        kept_lines_by_chain.setdefault(chain, []).append(line)
  
    # Summary of ambiguous residues
    for chain in total_counts:
        amb = ambiguous_counts.get(chain, 0)
        total = total_counts[chain]
        print(f" Chain {chain}: ambiguous={amb}, total={total}, ratio={amb/total:.2%}")
   
    # Filter chains passing ambiguity threshold
    chains_ok = [
        c for c in sequences
        if (ambiguous_counts.get(c, 0) / total_counts.get(c, 1)) <= ambiguity_threshold
    ]
    if not chains_ok:
        print(f" No chains passed ambiguity threshold in {pdb_id}")
        print("Chains OK:", chains_ok)
   
    # Write outputs for each valid chain
    output_data = []
    for chain in chains_ok:
        key = f"{pdb_id}_chain_{chain}"
        with open(os.path.join(fasta_dir, f"{key}.fasta"), "w") as f_out:
            f_out.write(f">{key}\n")
            f_out.write("".join(sequences1L[chain]) + "\n")

        if mapping_dir:
            df_map = pd.DataFrame(mapping[chain], columns=["label_seq_id", "label_asym_id", "auth_seq_id", "auth_asym_id"])
            df_map.to_csv(os.path.join(mapping_dir, f"{key}_author_label_mapping.csv"), index=False)

        with open(os.path.join(out_dir, f"{key}.cif"), "w") as fout:
            # Write minimal CIF header so PyMOL can parse it
            fout.write(f"data_{key}\n")
            fout.write("#\n")
            fout.write(f"_entry.id   {pdb_id}\n")
            fout.write("#\n")
        
            fout.write("loop_\n")
            for field in fields:
                fout.write(f"{field}\n")
            for line in kept_lines_by_chain[chain]:
                fout.write(line if line.endswith("\n") else line + "\n")
        
            for line in ligand_lines:
                if line.strip().startswith("ATOM") or line.strip().startswith("HETATM"):
                    fout.write(line if line.endswith("\n") else line + "\n")
        

        ligands_str = ",".join(sorted(ligand_ids)) if ligand_ids else "NO_LIGAND"
        output_data.append([
            pdb_id, chain,
            "-".join(sequences[chain]),
            "".join(sequences1L[chain]),
            "-".join(positions[chain]),
            ligands_str
        ])

    return output_data

# Batch processing: iterate through all CIF files in directory
def process_all_cifs(cif_dir, out_dir, fasta_dir, mapping_dir=None):
    os.makedirs(cif_dir, exist_ok=True)
    all_rows = []

    for filename in os.listdir(cif_dir):
        if not filename.lower().endswith(".cif") or filename.startswith("~$"):
            continue
        cif_path = os.path.join(cif_dir, filename)
        rows = process_cif(cif_path, out_dir, fasta_dir, mapping_dir)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=["PDB ID", "Chain ID", "SEQUENCE_3L_STD", "SEQUENCE_1L", "POSITIONS", "LIGAND_ID"])
    df.to_csv("ligands_per_chain.csv", index=False)
    print("\n Done! ligands_per_chain.csv created.")
    return df

# Run

process_all_cifs("cif", "split_cif_chains", "fasta_sequences", "author_label_mappings")
