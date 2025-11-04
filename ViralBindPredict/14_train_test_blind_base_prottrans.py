import os
import h5py
import pandas as pd
import random
from sklearn.model_selection import GroupShuffleSplit

def set_seed(seed=42):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# Helper functions
def get_pdb(i): return i.split(':')[0]
def get_chain(i): return i.split(':')[1]
def get_lig(i): return i.split(':')[2]
def get_prot_chain(i): return f"{get_pdb(i)}:{get_chain(i)}"
def lig_no_chain(i): return get_lig(i).split('_')[0]

def get_all_interaction_ids(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        return list(f['interactions'].keys())

def save_split_hdf5(original_path, output_path, interaction_ids, split_label):
    with h5py.File(original_path, 'r') as orig, h5py.File(output_path, 'w') as out:
        if "mordred_column_names" in orig:
            out.create_dataset("mordred_column_names", data=orig["mordred_column_names"][:])
        out.create_group('interactions')
        out.create_group('proteins')
        out.create_group('ligands')
        for i in interaction_ids:
            prot, chain, lig = i.split(':')
            orig_int = orig['interactions'][i]
            int_grp = out['interactions'].create_group(i)
            int_grp.attrs['ligand'] = orig_int.attrs['ligand']
            int_grp.attrs['protein'] = orig_int.attrs['protein']
            int_grp.attrs['split'] = split_label
            int_grp.create_dataset('targets', data=orig_int['targets'][()])
            # Proteins
            if prot not in out['proteins']:
                out['proteins'].create_group(prot)
            if chain not in out[f'proteins/{prot}']:
                data = orig[f'proteins/{prot}/{chain}/features'][()]
                out[f'proteins/{prot}'].create_group(chain)
                out[f'proteins/{prot}/{chain}'].create_dataset('features', data=data)
            # Ligands
            if lig not in out['ligands']:
                data = orig[f'ligands/{lig}/features'][()]
                out['ligands'].create_group(lig)
                out[f'ligands/{lig}'].create_dataset('features', data=data)

def write_sets_to_files(interactions, outdir, seq_mapping, tag=""):
    os.makedirs(outdir, exist_ok=True)
    prot_c = {f"{get_pdb(i)}:{get_chain(i)}" for i in interactions}
    prot = {get_pdb(i) for i in interactions}
    ligs = {get_lig(i) for i in interactions}
    lig_no_chain_set = {lig_no_chain(i) for i in interactions}
    seqs = {seq_mapping.get(get_prot_chain(i)) for i in interactions if get_prot_chain(i) in seq_mapping}
    with open(os.path.join(outdir, f'unique_prot_c{tag}.txt'), 'w') as f:
        f.write('\n'.join(sorted(prot_c)))
    with open(os.path.join(outdir, f'unique_prot{tag}.txt'), 'w') as f:
        f.write('\n'.join(sorted(prot)))
    with open(os.path.join(outdir, f'unique_ligands{tag}.txt'), 'w') as f:
        f.write('\n'.join(sorted(ligs)))
    with open(os.path.join(outdir, f'unique_ligand_ids{tag}.txt'), 'w') as f:
        f.write('\n'.join(sorted(lig_no_chain_set)))
    with open(os.path.join(outdir, f'unique_sequences{tag}.txt'), 'w') as f:
        f.write('\n'.join(sorted(seqs)))
    with open(os.path.join(outdir, f'unique_prot_c_lig{tag}.txt'), 'w') as f:
        f.write('\n'.join(sorted(interactions)))

def split_all(
    original_hdf5,
    seq_mapping_excel,
    output_dir,
    test_size=0.2,
    blind_frac=0.05,
    val_size=0.1,
    seed=44
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Load sequence mapping
    df = pd.read_excel(seq_mapping_excel)
    df['PDB:Chain'] = df['PDB ID'].astype(str) + ':' + df['Chain ID'].astype(str)
    pdb_chain_to_seq = dict(zip(df['PDB:Chain'], df['SEQUENCE_1L']))
    
    print(f" Loading interaction IDs from {original_hdf5}...")
    all_ids = get_all_interaction_ids(original_hdf5)
    print(f" Total interactions in dataset: {len(all_ids)}")

    # Blind sequences and ligands
    all_sequences = sorted(set(
        pdb_chain_to_seq.get(get_prot_chain(i))
        for i in all_ids if get_prot_chain(i) in pdb_chain_to_seq
    ))
    all_lig_base = sorted(set(lig_no_chain(i) for i in all_ids))

    print(f" Unique protein sequences: {len(all_sequences)}")
    print(f" Unique ligands (no chain): {len(all_lig_base)}")
    
    sampled_sequences = set(random.sample(all_sequences, max(1, int(blind_frac * len(all_sequences)))))
    sampled_ligands_base = set(random.sample(all_lig_base, max(1, int(blind_frac * len(all_lig_base)))))

    print(f" Sampled {len(sampled_sequences)} ({blind_frac*100:.1f}%) sequences for blind set")
    print(f" Sampled {len(sampled_ligands_base)} ({blind_frac*100:.1f}%) ligands for blind set")

    # Collateral blind PDBs
    blind_pdb_ids = set()
    blind_sequences = set(sampled_sequences)
    print(f" Initial blind sequences: {len(blind_sequences)}")
    for seq in sampled_sequences:
        chains = [pc for pc, s in pdb_chain_to_seq.items() if s == seq]
        pdbs = {get_pdb(pc) for pc in chains}
        blind_pdb_ids.update(pdbs)
        for pdb in pdbs:
            pdb_chains = [f"{pdb}:{chain}" for chain in set(get_chain(i) for i in all_ids if get_pdb(i) == pdb)]
            blind_sequences.update(pdb_chain_to_seq.get(pc) for pc in pdb_chains if pc in pdb_chain_to_seq)

    print(f" Blind PDB IDs: {len(blind_pdb_ids)}")
    print(f" Blind sequences after collateral PDB inclusion: {len(blind_sequences)}")

    # Define clusters (PDB:SEQUENCE:LIG:LIGCHAIN)
    clusters = {}
    for i in all_ids:
        seq = pdb_chain_to_seq.get(get_prot_chain(i))

        ckey = f"{get_pdb(i)}:{seq}:{get_lig(i)}"  # PDB:SEQUENCE:LIG:LIGCHAIN
        clusters.setdefault(ckey, []).append(i)

    # Assign clusters
    blind_ids, blind_protein_ids, blind_ligand_ids, test_ids_extra, train_test_candidates = [], [], [], [], []
    for ckey, members in clusters.items():
        pdb = get_pdb(members[0])
        lig_base = lig_no_chain(members[0])
        seqs = {pdb_chain_to_seq.get(get_prot_chain(m)) for m in members if get_prot_chain(m) in pdb_chain_to_seq}

        if pdb in blind_pdb_ids and lig_base in sampled_ligands_base:
            blind_ids.extend(members)
        elif pdb in blind_pdb_ids:
            blind_protein_ids.extend(members)
        elif lig_base in sampled_ligands_base:
            if any(s in blind_sequences for s in seqs):
                test_ids_extra.extend(members)
            else:
                blind_ligand_ids.extend(members)
        elif any(s in blind_sequences for s in seqs):
            test_ids_extra.extend(members)
        else:
            train_test_candidates.extend(members)

    print(f"\n Blind set interactions: {len(blind_ids)}")
    print(f" Blind protein-only interactions: {len(blind_protein_ids)}")
    print(f" Blind ligand-only interactions: {len(blind_ligand_ids)}")
    print(f" Test collateral (test_ids_extra): {len(test_ids_extra)}")
    print(f" Train/Test candidates (before split): {len(train_test_candidates)}")

     # ---- New: Split candidates into TRAIN / VAL / TEST cluster-wise ----
    # Prepare candidate clusters list
    candidate_cluster_items = []
    for ckey, members in clusters.items():
        if any(m in train_test_candidates for m in members):
            candidate_cluster_items.append((ckey, [m for m in members if m in train_test_candidates]))

    random.shuffle(candidate_cluster_items)
    total_cand = sum(len(members) for _, members in candidate_cluster_items)
    target_test = int(round(test_size * total_cand))
    target_val  = int(round(val_size  * total_cand))
    target_train = total_cand - target_test - target_val
    print(f" Target counts → TRAIN={target_train}, VAL={target_val}, TEST={target_test} (from candidates only)")

    train_ids, val_ids, test_ids = [], [], []
    interaction_keys_seen = set()  # Tracks PDB:CHAIN:LIG globally to prevent any overlap accident

    def assign_cluster(members):
        nonlocal train_ids, val_ids, test_ids
        # Compute current deficits
        rem_train = target_train - len(train_ids)
        rem_val   = target_val  - len(val_ids)
        rem_test  = target_test - len(test_ids)

        # Priority: fill the largest deficit first
        deficits = [('train', rem_train), ('val', rem_val), ('test', rem_test)]
        deficits.sort(key=lambda x: x[1], reverse=True)
        destination = deficits[0][0]
        if destination == 'train':
            train_ids.extend(members)
        elif destination == 'val':
            val_ids.extend(members)
        else:
            test_ids.extend(members)

    for ckey, cluster_members in candidate_cluster_items:
        cluster_interaction_keys = {f"{get_pdb(i)}:{get_chain(i)}:{get_lig(i)}" for i in cluster_members}

        # Overlap prevention: if overlaps seen keys, push to TEST
        if cluster_interaction_keys & interaction_keys_seen:
            test_ids.extend(cluster_members)
        else:
            assign_cluster(cluster_members)
            interaction_keys_seen.update(cluster_interaction_keys)

    # Add collateral exclusively to TEST
    test_ids.extend(test_ids_extra)

    print(f" Train set: {len(train_ids)} interactions")
    print(f" Val set: {len(val_ids)} interactions")
    print(f" Test set: {len(test_ids)} interactions (including collateral)")
    print(" Validating splits...")

    # Validate interaction uniqueness
    all_splits = {
        "train": set(f"{get_pdb(i)}:{get_chain(i)}:{get_lig(i)}" for i in train_ids),
        "val": set(f"{get_pdb(i)}:{get_chain(i)}:{get_lig(i)}" for i in val_ids),
        "test": set(f"{get_pdb(i)}:{get_chain(i)}:{get_lig(i)}" for i in test_ids),
        "blind": set(f"{get_pdb(i)}:{get_chain(i)}:{get_lig(i)}" for i in blind_ids),
        "blind_protein": set(f"{get_pdb(i)}:{get_chain(i)}:{get_lig(i)}" for i in blind_protein_ids),
        "blind_ligand": set(f"{get_pdb(i)}:{get_chain(i)}:{get_lig(i)}" for i in blind_ligand_ids),
    }
    for s1, ids1 in all_splits.items():
        for s2, ids2 in all_splits.items():
            if s1 != s2:
                overlap = ids1 & ids2
                if overlap:
                    raise ValueError(f" Overlap between {s1} and {s2}: {len(overlap)} interactions")

    # Save splits
    save_split_hdf5(original_hdf5, os.path.join(output_dir, "prottrans_train.hdf5"), train_ids, "train")
    save_split_hdf5(original_hdf5, os.path.join(output_dir, "prottrans_val.hdf5"),   val_ids,   "val")   # NEW
    save_split_hdf5(original_hdf5, os.path.join(output_dir, "prottrans_test.hdf5"),  test_ids,  "test")
    save_split_hdf5(original_hdf5, os.path.join(output_dir, "prottrans_blind.hdf5"), blind_ids, "blind")
    save_split_hdf5(original_hdf5, os.path.join(output_dir, "prottrans_blind_protein.hdf5"), blind_protein_ids, "blind_protein")
    save_split_hdf5(original_hdf5, os.path.join(output_dir, "prottrans_blind_ligand.hdf5"),  blind_ligand_ids,  "blind_ligand")

    # Helper TXT exports
    write_sets_to_files(train_ids, output_dir, pdb_chain_to_seq, tag="_train")
    write_sets_to_files(val_ids,   output_dir, pdb_chain_to_seq, tag="_val")      # NEW
    write_sets_to_files(test_ids,  output_dir, pdb_chain_to_seq, tag="_test")
    write_sets_to_files(blind_ids, output_dir, pdb_chain_to_seq, tag="_blind")
    write_sets_to_files(blind_protein_ids, output_dir, pdb_chain_to_seq, tag="_blind_protein")
    write_sets_to_files(blind_ligand_ids,  output_dir, pdb_chain_to_seq, tag="_blind_ligand")

    # Print summary
    print("\n Split Summary:")
    print(f"Train: {len(train_ids)} interactions")
    print(f"Val:   {len(val_ids)} interactions")
    print(f"Test:  {len(test_ids)} interactions (including {len(test_ids_extra)} collateral)")
    print(f"Blind: {len(blind_ids)} interactions")
    print(f"Blind Protein: {len(blind_protein_ids)} interactions")
    print(f"Blind Ligand:  {len(blind_ligand_ids)} interactions")
    print("\n All splits and metadata saved successfully.")
    
split_all("viralbindpredictDB-prottrans-cleaned.hdf5", "deduplicated_filtered_prottrans_df.xlsx", output_dir="splits_prottrans_output", test_size=0.05, blind_frac=0.07, val_size= 0.1, seed=42)

# Validate feature shapes
def validate_first3_shapes(hdf5_path):
    print(f"\nValidating first 3 protein and ligand feature shapes in {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        # Proteins
        prot_printed = 0
        for prot in f['proteins']:
            for chain in f[f'proteins/{prot}']:
                data = f[f'proteins/{prot}/{chain}/features'][()]
                print(f"Protein {prot}:{chain} shape: {data.shape}")
                prot_printed += 1
                if prot_printed >= 3:
                    break
            if prot_printed >= 3:
                break

        # Ligands
        lig_printed = 0
        for lig in f['ligands']:
            data = f[f'ligands/{lig}/features'][()]
            print(f"Ligand {lig} shape: {data.shape}")
            lig_printed += 1
            if lig_printed >= 3:
                break

# Validate splits
for split_file in [
    'prottrans_train.hdf5',
    'prottrans_test.hdf5',
    'prottrans_val.hdf5',
    'prottrans_blind.hdf5',
    'prottrans_blind_ligand.hdf5',
    'prottrans_blind_protein.hdf5'
]:
    validate_first3_shapes(os.path.join("splits_prottrans_output", split_file))