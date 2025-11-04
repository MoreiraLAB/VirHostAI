import os
import h5py
import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Set seed for reproducibility
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

def write_list(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for x in sorted(items):
            f.write(str(x) + "\n")

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
    with open(os.path.join(outdir, f'unique_prot_c{tag}_nored_90%_cluster90.txt'), 'w') as f:
        f.write('\n'.join(sorted(prot_c)))
    with open(os.path.join(outdir, f'unique_prot{tag}_nored_90%_cluster90.txt'), 'w') as f:
        f.write('\n'.join(sorted(prot)))
    with open(os.path.join(outdir, f'unique_ligands{tag}_nored_90%_cluster90.txt'), 'w') as f:
        f.write('\n'.join(sorted(ligs)))
    with open(os.path.join(outdir, f'unique_ligand_ids{tag}_nored_90%_cluster90.txt'), 'w') as f:
        f.write('\n'.join(sorted(lig_no_chain_set)))
    with open(os.path.join(outdir, f'unique_sequences{tag}_nored_90%_cluster90.txt'), 'w') as f:
        f.write('\n'.join(sorted(seqs)))
    with open(os.path.join(outdir, f'unique_prot_c_lig{tag}_nored_90%_cluster90.txt'), 'w') as f:
        f.write('\n'.join(sorted(interactions)))

# MMSEQS2 clusters --> used for clutering of protein sequences and assgning different clusters to blind and blind protein
def load_mmseqs_clusters(mmseqs_file_path):
    cluster_map = {}
    cluster_contents = defaultdict(list)
    current_cluster = None

    with open(mmseqs_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            # Lines like: "Cluster#   1"
            if line.startswith("Cluster#"):
                # grab the number after '#'
                try:
                    current_cluster = int(line.split("#", 1)[1].strip())
                except ValueError:
                    current_cluster = line.split("#", 1)[1].strip()
                continue

            # Lines like: ">7t7h_B"
            if line.startswith(">"):
                entry = line[1:].split()[0]  # e.g. "7t7h_B"
                # Convert to PDB:Chain
                if "_" in entry:
                    pdb_chain = entry.replace("_", ":", 1)
                elif ":" in entry:
                    pdb_chain = entry
                else:
                    continue

                if current_cluster is not None:
                    cluster_map[pdb_chain] = current_cluster
                    cluster_contents[current_cluster].append(pdb_chain)

    print(f"\n Total MMseqs2 clusters: {len(cluster_contents)}")
    if cluster_contents:
        sizes = [len(v) for v in cluster_contents.values()]
        print(f" Cluster size (min/avg/max): {min(sizes)} / {sum(sizes)//len(sizes)} / {max(sizes)}")
        print("\n Example clusters:")
        for cid, members in list(cluster_contents.items())[:5]:
            print(f"  Cluster {cid}: {len(members)} chains → {members[:6]}{' ...' if len(members)>6 else ''}")

    return cluster_map

# Main function to split in training, testing, blind protein, blind ligand and blind sets
def split_all(
    original_hdf5,
    seq_mapping_excel,
    output_dir,
    test_size=0.2,
    blind_frac=0.05,
    val_size=0.1,
    seed=44,
    mmseqs_clusters_path="mmseqs2_90cluster.txt"
):

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Load sequence & MMseqs clusters
    df = pd.read_excel(seq_mapping_excel)
    df['PDB:Chain'] = df['PDB ID'].astype(str) + ':' + df['Chain ID'].astype(str)
    pdb_chain_to_cluster = load_mmseqs_clusters(mmseqs_clusters_path)
    df['ClusterID'] = df['PDB:Chain'].map(pdb_chain_to_cluster)
    pdb_chain_to_seq = dict(zip(df['PDB:Chain'], df['SEQUENCE_1L']))

    print("\n Total PDB:Chain entries:", len(df))
    print(" Mapped to cluster:", df['ClusterID'].notna().sum())
    print(" Not mapped:", df['ClusterID'].isna().sum())

    # Load all interaction IDs
    all_ids = get_all_interaction_ids(original_hdf5)
    print(f" Total interactions in dataset: {len(all_ids)}")

    # Ligand → SMILES → BM scaffold; some non-scaffold ligands produce None and are not used for blind
    ligand_to_smiles = dict(zip(df['LIGAND_ID'], df['SMILES']))

    def bm_scaffold_exact(smi):
        if not isinstance(smi, str) or not smi:
            return None
        try:
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                return None
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            if scaf and scaf.GetNumAtoms() > 0:
                # exact BM scaffold
                return Chem.MolToSmiles(scaf)
            return None  # empty scaffold
        except Exception:
            return None

    # Only consider ligands that actually appear in the interactions
    all_lig_bases = {lig_no_chain(i) for i in all_ids}
    lig_to_scaffold = {
        lig: bm_scaffold_exact(smi)
        for lig, smi in ligand_to_smiles.items()
        if lig in all_lig_bases
    }
    
    total_ligs_in_interactions = len(lig_to_scaffold)
    no_scaffold_count = sum(1 for scaf in lig_to_scaffold.values() if not scaf)
    
    print(f" Total ligands in interactions: {total_ligs_in_interactions}")
    print(f" Ligands with no scaffold: {no_scaffold_count} ({no_scaffold_count/total_ligs_in_interactions:.1%})")

    # keep only ligands with a valid (non-empty) BM scaffold
    lig_to_scaffold = {lig: scaf for lig, scaf in lig_to_scaffold.items() if scaf}

    unique_scaffolds = sorted(set(lig_to_scaffold.values()))
    print(f" Ligands present in interactions (with scaffold): {len(lig_to_scaffold)}")
    print(f" Unique exact BM scaffolds: {len(unique_scaffolds)}")

    # Sample blind protein clusters (MMseqs)
    unique_clusters = df['ClusterID'].dropna().unique()
    if len(unique_clusters) == 0:
        print(" No MMseqs clusters found; blind_protein will be empty.")
        sampled_clusters = set()
    else:
        sampled_clusters = set(random.sample(list(unique_clusters),
                                             max(1, int(blind_frac * len(unique_clusters)))))
    blind_cluster_chains = set(df[df['ClusterID'].isin(sampled_clusters)]['PDB:Chain'])
    
    # Expand to all chains from the same PDBs
    blind_pdb_ids = {get_pdb(pc) for pc in blind_cluster_chains}
    if blind_pdb_ids:
        blind_cluster_chains = set(df[df['PDB ID'].isin(blind_pdb_ids)]['PDB:Chain'])

    # Sample blind scaffolds and derive blind ligands (non-scaffold ligands not included)
    if len(unique_scaffolds) == 0:
        print(" No ligands with BM scaffolds; blind_ligand will be empty.")
        sampled_scaffolds = set()
        sampled_ligands_base = set()
    else:
        num_blind_scaffolds = max(1, int(blind_frac * len(unique_scaffolds)))
        sampled_scaffolds = set(random.sample(unique_scaffolds, num_blind_scaffolds))
        sampled_ligands_base = {lig for lig, scaf in lig_to_scaffold.items()
                                if scaf in sampled_scaffolds}
    print(f" Sampled {len(sampled_scaffolds)} scaffolds → "
          f"{len(sampled_ligands_base)} ligands for blind-ligand")

    # Pre-filter interactions into blind/BL-only/PL-only and train/test candidates
    blind_chains_set = set(blind_cluster_chains)
    blind_ligands_set = set(sampled_ligands_base)

    blind_ids, blind_protein_ids, blind_ligand_ids, train_test_candidates = [], [], [], []

    # Group by (PDB:SEQUENCE:LIG_LIGCHAIN) so related interactions move together
    clusters_interactions = {}
    for i in all_ids:
        seq = pdb_chain_to_seq.get(get_prot_chain(i))
        ckey = f"{get_pdb(i)}:{seq}:{get_lig(i)}"
        clusters_interactions.setdefault(ckey, []).append(i)

    for ckey, members in clusters_interactions.items():
        ligand_base = lig_no_chain(members[0])
        member_chains = {get_prot_chain(m) for m in members}
        if any(chain in blind_chains_set for chain in member_chains):
            if ligand_base in blind_ligands_set:
                blind_ids.extend(members)            # both protein-cluster & scaffold-blind
            else:
                blind_protein_ids.extend(members)    # protein-blind only
        elif ligand_base in blind_ligands_set:
            blind_ligand_ids.extend(members)         # scaffold-blind only
        else:
            train_test_candidates.extend(members)

    print("\n Blind set interactions:", len(blind_ids))
    print(" Blind protein-only:", len(blind_protein_ids))
    print(" Blind ligand-only:", len(blind_ligand_ids))
    print(" Train/Val/Test candidates:", len(train_test_candidates))

    # Remove candidates sharing PROTEIN clusters with blind (MMseqs)
    blind_clusters = set(df[df['PDB:Chain'].isin(blind_cluster_chains)]['ClusterID'].dropna())
    seen_chains = set(df[df['ClusterID'].isin(blind_clusters)]['PDB:Chain'])
    clean_candidates = [i for i in train_test_candidates if get_prot_chain(i) not in seen_chains]

    # Forbid SCAFFOLD leakage from blind/blind_ligand into train/val
    blind_scaffolds = sampled_scaffolds

    scaf_collateral = [i for i in clean_candidates
                       if lig_to_scaffold.get(lig_no_chain(i)) in blind_scaffolds]
    clean_candidates = [i for i in clean_candidates
                        if lig_to_scaffold.get(lig_no_chain(i)) not in blind_scaffolds]

    # TRAIN/VAL/TEST split on clean candidates, grouped by (PPDB:CHAIN:LIG_LIGCHAIN)
    train_ids, val_ids, test_ids = [], [], []
    test_ids_collateral = []

    random.shuffle(clean_candidates)

    candidate_cluster_items = []
    for ckey, members in clusters_interactions.items():
        mems = [m for m in members if m in clean_candidates]
        if mems:
            candidate_cluster_items.append((ckey, mems))

    random.shuffle(candidate_cluster_items)
    total_cand = sum(len(mems) for _, mems in candidate_cluster_items)
    train_frac = max(0.0, 1.0 - test_size - val_size)
    target_train = int(round(train_frac * total_cand))
    target_val   = int(round(val_size  * total_cand))
    target_test  = total_cand - target_train - target_val

    interaction_keys_seen = set()

    def assign_cluster(members):
        deficits = [
            ('train', target_train - len(train_ids)),
            ('val',   target_val   - len(val_ids)),
            ('test',  target_test  - len(test_ids)),
        ]
        dest = max(deficits, key=lambda x: x[1])[0]
        if dest == 'train': train_ids.extend(members)
        elif dest == 'val': val_ids.extend(members)
        else: test_ids.extend(members)

    for ckey, members in candidate_cluster_items:
        cluster_interaction_keys = {f"{get_pdb(i)}:{get_chain(i)}:{get_lig(i)}" for i in members}
        if cluster_interaction_keys & interaction_keys_seen:
            test_ids.extend(members)
        else:
            assign_cluster(members)
            interaction_keys_seen.update(cluster_interaction_keys)

    # Add scaffold-collateral & PDB-collateral to TEST
    already_assigned = set(train_ids + val_ids + test_ids +
                           blind_ids + blind_protein_ids + blind_ligand_ids)

    scaf_collateral = [i for i in scaf_collateral if i not in already_assigned]
    test_ids.extend(scaf_collateral)
    test_ids_collateral.extend(scaf_collateral)

    affected_clusters = set(df[df['PDB:Chain'].isin(blind_cluster_chains)]['ClusterID'].dropna())
    affected_chains = set(df[df['ClusterID'].isin(affected_clusters)]['PDB:Chain'])
    pdb_collateral = [i for i in all_ids
                      if (i not in already_assigned) and (get_prot_chain(i) in affected_chains)]
    test_ids.extend(pdb_collateral)
    test_ids_collateral.extend(pdb_collateral)

    # no scaffold leakage into train/val
    def scaffolds_of(ids):
        return {lig_to_scaffold.get(lig_no_chain(i))
                for i in ids if lig_to_scaffold.get(lig_no_chain(i)) is not None}

    train_val_scaffolds = scaffolds_of(train_ids) | scaffolds_of(val_ids)
    assert not (train_val_scaffolds & blind_scaffolds), "Scaffold leakage into train/val detected."

    # Validate splits: no overlap
    all_splits = {
        "train": set(f"{get_pdb(i)}:{get_chain(i)}:{get_lig(i)}" for i in train_ids),
        "val":   set(f"{get_pdb(i)}:{get_chain(i)}:{get_lig(i)}" for i in val_ids),
        "test":  set(f"{get_pdb(i)}:{get_chain(i)}:{get_lig(i)}" for i in test_ids),
        "blind": set(f"{get_pdb(i)}:{get_chain(i)}:{get_lig(i)}" for i in blind_ids),
        "blind_protein": set(f"{get_pdb(i)}:{get_chain(i)}:{get_lig(i)}" for i in blind_protein_ids),
        "blind_ligand":  set(f"{get_pdb(i)}:{get_chain(i)}:{get_lig(i)}" for i in blind_ligand_ids),
    }
    for s1, ids1 in all_splits.items():
        for s2, ids2 in all_splits.items():
            if s1 != s2:
                overlap = ids1 & ids2
                if overlap:
                    raise ValueError(f" Overlap between {s1} and {s2}: {len(overlap)} interactions")

    # Save splits
    save_split_hdf5(original_hdf5, os.path.join(output_dir, "esm2_train_nored_90%_cluster90.hdf5"), train_ids, "train")
    save_split_hdf5(original_hdf5, os.path.join(output_dir, "esm2_val_nored_90%_cluster90.hdf5"),   val_ids,   "val")
    save_split_hdf5(original_hdf5, os.path.join(output_dir, "esm2_test_nored_90%_cluster90.hdf5"),  test_ids, "test")
    save_split_hdf5(original_hdf5, os.path.join(output_dir, "esm2_blind_nored_90%_cluster90.hdf5"), blind_ids, "blind")
    save_split_hdf5(original_hdf5, os.path.join(output_dir, "esm2_blind_protein_nored_90%_cluster90.hdf5"), blind_protein_ids, "blind_protein")
    save_split_hdf5(original_hdf5, os.path.join(output_dir, "esm2_blind_ligand_nored_90%_cluster90.hdf5"),  blind_ligand_ids,  "blind_ligand")

    # Write helper text files
    write_sets_to_files(train_ids, output_dir, pdb_chain_to_seq, tag="_train")
    write_sets_to_files(val_ids,   output_dir, pdb_chain_to_seq, tag="_val")
    write_sets_to_files(test_ids,  output_dir, pdb_chain_to_seq, tag="_test")
    write_sets_to_files(blind_ids, output_dir, pdb_chain_to_seq, tag="_blind")
    write_sets_to_files(blind_protein_ids, output_dir, pdb_chain_to_seq, tag="_blind_protein")
    write_sets_to_files(blind_ligand_ids,  output_dir, pdb_chain_to_seq, tag="_blind_ligand")

    # Summary
    print("\n Split Summary:")
    print(f"Train: {len(train_ids)} interactions")
    print(f"Val:   {len(val_ids)} interactions")
    print(f"Test:  {len(test_ids)} interactions (including {len(test_ids_collateral)} collateral)")
    print(f"Blind: {len(blind_ids)} interactions")
    print(f"Blind Protein: {len(blind_protein_ids)} interactions")
    print(f"Blind Ligand: {len(blind_ligand_ids)} interactions")
    print("\n All splits and metadata saved successfully.")

split_all("viralbindpredictDB-esm2-cleaned_no_red.hdf5", "deduplicated_esm2_mordred_filtered_df_nored.xlsx", output_dir="splits_esm2_output_nored_90%_cluster90", test_size=0.1, blind_frac=0.1, val_size=0.1, seed=98, mmseqs_clusters_path="mmseqs2_90clusters.txt")

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
    'esm2_train_nored_90%_cluster90.hdf5',
    'esm2_test_nored_90%_cluster90.hdf5',
    'esm2_val_nored_90%_cluster90.hdf5',
    'esm2_blind_nored_90%_cluster90.hdf5',
    'esm2_blind_ligand_nored_90%_cluster90.hdf5',
    'esm2_blind_protein_nored_90%_cluster90.hdf5'
]:
    validate_first3_shapes(os.path.join("splits_esm2_output_nored_90%_cluster90", split_file))