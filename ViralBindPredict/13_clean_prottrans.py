import os
import h5py
import numpy as np
import pandas as pd
import shutil

# Check Nan, inf and 0 variance
def get_valid_feature_indices(feature_list):
    stacked = np.vstack(feature_list)
    mask_nan = np.any(np.isnan(stacked), axis=0)
    mask_inf = np.any(np.isinf(stacked), axis=0)
    mask_zero_var = np.all(stacked == stacked[0, :], axis=0)
    invalid_mask = mask_nan | mask_inf | mask_zero_var
    valid_indices = np.where(~invalid_mask)[0]
    return valid_indices

def analyze_interactions(file_path, results_dir="Results"):
    os.makedirs(results_dir, exist_ok=True)
    counts_interacting, counts_non_interacting = {}, {}
    unique_prot_c, unique_prot, unique_ligands, unique_ligands_no_chain, unique_prot_c_lig = set(), set(), set(), set(), set()

    with h5py.File(file_path, 'r') as hdf:
        if 'interactions' in hdf:
            for prot_chain_lig in hdf['interactions']:
                subgroup = hdf['interactions'][prot_chain_lig]
                unique_prot_c_lig.add(prot_chain_lig)
                prot_c_id = ":".join(prot_chain_lig.split(':')[:2])
                unique_prot_c.add(prot_c_id)
                prot_id = prot_chain_lig.split(':')[0]
                unique_prot.add(prot_id)
                ligand = prot_chain_lig.split(':')[-1]
                unique_ligands.add(ligand)
                ligand_no_chain = ligand.split('_')[0]  
                unique_ligands_no_chain.add(ligand_no_chain)

                if 'targets' in subgroup:
                    data_array = subgroup['targets'][:]
                    counts_interacting[prot_chain_lig] = (data_array[:, 0] == 1).sum()
                    counts_non_interacting[prot_chain_lig] = (data_array[:, 0] == 0).sum()

    counts_df_interacting = pd.DataFrame(list(counts_interacting.items()), columns=['Interaction Group', 'Interacting Residues'])
    counts_df_non_interacting = pd.DataFrame(list(counts_non_interacting.items()), columns=['Interaction Group', 'Non-Interacting Residues'])

    with open(os.path.join(results_dir, 'unique_prot_c-prottrans.txt'), 'w') as f:
        f.write("\n".join(sorted(unique_prot_c)))
    with open(os.path.join(results_dir, 'unique_prot-prottrans.txt'), 'w') as f:
        f.write("\n".join(sorted(unique_prot)))
    with open(os.path.join(results_dir, 'unique_ligands-prottrans.txt'), 'w') as f:
        f.write("\n".join(sorted(unique_ligands)))
    with open(os.path.join(results_dir, 'unique_prot_c_lig-prottrans.txt'), 'w') as f:
        f.write("\n".join(sorted(unique_prot_c_lig)))
    with open(os.path.join(results_dir, 'unique_ligands_no_chain_prottrans.txt'), 'w') as f:
        f.write("\n".join(sorted(unique_ligands_no_chain)))
        
    counts_df_interacting.to_csv(os.path.join(results_dir, 'interacting_residues_counts-prottrans.csv'), index=False)
    counts_df_non_interacting.to_csv(os.path.join(results_dir, 'non_interacting_residues_counts-prottrans.csv'), index=False)

    print(counts_df_interacting)
    print(f"Total number of interacting residues: {counts_df_interacting['Interacting Residues'].sum()}")
    print(counts_df_non_interacting)
    print(f"Total number of non-interacting residues: {counts_df_non_interacting['Non-Interacting Residues'].sum()}")
    print(f"Total unique PROT:C chains: {len(unique_prot_c)}")
    print(f"Total unique PROT: {len(unique_prot)}")
    print(f"Total unique ligands (considering ligand chain id): {len(unique_ligands)}")
    print(f"Total unique ligands (not considering ligand chain id): {len(unique_ligands_no_chain)}")
    print(f"Total unique PROT:C:LIG combinations: {len(unique_prot_c_lig)}")

    #with h5py.File(file_path, 'r') as hdf:
    #    print("\nLigand Features:")
    #    for ligand in hdf.get('ligands', {}):
    #        features = hdf[f'ligands/{ligand}/features'][:]
    #        print(f"Ligand: {ligand}, Shape: {features.shape}")
    #        print(features)

     #   print("\nProtein Features:")
      #  for protein in hdf.get('proteins', {}):
       #     for chain in hdf[f'proteins/{protein}']:
        #        features = hdf[f'proteins/{protein}/{chain}/features'][:]
         #       print(f"Protein: {protein}, Chain: {chain}, Shape: {features.shape}")
          #      print(features)

def verify_cleaned_file(file_path):
    print("\n Verifying cleaned HDF5 file for NaN, Inf, and zero-variance columns...")

    def verify_features(feature_arrays, label):
        all_data = np.vstack(feature_arrays)
        has_nan = np.isnan(all_data).any()
        has_inf = np.isinf(all_data).any()
        zero_var = np.all(all_data == all_data[0, :], axis=0).any()
        print(f"{label} - Has NaN: {has_nan}, Has Inf: {has_inf}, Any zero-variance columns: {zero_var}")

    ligand_features = []
    protein_features = []

    with h5py.File(file_path, 'r') as hdf:
        for ligand in hdf.get('ligands', {}):
            features = hdf[f'ligands/{ligand}/features'][:]
            ligand_features.append(features)

        for protein in hdf.get('proteins', {}):
            for chain in hdf[f'proteins/{protein}']:
                features = hdf[f'proteins/{protein}/{chain}/features'][:]
                protein_features.append(features)

    verify_features(ligand_features, "Ligand features")
    verify_features(protein_features, "Protein features")
    print(" Verification complete.\n")

def copy_cleaned_features(original_path, new_cleaned_path):
    with h5py.File(original_path, "r") as orig_f, h5py.File(new_cleaned_path, "w") as new_f:
        ligand_feats = [orig_f[f"ligands/{lig}/features"][:] for lig in orig_f["ligands"]]
        ligand_valid_cols = get_valid_feature_indices(ligand_feats)
        new_f.attrs["ligand_valid_cols"] = ligand_valid_cols
        lig_group = new_f.create_group("ligands")
        for lig in orig_f["ligands"]:
            grp = lig_group.create_group(lig)
            cleaned = orig_f[f"ligands/{lig}/features"][:, ligand_valid_cols]
            grp.create_dataset("features", data=cleaned.astype(np.float32))
            
        # Update mordred_column_names dataset
        if "mordred_column_names" in orig_f:
            # Read original column names
            all_columns = [c.decode("utf-8") for c in orig_f["mordred_column_names"][:]]
            filtered_columns = [all_columns[i] for i in ligand_valid_cols]
            # Save filtered column names
            col_bytes = np.array([c.encode("utf-8") for c in filtered_columns], dtype="S")
            new_f.create_dataset("mordred_column_names", data=col_bytes)
            print(f" Updated mordred_column_names with {len(filtered_columns)} columns.")

        protein_feats = []
        for prot in orig_f["proteins"]:
            for chain in orig_f[f"proteins/{prot}"]:
                feat = orig_f[f"proteins/{prot}/{chain}/features"][:]
                protein_feats.append(feat)
        protein_valid_cols = get_valid_feature_indices(protein_feats)
        new_f.attrs["protein_valid_cols"] = protein_valid_cols
        prot_group = new_f.create_group("proteins")
        for prot in orig_f["proteins"]:
            prot_subgroup = prot_group.create_group(prot)
            for chain in orig_f[f"proteins/{prot}"]:
                cleaned = orig_f[f"proteins/{prot}/{chain}/features"][:, protein_valid_cols]
                prot_subgroup.create_group(chain).create_dataset("features", data=cleaned.astype(np.float32))

        # Copy interactions group as-is
        orig_f.copy("interactions", new_f)

# Run cleaning and analysis
original_path = "interaction_dataset_prottrans_mordred_deduplicated.hdf5"
cleaned_path = "viralbindpredictDB-prottrans-cleaned.hdf5"

copy_cleaned_features(original_path, cleaned_path) 
analyze_interactions(cleaned_path)
verify_cleaned_file(cleaned_path)