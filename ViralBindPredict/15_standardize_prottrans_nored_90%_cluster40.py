import os
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
import joblib

random.seed(42)
np.random.seed(42)

def get_feature_arrays(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        # Get all protein features
        prot_feats = []
        for prot in f['proteins']:
            for chain in f[f'proteins/{prot}']:
                data = f[f'proteins/{prot}/{chain}/features'][()]
                prot_feats.append(data)

        # Get all ligand features
        lig_feats = []
        for lig in f['ligands']:
            data = f[f'ligands/{lig}/features'][()]
            lig_feats.append(data)

    return np.vstack(prot_feats), np.vstack(lig_feats)

def apply_and_save_standardized(input_path, output_path, prot_scaler, lig_scaler, label):
    with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        if "mordred_column_names" in f_in:
            f_out.create_dataset("mordred_column_names", data=f_in["mordred_column_names"][:])
        f_out.create_group('interactions')
        f_out.create_group('proteins')
        f_out.create_group('ligands')

        # Copy interactions
        for i in f_in['interactions']:
            int_grp = f_out['interactions'].create_group(i)
            int_grp.attrs['ligand'] = f_in['interactions'][i].attrs['ligand']
            int_grp.attrs['protein'] = f_in['interactions'][i].attrs['protein']
            int_grp.attrs['split'] = label
            int_grp.create_dataset('targets', data=f_in['interactions'][i]['targets'][()])

        # Standardize and save proteins
        for prot in f_in['proteins']:
            f_out['proteins'].create_group(prot)
            for chain in f_in[f'proteins/{prot}']:
                data = f_in[f'proteins/{prot}/{chain}/features'][()]
                standardized = prot_scaler.transform(data)
                f_out[f'proteins/{prot}'].create_group(chain)
                f_out[f'proteins/{prot}/{chain}'].create_dataset('features', data=standardized)

        # Standardize and save ligands
        for lig in f_in['ligands']:
            data = f_in[f'ligands/{lig}/features'][()]
            standardized = lig_scaler.transform(data)
            f_out['ligands'].create_group(lig)
            f_out[f'ligands/{lig}'].create_dataset('features', data=standardized)

def validate_first3_examples(hdf5_path):
    print(f"\n Validation: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        print("Proteins:")
        printed = 0
        for prot in f['proteins']:
            for chain in f[f'proteins/{prot}']:
                data = f[f'proteins/{prot}/{chain}/features'][()]
                print(f"{prot}:{chain} shape={data.shape}, mean={np.mean(data):.3f}, std={np.std(data):.3f}")
                printed += 1
                if printed >= 3:
                    break
            if printed >= 3:
                break

        print("Ligands:")
        printed = 0
        for lig in f['ligands']:
            data = f[f'ligands/{lig}/features'][()]
            print(f"{lig} shape={data.shape}, mean={np.mean(data):.3f}, std={np.std(data):.3f}")
            printed += 1
            if printed >= 3:
                break

input_dir = "splits_prottrans_output_nored_90%_cluster40"
output_dir = "splits_prottrans_output_standardized_nored_90%_cluster40"
os.makedirs(output_dir, exist_ok=True)

splits = {
    "train": "prottrans_train_nored_90%_cluster40.hdf5",
    "test": "prottrans_test_nored_90%_cluster40.hdf5",
    "val": "prottrans_val_nored_90%_cluster40.hdf5",
    "blind": "prottrans_blind_nored_90%_cluster40.hdf5",
    "blind_ligand": "prottrans_blind_ligand_nored_90%_cluster40.hdf5",
    "blind_protein": "prottrans_blind_protein_nored_90%_cluster40.hdf5",
}

# Fit scalers on training set
train_path = os.path.join(input_dir, splits["train"])
prot_data, lig_data = get_feature_arrays(train_path)
prot_scaler = StandardScaler().fit(prot_data)
lig_scaler = StandardScaler().fit(lig_data)

# Standardize all splits and save
for label, fname in splits.items():
    input_path = os.path.join(input_dir, fname)
    output_path = os.path.join(output_dir, fname.replace(".hdf5", "_standardized.hdf5"))
    apply_and_save_standardized(input_path, output_path, prot_scaler, lig_scaler, label)
    validate_first3_examples(output_path)

# Save scalers
joblib.dump(prot_scaler, os.path.join(output_dir, "prot_scaler_prottrans_90%_cluster40.pkl"))
joblib.dump(lig_scaler, os.path.join(output_dir, "lig_scaler_prottrans_90%_cluster40.pkl"))

print("\n All splits standardized and saved.")

def validate_first5_shapes(hdf5_path):
    print(f"\nValidating first 5 protein and ligand feature shapes in {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        # Proteins
        prot_printed = 0
        for prot in f['proteins']:
            for chain in f[f'proteins/{prot}']:
                data = f[f'proteins/{prot}/{chain}/features'][()]
                print(f"Protein {prot}:{chain} shape: {data.shape}")
                print(f"Values (last 10 features):\n{data[0, :10] if data.ndim == 2 else data[:10]}")
                prot_printed += 1
                if prot_printed >= 5:
                    break
            if prot_printed >= 5:
                break

        # Ligands
        lig_printed = 0
        for lig in f['ligands']:
            data = f[f'ligands/{lig}/features'][()]
            print(f"Ligand {lig} shape: {data.shape}")
            print(f"Values (last 10 features):\n{data[0, :10] if data.ndim == 2 else data[:10]}")
            lig_printed += 1
            if lig_printed >= 5:
                break

for split_file in [
    'prottrans_train_nored_90%_cluster40_standardized.hdf5',
    'prottrans_test_nored_90%_cluster40_standardized.hdf5',
    'prottrans_val_nored_90%_cluster40_standardized.hdf5',
    'prottrans_blind_nored_90%_cluster40_standardized.hdf5',
    'prottrans_blind_ligand_nored_90%_cluster40_standardized.hdf5',
    'prottrans_blind_protein_nored_90%_cluster40_standardized.hdf5'
]:
    validate_first5_shapes(os.path.join("splits_prottrans_output_standardized_nored_90%_cluster40", split_file))