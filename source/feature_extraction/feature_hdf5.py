import h5py
import pandas as pd
from source.feature_extraction.ViralBindPredict_variables import PREDICTION_FOLDER, H5_FOLDER, open_txt_file
import re
import numpy as np


class NewFeatureFile():
    def __init__(self):
        pass
        
    def ligand_transform(self, mordred_file):
        # Input file path
        input_file = "example_datasets/transformations-logs/example-train.txt"

        # Read the input file and process the lines
        with open(input_file, 'r') as file:
            lines = file.readlines()

        # Initialize the transformation lists
        lig_transf_1 = []
        lig_transf_2 = []

        # Extract indices from the lines starting with "deleted - ligand features"
        for line in lines:
            if line.startswith("deleted - ligand features"):
                # Extract numbers in brackets
                indices = re.findall(r'\[(.*?)\]', line)
                if indices:
                    numbers = list(map(int, indices[0].split(',')))
                    if not lig_transf_1:
                        lig_transf_1 = sorted(numbers)  # Assign to lig_transf_1 if empty
                    else:
                        lig_transf_2 = sorted(numbers)  # Otherwise assign to lig_transf_2

        # Validate that both transformation lists were found
        if not lig_transf_1 or not lig_transf_2:
            raise ValueError("Transformation lists lig_transf_1 or lig_transf_2 could not be extracted.")

        # Open the HDF5 file for read/write operations
        with h5py.File(mordred_file, 'r+') as mordred_h5:
            for key in mordred_h5.keys():
                print(f"Processing key: {key}")

                # Step 1: Apply the first transformation (remove columns in lig_transf_1)
                dataset = mordred_h5[key][:]
                mask_1 = np.ones(dataset.shape[1], dtype=bool)  # Create a mask for lig_transf_1
                mask_1[lig_transf_1] = False  # Set indices in lig_transf_1 to False
                filtered_dataset_1 = dataset[:, mask_1]  # Apply the mask
                # Step 2: Apply the second transformation (remove columns in lig_transf_2)
                mask_2 = np.ones(filtered_dataset_1.shape[1], dtype=bool)  # Create a mask for lig_transf_2
                mask_2[lig_transf_2] = False  # Set indices in lig_transf_2 to False
                filtered_dataset_2 = filtered_dataset_1[:, mask_2]  # Apply the mask
                # Replace the dataset in the HDF5 file
                del mordred_h5[key]  # Delete the original dataset
                mordred_h5.create_dataset(key, data=filtered_dataset_2)  # Save the final dataset

        print("mordred.hdf5 data was transformed successfully.")

    def new_prediction_hdf5(self, mordred_file, spotone_file, pssm_file, smi_file, fasta_file, output_file):
        # Open the HDF5 files
        with h5py.File(mordred_file, 'r') as mordred_h5, \
            h5py.File(spotone_file, 'r') as spotone_h5, \
            h5py.File(pssm_file, 'r') as pssm_h5, \
            h5py.File(output_file, 'w') as combined_h5:

            # Create the 'ligands' group in the output file
            ligands_group = combined_h5.create_group('ligands')

            # Copy keys from the mordred file to 'ligands'
            for key in mordred_h5.keys():
                ligand_group = ligands_group.create_group(key)  # Create a group for each ligand (e.g., 'BME')
                ligand_group.create_dataset('features', data=np.nan_to_num(mordred_h5[key],nan=0))  # Save the dataset

            # Create the 'proteins' group in the output file
            proteins_group = combined_h5.create_group('proteins')
            protein_dict = {}
            for protein_key in spotone_h5.keys():
                pdb_id, chain = protein_key.split(":")
                if pdb_id not in protein_dict:
                    protein_dict[pdb_id] = []
                protein_dict[pdb_id].append(chain)

            for pdb_id, chains in protein_dict.items():
                protein_group = proteins_group.create_group(pdb_id)
                for chain_key in chains:
                    # Create a group for each chain (e.g., 'A')
                    chain_group = protein_group.create_group(chain_key)

                    # Load tables from spotone and pssm
                    spotone_data = pd.DataFrame(spotone_h5[f"{pdb_id}:{chain_key}"])
                    pssm_data = pd.DataFrame(pssm_h5[f"{pdb_id}:{chain_key}"])

                    # Concatenate the tables and add a column of zeros
                    zeros_column = np.zeros((spotone_data.shape[0], 1))  # Create a column of zeros
                    concatenated_data = pd.concat([spotone_data, pssm_data, pd.DataFrame(zeros_column, columns=['zeros'])], axis=1)
                    # Save the concatenated table in the new file
                    chain_group.create_dataset('features', data=concatenated_data.to_numpy())

            # Create the 'interactions' group in the output file
            interactions_group = combined_h5.create_group('interactions')

            # Load ligand and protein interaction data
            with open(smi_file, 'r') as smi_f, open(fasta_file, 'r') as fasta_f:
                ligands = [line.split()[-1].strip() for line in smi_f.readlines() if line.strip()]
                proteins = [line.strip() for line in fasta_f.readlines() if line.startswith('>') and line.strip()]

            # Ensure the number of ligands matches the number of proteins
            if len(ligands) != len(proteins):
                raise ValueError("Number of ligands and proteins do not match")

            # Create interaction groups
            for i, (ligand, protein) in enumerate(zip(ligands, proteins)):
                protein = protein.replace(">", "")
                pdb_id, chain = protein_key.split(":")
                interaction_name = f"{pdb_id}:{chain}:{ligand}"
                interaction_group = interactions_group.create_group(interaction_name)
                # Add attributes to the interaction group
                interaction_group.attrs['protein'] = f"proteins/{pdb_id}/{chain}"
                interaction_group.attrs['ligand'] = f"ligands/{ligand}"
                interaction_group.attrs['split'] = 'test'  # Add 'test' attribute
                # Get the number of amino acids from spotone_data or pssm_data
                spotone_data = pd.DataFrame(spotone_h5[protein])
                num_amino_acids = spotone_data.shape[0]  # Number of rows corresponds to the number of amino acids
                
                # Create a dataset of zeros with a single column and rows equal to the number of amino acids
                interaction_group.create_dataset('targets', data=[[0]] * num_amino_acids)

            print("HDF5 file with interactions created successfully.")
          
        
    
    def execute(self):
        # Open existing HDF5 files
        mordred_file = H5_FOLDER+"mordred.hdf5"
        spotone_file = H5_FOLDER+"spotone.hdf5"
        pssm_file = H5_FOLDER+"pssm.hdf5"
        output_file = PREDICTION_FOLDER+"new_prediction.hdf5"
        smi_file = PREDICTION_FOLDER +"ligand_predict.smi"
        fasta_file = PREDICTION_FOLDER+"protein_predict.fasta"
        print("Initiating Ligand data transformation...")
        self.ligand_transform(mordred_file)
        self.new_prediction_hdf5(mordred_file, spotone_file, pssm_file, smi_file, fasta_file, output_file)