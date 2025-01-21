__author__ = "C. Marques-Pereira"
__email__ = "amarques@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__project__ = "ViralBindPredict: Empowering Viral Protein-Ligand Binding Sites through Deep Learning and Protein Sequence-Derived Insights"

# mordred_features.py
# INPUT FILES: (path: "./NEW_PREDICTIONS/") "ligand_predict.smi"
# OUTPUT FILES: (path: "./FEATURES/") - Folder with descriptors and feature information
#                                      - "mordred_descriptors.txt" - List of mordred descriptors
#                                      - "mordred_error.csv" - IDs without mordred features due to error
#              (path: "./H5_FILES/")- "mordred.hdf5" - H5 file with Mordred descriptors, keys correspond to PDB ligand ID
#                                   - "mordred_keys.txt" - txt with mordred keys
#              (path: "./FEATURES/MORDRED/") - Folder with Mordred descriptors for all ligand IDs, in .csv

from source.feature_extraction.ViralBindPredict_variables import PREDICTION_FOLDER, MORDRED_FOLDER, FEATURES_FOLDER, H5_FOLDER, write_txt, h5_keys, remove_descriptors
import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
import csv
import os
import h5py
import warnings


class MordredFeaturesExtractor():
    def __init__(self):
        warnings.filterwarnings("ignore", category=FutureWarning)

        #print("######## ViralBindPredict- Mordred Features ########")
        # Create a new folder to save Mordred descriptors
        if not os.path.exists(MORDRED_FOLDER):
            os.makedirs(MORDRED_FOLDER)
            print("- Directory Created: " + MORDRED_FOLDER)
        if not os.path.exists(H5_FOLDER):
            os.makedirs(H5_FOLDER)
            print("- Directory Created: " + H5_FOLDER)

    def execute(self, ligand_input_path):
        # ligand ids and smile list from clean dictionary with ligands of interest
        ligands = pd.read_csv(ligand_input_path, sep="\t", names=["smile", "ID"])
        unique_ligands = ligands.drop_duplicates()

        # extract Mordred features for each ligand id
        self.drug_feature_extraction(list(unique_ligands["ID"]), list(unique_ligands["smile"]), MORDRED_FOLDER)

        print("- H5 file with Mordred descriptors saved in " + FEATURES_FOLDER + "mordred.hdf5")
        print("- Total Ligands: " + str(len(unique_ligands["ID"])))
        print("- Mordred keys are the ligand PDB ID")
        print("#######################################")

    # Function to calculate Mordred descriptors and save them in MORDRED folder
    # Descriptors are also saved in an H5 file ("mordred.h5py") in FEATURES folder
    def drug_feature_extraction(self, input_ids, input_smiles, path_to_ligands_features):
        # Open and read the file
        with open('ViralBindDB/mordred_descriptors.txt', 'r') as file:
            descriptor_file = file.readlines()

        # Remove trailing newline characters
        descriptors_list = [line.strip() for line in descriptor_file]

        calc = Calculator(descriptors, ignore_3D=True)
        
        mordred_error = open(H5_FOLDER + "mordred_error.csv", "w")
        mordred_error = csv.writer(mordred_error, delimiter=";")
        keys_written = False
        counter = 0

        if not os.path.exists(str(path_to_ligands_features + input_ids[counter] + '.csv')):
            for smile in input_smiles:
                mol = Chem.MolFromSmiles(smile)

                if mol is None:
                    mordred_error.writerow([input_ids[counter], smile])
                else:
                    feature = calc(mol)
                    feature_values = [str(value) if isinstance(value, (str, object)) else value for value in feature.values()]
                    output_file = open(str(path_to_ligands_features + input_ids[counter] + '.csv'), "w")
                    writer = csv.writer(output_file, delimiter=";")
                    writer.writerow(feature.keys())
                    writer.writerow(feature.values())
                    output_file.close()
                    counter +=1
                    

        h5_file = h5py.File(H5_FOLDER + "mordred.hdf5", "w")

        dataset = None
        for lig in os.listdir(MORDRED_FOLDER):
            if lig.split(".")[0] not in h5_file.keys():
                dataset = pd.read_csv(MORDRED_FOLDER+lig, sep=";")
                dataset = dataset.apply(pd.to_numeric, errors='coerce')
                for column in dataset.columns:
                    if column not in descriptors_list:
                        dataset.drop(columns=[column], inplace=True)
                dataset = dataset.to_numpy(dtype=np.float32)
                h5_file.create_dataset(lig.split(".")[0], data = dataset)
                print(f"- Total number of Mordred features in {lig}: ", dataset.shape[1])
        write_txt(H5_FOLDER + "mordred_keys.txt", list(h5_file.keys()))
