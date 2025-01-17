#spotone_features.py
#INPUT FILES: (path: "./NEW_PREDICTION/") "protein_predict.fasta" - FASTA file with protein sequences
#OUTPUT FILES: (path: "./FEATURES/SPOTONE/") - Creates SPOTONE folder with individual fasta for each chain and spotone features
#              (path: "./H5_FILES/") - "spotone.hdf5" - H5 file with spotone descriptors, key is PDB:Chain
#                                    - "spotone_keys.txt" - txt with spotone keys

from source.feature_extraction.ViralBindPredict_variables import PREDICTION_FOLDER, FEATURES_FOLDER, SPOTONE_FOLDER, H5_FOLDER, RESOURCES_FOLDER, write_txt, h5_keys
import pandas as pd
import numpy as np
import time
from ast import literal_eval
import os
import csv
import h5py
from source.feature_extraction.run_spotone import SpotOne

class SpotOneFeatureExtractor():
    def __init__(self):
        # Create a new folder to save SPOTONE descriptors
        if not os.path.exists(SPOTONE_FOLDER):
            os.makedirs(SPOTONE_FOLDER)
            print("- Directory Created: "+SPOTONE_FOLDER)

    def split_fasta(self, fasta_file, path_to_fasta):
        with open(fasta_file, "r") as file:
            fasta_content = file.read().splitlines()
        protein_list = []
        for l in range(len(fasta_content)):
            line = fasta_content[l].strip()
            if line.startswith(">"):
                    sequence_name = line[1:] # Remove ">" from the header
                    if sequence_name not in protein_list:
                        protein_list.append(sequence_name)
                        output_file_path = os.path.join(path_to_fasta, f"{sequence_name}.fasta")
                        with open(output_file_path, "w") as output:
                            output.write(f">{sequence_name}\n")
                            output.write(fasta_content[l+1] + "\n")     

    def get_entries(self, input):
        proteins_fasta=open(input,"r").readlines()
        output = [element.replace(">","").replace("\n","") for element in proteins_fasta if ">" in element]
        return output

    def join_spotone_results(self, file, list, input_names, h5_file):
        spotone_drop = ["Input ID","Chain","Residue number","Residue name"]
        for element in list:
            dataset = file.loc[file["Input ID"] == element.split(":")[0]].loc[file["Chain"] == element.split(":")[1]]
            for column in dataset.columns:
                    if column in spotone_drop:
                        dataset.drop(columns=[column], inplace=True)
            dataset_column= dataset.columns
            dataset = np.array(dataset.astype("float32").round(10))
            try:
                index = input_names.index(element)
                dataset_name = input_names[index]
            except:
                dataset_name = [i for i in input_names if element in i][0]
            #print(dataset)
            if dataset_name not in h5_file.keys():
                h5_file.create_dataset(dataset_name, data = dataset)
            print("- Total number of Spotone features: ", len(dataset_column))
        write_txt(H5_FOLDER+"spotone_keys.txt", h5_file.keys())
        return

    def execute(self, protein_input_path):
        self.split_fasta(protein_input_path, SPOTONE_FOLDER)

        h5_spotone= h5py.File(H5_FOLDER+"spotone.hdf5", "w")

        list_of_files = [i for i in os.listdir(SPOTONE_FOLDER) if i.endswith(".fasta")]
        #python_raw_command = "nohup python3 -u run_spotone.py "

        spot_one_tools = SpotOne()

        i = 1
        number_of_files = len(list_of_files)
        for file in list_of_files:
            if not os.path.exists(SPOTONE_FOLDER+file+"_features.csv"):
                print(f"Processing protein chain {file} ({i} out of {number_of_files})")
                spot_one_tools.execute(file)
            else:
                print(f"Protein chain {file} not found")
            i += 1

        files = os.listdir(SPOTONE_FOLDER)
        names = self.get_entries(PREDICTION_FOLDER+"protein_predict.fasta")

        for file in files:
            if "fasta_features.csv" in file:
                file = pd.read_csv(SPOTONE_FOLDER+file, sep=",")
                list = file['Input ID'] + ':' + file['Chain']
                self.join_spotone_results(file, list.unique(), names, h5_spotone)

        print("- H5 file with SPOTONE descriptors saved in "+FEATURES_FOLDER+"spotone.hdf5")
        print("- Total Chains: "+ str(len(list_of_files)))
        print("- SPOTONE keys are the PDB ID and protein chains: PDB:Chains")
        print("#######################################")
