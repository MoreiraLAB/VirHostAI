#spotone_features.py
#INPUT FILES: (path: "./NEW_PREDICTION/") "protein_predict.fasta" - FASTA file with protein sequences
#OUTPUT FILES: (path: "./FEATURES/SPOTONE/") - Creates SPOTONE folder with individual fasta for each chain and spotone features
#              (path: "./H5_FILES/") - "spotone.hdf5" - H5 file with spotone descriptors, key is PDB:Chain
#                                    - "spotone_keys.txt" - txt with spotone keys

__author__ = "C. Marques-Pereira, A.T. Gaspar"
__email__ = "amarques@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "ViralBindPredict: Empowering Viral Protein-Ligand Binding Sites through Deep Learning and Protein Sequence-Derived Insights"

from ViralBindPredict_variables import PREDICTION_FOLDER, FEATURES_FOLDER, SPOTONE_FOLDER, H5_FOLDER, RESOURCES_FOLDER, write_txt, h5_keys
import pandas as pd
import numpy as np
import time
from ast import literal_eval
import os
import csv
import h5py

print("######## ViralBindPredict- SPOTONE Features ########")

# Create a new folder to save SPOTONE descriptors
if not os.path.exists(SPOTONE_FOLDER):
    os.makedirs(SPOTONE_FOLDER)
    print("- Directory Created: "+SPOTONE_FOLDER)
if not os.path.exists(H5_FOLDER):
    os.makedirs(H5_FOLDER)
    print("- Directory Created: " + H5_FOLDER)

def split_fasta(fasta_file, path_to_fasta):
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

def get_entries(input):
    proteins_fasta=open(input,"r").readlines()
    output = [element.replace(">","").replace("\n","") for element in proteins_fasta if ">" in element]
    return output

def join_spotone_results(file, list, input_names, h5_file):
	for element in list:
		dataset = file.loc[file["Input ID"] == element.split(":")[0]].loc[file["Chain"] == element.split(":")[1]]
		dataset = dataset.drop(columns="Chain")
		dataset_column= dataset.columns
		dataset = np.array(dataset.iloc[:,4:].astype("float64").round(10))
		try:
			index = input_names.index(element)
			dataset_name = input_names[index]
		except:
			dataset_name = [i for i in input_names if element in i][0]
		if dataset_name not in h5_file.keys():
			h5_file.create_dataset(dataset_name, data = dataset)
		print("- Total number of Spotone features: ", len(dataset_column))
	write_txt(H5_FOLDER+"spotone_keys.txt", h5_file.keys())
	return


split_fasta(PREDICTION_FOLDER+"protein_predict.fasta", SPOTONE_FOLDER)

if os.path.exists(H5_FOLDER + "spotone.hdf5"):
    h5_spotone = h5py.File(H5_FOLDER+"spotone.hdf5", "a")
else:
    h5_spotone= h5py.File(H5_FOLDER+"spotone.hdf5", "w")

list_of_files = [i for i in os.listdir(SPOTONE_FOLDER) if i.endswith(".fasta")]
python_raw_command = "nohup python3 -u run_spotone.py "

for file in list_of_files:
    if not os.path.exists(SPOTONE_FOLDER+file+"_features.csv"):
        current_command = python_raw_command + str(file) + " &"
        os.system(current_command)
        print(current_command)

files = os.listdir(SPOTONE_FOLDER)
names = get_entries(PREDICTION_FOLDER+"protein_predict.fasta")

for file in files:
    if "fasta_features.csv" in file:
        file = pd.read_csv(SPOTONE_FOLDER+file, sep=",")
        list = file['Input ID'] + ':' + file['Chain']
        join_spotone_results(file, list.unique(), names, h5_spotone)

print("- H5 file with SPOTONE descriptors saved in "+FEATURES_FOLDER+"spotone.hdf5")
print("- Total Chains: "+ str(len(list_of_files)))
print("- SPOTONE keys are the PDB ID and protein chains: PDB:Chains")
print("#######################################")