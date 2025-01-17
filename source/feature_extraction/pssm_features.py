#pssm_features.py
#INPUT FILES: (path: "./NEW_PREDICTION/") "protein_predict.fasta" - FASTA file with protein sequences
#OUTPUT FILES: (path: "./FEATURES/PSSM/") - Creates PSSM folder with individual fasta for each chain and spotone features
#              (path: "./H5_FILES/") - "pssm.hdf5" - H5 file with spotone descriptors, key is PDB:Chain
#                                    - "pssm_keys.txt" - txt with spotone keys
#
#IMPORTANT! PSSM uses a database (SwissProt) that ocupies 206 GB

from source.feature_extraction.ViralBindPredict_variables import PREDICTION_FOLDER, FEATURES_FOLDER, SPOTONE_FOLDER, PSSM_FOLDER, BLAST_FOLDER, H5_FOLDER, write_txt, h5_keys
import os
import subprocess
import urllib.request
import tarfile
import concurrent.futures
import h5py
import pandas as pd
import numpy as np

output_file = BLAST_FOLDER + "uniref50.fasta.gz"
blast_db_dir = BLAST_FOLDER
#ftp_url = "ftp://ftp.ebi.ac.uk/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"

class PSSMFeatureExtraction():
    def __init__(self):
        if not os.path.exists(BLAST_FOLDER):
            os.makedirs(BLAST_FOLDER)
            print("- Directory Created: "+BLAST_FOLDER)

    # Function to download the database file
    def download_database(self, url, output_file):
        print(f"Downloading the database from {url}...")
        urllib.request.urlretrieve(url, output_file)
        print(f"Download completed: {output_file}")

    # Function to extract the .tar.gz file
    def extract_database(self, tar_gz_file, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Extracting {tar_gz_file} to {output_dir}...")
        with tarfile.open(tar_gz_file, "r:gz") as tar:
            tar.extractall(path=output_dir)
        print(f"Extraction completed. Database available in {output_dir}")

    def run_psiblast(self, command):
        subprocess.run(command, shell=True)

    def submit_pssm_run(self, input_file, psiblast_commands, pssm_variables = {}, remove_files = False):
        raw_base_name = input_file.split(".")[0]
        only_file_name = input_file.split("/")[-1]
        output_txt_name = raw_base_name.split("/")[-1] + ".txt"
        output_pssm_name = raw_base_name.split("/")[-1] + ".pssm"
        if os.path.exists(PSSM_FOLDER+output_pssm_name):
            return "done"
        else:
            print("Retrieving PSSM: ", only_file_name)
            """
            Initiate PSSM run
            """
            run_psiblast_command = "psiblast -query " + pssm_variables["target_folder"]+only_file_name + \
                        " -evalue " + pssm_variables["evaluation_threshold"] + \
                        " -db " + pssm_variables["file_location"]+pssm_variables["database_relative_location"] + \
                        " -outfmt " + pssm_variables["output_format"] + \
                        " -out " + PSSM_FOLDER+output_txt_name + \
                        " -out_ascii_pssm " + PSSM_FOLDER +output_pssm_name + \
                        " -num_iterations " + pssm_variables["number_of_iterations"] + \
                        " -num_threads " + pssm_variables["number_of_threads"] 
            return run_psiblast_command
   
    def process_pssm_file(self, file_name, dir = ""):
        """This functions process one single .pssm file"""

        AMINO_ACIDS_CONVERTER = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", \
                                                "C": "CYS", "B": "ASX", "E": "GLU", "Q": "GLN", \
                                                "Z": "GLX", "G": "GLY", "H": "HIS", "I": "ILE", \
                                                "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", \
                                                "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", \
                                                "Y": "TYR", "V": "VAL"}

        with open(PSSM_FOLDER + file_name) as file:
            lines = file.readlines()
            #lines = [i.decode("utf-8") for i in lines if i.strip()]
        output_table = []
        for index, row in enumerate(lines[3:-6], start=1):
            row = row.split()
            output_table.append([index]+row)
        header = ["indexed_number","original_residue_number","residue_name"] + \
                ["pssm_" + str(x) for x in range(1,21)] + \
                ["psfm_" + str(x) for x in range(1,21)] + \
                ["a", "b"]
        output_table = pd.DataFrame(output_table, columns = header)
        output_table["residue_name"].replace(AMINO_ACIDS_CONVERTER, inplace=True)
        return output_table

    def open_txt(self, txt_file):
        with open(txt_file, 'r') as file:
            content = [line.strip() for line in file]
            return content

    def execute(self, blast_database_path):

        # Download database
        #if not os.path.exists(output_file):
            #self.download_database(ftp_url, output_file)
            #self.extract_database(output_file, blast_db_dir)
            #print("- Database downloaded: "+BLAST_FOLDER+output_file)

        directory_path, filename = blast_database_path.rsplit('/', 1)
        directory_path += '/'

        pssm_variables = {"file_location": directory_path,
                        "target_folder": SPOTONE_FOLDER, #change
                        "database_relative_location": filename, 
                        "number_of_iterations": "3",
                        "output_format": "5",
                        "evaluation_threshold": "0.001",
                        "number_of_threads": "12"}

        # Run PSSM for fasta files in spotone folder
        prt_files = os.listdir(SPOTONE_FOLDER)
        psiblast_commands=[]
        for prt in prt_files:
            if prt.endswith(".fasta"):
                psiblast_command = self.submit_pssm_run(prt, psiblast_commands, pssm_variables, remove_files = True)
                if psiblast_command != "done":
                    psiblast_commands.append(psiblast_command)

        index = 0
        max_processes = 2
        while index < len(psiblast_commands):
            running_processes = []

            while len(running_processes) < max_processes and index < len(psiblast_commands):
                command = psiblast_commands[index]
                process = subprocess.Popen(command, shell=True)
                running_processes.append(process)
                index += 1
            
            for process in running_processes:
                process.wait()

        # Join PSSM in h5 file        
        h5_pssm= h5py.File(H5_FOLDER+"pssm.hdf5", "w")

        drop_columns = ["indexed_number", "original_residue_number", "residue_name"]
        files = os.listdir(PSSM_FOLDER)
        for f_name in files:
            if f_name.endswith(".pssm"):
                table = self.process_pssm_file(f_name)
                for column in table.columns:
                    if column in drop_columns:
                        table.drop(columns=[column], inplace=True)
                data = table.to_numpy(dtype=np.float32)
                print(f"Number of rows: {data.shape[0]}, Number of columns: {data.shape[1]}")
                h5_pssm.create_dataset(f_name.replace(".pssm",""), data = data)

        write_txt(H5_FOLDER+"pssm_keys.txt", h5_pssm.keys())

        print("- H5 file with PSSM descriptors saved in "+FEATURES_FOLDER+"pssm.hdf5")
        print("- PSSM keys are the PDB ID and protein chains: PDB:Chains")
        print("#######################################")
