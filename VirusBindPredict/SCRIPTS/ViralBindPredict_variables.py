import csv
import h5py
"""
Variables useful for the bulk of ProLigRes project
"""

__author__ = "C. Marques-Pereira"
__email__ = "amarques@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "ViralBindPredict: Empowering Viral Protein-Ligand Binding Sites through Deep Learning and Protein Sequence-Derived Insights"

"""
Folder locations
"""

DEFAULT_LOCATION = "/home/user/ProLigResDB/viral/GITHUB/" #"Change/to/user/location/ViralBindPredict/"
SCRIPT_FOLDER = DEFAULT_LOCATION + "SCRIPTS/"
PREDICTION_FOLDER = DEFAULT_LOCATION + "NEW_PREDICTION/"
H5_FOLDER = DEFAULT_LOCATION + "H5_FILES/"
FEATURES_FOLDER = DEFAULT_LOCATION + "FEATURES/"
MORDRED_FOLDER = FEATURES_FOLDER + "MORDRED/"
SPOTONE_FOLDER = FEATURES_FOLDER + "SPOTONE/"
PSSM_FOLDER = FEATURES_FOLDER + "PSSM/"
BLAST_FOLDER = PSSM_FOLDER + "BLAST_DB/"
RESOURCES_FOLDER = SCRIPT_FOLDER + "RESOURCES/"


# Used functions
def open_txt_file(txt_file):
    file = open(txt_file, "r")
    lines = file.readlines()
    list = []
    for line in lines:
        line = line.replace("\n", "")
        list.append(str(line))
    file.close()
    return list

def h5_keys(path):
    with h5py.File(path, "r") as f:
        data = list(f.keys())
    return data

def write_csv(file_name, info_list, header_list):
    with open(file_name, "w", newline = "") as csvfile:
        writer = csv.writer(csvfile, delimiter = ";")
        if info_list[0] not in header_list:
            writer.writerow(header_list)
        for line in info_list:
            if type(line) is not list:
                line = line.split(";")
            else:
                line = [element.replace('"', "") for element in line]
            writer.writerow(line)

def write_txt(path, keys):
    with open(path, "w") as keys_file:
        keys_file.write("\n".join(map(str, keys)))

# SPOTONE variables
HEADER_COLUMNS = ["Input ID","Chain","Residue number","Residue name"]
INTERMEDIATE_SEP = "_"
SYSTEM_SEP = "/"
SEP = ","
COMPLEX_NAME_COL = "Input ID"
CHAIN_NAME_COL = "Chain"
RES_NUMBER_COL = "Residue number"
ENCODING_FILE = RESOURCES_FOLDER + "encoding.csv"
ENCODING_RESIDUE_NAME = "res_letter"
RES_NAME_COL = "Residue name"
THREE_TO_ONE_CODE_CONVERTER = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
ONE_TO_THRE_CODE_CONVERTER = {value:key for key, value in THREE_TO_ONE_CODE_CONVERTER.items()}
AMINO_PROPERTIES_FILE = RESOURCES_FOLDER + "amino_properties.csv"
AMINO_PROPERTIES_THREE_CODE_COL = "three_code"
FEATURES_HOLDER_COLS = ['Helix_propensity', 'Sheet_propensity', 'Helix_propensity_values', 'Sheet_propensity_values', 'MW', 'pka_carboxylate', 'pka_amine', 'pka_side_chain', 'C_number', 'H_number', 'N_number', 'O_number', 'S_number', 'Standard_area_free', 'Standard_area_protein', 'folded_buried_area', 'mean_fractional_area_loss', 'residue_mass', 'monoisotopic_mass']
CENTRED_FEATURES_WINDOWS = [2, 5, 7, 10, 25, 50, 75]
SECTIONS_FEATURES_SPLITS = [100]
PROCESSED_TERMINATION = INTERMEDIATE_SEP + "processed.csv" 
FEATURES_TERMINATION = INTERMEDIATE_SEP + "features.csv"

# PSSM variables
pssm_variables = {"file_location": BLAST_FOLDER,
                  "target_folder": SPOTONE_FOLDER, #change
                  "database_relative_location": "uniref100", 
                  "number_of_iterations": "3",
                  "output_format": "5",
                  "evaluation_threshold": "0.001",
                  "number_of_threads": "8"}
