__author__ = "C. Marques-Pereira"
__email__ = "amarques@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__project__ = "ViralBindPredict: Empowering Viral Protein-Ligand Binding Sites through Deep Learning and Protein Sequence-Derived Insights"

import sys
import argparse
import os
import traceback
from source.feature_extraction.mordred_features import MordredFeaturesExtractor
from source.feature_extraction.spotone_features import SpotOneFeatureExtractor
from source.feature_extraction.pssm_features import PSSMFeatureExtraction
from source.feature_extraction.feature_hdf5 import NewFeatureFile
import time

parser = argparse.ArgumentParser(description='Parameters for Feature Extraction.')

parser.add_argument('--ligand_input_path', type=str, help='Path to the .smi file with the ligands\' smile', default="prediction/new_prediction/ligand_predict.smi")
parser.add_argument('--protein_input_path', type=str, help='Path to the .fasta file with the proteins\' sequence', default="prediction/new_prediction/protein_predict.fasta")
parser.add_argument('--blast_database_path', type=str, help='Path to the .fasta file with the blast database (uniref100)')

try:
    args = parser.parse_args()

    if not os.path.exists(args.ligand_input_path):
        raise Exception("Ligand input file not found.")
    else: 
        ligand_input_path = args.ligand_input_path

    if not os.path.exists(args.protein_input_path):
        raise Exception("Protein input file not found.")
    else: 
        protein_input_path = args.protein_input_path

    #if not os.path.exists(args.blast_database_path):
    #    raise Exception("Blast databaset input file not found.")
    #else: 
    #    blast_database_path = args.blast_database_path
    blast_database_path = args.blast_database_path

except Exception as e:
    print(f"Failed to parse arguments: {e}")
    sys.exit(1)

def format_time(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

print("\n\n===============================================")
print(f"Using ligand input file from: {ligand_input_path}")
print(f"Using protein input file from: {protein_input_path}")
print("===============================================\n\n")

start_time = time.time()

try:
    print("===============================================")
    print("Starting Mordred feature extraction...")
    print("===============================================\n\n")
    mordred_extractor = MordredFeaturesExtractor()

    mordred_extractor.execute(ligand_input_path)

    print("\n\n===============================================")
    print("Mordred feature extraction completed!")
    print("===============================================\n\n")
except Exception as e:
    print(f"Failed to extract Mordred features: {e}")
    traceback.print_exc()
    print("\n\n===============================================")
    print("Mordred feature extraction failed.")
    print("===============================================\n\n")
    sys.exit(1)

try:
    print("===============================================")
    print("Starting SpotOne feature extraction...")
    print("===============================================\n\n")
    spotone_extractor = SpotOneFeatureExtractor()

    spotone_extractor.execute(protein_input_path)

    print("\n\n===============================================")
    print("SpotOne feature extraction completed!")
    print("===============================================\n\n")
except Exception as e:
    print(f"Failed to extract SpotOne features: {e}")
    traceback.print_exc()
    print("\n\n===============================================")
    print("SpotOne feature extraction failed.")
    print("===============================================\n\n")
    sys.exit(1)

try:
    print("===============================================")
    print("Starting PSSM feature extraction...")
    print("===============================================\n\n")
    pssm_extractor = PSSMFeatureExtraction()

    pssm_extractor.execute(blast_database_path)

    print("\n\n===============================================")
    print("PSSM feature extraction completed!")
    print("===============================================\n\n")
except Exception as e:
    print(f"Failed to extract PSSM features: {e}")
    traceback.print_exc()
    print("\n\n===============================================")
    print("PSSM feature extraction failed.")
    print("===============================================\n\n")
    sys.exit(1)

try:
    print("===============================================")
    print("Creating Feature file for new prediction...")
    print("===============================================\n\n")
    new_feature_file = NewFeatureFile()

    new_feature_file.execute()

    print("\n\n===============================================")
    print("New Feature file created.")
    print("===============================================\n\n")
except Exception as e:
    print(f"Failed to create new feature file: {e}")
    traceback.print_exc()
    print("\n\n===============================================")
    print("New Feature file failed.")
    print("===============================================\n\n")
    sys.exit(1)

elapsed_time = time.time() - start_time
print(f"Total execution time: {format_time(elapsed_time)}")
