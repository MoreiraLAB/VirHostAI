__author__ = "C. Marques-Pereira"
__email__ = "amarques@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__project__ = "ProLigResDB: A Comprehensive Repository of Experimental Protein Residue-Ligand Interactions from Protein Data Bank"

import sys
import argparse
import os
import traceback
from source.autoencoder_mlp import AE_MLP

parser = argparse.ArgumentParser(description='Parameters for New Prediction.')

parser.add_argument('--hdf5_file_path', type=str, help='Path to the .hdf5 file with ligand and protein features', default="prediction/new_prediction/new_prediction.hdf5")
parser.add_argument('--protein_input_path', type=str, help='Path to the .fasta file with the proteins\' sequence', default="prediction/new_prediction/protein_predict.fasta")
parser.add_argument('--autoencoder_model_path', type=str, help='Path to the autoencoder model', default="models/ae-fanciful-sweep-308.pt")
parser.add_argument('--mlp_model_path', type=str, help='Path to the MLP model', default="models/mlp-silvery-sweep-16.pt")
parser.add_argument('--cuda_device_idx', type=str, help='Cuda device index if available', default="0")

try:
    args = parser.parse_args()

    if not os.path.exists(args.hdf5_file_path):
        raise Exception("hdf5 input file not found.")
    else: 
        hdf5_file_path = args.hdf5_file_path

    if not os.path.exists(args.autoencoder_model_path):
        raise Exception("autoencoder model path not found.")
    else: 
        autoencoder_model_path = args.autoencoder_model_path

    if not os.path.exists(args.mlp_model_path):
        raise Exception("MLP model path not found.")
    else: 
        mlp_model_path = args.mlp_model_path
    
    if not os.path.exists(args.protein_input_path):
        raise Exception("Protein input path not found.")
    else: 
        protein_input_path = args.protein_input_path

    cuda_device_idx = args.cuda_device_idx

except Exception as e:
    print(f"Failed to parse arguments: {e}")
    sys.exit(1)

print("\n\n===============================================")
print(f"Using hdf5 input file from: {hdf5_file_path}")
print("===============================================\n\n")

try:
    print("===============================================")
    print("Starting MLP Prediction...")
    print("===============================================\n\n")
    
    mlp_prediction = AE_MLP(mlp_model_path, cuda_device_idx)
    mlp_prediction.execute(hdf5_file_path,autoencoder_model_path,protein_input_path)

    print("\n\n===============================================")
    print("MLP Prediction completed!")
    print("===============================================\n\n")
except Exception as e:
    print(f"Failed to complete MLP Prediction: {e}")
    traceback.print_exc()
    print("\n\n===============================================")
    print("MLP Prediction failed.")
    print("===============================================\n\n")
    sys.exit(1)
