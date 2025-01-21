__author__ = "C. Marques-Pereira"
__email__ = "amarques@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__project__ = "ViralBindPredict: Empowering Viral Protein-Ligand Binding Sites through Deep Learning and Protein Sequence-Derived Insights"

import source.transformations as transformations
import multilayer_perceptron as mlp
import h5py
import csv
import os 

class AE_MLP():
    def __init__(self, mlp_model_path, cuda_device_idx='0'):
        #self.device = torch.device(f'cuda:{cuda_device_idx}' if torch.cuda.is_available() else 'cpu')
        self.device = "cpu"
        self.mlp_model_path = mlp_model_path
        #self.mlp = torch.load(mlp_model_path, pickle_module=dill, map_location=self.device)

    def autoencoder(self,hdf5_file_path, autoencoder_model):
        transformed_file = hdf5_file_path.split(".")[0]+"_transformed.hdf5"
        transformations.reduce_dimensionality_ae_proteins(hdf5_file_path,transformed_file,autoencoder_model)

    def predict(self, x_file_path):
        from argparse import Namespace
        if self.device == "cpu":
            args = Namespace(
                dataset=[x_file_path,"interactions","chain"],
                model=self.mlp_model_path,
                wandb="offline",
                device=self.device,
                config=["models/config-files/config-mlp-example-predict.yaml","1"]
            )
        else:            
            args = Namespace(
                dataset=[x_file_path,"interactions","chain"],
                model=self.mlp_model_path,
                wandb="offline",
                device="cuda:"+self.device,
                config=["models/config-files/config-mlp-example-predict.yaml","1"]
            )
        mlp.main(args)
    
    # Function to read a FASTA file without BioPython
    def read_fasta(self, file_path):
        sequences = []
        with open(file_path, 'r') as file:
            sequence = ""
            for line in file:
                if line.startswith('>'):
                    if sequence:
                        sequences.append(sequence)
                        sequence = ""
                else:
                    sequence += line.strip()
            if sequence:
                sequences.append(sequence)
        return sequences

    # Main function to process the interactions and generate CSVs
    def process_interactions(self, h5_file, fasta_file, predictions_file):
        # Open the HDF5 file and extract interaction keys
        with h5py.File(h5_file, 'r') as h5:
            interaction_keys = list(h5['interactions'].keys())

        # Read the sequences from the FASTA file
        sequences = self.read_fasta(fasta_file)

        # Read the predictions from the CSV file
        with open(predictions_file, 'r') as file:
            reader = csv.DictReader(file)
            predictions = list(reader)

        # Iterate over each interaction
        for idx, interaction_key in enumerate(interaction_keys):
            # Get the corresponding sequence and prediction row
            sequence = sequences[idx]
            prediction_row = predictions[idx]

            # Extract probabilities and predictions as lists
            #probabilities = eval(prediction_row['probabilities'])
            predictions_list = eval(prediction_row['predictions'])
            path = os.path.dirname(h5_file)
            # Create the output CSV for the interaction
            output_file = f"{path}/{interaction_key}.csv"
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write the header
                writer.writerow(["Amino Acid", "Prediction"])
                # Write rows with sequence, probabilities, and predictions
                for aa, pred in zip(sequence, predictions_list):
                    writer.writerow([aa, pred[0]])

            print(f"\n\nCreated file with predictions: {output_file}")


    def execute(self, hdf5_file_path, autoencoder_model, protein_file_path):
        self.autoencoder(hdf5_file_path, autoencoder_model)
        self.predict(hdf5_file_path.split(".")[0]+"_transformed.hdf5")
        self.process_interactions(hdf5_file_path, protein_file_path,"source/predictions.csv")
