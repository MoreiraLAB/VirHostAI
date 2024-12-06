# ViralBindPredict

## Abstract
The development of a single drug can cost up to 1.8 billion USD and require over a decade, timelines that contemporary pharmaceutical companies seek to expedite. Computational methodologies have become integral to drug discovery; however, traditional approaches, such as docking simulations, often rely on protein and ligand structures that are unavailable, incomplete, or lack sufficient accuracy. Notwithstanding advances such as AlphaFold in predicting protein structures, these models are not always sufficiently precise for identifying Ligand-Binding Sites (LBS) or Drug-Target Interactions (DTI). In this study, we introduce ViralBindPredict, an innovative Deep-Learning (DL) model designed to predict LBS in viral proteins using sequence-based data. Our approach leverages sequence-derived information from the Protein Data Bank (PDB), offering a faster and more accessible alternative to structure-based methods. ViralBindPredict classifies viral protein residues as interacting or non-interacting with ligands based on a 5 Å threshold, unlocking new avenues for antiviral drug discovery. To enhance performance, we extracted advanced descriptors from protein-ligand complexes and applied autoencoders for dimensionality reduction of protein features. ViralBindPredict was rigorously evaluated across key metrics, achieving an accuracy of 0.68, AUC-ROC of 0.74, F1-Score of 0.65, precision of 0.69, and recall of 0.62. These results establish ViralBindPredict as an effective instrument for accelerating drug development, especially in the realm of antiviral treatment, where time and resource limitations are often crucial. The ability of the model to overcome conventional constraints and generate dependable predictions demonstrates its potential to substantially influence the pharmaceutical industry.

__Keywords:__ Viral Drug Discovery; Viral Drug-Target Interactions; Viral Ligand Binding Site; Deep Learning; Supervised Learning; Neural Networks.

![Graphical Abstract]()

## Prerequisites
Python libraries:
* python - 3.11.6
* pytorch - 2.1.0
* torchmetrics - 1.2.0
* torchinfo - 1.8.0
* numpy - 1.26.0
* h5py  - 3.10.0
* imbalanced-learn - 0.11.0
* scikit-learn - 1.3.1
* scipy - 1.11.3
* tqdm - 4.66.1
* wandb - 0.15.12
* dill - 0.3.7
* MORDRED - version 1.2.0
* RDKit - version 2023.9.4

We recommend creating an isolated Conda environment to run our pipeline, which can be performed using the following code:
```bash
conda create --name ViralBindPredict python=3.11.6

# installing pytorch: https://pytorch.org
# our pytorch installation: conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install torchmetrics torchinfo numpy h5py tqdm dill wandb imbalanced-learn scikit-learn scipy pyarrow rdkit mordred requests biopython pandas

conda activate ViralBindPredict
```

__Note:__ The environment name, defined after the "--name" argument in the first step, can be whatever the user desires.

__Note:__ Required information to replicate and run ViralBindPredict is described in this repository.

__Note:__ To use WandB, you need to create an account at https://wandb.ai/ and follow the instructions to set up your account.

### ViralBindPredictDB
Dataset __viralbindpredictDB.hdf5__ regarding protein chain/residue classification, Mordred, SPOTONE and PSSM features is available at [TODO]. Folder [ViralBindPredictDB](./ViralBindPredictDB/) contains text files (_.txt_) with keys and descriptors from viralbindpredictDB.hdf5.
- [class_keys.txt](./ViralBindPredictDB/class_keys.txt): file with the 20.441 class keys with nomenclature _PDB ID (4 letters) : Chain ID (1 letter) _ PDB Compound ID (3 letters)_, e.g., 102l:A_0BU.
- [ligands.txt](./ViralBindPredictDB/ligands.txt): file with the 2.066 ligand keys with nomenclature _PDB Compound ID (3 letters)_, e.g., 0BU.
- [mordred_descriptors.txt](./ViralBindPredictDB/mordred_descriptors.txt): file with the 1514 Mordred features.
- [proteins.txt](./ViralBindPredictDB/proteins.txt): file with the 12.824 protein keys with nomenclature _PDB ID (4 letters) : Chain ID (1 letter)_, e.g., 102l:A.
- [spotone_descriptors.txt](./ViralBindPredictDB/spotone_descriptors.txt): file with the 173 SPOTONE features.
- [pssm_descriptors.txt](./ViralBindPredictDB/pssm_descriptors.txt): file with the 42 PSSM features.

### datasets
This folder contains an example of a dataset (dataset-example.hdf5) with a single interaction to run the ViralBindPredict pipeline. The dataset is stored in the HDF5 format and only includes protein and ligand features, i.e., there is not interaction target information.

This folder also contains a subfolder where transformations are logged for future reference. The transformations (dataset-example.txt) undertaken include:
[TODO]

### models/config-files:
This folder includes configuration files that set up the hyperparameter searches for the Multi-Layer Perceptron (MLP) and Autoencoder (AE) models. Each configuration defines the search spaces and parameters used to optimize the models’ training and performance.

### config-ae.yaml - Autoencoder Configuration:
This file outlines the hyperparameter search for the Autoencoder model, using a random search approach aimed at minimizing the loss across training epochs. Key parameters include:
- encoder_layers: Specifies different layer architectures for the encoder, offering a range of layer sizes to adjust model depth and feature extraction capacity.
- latent_vector: Sets possible latent space dimensions, ranging from 180 down to 40.
- activation: Tests various activation functions (relu, leaky-relu, gelu) to find the most effective non-linear transformation.
- criterion: Uses Mean Squared Error (mse) as the loss function for model training.
- optimizer: Applies the adam optimizer to enable adaptive learning rates.
- learning_rate: Explores a range between 0.0001 and 0.01 for learning rate selection.
- splits: Sets a data split ratio of [0.9, 0.1, 0], allocating 90% of the data for training and 10% for validation. First value is for training, second for validation and third for test. All values must add up to 1.
- epochs: Tests epochs ranging from 6 to 14.
- batch_size: Evaluates batch sizes between 32 and 256 to ensure stability during training.
- shuffle: Shuffles data at each epoch to improve model generalization.

### config-mlp.yaml - MLP Configuration:
This file configures the hyperparameter search for the MLP model, using random search to minimize training epoch loss. Primary parameters include:

- layers: Defines options for MLP layer architecture and depth, such as num_layers, with flexible architecture types (e.g., =, <, >, <>).
- activation: Tests multiple activation functions (relu, leaky-relu, gelu, tanh, sigmoid) to evaluate non-linearity effects.
- optimizer: Tests both adam and sgd optimizers.
- learning_rate: Adjusts learning rates over a range from 0 to 0.1.
- splits: Data split configurations include options like [0.7, 0.3, 0] and [0.8, 0.2, 0]. First value is for training, second for validation and third for test. All values must add up to 1.
- epochs: Ranges between 4 and 10 epochs.
- batch_size: Tests batch sizes from 32 to 512 to optimize training throughput.
- shuffle: Ensures data shuffling to enhance training robustness.

These configuration files allow extensive hyperparameter tuning, providing flexibility to adapt both models to the dataset’s specific requirements for optimal performance.

### Script files
#### autoencoder.py
This Python script is designed to train an Autoencoder model that encodes and reconstructs molecular descriptors for proteins. Model and configuration settings can be provided as command-line arguments.

```bash
usage: autoencoder.py [--dataset DATASET DATASET DATASET] [--model MODEL] [--config CONFIG CONFIG] [--device DEVICE] [--wandb {online,offline,disabled}]

--dataset DATASET DATASET DATASET
          <hdf5 dataset filepath> {interactions, proteins, ligands} {residue, chain}
--model MODEL
        <torch model filepath>
--config CONFIG CONFIG
         {<yaml sweep config filepath>, <wandb sweep author/project/id>} <number of runs>
--device DEVICE
         <torch device>
--wandb {online,offline,disabled}

#example 1: create new model, use local config file and upload results to wandb (autoencode protein features, chain granularity)
python autoencoder.py --dataset dataset-example.hdf5 proteins chain --config models/config-files/config-ae.yaml 10 --device cuda:0 --wandb online

#example 2: reuse model, associate run to already existing sweep and save results locally (autoencode protein features, residue granularity)
python autoencoder.py --dataset dataset-example.hdf5 proteins residue --model models/ae-model.pt --config wandb_user/wandb_project/wandb_sweep_id 10 --device cpu --wandb offline

# example 3: create new model, associate run to already existing sweep and upload results to wandb (autoencode protein and ligand features, residue granularity)
python autoencoder.py --dataset dataset-example.hdf5 interactions residue --config wandb_user/wandb_project/wandb_sweep_id 10 --device cuda:0 --wandb online
```

#### dataset.py
The script defines custom dataset classes that load, process, and manage protein-ligand interaction data. These classes are designed to support:
- Flexible Granularity: Datasets can be accessed at different levels (residues or chains).
- Data Splits: Supports train, validation, and test splits as specified in the dataset.
- Balanced Datasets: Additional support for balanced batch handling with the BalancedInteractionsDataset.

### multilayer_perceptron.py
This Python script trains a Multilayer Perceptron (MLP) model on interaction datasets with support for hyperparameter tuning via Weights & Biases (WandB). The script is designed for binary classification tasks involving interaction data, providing configurable architectures, training metrics, and options for balanced datasets.

```bash
usage: multilayer_perceptron.py [--dataset DATASET DATASET DATASET] [--model MODEL] [--config CONFIG CONFIG] [--device DEVICE] [--wandb {online,offline,disabled}]

--dataset DATASET DATASET DATASET
          <hdf5 dataset filepath> {interactions, balanced-interactions} {residue, chain}
--model MODEL
        <torch model filepath>
--config CONFIG CONFIG
         {<yaml sweep config filepath>, <wandb sweep author/project/id>} <number of runs>
--device DEVICE
         <torch device>
--wandb {online,offline,disabled}

#example 1: create new model, use local config file and upload results to wandb (train on protein and ligand features, residue granularity)
python multilayer_perceptron.py --dataset dataset-example.hdf5 interactions residue --config models/config-files/config-mlp.yaml 10 --device cuda:0 --wandb online

#example 2: reuse model, associate run to already existing sweep and save results locally (train on protein and ligand features, chain granularity)
python multilayer_perceptron.py --dataset dataset-example.hdf5 interactions chain --model models/mlp-model.pt --config wandb_user/wandb_project/wandb_sweep_id 10 --device cpu --wandb offline

# example 3: create new model, associate run to already existing sweep and upload results to wandb (train on balanced protein and ligand features, residue granularity) (balanced datasets have an extra root group in the HDF5 file called balanced-batches)
python multilayer_perceptron.py --dataset some-balanced-dataset.hdf5 balanced-interactions residue --config wandb_user/wandb_project/wandb_sweep_id 10 --device cuda:0 --wandb online
```

#### torch_map.py
This script defines mappings for common activation functions, loss functions (criterions), and optimizers to streamline model configuration.

#### transformations.py
This script defines several functions for managing and transforming datasets. It includes support for removing or updating specific dataset features, handling missing values, and rebalancing data through oversampling and SMOTE. Additionally, it logs each transformation, providing transparency and reproducibility.

There is no `__main__` function in this script. It is designed to be imported and used in other scripts for data transformation operations. We recommend using the provided transformations-logs to understand the transformations applied to the dataset-example.hdf5.

[TODO]

### If you are running this project on Windows and get an encoding related error, please run the following command in the terminal before running the project.
```bash
set PYTHONUTF8=1
```

### If you use ViralBindPredict, please cite the following.
[ViralBindPredict: Empowering Viral Protein-Ligand Binding Sites through Deep Learning and Protein Sequence-Derived Insights] PENDING CITATION