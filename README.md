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

#### HDF5 format
- <u>interactions/</u>:
	- Type: Group
	- Attributes: None
	- Group Members:
		- <u>PROT:C:LIG/</u>:
			- Type: Group
			- Attributes:
				- ligand:
					- Type: string
					- Value: ligands/LIG
				- protein:
					- Type: string
					- Value: proteins/PROT/C
				- split:
					- Type: string
					- Value: {train, test, validation}
			- Group Members:
				- <u>targets</u>:
					- Type: Dataset
					- Attributes: None
					- Dataset Dataspace:
						- Dimensions: 2
						- Shape: (|PROT:C residues|, 1)
					- Dataset Data Type:
						- 8-bit integer (numpy.int8)
					- Interpretation:
						- Each row represents a residue in the protein chain PROT:C.
						- The only column represents the target feature of the problem, i.e., whether or not the residue is interacting with the ligand LIG.
						- If proteins/PROT/C/features\[i\] is not interacting with LIG then interactions/PROT:C:LIG/targets\[i, 0\] = 0.
						- If proteins/PROT/C/features\[i\] is interacting with LIG then interactions/PROT:C:LIG/targets\[i, 0\] = 1.
- <u>ligands/</u>:
	- Type: Group
	- Attributes: None
	- Group Members:
		- <u>LIG/</u>:
			- Type: Group
			- Attributes: None
			- Group Members:
				- <u>features</u>:
					- Type: Dataset
					- Attributes: None
					- Dataset Dataspace:
						- Dimensions: 2
						- Shape: (1, |ligand features|)
					- Dataset Data Type:
						- 32-bit floating-point (numpy.float32) (__DEFAULT__; Can be changed through transformations.py)
					- Interpretation:
						- The only row represents the ligand (molecule) LIG.
						- Each column represents a ligand feature (1514 total). (__DEFAULT__; Can be changed through transformations.py)
						- The ligand features are taken from Mordred.
- <u>proteins/</u>:
	- Type: Group
	- Attributes: None
	- Group Members:
		- <u>PROT/</u>:
			- Type: Group
			- Attributes: None
			- Group Members:
				- <u>C/</u>:
					- Type: Group
					- Attributes: None
					- Group Members:
						- <u>features</u>:
							- Type: Dataset
							- Attributes: None
							- Dataset Dataspace:
								- Dimensions: 2
								- Shape: (|PROT:C residues|, |protein features|)
							- Dataset Data Type:
								- 32-bit floating-point (numpy.float32) (__DEFAULT__; Can be changed through transformations.py)
							- Interpretation:
								- Each row represents a residue in the protein chain PROT:C.
								- Each column represents a protein feature (216 total). (__DEFAULT__; Can be changed through transformations.py)
								- The protein features are taken from SPOTONE (proteins/PROT/C/features\[:, :173\]) and PSSM (proteins/PROT/C/features\[:, 173:\]). (__DEFAULT__; Can be changed through transformations.py)

#### BaseDataset(abc.ABC)
##### Methods
- \_\_init\_\_:
	- Opens the dataset in reading mode.
	- Asserts that the root keys are in accordance with the defined format.
- close:
	- Properly releases resources associated with the dataset.
- clone:
	- Clones dataset to a new HDF5 file.
	- Useful in automatization of dataset transformations.
- get_dataloader:
	- Returns a PyTorch DataLoader.
	- Abstract method therefore must be implemented by subclasses.
- get_splits_dataloaders:
	- Returns a list of PyTorch DataLoaders
	- Abstract method therefore must be implemented by subclasses.
#### InteractionsDataset(BaseDataset, torch.utils.data.Dataset)
- Dataset that deals only with interaction entries (interactions/ group).
##### Methods
- \_\_init\_\_:
	- Creates a residue_map:
		- residue_map\[split\]\[global residue index\] = (PROT:C:LIG, local residue index).
		- global vs. local residue index:
			- global: Index of a residue within the scope of all protein chains.
			- local: Index of a residue within the scope of its respective protein chain.
	- Creates a chain_map:
		- chain_map\[split\]\[global chain index\] = PROT:C:LIG.
		- global chain index: Index of a protein chain within the scope of all protein.
	- Asserts that the attribute split of each group interactions/PROT:C:LIG/, if it exists, is "train" or "test" or "validation" and, if it doesn't exist, is set to "dataset".
	- Asserts that all feature values in targets are either 0 (not interacting) or 1 (interacting).
	- Asserts that either all interactions/PROT:C:LIG/ have a split attribute or none has it.
	- Sets split to "dataset":
		- In case the dataset is pre split (all interactions/PROT:C:LIG/ have a split attribute), this allows us to deal with only the data of a certain split or all the data.
		- Can be "train" or "test" or "validation" or "dataset" (no split, i.e., dataset as a whole).
		- Since the attribute split is optional the default split is "dataset".
	- Sets residue_granularity to True, i.e., residue (not chain) granularity:
		- Can be True (residue) or False (chain).
		- True (residue) means that each data entry corresponds to a single residue of a PROT:C within each interaction PROT:C:LIG in the currently set split. (if a certain PROT:C is in two separate interactions in the currently set split, then there is repetition of data). We can load batches of residues (batch size = number of residues per batch) using the residue_map.
		- False (chain) means that each data entry corresponds to a single C of a PROT within each interaction PROT:C:LIG in the currently set split, i.e., including all the residues in that chain (if a certain PROT:C is in two separate interactions in the currently set split, then there is repetition of data). We can load batches of chains (batch size = number of chains per batch) using the chain_map.
		- Default residue_granularity is True, i.e., residue (implementation decision).
```js
residue_map = {

// [] if the dataset is not pre split
'train': [(AAAA:A:111, 0), (AAAA:A:111, 1), ..., (AAAA:B:111, 0), ..., (AAAB:A:444, 0), ...],

// [] if the dataset is not pre split
'test': [(BBBB:B:222, 0), (BBBB:B:222, 1), ...],

// [] if the dataset is not pre split
'validation': [(CCCC:C:333, 0), (CCCC:C:333, 1), ...],

'dataset': [(AAAA:A:111, 0), ..., (BBBB:B:222, 0), ..., (CCCC:C:333, 0), ...]

}

chain_map = {

// [] if the dataset is not pre split or if the "train" split is empty
'train': [AAAA:A:111, AAAA:B:111, ..., AAAB:A:444, ...],

// [] if the dataset is not pre split or if the "test" split is empty
'test': [BBBB:B:222, BBBB:A:222, ...],

// [] if the dataset is not pre split or if the "validation" split is empty
'validation': [CCCC:C:333, CCCC:A:333, ...],

'dataset': [AAAA:A:111, AAAA:B:111, ..., AAAB:A:444, ..., BBBB:B:222, BBBB:A:222, ..., CCCC:C:333, CCCC:A:333, ...]

}
```
- \_\_len\_\_:
	- Must be implemented because this is a PyTorch Dataset subclass.
	- Returns len(residue_map\[split\]) if residue_granularity is True (residue), i.e., the sum of the number of residues of each PROT:C:LIG in the split.
	- Returns len(chain_map\[split\]) if residue_granularity is False (chain), i.e., the number of PROT:C:LIG in the split.
	- this lets the DataLoader know how many data entries are there.
- \_\_getitem\_\_:
	- Must be implemented because this is a PyTorch Dataset subclass.
	- Given the index of a data entry returns its data.
	- If residue_granularity is True (residue):
		- Protein features of residue at global index i (residue_map\[split\]\[i\] = (PROT:C:LIG, k)):
			- numpy array because we use indexing on an h5py Dataset (...\['features'\]\[residue\[1\]\]).
			- shape: (number of protein features), 1 dimension.
		- Ligand features of LIG: 
			- numpy array because we use indexing on an h5py Dataset (...\['features'\]\[0\]).
			- shape: (number of ligand features), 1 dimension.
		- Concatenate protein and ligand features:
			- numpy array.
			- shape: (number of protein features + number of ligand features), 1 dimension.
		- Target feature value of the interaction between the residue at global index i and the ligand LIG:
			- numpy array because we use indexing on an h5py Dataset (...\['targets'\]\[residue\[1\]\]).
			- shape: (1), 1 dimension.
	- if residue_granularity is False (chain):
		- Protein features of all residues in chain at global index i (chain_map\[split\]\[i\] = PROT:C:LIG):
			- h5py Dataset because we do not use indexing (...\['features'\]...) therefore conversion to numpy array is needed.
			- shape: (number of residues in PROT:C, number of protein features), 2 dimensions.
		- Ligand features of LIG: 
			- h5py Dataset because we do not use indexing (...\['features'\]...) therefore conversion to numpy array is needed.
			- repeat the features values of LIG once for each residue in PROT:C (numpy.repeat which also does the conversion to numpy array needed).
			- shape: (number of residues in __PROT:C__, number of ligand features), 2 dimensions
		- Concatenate protein and ligand features:
			- numpy array
			- shape: (number of residues in PROT:C, number of protein features + number of ligand features), 2 dimensions.
		- Target feature values of the interactions between each residue in chain at global index i and the ligand LIG:
			- h5py Dataset because we do not use indexing (...\['targets'\]...) therefore conversion to numpy array is needed.
			- shape: (number of residues in PROT:C, 1), 2 dimensions.
	- Return torch.tensor(features), torch.tensor(targets).
	- This lets the DataLoader use indexing in a InteractionsDataset object to get the data entries of a batch.
- get_dataloader:
	- if G = "residue":
		- if argument "split" is None, use current split S of self
		- if argument "split" is a PyTorch Subset, use it instead of self (this allows us to re split the dataset as we wish, PyTorch random_split, i.e., allows to use newly made splits instead of only the ones defined by the self.split)
		- return a PyTorch DataLoader for data in the obtained split where batch size is the number of residues per batch
		- here each data entry has shape __((number of protein features + number of ligand features), (1))__ (one for the features, one for the targets), i.e., all data entries have the same constant shape because "number of protein features" and "number of ligand features" doesn't change. therefore the dataloader can automatically stack them vertically to form each batch of shape __((number of residues in the batch, number of protein features + number of ligand features), (number of residues in the batch, 1))
	- if G = "chain":
		- if argument "split" is None, use current split S of self
		- if argument "split" is a PyTorch Subset, use it instead of self (this allows us to re split the dataset as we wish, PyTorch random_split)
		- return a PyTorch DataLoader for data in the obtained split where batch size is the number of chains per batch
		- here each data entry has shape __((number of residues in PROT:C, number of protein features + number of ligand features), (number of residues in PROT:C, 1))__ (one for the features, one for the targets), i.e., all data entries have variable shape because "number of residues in PROT:C" varies depending on the specific PROT:C. therefore the dataloader can't automatically stack them vertically to form each batch. so we do it manually with collate_fn to form batches of shape __((number of residues in the batch, number of protein features + number of ligand features), (number of residues in the batch, 1))__
	- what the dataloader essentially does when it receives a subclass of PyTorch Dataset is use the \_\_len\_\_ and \_\getitem\_\_ functions defined by us to iterate and get the correct data entries. when it receives a PyTorch Subset it uses the list of indexes this object keeps in self to know which indexes retrieve.
- get_splits_dataloaders:
	- similar to get_dataloader but useful to get a list of dataloaders
	- if argument "split" is None retrieve dataloaders for each split if they have at least one data entry otherwise retrieve None for that particular list index
	- if argument "split" is admissable to PyTorch random_split, uses it in a PyTorch random_split to retrieve dataloaders with the specified lengths for the current split S of self. allows for "empty" dataloaders which are None in the respective list index.
- set_split: _self explanatory_
	- asserts that the passed split argument is a valid split
	- asserts that the passed split argument is not an empty split
- set_granularity: _self explanatory_

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