## ViralBindPredict

## Abstract:
The development of a single drug can cost up to 1.8 billion USD and require over a decade, timelines that contemporary pharmaceutical companies seek to expedite. Computational methodologies have become
integral to drug discovery; however, traditional approaches, such as docking simulations, often rely on protein   and   ligand   structures   that   are   unavailable,   incomplete,   or   lack   sufficient   accuracy.
Notwithstanding advances such as AlphaFold in predicting protein structures, these models are not always sufficiently precise for identifying Ligand-Binding Sites (LBS) or Drug-Target Interactions (DTI).
In this study, we introduce ViralBindPredict, an innovative Deep-Learning (DL) model designed to predict LBS   in   viral   proteins   using   sequence-based   data.   Our   approach   leverages   sequence-derived
information from the Protein Data Bank (PDB), offering a faster and more accessible alternative to structure-based   methods.   ViralBindPredict   classifies   viral   protein   residues   as   interacting   or   non-
interacting with ligands based on a 5 Å threshold, unlocking new avenues for antiviral drug discovery. To enhance performance, we extracted advanced descriptors from protein-ligand complexes and
applied autoencoders for dimensionality reduction of protein features. ViralBindPredict was rigorously evaluated across key metrics, achieving an accuracy of 0.68, AUC-ROC of 0.74, F1-Score of 0.65,
precision of 0.69, and recall of 0.62. These results establish ViralBindPredict as an effective instrument for accelerating drug development,
especially in the realm of antiviral treatment, where time and resource limitations are often crucial. The ability   of   the   model   to   overcome   conventional   constraints   and   generate   dependable   predictions
demonstrates its potential to substantially influence the pharmaceutical industry .

Key words: Viral   Drug   Discovery;   Viral   Drug-Target   Interactions;   Viral   Ligand   Binding   Site;   Deep   Learning;
Supervised Learning; Neural Networks.

![Graphical Abstract](Graphical_Abstract.png)

### Prerequisites:
ViralBindPredict was developed and tested as follows:
* Python 3.11
* MORDRED - version 1.2.0
* RDKit - version 2023.9.4
* numpy - version 1.26.3
* pandas - version 2.2.0 
* scipy - version 1.12.0
* h5py - version 3.10.0

We recommend creating an isolated Conda environment to run our pipeline, which can be performed using the following code:
```bash
conda create --name ViralBindPredict python=3.9.16 -c conda-forge biopython pandas scipy h5py pyarrow numpy rdkit mordred requests
conda activate ViralBindPredict
```
Note: The environment name, defined after the "--name" argument in the first step, can be whatever the user desires.

Required information to replicate and run ViralBindPredict is described in this Repository.

### ViralBindPredictDB:
Data viralbindpredictDB.hdf5 file regardinig protein chain residue classification, Mordred descriptors and SPOTONE and PSSM features are available in this [link]().
./ViralBindPredictDB/ folder contains .txt files with keys and descriptors from viralbindpredictDB.hdf5:

 1) "class_keys.txt"- file with the 20.441 class keys (PDB:chain_compound).
 2) "mordred_keys.txt"- file with the 2.066 Mordred keys (PDB compound ID).
 3) "mordred_descriptors.txt"- file with the 1514 Mordred descriptors.
 4) "spotone_keys.txt"- file with the 12.824 SPOTONE keys (PDB:chain).
 5) "spotone_descriptors.txt"- file with the 173 SPOTONE descriptors.
 6) "pssm_keys.txt"- file with the 12.824 PSSM keys (PDB:chain).
 7) "pssm_descriptors.txt"- file with the 173 PSSM descriptors.

### BU48dataset:
### A) dataset-bu48-transformed-1.hdf5:
Dataset with BU48 descriptor information (Mordred, SPOTONE, PSSM)

### B) transformations-logs:
Includes a txt with information regarding transformations on BU48 dataset.

### models/config-files:
Details on MLP model and Autoencoder hyperparameter search

### Script files:
### autoencoder.py
Script to run the autoencoder in a input data

### dataset.py
Script to access input data and transform data into balanced data, if needed.

### multilayer_perceptron.py
Script to run a Multilayer Perceptron

### torch_map.py
Script with a hyperparameters list used for MLP

### transformations.py
Script to clean dataset

```bash
python3 5-spotone_features.py
```

### If you use our predictor, please cite the following.

[ViralBindPredict: Empowering Viral Protein-Ligand Binding Sites through Deep Learning and Protein Sequence-Derived Insights] PENDING CITATION
