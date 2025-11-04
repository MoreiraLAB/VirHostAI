# ViralBindPredict

> This repository is structured in 4 parts: 
> **[Install Requirements](#install-requirements)**, 
> **[ViralBindPredict Database](#viralbindpredict-database)**, 
> **[Model Optimization & Training](#model-optimization--training)**, and 
> **[New Prediction](#new-prediction)**.

```bash
$HOME/VirHostAI/
│   ├── results_models/
│   ├── results_models_40%/
│   ├── results_models_90%/
│   ├── results_models_nored_90%_cluster40/
│   ├── results_models_nored_90%_cluster90/
│   ├── ViralBindPredict/
│   │   ├── splits_esm2_output/
│   │   ├── splits_esm2_output_40%/
│   │   ├── splits_esm2_output_90%/
│   │   ├── splits_esm2_output_nored_90%_cluster90/
│   │   ├── splits_esm2_output_nored_90%_cluster40/
│   │   ├── splits_prottrans_output/
│   │   ├── splits_prottrans_output_40%/
│   │   ├── splits_prottrans_output_90%/
│   │   ├── splits_prottrans_output_nored_90%_cluster90/
│   │   ├── splits_prottrans_output_nored_90%_cluster40/
│   │   ├── splits_esm2_output_standardized/
│   │   ├── splits_esm2_output_standardized_40%/
│   │   ├── splits_esm2_output_standardized_90%/
│   │   ├── splits_esm2_output_standardized_nored_90%_cluster40/
│   │   ├── splits_esm2_output_standardized_nored_90%_cluster90/
│   │   ├── splits_prottrans_output_standardized/
│   │   ├── splits_prottrans_output_standardized_40%/
│   │   ├── splits_prottrans_output_standardized_90%/
│   │   ├── splits_prottrans_output_standardized_nored_90%_cluster40/
│   │   ├── splits_prottrans_output_standardized_nored_90%_cluster90/
│   │   ├── binary_interactions_csv/
│   │   ├── interaction_details_csv_cif/
│   │   ├── cif/
│   │   ├── split_cif_chains/
│   │   ├── fasta_sequences/
│   │   ├── fasta_sequences_filtered/
│   │   └── split_cif_chains_filtered/
│   └── new_prediction/
```

An isolated Conda environment can be created using the following code:

```bash
# Create a new environment named "ViralBindPredict"
conda create -n ViralBindPredict python=3.11.3

# Activate environment
conda activate ViralBindPredict
```

## Install Requirements 

> After cloning this repository and activating the `ViralBindPredict` Conda environment, make sure you are in the project directory (where `requirements.txt` or `requirements_cpu.txt` are located) before running the commands below:

```bash
# To use GPU
pip install -r requirements.txt

# CPU-only
pip install -r requirements_cpu.txt
```

> Note: If LightGBM raises a `libomp` or `Library not loaded: @rpath/libomp.dylib` error on macOS, run:
> ```bash
> brew update
> brew install libomp
> ```

## ViralBindPredict Database

To recreate the ViralBindPredict database, navigate to the ViralBindPredict folder (where the scripts are located) using:
> **Note:** This step requires approximately 400G of available disk space. Make sure you have sufficient memory and storage to complete all database generation steps successfully. 

```bash
cd ViralBindPredict
```

All data frames and folders generated during this process are provided at the following [link](https://1drv.ms/f/c/6645cc99a95f711d/EtRIfMI6bSxBuY5PVGIRtgUBL6lfH1W9iuBLVbHXlwd8Uw?e=EinOKH). 

**Before starting:**
**Please download the components_18_06.cif file from the ViralBindPredict/ViralBindPredict/ folder in the [link](https://1drv.ms/f/c/6645cc99a95f711d/EhugUMOUp49Hqx0gKJi-lDUBnIpQWpGjPzf5RiiDvkM5Cw?e=pf6sef), and save it to the ./ViralBindPredict directory of this repository.**

Then, run the following commands:

1. **```1_download_cifs.py```** -> Reads PDB IDs from **virus_pdb_ids_18_06.txt**, downloads CIF files for protein-containing PDB entries into a cif/ folder, retrieves metadata (experimental method, resolution) from the RCSB, and obtains chain → organism mappings from the PDB, saving **protein_data.csv** with entry-level metadata and **chain_organisms.csv** with chain-level organism information.

```bash
python 1_download_cifs.py
```

> Note: This PDB list was obtained on 18/06/2025. Some PDB entries may become obsolete over time and may no longer be available for download; such entries can be found in the obsolete PDB archive. However, we provide the downloaded CIF files in compressed format at [link](https://1drv.ms/f/c/6645cc99a95f711d/EhugUMOUp49Hqx0gKJi-lDUBnIpQWpGjPzf5RiiDvkM5Cw?e=pf6sef).

2. **```2_preprocess_chains.py```** -> Processes all CIF files in the cif/ folder. For each protein chain, it converts modified residues (listed in _pdbx_struct_mod_residue) from HETATM to ATOM and remaps them to standard residues using parent_comp_id; filters out chains containing ambiguous amino acids (e.g., UNK); and tracks ligands (HETATM entries excluding water). **Outputs:**
split_cif_chains/;fasta_sequences/; and ligands_per_chain.csv


```bash
python 2_preprocess_chains.py
```

3. **```3_filter_files.py```** -> Loads **ligands_per_chain.csv** (chains + ligands), **chain_organisms.csv** (organisms per chain), and **protein_data.csv** (PDB-level metadata); retains only chains with ligands and sequences ≥ 30 amino acids; matches chains from ligands_per_chain.csv with organism information; and saves chain_organisms_filtered.xlsx (chains with ligands and organisms) and protein_data_filtered.xlsx (PDB entries with ligands and metadata).

```bash
python 3_filter_files.py
```

4. **```4_keep_viral_only.py```** -> Loads **chain_organisms_filtered.xlsx** (chains + organisms), **keep_chain_organisms.xlsx** (trusted corrections for “unknown” organisms), and **ligands_per_chain.csv** (ligands per chain); applies a viral keyword list, fills “unknown” organisms using the manually reviewed corrections in keep_chain_organisms.xlsx, and flags chains only when all associated organisms are viral; saves the filtered **outputs** as chain_organisms_viral_only.xlsx (viral chains only), ligands_per_chain_viral_only.csv (viral chains with ligands), unique_viral_organisms.txt (unique viral organism names), and excluded_non_viral_organisms.txt (all excluded organisms: non-viral or chimeric).

```bash
python 4_keep_viral_only.py
```

5. **```5_preprocess_ligands.py```** -> Loads data: **components_18_06.cif** (ligand chemical info retrivered from PDB on 18/06/2025), **ligands_per_chain_viral_only.csv** (ligands bound to viral chains), and predefined exclusions lists (ions, solvents, possible artifacts). Applies filtering rules and steps, including basic exclusions, SMILES & Structure quality, conflict rules. **Outputs:** ligands_filtered.csv: Final curated ligand list with SMILES, InChIKey, etc; ligands_excluded.csv: All excluded ligands with reasons; ligands_excluded_conflicts.csv: Ligands removed due to SMILES/InChI conflicts; excluded_counts_by_reason.csv: Count summary of all exclusion reasons; question_type_valid_chemtype_ligands.txt: Log of ligands with missing TYPE but valid CHEM_TYPE. 

> Note: Please **download components_18_06.cif** file from the ViralBindPredict/ViralBindPredict/ folder in the [link](https://1drv.ms/f/c/6645cc99a95f711d/EhugUMOUp49Hqx0gKJi-lDUBnIpQWpGjPzf5RiiDvkM5Cw?e=pf6sef), and save it to the ./ViralBindPredict directory of this repository.

```bash
python 5_preprocess_ligands.py
```

6. **```6_filter_files_by_ligands.py```** -> Loads **ligands_per_chain_viral_only.csv** → chains and ligands; **ligands_filtered.csv** → ligands previously marked as relevant; **protein_data_filtered.xlsx** → metadata about PDB entries; and **chain_organisms_viral_only.xlsx** → organisms per chain. It filters to keep only ligands in ligands_filtered.csv. Adds ligand details (TYPE, CHEM_TYPE, SMILES, InChIKey), joins protein metadata (Experimental Method, Resolution), and adds organism information for each chain. **Output:** filtered_df.xlsx.

 
```bash
python 6_filter_files_by_ligands.py
```

7. **```7_filter_fasta_cifs.py```** -> Filters FASTA and CIF files to retain only those referenced in filtered_df.xlsx, and writes the results to **split_cif_chains_filtered/** and **fasta_sequences_filtered/**.

```bash
python 7_filter_fasta_cifs.py
```

8. **```8_class_construction.py```** ->  **Input:** Filtered protein–ligand chain list from filtered_df.xlsx and corresponding chain-specific CIF files. Interaction calculation (per ligand site): For each residue in the chain, selects all non-hydrogen atoms with that residue index; computes all distances to ligand non-hydrogen atoms; and classifies each pair as: "Interacting": ≤ 4.5 Å or "Non-Interacting": > 4.5 Å. **Output:** Writes a CSV to **interaction_details_csv_cif/** for each PDB–chain–ligand–site combination. Includes protein + ligand atom names and coordinates; Residue identity, sequence position, ligand metadata; and Distance and interaction label.

```bash
python 8_class_construction.py
```

9. **```9_binary_csv_interactions.py```** -> Reads all **_interactions.csv** files in **interaction_details_csv_cif**. Converts “Classification” (Interacting → 1, Non-Interacting → 0). If at least one atom in the residue interacts, then the residue is marked as 1. If no atoms in the residue interact with the ligand, then it is 0. Saves a binary.csv file in **binary_interactions_csv/** only if there’s at least one interacting residue.

```bash
python 9_binary_csv_interactions.py
```

> From here, two datasets were generated: one with **Mordred + ESM2 features** and another with **Mordred + ProtTrans** features.

10. **```10_feature_extraction_mordred_esm2.py```** or **```10_feature_extraction_mordred_prottrans.py```** -> Loads filtered residue-ligand binary interaction CSVs (*_binary.csv) from **binary_interactions_csv**. For each protein chain: Loads the corresponding FASTA sequence from fasta_sequences_filtered. Generates protein features: ESM-2 or ProtTrans T5, and ligand features: Computes Mordred descriptors from the SMILES string. **Output:** interaction_dataset.hdf5 (ESM2 + Mordred) and interaction_dataset_prottrans_mordred.hdf5 (Prottrans + Mordred).

```bash
# For Mordred + ESM2 dataset
python 10_feature_extraction_mordred_esm2.py

# Or for Mordred + Prottrans dataset 
python 10_feature_extraction_mordred_prottrans.py
```

11. **```11_deduplication_esm2.py```** or **```11_deduplication_prottrans.py```** -> Keeps only interactions (PDB CHAINS) with resolution lower than 5Å, excludes interactions where: fewer than 2 interacting residues (only 1 interacting residue), or all interacting residues are consecutive (even if there are more than 2). Then it performs **intra-PDB deduplication** (For chains in the same PDB), and **inter-PDB deduplication** (for chains in different PDBs). **Output:** interaction_dataset_esm2_mordred_deduplicated.hdf5 and deduplicated_esm2_mordred_filtered_df.xlsx (ESM2 + Mordred); and interaction_dataset_prottrans_mordred_deduplicated.hdf5 and deduplicated_filtered_prottrans_df.xlsx (Prottrans + Mordred).

```bash
# For Mordred + ESM2 dataset
python 11_deduplication_esm2.py

# Or for Mordred + Prottrans dataset 
python 11_deduplication_prottrans.py
```

> Another two datasets were generated, where **90% sequence redundancy** was removed for each Mordred + ProtTrans and Mordred + ESM2 combination.

12. **```12_filter_redundancy_90%_esm2.py```** or **```12_filter_redundancy_90%_prottrans.py```** -> receives as input the mmseqs fasta file (after 90% sequence similarity removal) and filters the hdf5 files from previous step to retain only the chains that contain the non redundant set of sequences. **Output:** interaction_dataset_esm2_mordred_deduplicated_no_red.hdf5 and deduplicated_esm2_mordred_filtered_df.xlsx (ESM2 + Mordred); and interaction_dataset_prottrans_mordred_deduplicated_no_red.hdf5 and deduplicated_filtered_prottrans_df.xlsx (Prottrans + Mordred).

```bash
# For Mordred + ESM2 dataset
python 12_filter_redundancy_90%_esm2.py

# Or for Mordred + Prottrans dataset 
python 12_filter_redundancy_90%_prottrans.py
```

13. **```13_clean_esm2.py```** or **```13_clean_prottrans.py```** or **```13_clean_prottrans_nored_90%.py```** or **```13_clean_esm2_nored_90%.py```** -> Removes feature columns with: NaN values, Inf values, and Zero variance (same value for all samples). It saves cleaned HDF5 files: viralbindpredictDB-prottrans-cleaned.hdf5, viralbindpredictDB-esm2-cleaned.hdf5, viralbindpredictDB-esm2-cleaned_no_red.hdf5, and viralbindpredictDB-prottrans-cleaned_no_red.hdf5.

```bash
# For Mordred + ESM2 full dataset
python 13_clean_esm2.py

# Or for Mordred + Prottrans full dataset 
python 13_clean_prottrans.py

# For Mordred + ESM2 reduced dataset
python 13_clean_esm2_nored_90%.py

# Or for Mordred + Prottrans reduced dataset 
python 13_clean_prottrans_nored_90%.py
```

14. **```14_train_test_blind_base_esm2.py```** or  **```14_train_test_blind_esm2_cluster_90%.py```** or **```14_train_test_blind_esm2_cluster_40.py```** or **```14_train_test_blind_esm2_nored_90%_cluster90%.py```** or  **```14_train_test_blind_esm2_nored_90%_cluster40%.py```** or **```14_train_test_blind_base_prottrans.py```** or  **```14_train_test_blind_prottrans_cluster_90%.py```** or **```14_train_test_blind_prottrans_cluster_40.py```** or  **```14_train_test_blind_prottrans_nored_90%_cluster90%.py```** or **```14_train_test_blind_prottrans_nored_90%_cluster40%.py```** ->  Split the datasets into train, validation, test, blind protein (protein belongs to the blind protein list and ligand is not blind), blind ligand (ligand belongs to the blind ligand list and protein is not blind), and blind (protein belongs to the blind protein list and ligand belongs to the blind ligand list) datasets. 

> Note: The Base split corresponds to the Standard split using unique protein and ligand IDs. Cluster 90% or Cluster 40% splits use ligand scaffolds for ligand partitioning and MMseqs2 clusters for protein partitioning. NoRed90% + Cluster90% and NoRed90% + Cluster40% use the reduced dataset (after 90% sequence redundancy removal) and then apply MMseqs2 clustering for proteins and scaffold-based splitting for ligands. For more details, please refer to the Methodology section of the paper.

```bash
# --- For Mordred + ESM2 datasets ---
# Base split
python 14_train_test_blind_base_esm2.py

# Cluster 90% split
python 14_train_test_blind_esm2_cluster_90%.py

# Cluster 40% split
python 14_train_test_blind_esm2_cluster_40.py

# Non-redundant 90% + cluster 90%
python 14_train_test_blind_esm2_nored_90%_cluster90%.py

# Non-redundant 90% + cluster 40%
python 14_train_test_blind_esm2_nored_90%_cluster40%.py

# --- For Mordred + ProtTrans datasets ---
# Base split
python 14_train_test_blind_base_prottrans.py

# Cluster 90% split
python 14_train_test_blind_prottrans_cluster_90%.py

# Cluster 40% split
python 14_train_test_blind_prottrans_cluster_40.py

# Non-redundant 90% + cluster 90%
python 14_train_test_blind_prottrans_nored_90%_cluster90%.py

# Non-redundant 90% + cluster 40%
python 14_train_test_blind_prottrans_nored_90%_cluster40%.py
```

15. **```15_standardize_base_esm2.py```** or **```15_standardize_esm2_cluster_90%.py```** or **```15_standardize_esm2_cluster_40%.py```** or **```15_standardize_esm2_nored_90%_cluster90.py```** or **```15_standardize_esm2_nored_90%_cluster40.py```** or **```15_standardize_base_prottrans.py```** or **```15_standardize_prottrans_cluster_90%.py```** or  **```15_standardize_prottrans_nored_90%_cluster90.py```** or **```15_standardize_prottrans_nored_90%_cluster40.py```**. This step fits a `StandardScaler` on the training split and applies it to all other splits (validation, test, blind protein, blind ligand, and blind). Each experiment corresponds to a specific dataset configuration (ESM2 or ProtTrans) and clustering or redundancy filtering strategy.


```bash
# --- For Mordred + ESM2 datasets ---
# Base experiment
python 15_standardize_base_esm2.py

# Cluster 90% experiment
python 15_standardize_esm2_cluster_90%.py

# Cluster 40% experiment
python 15_standardize_esm2_cluster_40%.py

# NoRed 90% + Cluster 90% experiment
python 15_standardize_esm2_nored_90%_cluster90.py

# NoRed 90% + Cluster 40% experiment
python 15_standardize_esm2_nored_90%_cluster40.py

# --- For Mordred + ProtTrans datasets ---
# Base experiment
python 15_standardize_base_prottrans.py

# Cluster 90% experiment
python 15_standardize_prottrans_cluster_90%.py

# Cluster 40% experiment
python 15_standardize_prottrans_cluster_40%.py

# NoRed 90% + Cluster 90% experiment
python 15_standardize_prottrans_nored_90%_cluster90.py

# NoRed 90% + Cluster 40% experiment
python 15_standardize_prottrans_nored_90%_cluster40.py
```

## Model Optimization & Training


```bash
# Back to the main directory
cd ..

```

> **Note1:**  
> Running Optuna optimization is only required if you wish to tune a model for a specific dataset split.  
> This process is computationally intensive and may take a considerable amount of time depending on your hardware specifications.

> **Note2:** If you already have the datasets generated from the previous step, this stage will require an additional ~500 MB to 1 GB of space. 
> If you start from this step instead, make sure to have approximately 105 GB of available storage to download the necessary databases.

> **Hardware configurations used for optimization and training:**  
> - System A: 
>	- Operating System: Ubuntu 22.04.2 LTS
>	- Kernel: Linux 5.15.0-71-generic
>	- Architecture: x86_64
>	- NVIDIA Driver: 530.30.02
>	- CUDA Runtime Version: 12.1 

> - System B:
>	- Operating System: Ubuntu 22.04.2 LTS  
>	- Kernel: Linux 5.15.0-157-generic
>	- Architecture: x86_64
>	- NVIDIA Driver: 555.42.06
>	- CUDA Runtime Version: 12.5

16. Optuna (MLP & LightGBM)

Optuna-based hyperparameter optimization for both LightGBM and MLP models. Saves the resulting best hyperparameters to the appropriate output folder.

> Note: This table summarizes the training and validation inputs used for Optuna optimization, along with the output directories for the results. If you have already run the ViralBindPredict database creation pipeline, you should have all required inputs for this task. Otherwise, please refer to the datasets in [link](https://1drv.ms/f/c/6645cc99a95f711d/EtRIfMI6bSxBuY5PVGIRtgUBL6lfH1W9iuBLVbHXlwd8Uw?e=EinOKH) and download them to the indicated folders before running Optuna or training models.

| Dataset | Script/Algorithm | Experiment | Split | Input Folder | Train | Val | Output (models & hyperparameters) | Study Name | Database Name | json Name |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| Mordred + ESM2 | LGBM_optuna.py / LGBM | Base | Standard split using unique protein and ligand IDs | `ViralBindPredict/splits_esm2_output_standardized/` | `esm2_train_standardized.hdf5` | `esm2_val_standardized.hdf5` | `results_models/` | `lgbm_gpu_study_esm2_groupclusters` | `lgbm_optuna_esm2_groupclusters.db` | `best_lgbm_params_esm2_groupclusters.json` |
| Mordred + ESM2 | LGBM_optuna.py / LGBM | Cluster 90% | Scaffold split and clustering at 90% sequence identity | `ViralBindPredict/splits_esm2_output_standardized_90%/` | `esm2_train_90%_standardized.hdf5` | `esm2_val_90%_standardized.hdf5` | `results_models_90%/` | `lgbm_gpu_study_esm2_groupclusters_90%` | `lgbm_optuna_esm2_groupclusters_90%.db` | `best_lgbm_params_esm2_groupclusters_90%.json` |
| Mordred + ESM2 | LGBM_optuna.py / LGBM | Cluster 40% | Scaffold split and clustering at 40% sequence identity | `ViralBindPredict/splits_esm2_output_standardized_40%/` | `esm2_train_40%_standardized.hdf5` | `esm2_val_40%_standardized.hdf5` | `results_models_40%/` | `lgbm_gpu_study_esm2_groupclusters_40%` | `lgbm_optuna_esm2_groupclusters_40%.db` | `best_lgbm_params_esm2_groupclusters_40%.json` |
| Mordred + ESM2 | LGBM_optuna.py / LGBM | NoRed 90% + Cluster 90% | Scaffold split and Non-redundant 90% + Clustered at 90% identity | `ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/` | `esm2_train_nored_90%_cluster90_standardized.hdf5` | `esm2_val_nored_90%_cluster90_standardized.hdf5` | `results_models_nored_90%_cluster90/` | `lgbm_gpu_study_esm2_groupclusters_red_90%_cluster90` | `lgbm_optuna_esm2_groupclusters_red_90%_cluster90.db` | `best_lgbm_params_esm2_groupclusters_red_90%_cluster90.json` |
| Mordred + ESM2 | LGBM_optuna.py / LGBM | NoRed 90% + Cluster 40% | Scaffold split and Non-redundant 90% + Clustered at 40% identity | `ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster40/` | `esm2_train_nored_90%_cluster40_standardized.hdf5` | `esm2_val_nored_90%_cluster40_standardized.hdf5` | `results_models_nored_90%_cluster40/` | `lgbm_gpu_study_esm2_groupclusters_nored_90%_cluster40` | `lgbm_optuna_esm2_groupclusters_nored_90%_cluster40.db` | `best_lgbm_params_esm2_groupclusters_red_90%_cluster40.json` |
| Mordred + ESM2 | MLP_optuna.py / MLP | Base | Standard split using unique protein and ligand IDs | `ViralBindPredict/splits_esm2_output_standardized/` | `esm2_train_standardized.hdf5` | `esm2_val_standardized.hdf5` | `results_models/` | `mlp_study_esm2_clusters` | `optuna_mlp_esm2_clusters.db` | `best_mlp_params_esm2_clusters.json` |
| Mordred + ESM2 | MLP_optuna.py / MLP | Cluster 90% | Scaffold split and clustering at 90% sequence identity | `ViralBindPredict/splits_esm2_output_standardized_90%/` | `esm2_train_90%_standardized.hdf5` | `esm2_val_90%_standardized.hdf5` | `results_models_90%/` | `mlp_study_esm2_clusters_90%` | `optuna_mlp_esm2_clusters_90%.db` | `best_mlp_params_esm2_clusters_90%.json` |
| Mordred + ESM2 | MLP_optuna.py / MLP | Cluster 40% | Scaffold split and clustering at 40% sequence identity | `ViralBindPredict/splits_esm2_output_standardized_40%/` | `esm2_train_40%_standardized.hdf5` | `esm2_val_40%_standardized.hdf5` | `results_models_40%/` | `mlp_study_esm2_clusters_40%` | `optuna_mlp_esm2_clusters_40%.db` | `best_mlp_params_esm2_clusters_40%.json` |
| Mordred + ESM2 | MLP_optuna.py / MLP | NoRed 90% + Cluster 90% | Scaffold split and Non-redundant 90% + Clustered at 90% identity | `ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/` | `esm2_train_nored_90%_cluster90_standardized.hdf5` | `esm2_val_nored_90%_cluster90_standardized.hdf5` | `results_models_nored_90%_cluster90/` | `mlp_study_esm2_nored_90%_cluster90` | `optuna_mlp_esm2_clusters_nored_90%_cluster90.db` | `best_mlp_params_esm2_clusters_nored_90%_cluster90.json` |
| Mordred + ESM2 | MLP_optuna.py / MLP | NoRed 90% + Cluster 40% | Scaffold split and Non-redundant 90% + Clustered at 40% identity | `ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster40/` | `esm2_train_nored_90%_cluster40_standardized.hdf5` | `esm2_val_nored_90%_cluster40_standardized.hdf5` | `results_models_nored_90%_cluster40/` | `mlp_study_esm2_clusters_nored_90%_cluster40` | `optuna_mlp_esm2_clusters_nored_90%_cluster40.db` | `best_mlp_params_esm2_clusters_nored_90%_cluster40.json` |
| Mordred + ProtTrans | LGBM_optuna.py / LGBM| Base | Standard split using unique protein and ligand IDs | `ViralBindPredict/splits_prottrans_output_standardized/` | `prottrans_train_standardized.hdf5` | `prottrans_val_standardized.hdf5` | `results_models/` | `lgbm_gpu_study_prottrans_groupclusters` | `lgbm_optuna_prottrans_groupclusters.db` | `best_lgbm_params_prottrans_groupclusters.json` |
| Mordred + ProtTrans | LGBM_optuna.py / LGBM| Cluster 90% | Scaffold split and clustering at 90% sequence identity | `ViralBindPredict/splits_prottrans_output_standardized_90%/` | `prottrans_train_90%_standardized.hdf5` | `prottrans_val_90%_standardized.hdf5` | `results_models_90%/` | `lgbm_gpu_study_prottrans_groupclusters_90%` | `lgbm_optuna_prottrans_groupclusters_90%.db` | `best_lgbm_params_prottrans_groupclusters_90%.json` |
| Mordred + ProtTrans | LGBM_optuna.py / LGBM | Cluster 40% | Scaffold split and clustering at 40% sequence identity | `ViralBindPredict/splits_prottrans_output_standardized_40%/` | `prottrans_train_40%_standardized.hdf5` | `prottrans_val_40%_standardized.hdf5` | `results_models_40%/` | `lgbm_gpu_study_prottrans_groupclusters_40%` | `lgbm_optuna_prottrans_groupclusters_40%.db` | `best_lgbm_params_prottrans_groupclusters_40%.json` |
| Mordred + ProtTrans | LGBM_optuna.py / LGBM | NoRed 90% + Cluster 90% | Scaffold split and Non-redundant 90% + Clustered at 90% identity | `ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster90/` | `prottrans_train_nored_90%_cluster90_standardized.hdf5` | `prottrans_val_nored_90%_cluster90_standardized.hdf5` | `results_models_nored_90%_cluster90/` | `lgbm_gpu_study_prottrans_groupclusters_red_90%_cluster90` | `lgbm_optuna_prottrans_groupclusters_red_90%_cluster90.db` | `best_lgbm_params_prottrans_groupclusters_nored_90%_cluster90.json` |
| Mordred + ProtTrans | LGBM_optuna.py / LGBM | NoRed 90% + Cluster 40% | Scaffold split and Non-redundant 90% + Clustered at 40% identity | `ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster40/` | `prottrans_train_nored_90%_cluster40_standardized.hdf5` | `prottrans_val_nored_90%_cluster40_standardized.hdf5` | `results_models_nored_90%_cluster40/` | `lgbm_gpu_study_prottrans_groupclusters_nored_90%_cluster40` | `lgbm_optuna_prottrans_groupclusters_nored_90%_cluster40.db` | `best_lgbm_params_prottrans_groupclusters_nored_90%_cluster40.json` |
| Mordred + ProtTrans | MLP_optuna.py / MLP | Base | Standard split using unique protein and ligand IDs | `ViralBindPredict/splits_prottrans_output_standardized/` | `prottrans_train_standardized.hdf5` | `prottrans_val_standardized.hdf5` | `results_models/` | `mlp_study_prottrans_clusters` | `optuna_mlp_prottrans_clusters.db` | `best_mlp_params_prottrans_clusters.json` |
| Mordred + ProtTrans | MLP_optuna.py / MLP | Cluster 90% | Scaffold split and clustering at 90% sequence identity | `ViralBindPredict/splits_prottrans_output_standardized_90%/` | `prottrans_train_90%_standardized.hdf5` | `prottrans_val_90%_standardized.hdf5` | `results_models_90%/` | `mlp_study_prottrans_clusters_90%` | `optuna_mlp_prottrans_clusters_90%.db` | `best_mlp_params_prottrans_clusters_90%.json` |
| Mordred + ProtTrans | MLP_optuna.py / MLP | Cluster 40% | Scaffold split and clustering at 40% sequence identity | `ViralBindPredict/splits_prottrans_output_standardized_40%/` | `prottrans_train_40%_standardized.hdf5` | `prottrans_val_40%_standardized.hdf5` | `results_models_40%/` | `mlp_study_prottrans_clusters_40%` | `optuna_mlp_prottrans_clusters_40%.db` | `best_mlp_params_prottrans_clusters_40%.json` |
| Mordred + ProtTrans | MLP_optuna.py / MLP | NoRed 90% + Cluster 90% | Scaffold split and Non-redundant 90% + Clustered at 90% identity | `ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster90/` | `prottrans_train_nored_90%_cluster90_standardized.hdf5` | `prottrans_val_nored_90%_cluster90_standardized.hdf5` | `results_models_nored_90%_cluster90/` | `mlp_study_prottrans_nored_90%_cluster90` | `optuna_mlp_prottrans_clusters_nored_90%_cluster90.db` | `best_mlp_params_prottrans_clusters_nored_90%_cluster90.json` |
| Mordred + ProtTrans | MLP_optuna.py / MLP | NoRed 90% + Cluster 40% | Scaffold split and Non-redundant 90% + Clustered at 40% identity | `ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster40/` | `prottrans_train_nored_90%_cluster40_standardized.hdf5` | `prottrans_val_nored_90%_cluster40_standardized.hdf5` | `results_models_nored_90%_cluster40/` | `mlp_study_prottrans_nored_90%_cluster40` | `optuna_mlp_prottrans_clusters_nored_90%_cluster40.db` | `best_mlp_params_prottrans_clusters_nored_90%_cluster40.json` |

```bash 
# Notes:
# --train corresponds to the Input Folder + Train subset.
# --val corresponds to the Input Folder + Val subset.
# --results-dir defines the Output Folder where models and optimized hyperparameters will be saved.
# --study-name specifies the Optuna Study Name used for tracking optimization runs.
# --db-name sets the Optuna Database Name for storing results.
# --best-json-name defines the output JSON file name containing the best-found hyperparameters.
# --n-trials determines the number of Optuna trials to execute during optimization.

# Example to optimize MLP - Mordred + ESM2 - 90% Cluster
python MLP_optuna.py --train ViralBindPredict/splits_esm2_output_standardized_90%/esm2_train_90%_standardized.hdf5 --val ViralBindPredict/splits_esm2_output_standardized_90%/esm2_val_90%_standardized.hdf5 --results-dir results_models_90%/ --study-name mlp_study_esm2_clusters_90% --db-name optuna_mlp_esm2_clusters_90%.db --best-json-name best_mlp_params_esm2_clusters_90%.json --n-trials 200

# Example to optimize MLP - Mordred + Prottrans - 90% Cluster
python MLP_optuna.py --train ViralBindPredict/splits_prottrans_output_standardized_90%/prottrans_train_90%_standardized.hdf5 --val ViralBindPredict/splits_prottrans_output_standardized_90%/prottrans_val_90%_standardized.hdf5 --results-dir results_models_90%/ --study-name mlp_study_prottrans_clusters_90% --db-name optuna_mlp_prottrans_clusters_90%.db --best-json-name best_mlp_params_prottrans_clusters_90%.json --n-trials 200

# Example to optimize MLP - Mordred + ESM2 - NoRed 90% + Cluster 90%
python MLP_optuna.py --train ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/esm2_train_nored_90%_cluster90_standardized.hdf5 --val ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/esm2_val_nored_90%_cluster90_standardized.hdf5 --results-dir results_models_nored_90%_cluster90/ --study-name mlp_study_esm2_nored_90%_cluster90 --db-name optuna_mlp_esm2_clusters_nored_90%_cluster90.db --best-json-name best_mlp_params_esm2_clusters_nored_90%_cluster90.json --n-trials 200

# Example to optimize LGBM - Mordred + ESM2 - NoRed 90% + Cluster 90%
python LGBM_optuna.py --train ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/esm2_train_nored_90%_cluster90_standardized.hdf5 --val ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/esm2_val_nored_90%_cluster90_standardized.hdf5 --results-dir results_models_nored_90%_cluster90/ --study-name lgbm_gpu_study_esm2_groupclusters_red_90%_cluster90 --db-name lgbm_optuna_esm2_groupclusters_red_90%_cluster90.db --best-json-name best_lgbm_params_esm2_clusters_nored_90%_cluster90.json --n-trials 200

# For the remaining configurations, please refer to the table and use the corresponding input files.
```

17. Train MLP and LightGBM using best hyperparameters found during Optuna optimization

> Note: Before running this step, make sure you have the HDF5 files required for training, validation, and evaluation. These can be generated by following the pipeline provided in this repository, which includes the ViralBindPredict Database creation and the Model Optimization and Training steps. However, the input HDF5 files and the JSON files containing the best hyperparameters used for training and evaluating all models are also available at [link](https://1drv.ms/f/c/6645cc99a95f711d/EtRIfMI6bSxBuY5PVGIRtgUBL6lfH1W9iuBLVbHXlwd8Uw?e=EinOKH) in the corresponding ViralBindPredict/splits_*/ and results_*/ folders.

```bash

# --- MLP ---

# Train MLP - Mordred + ESM2 - Base
python MLP_besthyperparameters.py   --train ViralBindPredict/splits_esm2_output_standardized/esm2_train_standardized.hdf5   --val   ViralBindPredict/splits_esm2_output_standardized/esm2_val_standardized.hdf5   --eval "Test Set=ViralBindPredict/splits_esm2_output_standardized/esm2_test_standardized.hdf5"   --eval "Blind Set=ViralBindPredict/splits_esm2_output_standardized/esm2_blind_standardized.hdf5"   --eval "Blind Protein Set=ViralBindPredict/splits_esm2_output_standardized/esm2_blind_protein_standardized.hdf5"   --eval "Blind Ligand Set=ViralBindPredict/splits_esm2_output_standardized/esm2_blind_ligand_standardized.hdf5"   --params  results_models/best_mlp_params_esm2_clusters.json  --results-dir results_models  --model-name mlp_esm2

# Train MLP - Mordred + Prottrans - Base
python MLP_besthyperparameters.py   --train ViralBindPredict/splits_prottrans_output_standardized/prottrans_train_standardized.hdf5   --val   ViralBindPredict/splits_prottrans_output_standardized/prottrans_val_standardized.hdf5   --eval "Test Set=ViralBindPredict/splits_prottrans_output_standardized/prottrans_test_standardized.hdf5"   --eval "Blind Set=ViralBindPredict/splits_prottrans_output_standardized/prottrans_blind_standardized.hdf5"   --eval "Blind Protein Set=ViralBindPredict/splits_prottrans_output_standardized/prottrans_blind_protein_standardized.hdf5"   --eval "Blind Ligand Set=ViralBindPredict/splits_prottrans_output_standardized/prottrans_blind_ligand_standardized.hdf5"   --params  results_models/best_mlp_params_prottrans_clusters.json --results-dir results_models  --model-name mlp_prottrans

# Train MLP - Mordred + ESM2 - Cluster 90%
python MLP_besthyperparameters.py   --train ViralBindPredict/splits_esm2_output_standardized_90%/esm2_train_90%_standardized.hdf5   --val   ViralBindPredict/splits_esm2_output_standardized_90%/esm2_val_90%_standardized.hdf5   --eval "Test Set=ViralBindPredict/splits_esm2_output_standardized_90%/esm2_test_90%_standardized.hdf5"   --eval "Blind Set=ViralBindPredict/splits_esm2_output_standardized_90%/esm2_blind_90%_standardized.hdf5"   --eval "Blind Protein Set=ViralBindPredict/splits_esm2_output_standardized_90%/esm2_blind_protein_90%_standardized.hdf5"   --eval "Blind Ligand Set=ViralBindPredict/splits_esm2_output_standardized_90%/esm2_blind_ligand_90%_standardized.hdf5"   --params  results_models_90%/best_mlp_params_esm2_clusters_90%.json  --results-dir results_models_90%   --model-name mlp_esm2_90%

# Train MLP - Mordred + Prottrans - Cluster 90%
python MLP_besthyperparameters.py   --train ViralBindPredict/splits_prottrans_output_standardized_90%/prottrans_train_90%_standardized.hdf5   --val   ViralBindPredict/splits_prottrans_output_standardized_90%/prottrans_val_90%_standardized.hdf5   --eval "Test Set=ViralBindPredict/splits_prottrans_output_standardized_90%/prottrans_test_90%_standardized.hdf5"   --eval "Blind Set=ViralBindPredict/splits_prottrans_output_standardized_90%/prottrans_blind_90%_standardized.hdf5"   --eval "Blind Protein Set=ViralBindPredict/splits_prottrans_output_standardized_90%/prottrans_blind_protein_90%_standardized.hdf5"   --eval "Blind Ligand Set=ViralBindPredict/splits_prottrans_output_standardized_90%/prottrans_blind_ligand_90%_standardized.hdf5"   --params  results_models_90%/best_mlp_params_prottrans_clusters_90%.json  --results-dir results_models_90%   --model-name mlp_prottrans_90%

# Train MLP - Mordred + ESM2 - Cluster 40%
python MLP_besthyperparameters.py   --train ViralBindPredict/splits_esm2_output_standardized_40%/esm2_train_40%_standardized.hdf5   --val   ViralBindPredict/splits_esm2_output_standardized_40%/esm2_val_40%_standardized.hdf5   --eval "Test Set=ViralBindPredict/splits_esm2_output_standardized_40%/esm2_test_40%_standardized.hdf5"   --eval "Blind Set=ViralBindPredict/splits_esm2_output_standardized_40%/esm2_blind_40%_standardized.hdf5"   --eval "Blind Protein Set=ViralBindPredict/splits_esm2_output_standardized_40%/esm2_blind_protein_40%_standardized.hdf5"   --eval "Blind Ligand Set=ViralBindPredict/splits_esm2_output_standardized_40%/esm2_blind_ligand_40%_standardized.hdf5"   --params  results_models_40%/best_mlp_params_esm2_clusters_40%.json  --results-dir results_models_40%   --model-name mlp_esm2_40%

# Train MLP - Mordred + Prottrans - Cluster 40%
python MLP_besthyperparameters.py   --train ViralBindPredict/splits_prottrans_output_standardized_40%/prottrans_train_40%_standardized.hdf5   --val   ViralBindPredict/splits_prottrans_output_standardized_40%/prottrans_val_40%_standardized.hdf5   --eval "Test Set=ViralBindPredict/splits_prottrans_output_standardized_40%/prottrans_test_40%_standardized.hdf5"   --eval "Blind Set=ViralBindPredict/splits_prottrans_output_standardized_40%/prottrans_blind_40%_standardized.hdf5"   --eval "Blind Protein Set=ViralBindPredict/splits_prottrans_output_standardized_40%/prottrans_blind_protein_40%_standardized.hdf5"   --eval "Blind Ligand Set=ViralBindPredict/splits_prottrans_output_standardized_40%/prottrans_blind_ligand_40%_standardized.hdf5"   --params  results_models_40%/best_mlp_params_prottrans_clusters_40%.json  --results-dir results_models_40%   --model-name mlp_prottrans_40%

# Train MLP - Mordred + ESM2 - NoRed 90% + Cluster 90%
python MLP_besthyperparameters.py   --train ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/esm2_train_nored_90%_cluster90_standardized.hdf5   --val   ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/esm2_val_nored_90%_cluster90_standardized.hdf5   --eval "Test Set=ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/esm2_test_nored_90%_cluster90_standardized.hdf5"   --eval "Blind Set=ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/esm2_blind_nored_90%_cluster90_standardized.hdf5"   --eval "Blind Protein Set=ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/esm2_blind_protein_nored_90%_cluster90_standardized.hdf5"   --eval "Blind Ligand Set=ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/esm2_blind_ligand_nored_90%_cluster90_standardized.hdf5"   --params  results_models_nored_90%_cluster90/best_mlp_params_esm2_clusters_nored_90%_cluster90.json  --results-dir results_models_nored_90%_cluster90   --model-name mlp_esm2_nored_90%_cluster90

# Train MLP - Mordred + Prottrans - NoRed 90% + Cluster 90%
python MLP_besthyperparameters.py   --train ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster90/prottrans_train_nored_90%_cluster90_standardized.hdf5   --val   ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster90/prottrans_val_nored_90%_cluster90_standardized.hdf5   --eval "Test Set=ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster90/prottrans_test_nored_90%_cluster90_standardized.hdf5"   --eval "Blind Set=ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster90/prottrans_blind_nored_90%_cluster90_standardized.hdf5"   --eval "Blind Protein Set=ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster90/prottrans_blind_protein_nored_90%_cluster90_standardized.hdf5"   --eval "Blind Ligand Set=ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster90/prottrans_blind_ligand_nored_90%_cluster90_standardized.hdf5"   --params  results_models_nored_90%_cluster90/best_mlp_params_prottrans_clusters_nored_90%_cluster90.json  --results-dir results_models_nored_90%_cluster90   --model-name mlp_prottrans_nored_90%_cluster90

# Train MLP - Mordred + ESM2 - NoRed 90% + Cluster 40%
python MLP_besthyperparameters.py   --train ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster40/esm2_train_nored_90%_cluster40_standardized.hdf5   --val   ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster40/esm2_val_nored_90%_cluster40_standardized.hdf5   --eval "Test Set=ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster40/esm2_test_nored_90%_cluster40_standardized.hdf5"   --eval "Blind Set=ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster40/esm2_blind_nored_90%_cluster40_standardized.hdf5"   --eval "Blind Protein Set=ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster40/esm2_blind_protein_nored_90%_cluster40_standardized.hdf5"   --eval "Blind Ligand Set=ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster40/esm2_blind_ligand_nored_90%_cluster40_standardized.hdf5"   --params  results_models_nored_90%_cluster40/best_mlp_params_esm2_clusters_nored_90%_cluster40.json  --results-dir results_models_nored_90%_cluster40   --model-name mlp_esm2_nored_90%_cluster40

# Train MLP - Mordred + Prottrans - NoRed 90% + Cluster 40%
python MLP_besthyperparameters.py   --train ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster40/prottrans_train_nored_90%_cluster40_standardized.hdf5   --val   ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster40/prottrans_val_nored_90%_cluster40_standardized.hdf5   --eval "Test Set=ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster40/prottrans_test_nored_90%_cluster40_standardized.hdf5"   --eval "Blind Set=ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster40/prottrans_blind_nored_90%_cluster40_standardized.hdf5"   --eval "Blind Protein Set=ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster40/prottrans_blind_protein_nored_90%_cluster40_standardized.hdf5"   --eval "Blind Ligand Set=ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster40/prottrans_blind_ligand_nored_90%_cluster40_standardized.hdf5"   --params  results_models_nored_90%_cluster40/best_mlp_params_prottrans_clusters_nored_90%_cluster40.json  --results-dir results_models_nored_90%_cluster40   --model-name mlp_prottrans_nored_90%_cluster40


# --- LIGHTGBM ---

# Train LGBM - Mordred + ESM2 - Base
python LGBM_besthyperparameters.py   --train ViralBindPredict/splits_esm2_output_standardized/esm2_train_standardized.hdf5   --val   ViralBindPredict/splits_esm2_output_standardized/esm2_val_standardized.hdf5   --eval "Test Set=ViralBindPredict/splits_esm2_output_standardized/esm2_test_standardized.hdf5"   --eval "Blind Set=ViralBindPredict/splits_esm2_output_standardized/esm2_blind_standardized.hdf5"   --eval "Blind Protein Set=ViralBindPredict/splits_esm2_output_standardized/esm2_blind_protein_standardized.hdf5"   --eval "Blind Ligand Set=ViralBindPredict/splits_esm2_output_standardized/esm2_blind_ligand_standardized.hdf5"   --params  results_models/best_lgbm_params_esm2_groupclusters.json  --results-dir results_models  --model-name lgbm_esm2

# Train LGBM - Mordred + Prottrans - Base
python LGBM_besthyperparameters.py   --train ViralBindPredict/splits_prottrans_output_standardized/prottrans_train_standardized.hdf5   --val   ViralBindPredict/splits_prottrans_output_standardized/prottrans_val_standardized.hdf5   --eval "Test Set=ViralBindPredict/splits_prottrans_output_standardized/prottrans_test_standardized.hdf5"   --eval "Blind Set=ViralBindPredict/splits_prottrans_output_standardized/prottrans_blind_standardized.hdf5"   --eval "Blind Protein Set=ViralBindPredict/splits_prottrans_output_standardized/prottrans_blind_protein_standardized.hdf5"   --eval "Blind Ligand Set=ViralBindPredict/splits_prottrans_output_standardized/prottrans_blind_ligand_standardized.hdf5"   --params  results_models/best_lgbm_params_prottrans_groupclusters.json --results-dir results_models  --model-name lgbm_prottrans

# Train LGBM - Mordred + ESM2 - Cluster 90%
python LGBM_besthyperparameters.py   --train ViralBindPredict/splits_esm2_output_standardized_90%/esm2_train_90%_standardized.hdf5   --val   ViralBindPredict/splits_esm2_output_standardized_90%/esm2_val_90%_standardized.hdf5   --eval "Test Set=ViralBindPredict/splits_esm2_output_standardized_90%/esm2_test_90%_standardized.hdf5"   --eval "Blind Set=ViralBindPredict/splits_esm2_output_standardized_90%/esm2_blind_90%_standardized.hdf5"   --eval "Blind Protein Set=ViralBindPredict/splits_esm2_output_standardized_90%/esm2_blind_protein_90%_standardized.hdf5"   --eval "Blind Ligand Set=ViralBindPredict/splits_esm2_output_standardized_90%/esm2_blind_ligand_90%_standardized.hdf5"   --params  results_models_90%/best_lgbm_params_esm2_groupclusters_90%.json  --results-dir results_models_90%   --model-name lgbm_esm2_90%

# Train LGBM - Mordred + Prottrans - Cluster 90%
python LGBM_besthyperparameters.py   --train ViralBindPredict/splits_prottrans_output_standardized_90%/prottrans_train_90%_standardized.hdf5   --val   ViralBindPredict/splits_prottrans_output_standardized_90%/prottrans_val_90%_standardized.hdf5   --eval "Test Set=ViralBindPredict/splits_prottrans_output_standardized_90%/prottrans_test_90%_standardized.hdf5"   --eval "Blind Set=ViralBindPredict/splits_prottrans_output_standardized_90%/prottrans_blind_90%_standardized.hdf5"   --eval "Blind Protein Set=ViralBindPredict/splits_prottrans_output_standardized_90%/prottrans_blind_protein_90%_standardized.hdf5"   --eval "Blind Ligand Set=ViralBindPredict/splits_prottrans_output_standardized_90%/prottrans_blind_ligand_90%_standardized.hdf5"   --params  results_models_90%/best_lgbm_params_prottrans_groupclusters_90%.json  --results-dir results_models_90%   --model-name lgbm_prottrans_90%

# Train LGBM - Mordred + ESM2 - Cluster 40%
python LGBM_besthyperparameters.py   --train ViralBindPredict/splits_esm2_output_standardized_40%/esm2_train_40%_standardized.hdf5   --val   ViralBindPredict/splits_esm2_output_standardized_40%/esm2_val_40%_standardized.hdf5   --eval "Test Set=ViralBindPredict/splits_esm2_output_standardized_40%/esm2_test_40%_standardized.hdf5"   --eval "Blind Set=ViralBindPredict/splits_esm2_output_standardized_40%/esm2_blind_40%_standardized.hdf5"   --eval "Blind Protein Set=ViralBindPredict/splits_esm2_output_standardized_40%/esm2_blind_protein_40%_standardized.hdf5"   --eval "Blind Ligand Set=ViralBindPredict/splits_esm2_output_standardized_40%/esm2_blind_ligand_40%_standardized.hdf5"   --params  results_models_40%/best_lgbm_params_esm2_groupclusters_40%.json  --results-dir results_models_40%   --model-name lgbm_esm2_40%

# Train LGBM - Mordred + Prottrans - Cluster 40%
python LGBM_besthyperparameters.py   --train ViralBindPredict/splits_prottrans_output_standardized_40%/prottrans_train_40%_standardized.hdf5   --val   ViralBindPredict/splits_prottrans_output_standardized_40%/prottrans_val_40%_standardized.hdf5   --eval "Test Set=ViralBindPredict/splits_prottrans_output_standardized_40%/prottrans_test_40%_standardized.hdf5"   --eval "Blind Set=ViralBindPredict/splits_prottrans_output_standardized_40%/prottrans_blind_40%_standardized.hdf5"   --eval "Blind Protein Set=ViralBindPredict/splits_prottrans_output_standardized_40%/prottrans_blind_protein_40%_standardized.hdf5"   --eval "Blind Ligand Set=ViralBindPredict/splits_prottrans_output_standardized_40%/prottrans_blind_ligand_40%_standardized.hdf5"   --params  results_models_40%/best_lgbm_params_prottrans_groupclusters_40%.json  --results-dir results_models_40%   --model-name lgbm_prottrans_40%

# Train LGBM - Mordred + ESM2 - NoRed 90% + Cluster 90%
python LGBM_besthyperparameters.py --train ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/esm2_train_nored_90%_cluster90_standardized.hdf5 --val ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/esm2_val_nored_90%_cluster90_standardized.hdf5 --eval "Test Set=ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/esm2_test_nored_90%_cluster90_standardized.hdf5" --eval "Blind Set=ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/esm2_blind_nored_90%_cluster90_standardized.hdf5" --eval "Blind Protein Set=ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/esm2_blind_protein_nored_90%_cluster90_standardized.hdf5" --eval "Blind Ligand Set=ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/esm2_blind_ligand_nored_90%_cluster90_standardized.hdf5" --params results_models_nored_90%_cluster90/best_lgbm_params_esm2_groupclusters_red_90%_cluster90.json --results-dir results_models_nored_90%_cluster90 --model-name lgbm_esm2_nored_90%_cluster90

# Train LGBM - Mordred + Prottrans - NoRed 90% + Cluster 90%
python LGBM_besthyperparameters.py --train ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster90/prottrans_train_nored_90%_cluster90_standardized.hdf5 --val ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster90/prottrans_val_nored_90%_cluster90_standardized.hdf5 --eval "Test Set=ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster90/prottrans_test_nored_90%_cluster90_standardized.hdf5" --eval "Blind Set=ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster90/prottrans_blind_nored_90%_cluster90_standardized.hdf5" --eval "Blind Protein Set=ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster90/prottrans_blind_protein_nored_90%_cluster90_standardized.hdf5" --eval "Blind Ligand Set=ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster90/prottrans_blind_ligand_nored_90%_cluster90_standardized.hdf5" --params results_models_nored_90%_cluster90/best_lgbm_params_prottrans_groupclusters_red_90%_cluster90.json --results-dir results_models_nored_90%_cluster90 --model-name lgbm_prottrans_nored_90%_cluster90

# Train LGBM - Mordred + ESM2 - NoRed 90% + Cluster 40%
python LGBM_besthyperparameters.py --train ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster40/esm2_train_nored_90%_cluster40_standardized.hdf5 --val ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster40/esm2_val_nored_90%_cluster40_standardized.hdf5 --eval "Test Set=ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster40/esm2_test_nored_90%_cluster40_standardized.hdf5" --eval "Blind Set=ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster40/esm2_blind_nored_90%_cluster40_standardized.hdf5" --eval "Blind Protein Set=ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster40/esm2_blind_protein_nored_90%_cluster40_standardized.hdf5" --eval "Blind Ligand Set=ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster40/esm2_blind_ligand_nored_90%_cluster40_standardized.hdf5" --params results_models_nored_90%_cluster40/best_lgbm_params_esm2_groupclusters_nored_90%_cluster40.json --results-dir results_models_nored_90%_cluster40 --model-name lgbm_esm2_nored_90%_cluster40

# Train LGBM - Mordred + Prottrans - NoRed 90% + Cluster 40%
python LGBM_besthyperparameters.py --train ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster40/prottrans_train_nored_90%_cluster40_standardized.hdf5 --val ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster40/prottrans_val_nored_90%_cluster40_standardized.hdf5 --eval "Test Set=ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster40/prottrans_test_nored_90%_cluster40_standardized.hdf5" --eval "Blind Set=ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster40/prottrans_blind_nored_90%_cluster40_standardized.hdf5" --eval "Blind Protein Set=ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster40/prottrans_blind_protein_nored_90%_cluster40_standardized.hdf5" --eval "Blind Ligand Set=ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster40/prottrans_blind_ligand_nored_90%_cluster40_standardized.hdf5" --params results_models_nored_90%_cluster40/best_lgbm_params_prottrans_groupclusters_nored_90%_cluster40.json --results-dir results_models_nored_90%_cluster40 --model-name lgbm_prottrans_nored_90%_cluster40

```


## New Prediction

> Note: This table summarizes the input requirements for making new predictions using the MLP models trained in the previous steps. To perform a prediction, you need an Excel file formatted like the example provided in new_prediction/new_prediction.xlsx. Make sure the file includes all required columns, including SMILES and sequence, corresponding to the complex you wish to predict. Modify the entries as needed, and refer to the examples below for guidance. If you only wish to perform predictions without running the entire pipeline, all trained models, scalers, and reference HDF5 files are already provided in this repository or available through the [link](https://1drv.ms/f/c/6645cc99a95f711d/EtRIfMI6bSxBuY5PVGIRtgUBL6lfH1W9iuBLVbHXlwd8Uw?e=EinOKH). These files are located in the corresponding ViralBindPredict/, ViralBindPredict/splits_*/, and results_*/ folders.
Please download any missing files — especially the reference HDF5 files (Ref h5) — from the appropriate directory in the [link](https://1drv.ms/f/c/6645cc99a95f711d/EhugUMOUp49Hqx0gKJi-lDUBJRa0c0L5dzGBKvYLrgdnOg?e=arex95), place them in their designated ViralBindPredict/ folder, and then run the commands as shown in the examples below.

| Dataset | Experiment | Ref h5 | Protein Scaler | Ligand Scaler | Model | Output Directory |
|:--|:--|:--|:--|:--|:--|:--|
| Mordred + ESM2 | Base | ViralBindPredict/viralbindpredictDB-esm2-cleaned.hdf5 | ViralBindPredict/splits_esm2_output_standardized/prot_scaler_esm2_base.pkl | ViralBindPredict/splits_esm2_output_standardized/lig_scaler_esm2_base.pkl | results_models/mlp_esm2.pt | esm2_base |
| Mordred + Prottrans | Base | ViralBindPredict/viralbindpredictDB-prottrans-cleaned.hdf5 | ViralBindPredict/splits_prottrans_output_standardized/prot_scaler_prottrans_base.pkl | ViralBindPredict/splits_prottrans_output_standardized/lig_scaler_prottrans_base.pkl | results_models/mlp_prottrans.pt | prottrans_base |
| Mordred + ESM2 | Cluster 90% | ViralBindPredict/viralbindpredictDB-esm2-cleaned.hdf5 | ViralBindPredict/splits_esm2_output_standardized_90%/prot_scaler_esm2_90%.pkl | ViralBindPredict/splits_esm2_output_standardized_90%/lig_scaler_esm2_90%.pkl | results_models_90%/mlp_esm2_90%.pt | esm2_90 |
| Mordred + Prottrans | Cluster 90% | ViralBindPredict/viralbindpredictDB-prottrans-cleaned.hdf5 | ViralBindPredict/splits_prottrans_output_standardized_90%/prot_scaler_prottrans_90%.pkl | ViralBindPredict/splits_prottrans_output_standardized_90%/lig_scaler_prottrans_90%.pkl | results_models_90%/mlp_prottrans_90%.pt | prottrans_90 |
| Mordred + ESM2 | Cluster 40% | ViralBindPredict/viralbindpredictDB-esm2-cleaned.hdf5 | ViralBindPredict/splits_esm2_output_standardized_40%/prot_scaler_esm2_40%.pkl | ViralBindPredict/splits_esm2_output_standardized_40%/lig_scaler_esm2_40%.pkl | results_models_40%/mlp_esm2_40%.pt | esm2_40 |
| Mordred + Prottrans | Cluster 40% | ViralBindPredict/viralbindpredictDB-prottrans-cleaned.hdf5 | ViralBindPredict/splits_prottrans_output_standardized_40%/prot_scaler_prottrans_40%.pkl | ViralBindPredict/splits_prottrans_output_standardized_40%/lig_scaler_prottrans_40%.pkl | results_models_40%/mlp_prottrans_40%.pt | prottrans_40 |
| Mordred + ESM2 |  NoRed 90% + Cluster 90% | ViralBindPredict/viralbindpredictDB-esm2-cleaned_no_red.hdf5 | ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/prot_scaler_esm2_nored90%_cluster90.pkl | ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/lig_scaler_esm2_nored90%_cluster90.pkl | results_models_nored_90%_cluster90/mlp_esm2_nored_90%_cluster90.pt | esm2_nored90%_cluster90 |
| Mordred + Prottrans | NoRed 90% + Cluster 90% | ViralBindPredict/viralbindpredictDB-prottrans-cleaned_no_red.hdf5 | ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster90/prot_scaler_prottrans_nored90%_cluster90.pkl | ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster90/lig_scaler_prottrans_nored90%_cluster90.pkl | results_models_nored_90%_cluster90/mlp_prottrans_nored_90%_cluster90.pt | prottrans_nored90_cluster90 |
| Mordred + ESM2 |  NoRed 90% + Cluster 40% | ViralBindPredict/viralbindpredictDB-esm2-cleaned_no_red.hdf5 | ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster40/prot_scaler_esm2_nored90%_cluster40.pkl | ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster40/lig_scaler_esm2_nored90%_cluster40.pkl | results_models_nored_90%_cluster40/mlp_esm2_nored_90%_cluster40.pt | esm2_nored90%_cluster40 |
| Mordred + Prottrans | NoRed 90% + Cluster 40% | ViralBindPredict/viralbindpredictDB-prottrans-cleaned_no_red.hdf5 | ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster40/prot_scaler_prottrans_90%_cluster40.pkl | ViralBindPredict/splits_prottrans_output_standardized_nored_90%_cluster40/lig_scaler_prottrans_90%_cluster40.pkl | results_models_nored_90%_cluster40/mlp_prottrans_nored_90%_cluster40.pt | prottrans_nored90_cluster40 |


**Notes:**  
- Use **new_prediction_esm2.py** script if the dataset contains Mordred + ESM2 features  
- Use **new_prediction_prottrans.py** script if the dataset contains Mordred + ProtTrans features  

```bash
# --input-file corresponds to the input excel file (new_prediction.xlsx).
# --ref-h5 corresponds to a reference hdf5 file.
# --prot-scaler corresponds to the trained protein scaler.
# --lig-scaler corresponds to the trained ligand scaler.
# --model corresponds to the trained model.
# --outdir corresponds to the output directory.

# Predict using trained MLP - Mordred + ESM2 - Base
python new_prediction_esm2.py   --input-file new_prediction/new_prediction.xlsx   --ref-h5 ViralBindPredict/viralbindpredictDB-esm2-cleaned.hdf5   --prot-scaler ViralBindPredict/splits_esm2_output_standardized/prot_scaler_esm2_base.pkl   --lig-scaler ViralBindPredict/splits_esm2_output_standardized/lig_scaler_esm2_base.pkl   --model results_models/mlp_esm2.pt   --outdir esm2_base

# Predict using trained MLP - Mordred + ESM2 - Cluster 90%
python new_prediction_esm2.py   --input-file new_prediction/new_prediction.xlsx   --ref-h5 ViralBindPredict/viralbindpredictDB-esm2-cleaned.hdf5   --prot-scaler ViralBindPredict/splits_esm2_output_standardized_90%/prot_scaler_esm2_90%.pkl   --lig-scaler ViralBindPredict/splits_esm2_output_standardized_90%/lig_scaler_esm2_90%.pkl   --model results_models_90%/mlp_esm2_90%.pt   --outdir esm2_90

# Predict using trained MLP - Mordred + ESM2 - NoRed 90% + Cluster 90%
python new_prediction_esm2.py   --input-file new_prediction/new_prediction.xlsx   --ref-h5 ViralBindPredict/viralbindpredictDB-esm2-cleaned_no_red.hdf5   --prot-scaler ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/prot_scaler_esm2_nored90%_cluster90.pkl   --lig-scaler ViralBindPredict/splits_esm2_output_standardized_nored_90%_cluster90/lig_scaler_esm2_nored90%_cluster90.pkl   --model results_models_nored_90%_cluster90/mlp_esm2_nored_90%_cluster90.pt   --outdir esm2_nored90%_cluster90

# Predict using trained MLP - Mordred + Prottrans - Cluster 40%
python new_prediction_prottrans.py   --input-file new_prediction/new_prediction.xlsx   --ref-h5 ViralBindPredict/viralbindpredictDB-prottrans-cleaned.hdf5   --prot-scaler ViralBindPredict/splits_prottrans_output_standardized_40%/prot_scaler_40%_prottrans.pkl   --lig-scaler ViralBindPredict/splits_prottrans_output_standardized_40%/lig_scaler_40%_prottrans.pkl   --model results_models_40%/mlp_prottrans_40%.pt   --outdir prottrans_40
```

> Due to intrinsic stochasticity in model training and potential hardware-dependent numerical differences (e.g., GPU/CPU or CUDA variations), minor deviations in results may occur across systems.
