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