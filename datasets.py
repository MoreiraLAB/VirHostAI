from abc import ABC, abstractmethod
import h5py
import numpy as np
import shutil
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from tqdm.auto import tqdm
from typing import List, Union, Optional




class BaseDataset(ABC):
    
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.dataset = h5py.File(self.filepath, 'r')
        root_keys = set(list(self.dataset.keys()))
        root_keys.discard('balanced-batches')
        assert root_keys == set(['interactions', 'ligands', 'proteins']), 'ERROR: the root keys of the dataset must be "interactions", "ligands" and "proteins"'    


    def close(self) -> None:
        # after finishing using a dataset, close it to ensure that any resources associated with the file are properly released
        self.dataset.close()
    


    def clone(self, filepath) -> None:
        shutil.copy2(self.filepath, filepath)
    


    @abstractmethod
    def get_dataloader(self, batch_size: int, shuffle: bool = False, split: Optional[Subset] = None) -> DataLoader:
        pass



    @abstractmethod
    def get_splits_dataloaders(self, batch_size: int, shuffle: bool = False, splits: Optional[List[Union[int, float]]] = None) -> List[Optional[DataLoader]]:
        pass




class InteractionsDataset(BaseDataset, Dataset):

    def __init__(self, filepath: str) -> None:
        BaseDataset.__init__(self, filepath)

        # residue_map[split][global residue index] -> (protein:chain:ligand, local residue index), needed to load batches of residues (batch size = number of residues per batch)
        self.residue_map = {'train': [], 'test': [], 'validation': [], 'dataset': []}

        # chain_map[split][global chain index] -> protein:chain:ligand, needed to load batches of chains (batch size = number of chains per batch)
        self.chain_map = {'train': [], 'test': [], 'validation': [], 'dataset': []}

        chain_interactions = self.dataset['interactions']
        for chain_interaction in tqdm(list(chain_interactions.keys()), desc='iterating over all chain:ligand interactions: ', leave=False):
            split = chain_interactions[chain_interaction].attrs['split'] if 'split' in chain_interactions[chain_interaction].attrs.keys() else 'dataset'
            assert split in ['train', 'test', 'validation', 'dataset'], 'ERROR: if the dataset has splits they must be "train", "test" or "validation"'

            targets = chain_interactions[chain_interaction]['targets']
            assert np.all(np.isin(np.unique(targets), [0, 1])), 'ERROR: the dataset targets must be 0 (not interacting) or 1 (interacting)'

            self.residue_map[split] += [(chain_interaction, i) for i in range(len(targets))]
            self.chain_map[split].append(chain_interaction)
        
        dataset_has_splits = len(self.chain_map['train']) != 0 or len(self.chain_map['test']) != 0 or len(self.chain_map['validation']) != 0
        assert (dataset_has_splits and len(self.chain_map['dataset']) == 0) or (not dataset_has_splits and len(self.chain_map['dataset']) != 0), 'ERROR: either all chain:ligand interactions belong to a split or no chain:ligand interaction belongs to a split'

        # if the dataset has splits then update the split "dataset" (no split, dataset as a whole)
        if dataset_has_splits:
            for split in ['train', 'test', 'validation']:
                self.residue_map['dataset'] += self.residue_map[split]
                self.chain_map['dataset'] += self.chain_map[split]

        # set the current split to "dataset" (no split, dataset as a whole)
        self.set_split()

        # set the current granularity to "residue" (batch size = number of residues per batch)
        self.residue_granularity = True
    


    def __len__(self) -> int:
        # if the current granularity is set to "residue", then return the number of residue:ligand interactions in the current split
        if self.residue_granularity:
            return len(self.residue_map[self.split])
        # if the current granularity is set to "chain", then return the number of chain:ligand interactions in the current split
        else:
            return len(self.chain_map[self.split])



    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # if the current granularity is set to "residue"
        if self.residue_granularity:
            # get (protein:chain:ligand, local residue index)
            residue = self.residue_map[self.split][index]
            # get chain:ligand interaction
            interaction = self.dataset['interactions'][residue[0]]

            # get residue in chain:ligand interaction at index [local residue index] (because it is necessarily only 1 row, h5py returns a numpy array)
            protein_features = self.dataset[interaction.attrs['protein']]['features'][residue[1]]
            # get ligand in chain:ligand interaction (because it is necessarily only 1 row, h5py returns a numpy array)
            ligand_features = self.dataset[interaction.attrs['ligand']]['features'][0]

            # shape: [number of features (number of protein features + number of ligand features)] (1 dimension)
            features = np.concatenate((protein_features, ligand_features), axis=0)

            # shape: [1] (1 dimension), get target of the residue:ligand interaction
            targets = interaction['targets'][residue[1]]
        # if the current granularity is set to "chain"
        else:
            # get protein:chain:ligand
            chain = self.chain_map[self.split][index]
            # get chain:ligand interaction
            interaction = self.dataset['interactions'][chain]

            # get residues in chain:ligand interaction (because it is not necessarily only 1 row, h5py returns a h5py dataset, therefore, conversion to numpy array is needed)
            protein_features = np.array(self.dataset[interaction.attrs['protein']]['features'])
            # get ligand in chain:ligand interaction (because it is not necessarily only 1 row, h5py returns a h5py dataset) and repeat once per residue in chain:ligand interaction
            ligand_features = np.repeat(self.dataset[interaction.attrs['ligand']]['features'], protein_features.shape[0], axis=0)

            # shape: [number of residues in chain:ligand interaction, number of features (number of protein features + number of ligand features)] (2 dimensions)
            features = np.concatenate((protein_features, ligand_features), axis=1)

            # shape: [number of residues in chain:ligand interaction, 1] (2 dimensions), get targets of each residue:ligand interaction in chain:ligand interaction
            targets = np.array(interaction['targets'])

        return torch.tensor(features), torch.tensor(targets)
    


    def get_dataloader(self, batch_size: int, shuffle: bool = False, split: Optional[Subset] = None) -> DataLoader:
        # if the current granularity is set to "residue" (batch size = number of residues per batch)
        if self.residue_granularity:
            # here, each item added to a batch is ([number of features], [1]) 
            # therefore they are both already in 1 dimension and are automatically stacked vertically so that each batch becomes 
            # ([number of residue:ligand interactions of the batch, number of features], [number of residue:ligand interactions of the batch, 1]) (2 dimensions)
            # unlike when the current granularity is set to "chain" where the first dimension of the features returned 
            # is not only variable across different chains but also adds 1 dimension to each item of the batch
            dataloader = DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle) if split is None else DataLoader(dataset=split, batch_size=batch_size, shuffle=shuffle)
        # if the current granularity is set to "chain" (batch size = number of chains per batch)
        else:
            # here, each item added to a batch is ([number of residues in chain:ligand interaction, number of features], [number of residues in chain:ligand interaction, 1])
            # therefore we need to manually stack them vertically so that each batch becomes
            # ([number of residues in chain:ligand interactions of the batch, number of features], [number of residues in chain:ligand interactions of the batch, 1]) (2 dimensions)
            def collate_fn(batch):
                features, targets = zip(*batch)
                return torch.cat(features, dim=0), torch.cat(targets, dim=0)
            dataloader = DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn) if split is None else DataLoader(dataset=split, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        return dataloader
    


    def get_splits_dataloaders(self, batch_size: int, shuffle: bool = False, splits: Optional[List[Union[int, float]]] = None) -> List[Optional[DataLoader]]:
        assert splits is not None or len(self.chain_map['train']) != 0 or len(self.chain_map['test']) != 0 or len(self.chain_map['validation']) != 0, 'ERROR: to get the dataloaders of each split, either the dataset has splits or a list of new splits is passed ("splits" argument)'
        
        dataloaders = []
        if splits is not None:
            splits_subsets = random_split(dataset=self, lengths=[split for split in splits if split != 0])
            for split in splits:
                dataloaders.append(self.get_dataloader(batch_size, shuffle, splits_subsets.pop(0))) if split != 0 else dataloaders.append(None)
        else:
            current_split = self.split

            self.set_split('train')
            dataloaders.append(self.get_dataloader(batch_size, shuffle)) if len(self.chain_map['train']) != 0 else dataloaders.append(None)

            self.set_split('validation')
            dataloaders.append(self.get_dataloader(batch_size, shuffle)) if len(self.chain_map['validation']) != 0 else dataloaders.append(None)

            self.set_split('test')
            dataloaders.append(self.get_dataloader(batch_size, shuffle)) if len(self.chain_map['test']) != 0 else dataloaders.append(None)

            self.set_split(current_split)
        
        return dataloaders



    def set_split(self, split: str = 'dataset') -> None:
        # assert that the passed split is valid and is not empty
        assert split in self.chain_map, f'ERROR: {split} does not exist'
        assert len(self.chain_map[split]) != 0, f'ERROR: {split} is empty'

        # set split to passed split ("train", "test", "validation" or "dataset" (no split, dataset as a whole))
        self.split = split
    


    def set_granularity(self) -> None:
        # set residue_granularity (True = "residue", False = "chain")
        self.residue_granularity = not self.residue_granularity




class ProteinsDataset(BaseDataset, Dataset):

    def __init__(self, filepath: str) -> None:
        BaseDataset.__init__(self, filepath)

        self.residue_map = []
        self.chain_map = []

        proteins = self.dataset['proteins']
        for protein in tqdm(list(proteins.keys()), desc='iterating over all proteins: ', leave=False):
            for chain in list(proteins[protein].keys()):
                residues = proteins[protein][chain]['features']

                self.residue_map += [(f'proteins/{protein}/{chain}', i) for i in range(len(residues))]
                self.chain_map.append(f'proteins/{protein}/{chain}')

        self.residue_granularity = True
    


    def __len__(self) -> int:
        if self.residue_granularity:
            return len(self.residue_map)
        else:
            return len(self.chain_map)



    def __getitem__(self, index: int) -> torch.Tensor:
        if self.residue_granularity:
            residue = self.residue_map[index]
            protein_features = self.dataset[residue[0]]['features'][residue[1]]
        else:
            chain = self.chain_map[index]
            protein_features = np.array(self.dataset[chain]['features'])
        
        return torch.tensor(protein_features)
    


    def get_dataloader(self, batch_size: int, shuffle: bool = False, split: Optional[Subset] = None) -> DataLoader:
        if self.residue_granularity:
            dataloader = DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle) if split is None else DataLoader(dataset=split, batch_size=batch_size, shuffle=shuffle)
        else:
            def collate_fn(batch):
                return torch.cat(batch, dim=0)
            dataloader = DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn) if split is None else DataLoader(dataset=split, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        return dataloader
    


    def get_splits_dataloaders(self, batch_size: int, shuffle: bool = False, splits: Optional[List[Union[int, float]]] = None) -> List[Optional[DataLoader]]:
        assert splits is not None, 'ERROR: to get the dataloaders of each split, a list of splits must be passed ("splits" argument)'
        
        dataloaders = []
        splits_subsets = random_split(dataset=self, lengths=[split for split in splits if split != 0])
        for split in splits:
            dataloaders.append(self.get_dataloader(batch_size, shuffle, splits_subsets.pop(0))) if split != 0 else dataloaders.append(None)
        
        return dataloaders



    def set_granularity(self) -> None:
        self.residue_granularity = not self.residue_granularity




class LigandsDataset(BaseDataset, Dataset):

    def __init__(self, filepath: str) -> None:
        BaseDataset.__init__(self, filepath)

        self.ligand_map = []

        ligands = self.dataset['ligands']
        for ligand in tqdm(list(ligands.keys()), desc='iterating over all ligands: ', leave=False):
            self.ligand_map.append(f'ligands/{ligand}')
    


    def __len__(self) -> int:
        return len(self.ligand_map)



    def __getitem__(self, index: int) -> torch.Tensor:
        ligand = self.ligand_map[index]
        ligand_features = self.dataset[ligand]['features'][0]

        return torch.tensor(ligand_features)
    


    def get_dataloader(self, batch_size: int, shuffle: bool = False, split: Optional[Subset] = None) -> DataLoader:
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle) if split is None else DataLoader(dataset=split, batch_size=batch_size, shuffle=shuffle)
    


    def get_splits_dataloaders(self, batch_size: int, shuffle: bool = False, splits: Optional[List[Union[int, float]]] = None) -> List[Optional[DataLoader]]:
        assert splits is not None, 'ERROR: to get the dataloaders of each split, a list of splits must be passed ("splits" argument)'
        
        dataloaders = []
        splits_subsets = random_split(dataset=self, lengths=[split for split in splits if split != 0])
        for split in splits:
            dataloaders.append(self.get_dataloader(batch_size, shuffle, splits_subsets.pop(0))) if split != 0 else dataloaders.append(None)
        
        return dataloaders




class BalancedInteractionsDataset(InteractionsDataset):
    
    def __init__(self, filepath: str) -> None:
        # necessary to initialize it before calling InteractionsDataset.__init__ because set_split is called there but this class's set_split uses self.batch_map
        self.sample_map = {'train': [], 'test': [], 'validation': [], 'dataset': []}
        self.batch_map = {'train': [], 'test': [], 'validation': [], 'dataset': []}

        InteractionsDataset.__init__(self, filepath)

        assert 'balanced-batches' in list(self.dataset.keys()), 'ERROR: there must exist a root key "balanced-batches" in the dataset'

        batches = self.dataset['balanced-batches']
        for batch in tqdm(list(batches.keys()), desc='iterating over all balanced batches: ', leave=False):
            split = batches[batch].attrs['split'] if 'split' in batches[batch].attrs.keys() else 'dataset'
            assert split in ['train', 'test', 'validation', 'dataset'], 'ERROR: if the dataset has splits they must be "train", "test" or "validation"'

            targets = batches[batch]['targets']
            assert np.all(np.isin(np.unique(targets), [0, 1])), 'ERROR: the dataset targets must be 0 (not interacting) or 1 (interacting)'

            self.sample_map[split] += [(batch, i) for i in range(len(targets))]
            self.batch_map[split].append(batch)
        
        dataset_has_splits = len(self.batch_map['train']) != 0 or len(self.batch_map['test']) != 0 or len(self.batch_map['validation']) != 0
        assert (dataset_has_splits and len(self.batch_map['dataset']) == 0) or (not dataset_has_splits and len(self.batch_map['dataset']) != 0), 'ERROR: either all balanced batches belong to a split or no balanced batch belongs to a split'

        if dataset_has_splits:
            for split in ['train', 'test', 'validation']:
                self.sample_map['dataset'] += self.sample_map[split]
                self.batch_map['dataset'] += self.batch_map[split]
    


    def __len__(self) -> int:
        if self.residue_granularity:
            return len(self.residue_map[self.split]) + len(self.sample_map[self.split])
        else:
            return len(self.chain_map[self.split]) + len(self.batch_map[self.split])
    
    

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.residue_granularity:
            if index < len(self.residue_map[self.split]):
                return InteractionsDataset.__getitem__(self, index)
            else:
                sample = self.sample_map[self.split][index - len(self.residue_map[self.split])]
                batch = self.dataset['balanced-batches'][sample[0]]

                features = batch['features'][sample[1]]

                targets = batch['targets'][sample[1]]
        else:
            if index < len(self.chain_map[self.split]):
                return InteractionsDataset.__getitem__(self, index)
            else:
                batch = self.dataset['balanced-batches'][self.batch_map[self.split][index - len(self.chain_map[self.split])]]

                features = np.array(batch['features'])

                targets = np.array(batch['targets'])

        return torch.tensor(features), torch.tensor(targets)
    


    def get_splits_dataloaders(self, batch_size: int, shuffle: bool = False, splits: Optional[List[Union[int, float]]] = None) -> List[Optional[DataLoader]]:
        assert splits is not None or (len(self.chain_map['train']) + len(self.batch_map['train'])) != 0 or (len(self.chain_map['test']) + len(self.batch_map['test'])) != 0 or (len(self.chain_map['validation']) + len(self.batch_map['validation'])) != 0, 'ERROR: to get the dataloaders of each split, either the dataset has splits or a list of new splits is passed ("splits" argument)'
        
        dataloaders = []
        if splits is not None:
            splits_subsets = random_split(dataset=self, lengths=[split for split in splits if split != 0])
            for split in splits:
                dataloaders.append(self.get_dataloader(batch_size, shuffle, splits_subsets.pop(0))) if split != 0 else dataloaders.append(None)
        else:
            current_split = self.split

            self.set_split('train')
            dataloaders.append(self.get_dataloader(batch_size, shuffle)) if (len(self.chain_map['train']) + len(self.batch_map['train'])) != 0 else dataloaders.append(None)

            self.set_split('validation')
            dataloaders.append(self.get_dataloader(batch_size, shuffle)) if (len(self.chain_map['validation']) + len(self.batch_map['validation'])) != 0 else dataloaders.append(None)

            self.set_split('test')
            dataloaders.append(self.get_dataloader(batch_size, shuffle)) if (len(self.chain_map['test']) + len(self.batch_map['test'])) != 0 else dataloaders.append(None)

            self.set_split(current_split)
        
        return dataloaders
    


    def set_split(self, split: str = 'dataset') -> None:
        assert split in self.chain_map, f'ERROR: {split} does not exist'
        assert (len(self.chain_map[split]) + len(self.batch_map[split])) != 0, f'ERROR: {split} is empty'

        self.split = split
