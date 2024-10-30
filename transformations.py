from datasets import InteractionsDataset, ProteinsDataset, LigandsDataset
import dill
import h5py
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import math
import numpy as np
import torch
from tqdm.auto import tqdm
from typing import List, Union, Optional, Type




def remove_clean(filepath: str, masks: dict = {}) -> None:
    dataset = h5py.File(filepath, 'r+')

    log = open('./datasets/transformations-logs/transformations-log.txt', 'a')

    keep = {'proteins': set(), 'ligands': set()}
    for interaction in tqdm(list(dataset['interactions'].keys()), desc='iterating over all chain:ligand interactions: ', leave=False, disable=False):
        chain_path = dataset['interactions'][interaction].attrs['protein']
        ligand_path = dataset['interactions'][interaction].attrs['ligand']

        # delete interaction group if either its PROT:C or its LIG no longer exist
        if chain_path not in dataset or ligand_path not in dataset:
            del dataset['interactions'][interaction]
            log.write(f'deleted - {interaction}\n')
        # mark to keep PROT:C and LIG if both still exist
        else:
            keep['proteins'].add(chain_path)
            keep['ligands'].add(ligand_path)

            # remove corresponding targets of removed residues
            if chain_path in masks:
                targets = dataset['interactions'][interaction]['targets']
                targets_transformed = targets[~masks[chain_path]]
                del dataset['interactions'][interaction]['targets']
                dataset['interactions'][interaction].create_dataset('targets', data=targets_transformed)
    
    for protein in tqdm(list(dataset['proteins'].keys()), desc='iterating over all proteins: ', leave=False, disable=True):
        for chain in list(dataset['proteins'][protein].keys()):
            # delete PROT:C if not marked to keep (no interaction using it)
            if f'proteins/{protein}/{chain}' not in keep['proteins']:
                del dataset['proteins'][protein][chain]
                log.write(f'deleted - proteins/{protein}/{chain}\n')
        # check if PROT still has any C groups left and if not then delete PROT
        if not list(dataset['proteins'][protein].keys()):
            del dataset['proteins'][protein]
            log.write(f'deleted - proteins/{protein}\n')
    
    for ligand in tqdm(list(dataset['ligands'].keys()), desc='iterating over all ligands: ', leave=False, disable=True):
        # delete LIG if not marked to keep (no interaction using it)
        if f'ligands/{ligand}' not in keep['ligands']:
            del dataset['ligands'][ligand]
            log.write(f'deleted - ligands/{ligand}\n')

    dataset.close()

    log.close()




def remove_residues(original_filepath: str, transformed_filepath: str, value: float, features: Union[List[int], slice], any: bool = True) -> None:
    original = InteractionsDataset(original_filepath)
    original.clone(transformed_filepath)
    original.close()
    transformed = h5py.File(transformed_filepath, 'r+')

    log = open('./datasets/transformations-logs/transformations-log.txt', 'w')
    log.write(f'>>> remove_residues: original_filepath={original_filepath}, transformed_filepath={transformed_filepath}, value={value}, features={features}, any={any}\n')

    masks = {}
    for protein in tqdm(list(transformed['proteins'].keys()), desc='iterating over all proteins: ', leave=False, disable=False):
        for chain in list(transformed['proteins'][protein].keys()):
            protein_path = f'proteins/{protein}'
            chain_path = f'{protein_path}/{chain}'
            protein_features = transformed[chain_path]['features']
            
            if math.isnan(value) and any:
                # remove residue if ANY feature in "features" is "NaN"
                mask = np.any(np.isnan(protein_features[:, features]), axis=1)
            elif math.isnan(value) and not any:
                # remove residue if ALL features in "features" are "NaN"
                mask = np.all(np.isnan(protein_features[:, features]), axis=1)
            elif math.isinf(value) and any:
                # remove residue if ANY feature in "features" is "Inf"
                mask = np.any(np.isinf(protein_features[:, features]), axis=1)
            elif math.isinf(value) and not any:
                # remove residue if ALL features in "features" are "Inf"
                mask = np.all(np.isinf(protein_features[:, features]), axis=1)
            elif math.isfinite(value) and any:
                # remove residue if ANY feature in "features" is "value"
                mask = np.any(protein_features[:, features] == value, axis=1)
            elif math.isfinite(value) and not any:
                # remove residue if ALL features in "features" are "value"
                mask = np.all(protein_features[:, features] == value, axis=1)

            protein_features_transformed = protein_features[~mask]
            
            # delete features dataset in group "proteins/PROT/C" if no residues remain (delete the group "proteins/PROT/C" to delete its features dataset and not keep an empty group)
            if protein_features_transformed.size == 0:
                del transformed[chain_path]
                log.write(f'deleted - {chain_path}\n')
            # update features dataset in group "proteins/PROT/C" if some residues remain (must delete and recreate dataset because the shape changed)
            elif protein_features_transformed.size != protein_features.size:
                del transformed[chain_path]['features']
                transformed[chain_path].create_dataset('features', data=protein_features_transformed)
                masks[chain_path] = mask
                log.write(f'updated - {chain_path}\n')
            # keep features dataset in group "proteins/PROT/C" if all residues remain (do nothing)
    
    transformed.close()

    log.close()

    # delete interactions that use deleted groups in "proteins", delete empty "proteins/PROT" (no "proteins/PROT/C" left in it), delete ligands no longer used in at least one interaction
    remove_clean(transformed_filepath, masks)




def remove_chains(original_filepath: str, transformed_filepath: str, value: float, features: Union[List[int], slice], any: bool = True) -> None:
    original = InteractionsDataset(original_filepath)
    original.clone(transformed_filepath)
    original.close()
    transformed = h5py.File(transformed_filepath, 'r+')

    log = open('./datasets/transformations-logs/transformations-log.txt', 'w')
    log.write(f'>>> remove_chains: original_filepath={original_filepath}, transformed_filepath={transformed_filepath}, value={value}, features={features}, any={any}\n')

    for protein in tqdm(list(transformed['proteins'].keys()), desc='iterating over all proteins: ', leave=False, disable=False):
        for chain in list(transformed['proteins'][protein].keys()):
            protein_path = f'proteins/{protein}'
            chain_path = f'{protein_path}/{chain}'
            protein_features = transformed[chain_path]['features']

            # remove chain if ANY feature in "features" is "NaN"
            if math.isnan(value) and any and np.any(np.isnan(protein_features[:, features])):
                del transformed[chain_path]
                log.write(f'deleted - {chain_path}\n')
            # remove chain if ALL features in "features" are "NaN"
            elif math.isnan(value) and not any and np.all(np.isnan(protein_features[:, features])):
                del transformed[chain_path]
                log.write(f'deleted - {chain_path}\n')
            # remove chain if ANY feature in "features" is "Inf"
            elif math.isinf(value) and any and np.any(np.isinf(protein_features[:, features])):
                del transformed[chain_path]
                log.write(f'deleted - {chain_path}\n')
            # remove chain if ALL features in "features" are "Inf"
            elif math.isinf(value) and not any and np.all(np.isinf(protein_features[:, features])):
                del transformed[chain_path]
                log.write(f'deleted - {chain_path}\n')
            # remove chain if ANY feature in "features" is "value"
            elif math.isfinite(value) and any and np.any(protein_features[:, features] == value):
                del transformed[chain_path]
                log.write(f'deleted - {chain_path}\n')
            # remove chain if ALL features in "features" are "value"
            elif math.isfinite(value) and not any and np.all(protein_features[:, features] == value):
                del transformed[chain_path]
                log.write(f'deleted - {chain_path}\n')

    transformed.close()

    log.close()

    remove_clean(transformed_filepath)




def remove_ligands(original_filepath: str, transformed_filepath: str, value: float, features: Union[List[int], slice], any: bool = True) -> None:
    original = InteractionsDataset(original_filepath)
    original.clone(transformed_filepath)
    original.close()
    transformed = h5py.File(transformed_filepath, 'r+')

    log = open('./datasets/transformations-logs/transformations-log.txt', 'w')
    log.write(f'>>> remove_ligands: original_filepath={original_filepath}, transformed_filepath={transformed_filepath}, value={value}, features={features}, any={any}\n')

    for ligand in tqdm(list(transformed['ligands'].keys()), desc='iterating over all ligands: ', leave=False, disable=False):
        ligand_path = f'ligands/{ligand}'
        ligand_features = transformed[ligand_path]['features']

        # remove chain if ANY feature in "features" is "NaN"
        if math.isnan(value) and any and np.any(np.isnan(ligand_features[:, features])):
            del transformed[ligand_path]
            log.write(f'deleted - {ligand_path}\n')
        # remove chain if ALL features in "features" are "NaN"
        elif math.isnan(value) and not any and np.all(np.isnan(ligand_features[:, features])):
            del transformed[ligand_path]
            log.write(f'deleted - {ligand_path}\n')
        # remove chain if ANY feature in "features" is "Inf"
        elif math.isinf(value) and any and np.any(np.isinf(ligand_features[:, features])):
            del transformed[ligand_path]
            log.write(f'deleted - {ligand_path}\n')
        # remove chain if ALL features in "features" are "Inf"
        elif math.isinf(value) and not any and np.all(np.isinf(ligand_features[:, features])):
            del transformed[ligand_path]
            log.write(f'deleted - {ligand_path}\n')
        # remove chain if ANY feature in "features" is "value"
        elif math.isfinite(value) and any and np.any(ligand_features[:, features] == value):
            del transformed[ligand_path]
            log.write(f'deleted - {ligand_path}\n')
        # remove chain if ALL features in "features" are "value"
        elif math.isfinite(value) and not any and np.all(ligand_features[:, features] == value):
            del transformed[ligand_path]
            log.write(f'deleted - {ligand_path}\n')

    transformed.close()

    log.close()

    remove_clean(transformed_filepath)




def remove_protein_features(original_filepath: str, transformed_filepath: str, value: float, features: Union[List[int], slice], any: bool = True) -> None:
    original = ProteinsDataset(original_filepath)
    features = list(range(*features.indices(original[0].shape[0]))) if isinstance(features, slice) else features
    original.clone(transformed_filepath)
    original.close()
    transformed = h5py.File(transformed_filepath, 'r+')

    log = open('./datasets/transformations-logs/transformations-log.txt', 'w')
    log.write(f'>>> remove_protein_features: original_filepath={original_filepath}, transformed_filepath={transformed_filepath}, value={value}, features={features}, any={any}\n')

    mask = set() if any else set(features)
    for protein in tqdm(list(transformed['proteins'].keys()), desc='iterating over all proteins: ', leave=False, disable=False):
        for chain in list(transformed['proteins'][protein].keys()):
            protein_path = f'proteins/{protein}'
            chain_path = f'{protein_path}/{chain}'
            protein_features = transformed[chain_path]['features']
            
            for feature in features:
                # add to mask to be removed each feature that has ANY residue with "NaN"
                if math.isnan(value) and any and np.any(np.isnan(protein_features[:, [feature]])):
                    mask.add(feature)
                # remove from mask to be removed each feature that has not ALL residues with "NaN"
                elif math.isnan(value) and not any and not np.all(np.isnan(protein_features[:, [feature]])):
                    mask.discard(feature)
                # add to mask to be removed each feature that has ANY residue with "Inf"
                elif math.isinf(value) and any and np.any(np.isinf(protein_features[:, [feature]])):
                    mask.add(feature)
                # remove from mask to be removed each feature that has not ALL residues with "Inf"
                elif math.isinf(value) and not any and not np.all(np.isinf(protein_features[:, [feature]])):
                    mask.discard(feature)
                # add to mask to be removed each feature that has ANY residue with "value"
                elif math.isfinite(value) and any and np.any(protein_features[:, [feature]] == value):
                    mask.add(feature)
                # remove from mask to be removed each feature that has not ALL residues with "value"
                elif math.isfinite(value) and not any and not np.all(protein_features[:, [feature]] == value):
                    mask.discard(feature)
    
    cancel = False
    if mask:
        for protein in tqdm(list(transformed['proteins'].keys()), desc='iterating over all proteins: ', leave=False, disable=False):
            for chain in list(transformed['proteins'][protein].keys()):
                protein_path = f'proteins/{protein}'
                chain_path = f'{protein_path}/{chain}'
                protein_features = transformed[chain_path]['features']
                protein_features_transformed = np.delete(protein_features, list(mask), axis=1)

                if protein_features_transformed.size == 0:
                    log.write('all protein features are marked to be removed\n')
                    log.write('cancelling transformation\n')
                    cancel = True
                    break
                else:
                    del transformed[chain_path]['features']
                    transformed[chain_path].create_dataset('features', data=protein_features_transformed)
            if cancel:
                break
        if not cancel:
            log.write(f'deleted - protein features {list(mask)}\n')
    
    
    transformed.close()

    log.close()




def remove_ligand_features(original_filepath: str, transformed_filepath: str, value: float, features: Union[List[int], slice], any: bool = True) -> None:
    original = LigandsDataset(original_filepath)
    features = list(range(*features.indices(original[0].shape[0]))) if isinstance(features, slice) else features
    original.clone(transformed_filepath)
    original.close()
    transformed = h5py.File(transformed_filepath, 'r+')

    log = open('./datasets/transformations-logs/transformations-log.txt', 'w')
    log.write(f'>>> remove_ligand_features: original_filepath={original_filepath}, transformed_filepath={transformed_filepath}, value={value}, features={features}, any={any}\n')

    mask = set() if any else set(features)
    if value is not None:
        for ligand in tqdm(list(transformed['ligands'].keys()), desc='iterating over all ligands: ', leave=False, disable=False):
            ligand_path = f'ligands/{ligand}'
            ligand_features = transformed[ligand_path]['features']
                
            for feature in features:
                # add to mask to be removed each feature that has ANY ligand with "NaN"
                if math.isnan(value) and any and np.isnan(ligand_features[:, [feature]]).item():
                    mask.add(feature)
                # remove from mask to be removed each feature that has not ALL ligands with "NaN"
                elif math.isnan(value) and not any and not np.isnan(ligand_features[:, [feature]]).item():
                    mask.discard(feature)
                # add to mask to be removed each feature that has ANY ligand with "Inf"
                elif math.isinf(value) and any and np.isinf(ligand_features[:, [feature]]).item():
                    mask.add(feature)
                # remove from mask to be removed each feature that has not ALL ligands with "Inf"
                elif math.isinf(value) and not any and not np.isinf(ligand_features[:, [feature]]).item():
                    mask.discard(feature)
                # add to mask to be removed each feature that has ANY ligand with "value"
                elif math.isfinite(value) and any and ligand_features[:, [feature]].item() == value:
                    mask.add(feature)
                # remove from mask to be removed each feature that has not ALL ligands with "value"
                elif math.isfinite(value) and not any and not ligand_features[:, [feature]].item() == value:
                    mask.discard(feature)
    
    cancel = False
    if mask:
        for ligand in tqdm(list(transformed['ligands'].keys()), desc='iterating over all ligands: ', leave=False, disable=False):
            ligand_path = f'ligands/{ligand}'
            ligand_features = transformed[ligand_path]['features']
            ligand_features_transformed = np.delete(ligand_features, list(mask), axis=1)

            if ligand_features_transformed.size == 0:
                log.write('all ligand features are marked to be removed\n')
                log.write('cancelling transformation\n')
                cancel = True
                break
            else:
                del transformed[ligand_path]['features']
                transformed[ligand_path].create_dataset('features', data=ligand_features_transformed)
        if not cancel:
            log.write(f'deleted - ligand features {list(mask)}\n')
    
    transformed.close()

    log.close()




def set_dtype(original_filepath: str, transformed_filepath: str, features: Optional[Type[np.dtype]] = None, targets: Optional[Type[np.dtype]] = None) -> None:
    original = InteractionsDataset(original_filepath)
    original.clone(transformed_filepath)
    original.close()
    transformed = h5py.File(transformed_filepath, 'r+')

    log = open('./datasets/transformations-logs/transformations-log.txt', 'w')
    log.write(f'>>> set_dtype: original_filepath={original_filepath}, transformed_filepath={transformed_filepath}, features={features}, targets={targets}\n')

    if features is not None:
        for protein in tqdm(list(transformed['proteins'].keys()), desc='iterating over all proteins: ', leave=False, disable=False):
            for chain in list(transformed['proteins'][protein].keys()):
                protein_path = f'proteins/{protein}'
                chain_path = f'{protein_path}/{chain}'
                protein_features = transformed[chain_path]['features']
                protein_features_transformed = np.array(protein_features, dtype=features)

                del transformed[chain_path]['features']
                transformed[chain_path].create_dataset('features', data=protein_features_transformed)
        
        for ligand in tqdm(list(transformed['ligands'].keys()), desc='iterating over all ligands: ', leave=False, disable=False):
            ligand_path = f'ligands/{ligand}'
            ligand_features = transformed[ligand_path]['features']
            ligand_features_transformed = np.array(ligand_features, dtype=features)

            del transformed[ligand_path]['features']
            transformed[ligand_path].create_dataset('features', data=ligand_features_transformed)
        
        log.write(f'proteins and ligands features set to {features}\n')
    
    if targets is not None:
        for target in tqdm(list(transformed['interactions'].keys()), desc='iterating over all targets: ', leave=False, disable=False):
            target_path = f'interactions/{target}'
            target_feature = transformed[target_path]['targets']
            target_feature_transformed = np.array(target_feature, dtype=targets)

            del transformed[target_path]['targets']
            transformed[target_path].create_dataset('targets', data=target_feature_transformed)
        
        log.write(f'targets set to {targets}\n')

    if 'balanced-batches' in list(transformed.keys()):
        for batch in tqdm(list(transformed['balanced-batches'].keys()), desc='iterating over all batches: ', leave=False, disable=False):
            batch_path = f'balanced-batches/{batch}'

            if features is not None:
                batch_features = transformed[batch_path]['features']
                batch_features_transformed = np.array(batch_features, dtype=features)

                del transformed[batch_path]['features']
                transformed[batch_path].create_dataset('features', data=batch_features_transformed)
            
            if targets is not None:
                batch_targets = transformed[batch_path]['targets']
                batch_targets_transformed = np.array(batch_targets, dtype=targets)

                del transformed[batch_path]['targets']
                transformed[batch_path].create_dataset('targets', data=batch_targets_transformed)
        
        if features is not None:
            log.write(f'balanced-batches features set to {features}\n')
        if targets is not None:
            log.write(f'balanced-batches targets set to {targets}\n')
    
    transformed.close()

    log.close()




'''
def scale_zscore():
    pass




def reduce_dimensionality_pca():
    pass
'''



def reduce_dimensionality_ae_proteins(original_filepath: str, transformed_filepath: str, model_filepath:str):
    original = ProteinsDataset(original_filepath)
    original.clone(transformed_filepath)
    original.close()
    transformed = h5py.File(transformed_filepath, 'r+')

    model = torch.load(model_filepath, pickle_module=dill, map_location=torch.device('cpu'))
    model.eval()

    log = open('./datasets/transformations-logs/transformations-log.txt', 'w')
    log.write(f'>>> reduce_dimensionality_ae_proteins: original_filepath={original_filepath}, transformed_filepath={transformed_filepath}, model_filepath={model_filepath}\n')

    for protein in tqdm(list(transformed['proteins'].keys()), desc='iterating over all proteins: ', leave=False, disable=False):
        for chain in list(transformed['proteins'][protein].keys()):
            protein_path = f'proteins/{protein}'
            chain_path = f'{protein_path}/{chain}'
            protein_features = torch.tensor(np.array(transformed[chain_path]['features']))

            latent_vector = model.encoder(protein_features)

            del transformed[chain_path]['features']
            transformed[chain_path].create_dataset('features', data=latent_vector.detach())

    log.write(f'protein features went from {protein_features.shape[1]} to {latent_vector.shape[1]}\n')

    transformed.close()

    log.close()






'''
def reduce_dimensionality_umap():
    pass
'''



def sample_smote(original_filepath: str, transformed_filepath: str, residue_granularity: bool, batch_size: int, shuffle: bool = False, seed: int = 0):
    original = InteractionsDataset(original_filepath)
    original.clone(transformed_filepath)
    if not residue_granularity:
        original.set_granularity()
    dataloader = original.get_dataloader(batch_size=batch_size, shuffle=shuffle)
    transformed = h5py.File(transformed_filepath, 'r+')
    batches = transformed.require_group(f'balanced-batches')

    log = open('./datasets/transformations-logs/transformations-log.txt', 'w')
    log.write(f'>>> sample_smote: original_filepath={original_filepath}, transformed_filepath={transformed_filepath}, residue_granularity={residue_granularity}, batch_size={batch_size}, shuffle={shuffle}, seed={seed}\n')

    ignored = 0
    rjust_width = len(str(len(dataloader)))
    for batch, data in enumerate(tqdm(dataloader, desc='iterating over all batches: ', leave=False, disable=False)):
        features, targets = data
        
        n_residues = targets.shape[0]
        n_residues_interacting = torch.count_nonzero(targets).item()
        k_neighbors = (n_residues_interacting if n_residues - n_residues_interacting >= n_residues_interacting else n_residues - n_residues_interacting) - 1
        if k_neighbors <= 0:
            ignored += 1
            continue
        sampler = SMOTE(random_state=seed, k_neighbors=k_neighbors)
        transformed_features, transformed_targets = sampler.fit_resample(features, targets)
        transformed_features = torch.tensor(transformed_features)
        transformed_targets = torch.tensor(transformed_targets.reshape((-1, 1)))
        
        transformed_batch = batches.require_group(str(batch).rjust(rjust_width, '0'))
        transformed_batch.create_dataset('features', data=transformed_features[features.shape[0]:])
        transformed_batch.create_dataset('targets', data=transformed_targets[targets.shape[0]:])

    log.write(f'each batch of {"random" if shuffle else "ordered"} {batch_size} {"residues" if residue_granularity else "chains"} + ligands (i.e., interactions) has been SMOTEd\n')
    log.write(f'{ignored} batches were ignored due to not having at least 2 instances of the minority class within the batch\n')


    original.close()
    transformed.close()

    log.close()




def sample_over(original_filepath: str, transformed_filepath: str, residue_granularity: bool, batch_size: int, shuffle: bool = False, seed: int = 0):
    original = InteractionsDataset(original_filepath)
    original.clone(transformed_filepath)
    if not residue_granularity:
        original.set_granularity()
    dataloader = original.get_dataloader(batch_size=batch_size, shuffle=shuffle)
    transformed = h5py.File(transformed_filepath, 'r+')
    batches = transformed.require_group(f'balanced-batches')

    log = open('./datasets/transformations-logs/transformations-log.txt', 'w')
    log.write(f'>>> sample_over: original_filepath={original_filepath}, transformed_filepath={transformed_filepath}, residue_granularity={residue_granularity}, batch_size={batch_size}, shuffle={shuffle}, seed={seed}\n')

    sampler = RandomOverSampler(random_state=seed)

    ignored = 0
    rjust_width = len(str(len(dataloader)))
    for batch, data in enumerate(tqdm(dataloader, desc='iterating over all batches: ', leave=False, disable=False)):
        features, targets = data

        if len(torch.unique(targets)) != 2:
            ignored += 1
            continue
        
        transformed_features, transformed_targets = sampler.fit_resample(features, targets)
        transformed_features = torch.tensor(transformed_features)
        transformed_targets = torch.tensor(transformed_targets.reshape((-1, 1)))
        
        transformed_batch = batches.require_group(str(batch).rjust(rjust_width, '0'))
        transformed_batch.create_dataset('features', data=transformed_features[features.shape[0]:])
        transformed_batch.create_dataset('targets', data=transformed_targets[targets.shape[0]:])
    
    log.write(f'each batch of {"random" if shuffle else "ordered"} {batch_size} {"residues" if residue_granularity else "chains"} + ligands (i.e., interactions) has been oversampled\n')
    log.write(f'{ignored} batches were ignored due to not having at least 1 instance of each class within the batch\n')

    original.close()
    transformed.close()

    log.close()



