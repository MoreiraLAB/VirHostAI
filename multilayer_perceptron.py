import argparse
import csv
from source.datasets import InteractionsDataset, BalancedInteractionsDataset
import dill
import functools
import math
import numpy as np
import torch
from torch import nn
from source.torch_map import ACTIVATIONS, CRITERIONS, OPTIMIZERS
import torchinfo
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
from tqdm.auto import tqdm
from typing import List, Union, Optional
import wandb
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader




class MultilayerPerceptron(nn.Module):

    def __init__(self, layers: Union[List[int], dict], activation: str) -> None:
        super().__init__()
        
        if isinstance(layers, dict):
            assert set(layers.keys()) == set(['architecture', 'input_size', 'num_layers', 'proportion']), 'ERROR: if the layers argument is a dictionary then it must only contain the keys "architecture", "input_size", "num_layers" and "proportion"'
            
            architecture = layers['architecture']
            input_size = layers['input_size']
            num_layers = layers['num_layers']
            proportion = layers['proportion']

            assert num_layers > 0, 'ERROR: num_layers must be greater than 0'
            assert architecture == '=' or proportion >= 2, 'ERROR: if architecture is not "=" then proportion must be at least 2'

            layers = [input_size]
            for i in range(num_layers):
                if architecture == '=':
                    layers.append(int(input_size * proportion))
                elif architecture == '<':
                    layers.append(input_size * (proportion + i))
                elif architecture == '>':
                    layers.append(max(input_size // (proportion + i), 2))
                elif architecture == '<>':
                    layers.append(input_size * (proportion + i))
                    if i + 1 == num_layers:
                        layers = layers + layers[::-1][1:]
        
        mlp = []
        for i in range(len(layers) - 1):
            mlp.append(nn.Linear(layers[i], layers[i + 1]))
            mlp.append(ACTIVATIONS[activation]())
        mlp.append(nn.Linear(layers[-1], 1))

        self.mlp = nn.Sequential(*mlp)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return x




def run(dataset: Union[InteractionsDataset, BalancedInteractionsDataset], model: Optional[MultilayerPerceptron], device: torch.device, wandb_mode: str):
    with wandb.init(mode=wandb_mode) as wandb_run:
        dataloaders = dataset.get_splits_dataloaders(batch_size=wandb_run.config.batch_size, shuffle=wandb_run.config.shuffle, splits=wandb_run.config.splits)
        train_dataloader = dataloaders[0]
        validation_dataloader = dataloaders[1]
        test_dataloader = dataloaders[2]
        
        n_input = dataset[0][0].shape[-1]
        layers = wandb_run.config.layers
        if isinstance(layers, dict):
            layers['input_size'] = n_input
        else:
            layers = [n_input] + layers
        model = MultilayerPerceptron(layers=layers, activation=wandb_run.config.activation) if model is None else model
        assert isinstance(model, MultilayerPerceptron), 'ERROR: model must be a MultilayerPerceptron'

        model.to(device)
        criterion = CRITERIONS['bce-logits']()
        optimizer = OPTIMIZERS[wandb_run.config.optimizer](params=model.parameters(), lr=wandb_run.config.learning_rate)

        accuracy = BinaryAccuracy().to(device)
        precision = BinaryPrecision().to(device)
        recall = BinaryRecall().to(device)
        f1_score = BinaryF1Score().to(device)
        auroc = BinaryAUROC().to(device)

        if isinstance(wandb_run.config.model_log_frequency, int) and wandb_run.config.model_log_frequency > 0:
            wandb_run.watch(models=model, criterion=criterion, log='all', log_freq=wandb_run.config.model_log_frequency, log_graph=True)

        assert 0 <= wandb_run.config.train_batch_logs_per_epoch <= 1, "ERROR: 0 <= train_batch_logs_per_epoch <= 1"
        train_batch_log_frequency = int(1 / wandb_run.config.train_batch_logs_per_epoch) if wandb_run.config.train_batch_logs_per_epoch != 0 else float('inf')
        assert 0 <= wandb_run.config.validation_batch_logs_per_epoch <= 1, "ERROR: 0 <= validation_batch_logs_per_epoch <= 1"
        validation_batch_log_frequency = int(1 / wandb_run.config.validation_batch_logs_per_epoch) if wandb_run.config.validation_batch_logs_per_epoch != 0 else float('inf')
        assert 0 <= wandb_run.config.test_batch_logs_per_epoch <= 1, "ERROR: 0 <= test_batch_logs_per_epoch <= 1"
        test_batch_log_frequency = int(1 / wandb_run.config.test_batch_logs_per_epoch) if wandb_run.config.test_batch_logs_per_epoch != 0 else float('inf')
        
        if train_dataloader:
            wandb_run.define_metric('train/epoch')
            wandb_run.define_metric('train/epoch-loss', step_metric='train/epoch')
            wandb_run.define_metric('train/epoch-accuracy', step_metric='train/epoch')
            wandb_run.define_metric('train/epoch-precision', step_metric='train/epoch')
            wandb_run.define_metric('train/epoch-recall', step_metric='train/epoch')
            wandb_run.define_metric('train/epoch-f1-score', step_metric='train/epoch')
            wandb_run.define_metric('train/epoch-auroc', step_metric='train/epoch')
            wandb_run.define_metric('train/batch')
            wandb_run.define_metric('train/batch-loss', step_metric='train/batch')
            wandb_run.define_metric('train/batch-accuracy', step_metric='train/batch')
            wandb_run.define_metric('train/batch-precision', step_metric='train/batch')
            wandb_run.define_metric('train/batch-recall', step_metric='train/batch')
            wandb_run.define_metric('train/batch-f1-score', step_metric='train/batch')
            wandb_run.define_metric('train/batch-auroc', step_metric='train/batch')
        if validation_dataloader:
            wandb_run.define_metric('validation/epoch')
            wandb_run.define_metric('validation/epoch-loss', step_metric='validation/epoch')
            wandb_run.define_metric('validation/epoch-accuracy', step_metric='validation/epoch')
            wandb_run.define_metric('validation/epoch-precision', step_metric='validation/epoch')
            wandb_run.define_metric('validation/epoch-recall', step_metric='validation/epoch')
            wandb_run.define_metric('validation/epoch-f1-score', step_metric='validation/epoch')
            wandb_run.define_metric('validation/epoch-auroc', step_metric='validation/epoch')
            wandb_run.define_metric('validation/batch')
            wandb_run.define_metric('validation/batch-loss', step_metric='validation/batch')
            wandb_run.define_metric('validation/batch-accuracy', step_metric='validation/batch')
            wandb_run.define_metric('validation/batch-precision', step_metric='validation/batch')
            wandb_run.define_metric('validation/batch-recall', step_metric='validation/batch')
            wandb_run.define_metric('validation/batch-f1-score', step_metric='validation/batch')
            wandb_run.define_metric('validation/batch-auroc', step_metric='validation/batch')
        if test_dataloader:
            wandb_run.define_metric('test/batch')
            wandb_run.define_metric('test/batch-loss', step_metric='test/batch')
            wandb_run.define_metric('test/batch-accuracy', step_metric='test/batch')
            wandb_run.define_metric('test/batch-precision', step_metric='test/batch')
            wandb_run.define_metric('test/batch-recall', step_metric='test/batch')
            wandb_run.define_metric('test/batch-f1-score', step_metric='test/batch')
            wandb_run.define_metric('test/batch-auroc', step_metric='test/batch')
        
        if train_dataloader or validation_dataloader:
            for epoch in range(wandb_run.config.epochs):
                print()

                # train
                if train_dataloader:
                    epoch_loss = 0

                    epoch_accuracy = 0
                    epoch_precision = 0
                    epoch_recall = 0
                    epoch_f1_score = 0
                    epoch_auroc = 0

                    progress = tqdm(train_dataloader, desc=f'epoch [{epoch+1}/{wandb_run.config.epochs}]')
                    model.train()
                    for batch, data in enumerate(progress):
                        x, y = data[0], data[1]
                        x, y = x.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)

                        z = model(x)

                        loss = criterion(z, y)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        batch_loss = loss.item()
                        epoch_loss += batch_loss

                        probabilities = torch.sigmoid(z)
                        predictions = torch.round(probabilities)
                        batch_accuracy = accuracy(predictions, y)
                        epoch_accuracy += batch_accuracy
                        batch_precision = precision(predictions, y)
                        epoch_precision += batch_precision
                        batch_recall = recall(predictions, y)
                        epoch_recall += batch_recall
                        batch_f1_score = f1_score(predictions, y)
                        epoch_f1_score += batch_f1_score
                        batch_auroc = auroc(probabilities, y)
                        epoch_auroc += batch_auroc
                        
                        progress.set_postfix({'train/batch-loss': f'{batch_loss:.4f}'})
                        if (batch % train_batch_log_frequency) == 0 or (batch + 1) == len(train_dataloader):
                            wandb_run.log({'train/batch-loss': batch_loss, 
                                           'train/batch':batch + (len(train_dataloader) * epoch), 
                                           'train/batch-accuracy': batch_accuracy, 
                                           'train/batch-precision': batch_precision, 
                                           'train/batch-recall': batch_recall, 
                                           'train/batch-f1-score': batch_f1_score, 
                                           'train/batch-auroc': batch_auroc})

                        # early stop
                        if not math.isfinite(batch_loss):
                            progress.close()
                            print(f'ERROR: {batch_loss} train/batch-loss in epoch {epoch} batch {batch}, early stop triggered')
                            print()
                            return

                    epoch_loss /= len(train_dataloader)

                    epoch_accuracy /= len(train_dataloader)
                    epoch_precision /= len(train_dataloader)
                    epoch_recall /= len(train_dataloader)
                    epoch_f1_score /= len(train_dataloader)
                    epoch_auroc /= len(train_dataloader)
                    
                    print(f'epoch [{epoch+1}/{wandb_run.config.epochs}]: train/epoch-loss={epoch_loss:.4f}')
                    wandb_run.log({'train/epoch-loss': epoch_loss, 
                                   'train/epoch': epoch, 
                                   'train/epoch-accuracy': epoch_accuracy, 
                                   'train/epoch-precision': epoch_precision, 
                                   'train/epoch-recall': epoch_recall, 
                                   'train/epoch-f1-score': epoch_f1_score, 
                                   'train/epoch-auroc': epoch_auroc})

                # validation
                if validation_dataloader:
                    epoch_loss = 0

                    epoch_accuracy = 0
                    epoch_precision = 0
                    epoch_recall = 0
                    epoch_f1_score = 0
                    epoch_auroc = 0

                    progress = tqdm(validation_dataloader, desc=f'epoch [{epoch+1}/{wandb_run.config.epochs}]')
                    model.eval()
                    with torch.inference_mode():
                        for batch, data in enumerate(progress):
                            x, y = data[0], data[1]
                            x, y = x.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)

                            z = model(x)

                            loss = criterion(z, y)

                            batch_loss = loss.item()
                            epoch_loss += batch_loss

                            probabilities = torch.sigmoid(z)
                            predictions = torch.round(probabilities)
                            batch_accuracy = accuracy(predictions, y)
                            epoch_accuracy += batch_accuracy
                            batch_precision = precision(predictions, y)
                            epoch_precision += batch_precision
                            batch_recall = recall(predictions, y)
                            epoch_recall += batch_recall
                            batch_f1_score = f1_score(predictions, y)
                            epoch_f1_score += batch_f1_score
                            batch_auroc = auroc(probabilities, y)
                            epoch_auroc += batch_auroc
                            
                            progress.set_postfix({'validation/batch-loss': f'{batch_loss:.4f}'})
                            if (batch % validation_batch_log_frequency) == 0 or (batch + 1) == len(validation_dataloader):
                                wandb_run.log({'validation/batch-loss': batch_loss, 
                                               'validation/batch':batch + (len(validation_dataloader) * epoch), 
                                               'validation/batch-accuracy': batch_accuracy, 
                                               'validation/batch-precision': batch_precision, 
                                               'validation/batch-recall': batch_recall, 
                                               'validation/batch-f1-score': batch_f1_score, 
                                               'validation/batch-auroc': batch_auroc})
                        
                        epoch_loss /= len(validation_dataloader)

                        epoch_accuracy /= len(validation_dataloader)
                        epoch_precision /= len(validation_dataloader)
                        epoch_recall /= len(validation_dataloader)
                        epoch_f1_score /= len(validation_dataloader)
                        epoch_auroc /= len(validation_dataloader)
                        
                        print(f'epoch [{epoch+1}/{wandb_run.config.epochs}]: validation/epoch-loss={epoch_loss:.4f}')
                        wandb_run.log({'validation/epoch-loss': epoch_loss, 
                                       'validation/epoch': epoch, 
                                       'validation/epoch-accuracy': epoch_accuracy, 
                                       'validation/epoch-precision': epoch_precision, 
                                       'validation/epoch-recall': epoch_recall, 
                                       'validation/epoch-f1-score': epoch_f1_score, 
                                       'validation/epoch-auroc': epoch_auroc})
        
        # test
        epoch = 1
        dataset.set_split('test')
        if test_dataloader:
            if wandb_run.config.batch_size == 1:
                file = open('./outputs.csv', mode='w', newline='')
                writer = csv.writer(file)
                writer.writerow(['interaction', 'shape', 'probabilities', 'predictions'])
            
            print()

            epoch_loss = 0

            epoch_accuracy = 0
            epoch_precision = 0
            epoch_recall = 0
            epoch_f1_score = 0
            epoch_auroc = 0
            
            progress = tqdm(test_dataloader, desc=f'test')
            model.eval()
            with torch.inference_mode():
                for batch, data in enumerate(progress):
                    x, y = data[0], data[1]
                    x, y = x.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)

                    z = model(x)

                    loss = criterion(z, y)

                    batch_loss = loss.item()
                    epoch_loss += batch_loss

                    probabilities = torch.sigmoid(z)
                    predictions = torch.round(probabilities)
                    batch_accuracy = accuracy(predictions, y)
                    epoch_accuracy += batch_accuracy
                    batch_precision = precision(predictions, y)
                    epoch_precision += batch_precision
                    batch_recall = recall(predictions, y)
                    epoch_recall += batch_recall
                    batch_f1_score = f1_score(predictions, y)
                    epoch_f1_score += batch_f1_score
                    batch_auroc = auroc(probabilities, y)
                    epoch_auroc += batch_auroc

                    if wandb_run.config.batch_size == 1:
                        probabilities_str = np.array2string(probabilities.cpu().numpy(), separator=',', threshold=np.inf).replace('\n', '')
                        predictions_str = np.array2string(predictions.cpu().numpy(), separator=',', threshold=np.inf).replace('\n', '')
                        writer.writerow([batch, y.shape, probabilities_str, predictions_str])
                    
                    progress.set_postfix({'test/batch-loss': f'{batch_loss:.4f}'})
                    if (batch % test_batch_log_frequency) == 0 or (batch + 1) == len(test_dataloader):
                        wandb_run.log({'test/batch-loss': batch_loss, 
                                       'test/batch':batch + (len(test_dataloader) * epoch), 
                                       'test/batch-accuracy': batch_accuracy, 
                                       'test/batch-precision': batch_precision, 
                                       'test/batch-recall': batch_recall, 
                                       'test/batch-f1-score': batch_f1_score, 
                                       'test/batch-auroc': batch_auroc})
                
                epoch_loss /= len(test_dataloader)

                epoch_accuracy /= len(test_dataloader)
                epoch_precision /= len(test_dataloader)
                epoch_recall /= len(test_dataloader)
                epoch_f1_score /= len(test_dataloader)
                epoch_auroc /= len(test_dataloader)
                
                print(f'test: test/epoch-loss={epoch_loss:.4f}')
                print(f'      test/epoch-accuracy={epoch_accuracy:.4f}')
                print(f'      test/epoch-precision={epoch_precision:.4f}')
                print(f'      test/epoch-recall={epoch_recall:.4f}')
                print(f'      test/epoch-f1-score={epoch_f1_score:.4f}')
                print(f'      test/epoch-auroc={epoch_auroc:.4f}')
            
            if wandb_run.config.batch_size == 1:
                file.close()

        if wandb_run.config.save_model:
            print()
            artifact = wandb.Artifact(name='model', type='model')
            with artifact.new_file('model.pt', mode='wb') as file:
                torch.save(model, file, pickle_module=dill)
            with artifact.new_file('model-summary.txt') as file:
                file.write(str(torchinfo.summary(model, input_size=(1, n_input))))
            wandb_run.log_artifact(artifact)
        
        print()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=['datasets/dataset-viral-transformed-1.hdf5', 'interactions', 'residue'], nargs=3, help='<hdf5 dataset filepath> {interactions, balanced-interactions} {residue, chain}')
    parser.add_argument('--model', default=None, help='<torch model filepath>')
    parser.add_argument('--config', default=['models/config-files/config-mlp.yaml', '1'], nargs=2, help='{<yaml sweep config filepath>, <wandb sweep author/project/id>} <number of runs>')
    parser.add_argument('--device', default='cpu', help='<torch device>')
    parser.add_argument('--wandb', default='online', choices=['online', 'offline', 'disabled'])
    args = parser.parse_args()
    
    assert args.dataset[1] in ['interactions', 'balanced-interactions'], 'ERROR: dataset must be "interactions" or "balanced-interactions"'
    if args.dataset[1] == 'interactions':
        dataset = InteractionsDataset(args.dataset[0])
    elif args.dataset[1] == 'balanced-interactions':
        dataset = BalancedInteractionsDataset(args.dataset[0])
    
    assert args.dataset[2] in ['residue', 'chain'], 'ERROR: dataset granularity must be "residue" or "chain"'
    if args.dataset[2] == 'chain':
        dataset.set_granularity()

    device = torch.device(args.device)

    model = torch.load(args.model, pickle_module=dill, map_location=device) if args.model else None
    
    wandb.login()
    sweep_id = wandb.sweep(sweep=yaml.load(open(args.config[0], 'r'), Loader)) if args.config[0].split('.')[-1] in ['yaml', 'yml'] else args.config[0]
    sweep_agent_function = functools.partial(run, dataset, model, device, args.wandb)
    wandb.agent(sweep_id=sweep_id, function=sweep_agent_function, count=int(args.config[1]))
