import argparse
from datasets import InteractionsDataset, ProteinsDataset, LigandsDataset
import dill
import functools
import math
import torch
from torch import nn
from torch_map import ACTIVATIONS, CRITERIONS, OPTIMIZERS
import torchinfo
from tqdm.auto import tqdm
from typing import List, Union, Optional
import wandb
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader




class Autoencoder(nn.Module):

    def __init__(self, encoder_layers: List[int], latent_vector: int, activation: str) -> None:
        super().__init__()

        encoder = []
        decoder = []

        for i in range(len(encoder_layers) - 1):
            encoder.append(nn.Linear(encoder_layers[i], encoder_layers[i + 1]))
            encoder.append(ACTIVATIONS[activation]())
        encoder.append(nn.Linear(encoder_layers[-1], latent_vector))

        decoder.append(nn.Linear(latent_vector, encoder_layers[-1]))
        for i in range(len(encoder_layers) - 1, 0, -1):
            decoder.append(ACTIVATIONS[activation]())
            decoder.append(nn.Linear(encoder_layers[i], encoder_layers[i - 1]))
        
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x




def run(dataset: Union[InteractionsDataset, ProteinsDataset, LigandsDataset], model: Optional[Autoencoder], device: torch.device, wandb_mode: str):
    with wandb.init(mode=wandb_mode) as wandb_run:
        dataloaders = dataset.get_splits_dataloaders(batch_size=wandb_run.config.batch_size, shuffle=wandb_run.config.shuffle, splits=wandb_run.config.splits)
        train_dataloader = dataloaders[0]
        validation_dataloader = dataloaders[1]
        test_dataloader = dataloaders[2]
        
        batch_item_is_tuple = isinstance(dataset, InteractionsDataset)
        n_input = dataset[0][0].shape[-1] if batch_item_is_tuple else dataset[0].shape[-1]
        model = Autoencoder(encoder_layers=[n_input] + wandb_run.config.encoder_layers, latent_vector=wandb_run.config.latent_vector, activation=wandb_run.config.activation) if model is None else model
        assert isinstance(model, Autoencoder), 'ERROR: model must be an Autoencoder'

        model.to(device)
        criterion = CRITERIONS[wandb_run.config.criterion]()
        optimizer = OPTIMIZERS[wandb_run.config.optimizer](params=model.parameters(), lr=wandb_run.config.learning_rate)

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
            wandb_run.define_metric('train/batch')
            wandb_run.define_metric('train/batch-loss', step_metric='train/batch')
        if validation_dataloader:
            wandb_run.define_metric('validation/epoch')
            wandb_run.define_metric('validation/epoch-loss', step_metric='validation/epoch')
            wandb_run.define_metric('validation/batch')
            wandb_run.define_metric('validation/batch-loss', step_metric='validation/batch')
        if test_dataloader:
            wandb_run.define_metric('test/batch')
            wandb_run.define_metric('test/batch-loss', step_metric='test/batch')
        
        if train_dataloader or validation_dataloader:
            for epoch in range(wandb_run.config.epochs):
                print()

                # train
                if train_dataloader:
                    epoch_loss = 0
                    progress = tqdm(train_dataloader, desc=f'epoch [{epoch+1}/{wandb_run.config.epochs}]')
                    model.train()
                    for batch, data in enumerate(progress):
                        x = data[0] if batch_item_is_tuple else data
                        x = x.to(device=device, dtype=torch.float32)

                        z = model(x)

                        loss = criterion(z, x)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        batch_loss = loss.item()
                        epoch_loss += batch_loss
                        progress.set_postfix({'train/batch-loss': f'{batch_loss:.4f}'})
                        if (batch % train_batch_log_frequency) == 0 or (batch + 1) == len(train_dataloader):
                            wandb_run.log({'train/batch-loss': batch_loss, 'train/batch':batch + (len(train_dataloader) * epoch)})

                        # early stop
                        if not math.isfinite(batch_loss):
                            progress.close()
                            print(f'ERROR: {batch_loss} train/batch-loss in epoch {epoch} batch {batch}, early stop triggered')
                            print()
                            return

                    epoch_loss /= len(train_dataloader)
                    print(f'epoch [{epoch+1}/{wandb_run.config.epochs}]: train/epoch-loss={epoch_loss:.4f}')
                    wandb_run.log({'train/epoch-loss': epoch_loss, 'train/epoch': epoch})

                # validation
                if validation_dataloader:
                    epoch_loss = 0
                    progress = tqdm(validation_dataloader, desc=f'epoch [{epoch+1}/{wandb_run.config.epochs}]')
                    model.eval()
                    with torch.inference_mode():
                        for batch, data in enumerate(progress):
                            x = data[0] if batch_item_is_tuple else data
                            x = x.to(device)
                            
                            z = model(x)

                            loss = criterion(z, x)

                            batch_loss = loss.item()
                            epoch_loss += batch_loss
                            progress.set_postfix({'validation/batch-loss': f'{batch_loss:.4f}'})
                            if (batch % validation_batch_log_frequency) == 0 or (batch + 1) == len(validation_dataloader):
                                wandb_run.log({'validation/batch-loss': batch_loss, 'validation/batch':batch + (len(validation_dataloader) * epoch)})
                        
                        epoch_loss /= len(validation_dataloader)
                        print(f'epoch [{epoch+1}/{wandb_run.config.epochs}]: validation/epoch-loss={epoch_loss:.4f}')
                        wandb_run.log({'validation/epoch-loss': epoch_loss, 'validation/epoch': epoch})
        
        # test
        if test_dataloader:
            print()

            epoch_loss = 0
            progress = tqdm(test_dataloader, desc=f'test')
            model.eval()
            with torch.inference_mode():
                for batch, data in enumerate(progress):
                    x = data[0] if batch_item_is_tuple else data
                    x = x.to(device)
                    
                    z = model(x)

                    loss = criterion(z, x)

                    batch_loss = loss.item()
                    epoch_loss += batch_loss
                    progress.set_postfix({'test/batch-loss': f'{batch_loss:.4f}'})
                    if (batch % test_batch_log_frequency) == 0 or (batch + 1) == len(test_dataloader):
                        wandb_run.log({'test/batch-loss': batch_loss, 'test/batch':batch + (len(test_dataloader) * epoch)})
                
                epoch_loss /= len(test_dataloader)
                print(f'test: test/epoch-loss={epoch_loss:.4f}')

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
    parser.add_argument('--dataset', default=['datasets/dataset-viral.hdf5', 'proteins', 'residue'], nargs=3, help='<hdf5 dataset filepath> {interactions, proteins, ligands} {residue, chain}')
    parser.add_argument('--model', default=None, help='<torch model filepath>')
    parser.add_argument('--config', default=['config-1.yaml', '10'], nargs=2, help='{<yaml sweep config filepath>, <wandb sweep author/project/id>} <number of runs>')
    parser.add_argument('--device', default='cpu', help='<torch device>')
    parser.add_argument('--wandb', default='online', choices=['online', 'offline', 'disabled'])
    args = parser.parse_args()
    
    assert args.dataset[1] in ['interactions', 'ligands', 'proteins'], 'ERROR: dataset must be "interactions", "ligands" or "proteins"'
    if args.dataset[1] == 'interactions':
        dataset = InteractionsDataset(args.dataset[0])
    elif args.dataset[1] == 'ligands':
        dataset = LigandsDataset(args.dataset[0])
    elif args.dataset[1] == 'proteins':
        dataset = ProteinsDataset(args.dataset[0])
    
    assert args.dataset[2] in ['residue', 'chain'], 'ERROR: dataset granularity must be "residue" or "chain" (ignored in "ligands" dataset)'
    if args.dataset[2] == 'chain' and args.dataset[1] != 'ligands':
        dataset.set_granularity()

    model = torch.load(args.model, pickle_module=dill) if args.model else None

    device = torch.device(args.device)
    
    wandb.login()
    sweep_id = wandb.sweep(sweep=yaml.load(open(args.config[0], 'r'), Loader)) if args.config[0].split('.')[-1] in ['yaml', 'yml'] else args.config[0]
    sweep_agent_function = functools.partial(run, dataset, model, device, args.wandb)
    wandb.agent(sweep_id=sweep_id, function=sweep_agent_function, count=int(args.config[1]))
