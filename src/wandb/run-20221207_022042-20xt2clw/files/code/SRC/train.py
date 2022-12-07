from dataset import GNNDataset
from torch.utils.data import DataLoader
from GCN_LSTM_without_batching import GCN_LSTM
import torch
from tqdm.auto import tqdm
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

if __name__ == '__main__':
    dataset = GNNDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=10, pin_memory=False, persistent_workers=False) #takes care of shuffling
    model = GCN_LSTM(13, debug=lambda x: None, GCN_sizes = [16, 32, 32, 16], LSTM_num_layers=4)
    wandb_logger = WandbLogger(project='PGM Project', log_model="all")
    trainer = pl.Trainer(gpus=1, max_epochs=25, logger=wandb_logger)
    #trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model=model, train_dataloaders=dataloader)

# Notes
# Getting PyG to work:
# https://stackoverflow.com/questions/69952475/how-to-solve-the-pytorch-geometric-install-error-undefined-symbol-zn5torch3ji worked