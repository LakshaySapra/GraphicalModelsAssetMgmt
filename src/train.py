from dataset import GNNDataset
from torch.utils.data import DataLoader
from GCN_LSTM_without_batching import GCN_LSTM
import torch
import pytorch_lightning as pl

if __name__ == '__main__':
    dataset = GNNDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False, persistent_workers=False) #takes care of shuffling
    model = GCN_LSTM(13)
    #trainer = pl.Trainer(gpus=1, limit_train_batches=100, max_epochs=1)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model=model, train_dataloaders=dataloader)

# Notes
# Getting PyG to work:
# https://stackoverflow.com/questions/69952475/how-to-solve-the-pytorch-geometric-install-error-undefined-symbol-zn5torch3ji worked