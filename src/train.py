from dataset import GNNDataset
from torch.utils.data import DataLoader
from GCN_LSTM_without_batching import GCN_LSTM
from LSTM_only import LSTM_only
import torch
from tqdm.auto import tqdm
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler

if __name__ == '__main__':
    train_dataset = GNNDataset(end_idx=300)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=7, pin_memory=True, persistent_workers=True) #takes care of shuffling
    test_dataset = GNNDataset(begin_idx=301, end_idx=400)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=7, pin_memory=True, persistent_workers=True) #takes care of shuffling
    model = GCN_LSTM(13, debug=lambda x: None, GCN_sizes = [16, 16], LSTM_num_layers=2, train_avg=train_dataset.avg_rets, test_avg=test_dataset.avg_rets)
    #model = LSTM_only(13, debug=lambda x: None, LSTM_num_layers=4)
    wandb_logger = WandbLogger(project='PGM Project', log_model="all")
    profiler = None #AdvancedProfiler(dirpath=".", filename="perf_logs")
    trainer = pl.Trainer(gpus=1, max_epochs=1, logger=wandb_logger, accumulate_grad_batches=20, auto_lr_find=True, gradient_clip_val=0.5, profiler=profiler)
    #trainer.tune(model, train_dataloaders=train_dataloader)
    #trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
    final_dataset = GNNDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=7, pin_memory=True, persistent_workers=True)
    for i in train_dataloader:
        predicted_rets = model.validation_step(i, None)[:, -1]
        long_threshold_supliers = torch.quantile(predicted_rets, q=0.8)
        #short_threshold_supliers = torch.quantile(predicted_rets, q=0.8)
        long_threshold_data = i[4][:, -1] 

# Notes
# Getting PyG to work:
# https://stackoverflow.com/questions/69952475/how-to-solve-the-pytorch-geometric-install-error-undefined-symbol-zn5torch3ji worked