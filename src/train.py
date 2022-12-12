from dataset import GNNDataset
from torch.utils.data import DataLoader
from GCN_LSTM_without_batching import GCN_LSTM
from LSTM_only import LSTM_only
import torch
from tqdm.auto import tqdm
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler
import pandas as pd
import matplotlib.pyplot as plt

dropna = lambda x: x[~x.isnan()]

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
    trainer.fit(model=model, train_dataloaders=train_dataloader)
    final_dataset = GNNDataset(get_sp=True)
    final_dataloader = DataLoader(final_dataset, batch_size=1, shuffle=False, num_workers=7, pin_memory=True, persistent_workers=True)
    cumulative = {'date': [], 'returns':[]}
    for i in final_dataloader:
        predicted_rets, actual_rets = model.validation_step(i, None)
        date = final_dataset.date_lst[i[-1]]
        long_threshold_supliers = torch.quantile(predicted_rets, q=0.8)
        #short_threshold_supliers = torch.quantile(predicted_rets, q=0.8)
        long_threshold_data = (dropna(actual_rets[predicted_rets >= long_threshold_supliers]) - final_dataset.sp.loc[date].sprtrn).mean().detach().item()
        cumulative['date'].append(date)
        cumulative['returns'].append(long_threshold_data)
    cumulative = pd.DataFrame(cumulative).set_index('date')
    cumulative.to_csv('../results/final_returns.csv')
    print(cumulative)
    (cumulative+1).cumprod().plot()
    plt.savefig('../results/cumulative_plot.png')

# Notes
# Getting PyG to work:
# https://stackoverflow.com/questions/69952475/how-to-solve-the-pytorch-geometric-install-error-undefined-symbol-zn5torch3ji worked