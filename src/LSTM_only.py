import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.functional import r2_score

from typing import Union, List

class LSTM_only(pl.LightningModule):#(nn.Module):
    def __init__(self, LSTM_input_size:int, LSTM_output_size:int=16, LSTM_num_layers:int=2,
                 Linear_Sizes:Union[None, List[int]]=None, dropout=0.1, debug=lambda x: print(x)) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        if Linear_Sizes is None:
            Linear_Sizes = [16, 8, 4, 1]
        
        self.LSTM_input_size = LSTM_input_size
        self.LSTM_output_size = LSTM_output_size
        self.Linear_sizes = Linear_Sizes
        
        self.lstm   = nn.LSTM(self.LSTM_input_size, self.LSTM_output_size, LSTM_num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Sequential(*([j for input, output in zip([self.LSTM_output_size]+self.Linear_sizes[:-1], self.Linear_sizes) for j in [nn.Linear(input, output), nn.LeakyReLU()]][:-1]))
        self.debug = debug

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """forward function

        Args:
            x (torch.Tensor): Tensor of dimension n_stocks*time*features
            edge_index_stock (torch.Tensor): Tensor of dimension 2*edges0
            edge_index_supplies_to (torch.Tensor): Tensor of dimension 2*edges1
            edge_index_supplies_from (torch.Tensor): Tensor of dimension 2*edges2
        Returns:
            returns (torch.Tensor): Expected returns, dimension n_stocks*time
        """
        self.debug('a')
        n_stocks, time, features = x.shape
        x, _ = self.lstm(x) #Also outputs (final hidden states, the final cell state) https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        node_features = F.leaky_relu(x)
        
        self.debug('g')
        out = torch.squeeze(self.linear(node_features.reshape(n_stocks*time, -1)))
        
        return torch.tanh(out.reshape(n_stocks, time))
    
    def configure_optimizers (self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5) 
        return optimizer
    
    def training_step (self, train_batch, batch_idx):
        _, _, _, cur_hist_ret_df, cur_weekly_ret_df, mask = train_batch
        
        # Single element, remove batch argument
        cur_hist_ret_df = torch.nan_to_num(cur_hist_ret_df).squeeze(dim=0)
        cur_weekly_ret_df = cur_weekly_ret_df.squeeze(dim=0)
        mask = mask.squeeze(dim=0)[:, None] > 0.5
        
        #cur_hist_ret_df = torch.nan_to_num(cur_hist_ret_df, nan=-float('inf'))
        predicted_rets = self.forward(cur_hist_ret_df)
        time = predicted_rets.shape[1]
        predicted_rets = torch.masked_select(predicted_rets[:, (time//2):]   , mask)
        actual_rets    = torch.masked_select(cur_weekly_ret_df[:, (time//2):], mask)
        loss = (predicted_rets - actual_rets).square().mean().sqrt()
        self.log('train_loss', loss)
        r2   = r2_score(actual_rets, predicted_rets)
        self.log('train_r2', r2)
        return loss
    
    def validation_step(self, val_batch, batch_idx): 
        _, _, _, cur_hist_ret_df, cur_weekly_ret_df, mask = val_batch
        
        # Single element, remove batch argument
        cur_hist_ret_df = torch.nan_to_num(cur_hist_ret_df).squeeze(dim=0)
        cur_weekly_ret_df = cur_weekly_ret_df.squeeze(dim=0)
        mask = mask.squeeze(dim=0)[:, None] > 0.5
        
        #cur_hist_ret_df = torch.nan_to_num(cur_hist_ret_df, nan=-float('inf'))
        predicted_rets = self.forward(cur_hist_ret_df)
        time = predicted_rets.shape[1]
        predicted_rets = torch.masked_select(predicted_rets[:, (time//2):]   , mask)
        actual_rets    = torch.masked_select(cur_weekly_ret_df[:, (time//2):], mask)
        loss = (predicted_rets - actual_rets).square().mean().sqrt()
        self.log('val_loss', loss)
        r2   = r2_score(actual_rets, predicted_rets)
        self.log('val_r2', r2)
        return loss