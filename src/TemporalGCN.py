import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvLSTM
import pytorch_lightning as pl

from typing import Union

class RecurrentGCN(LightningModule):
    def __init__(self, input_features:int, LSTM_output:int) -> None:
        super(RecurrentGCN, self).__init__()
        self.recurrent_1 = GConvLSTM(input_features, LSTM_output, 5)
        self.recurrent_2 = GConvLSTM(LSTM_output, LSTM_output, 5)
        self.Linear_sizes = [16, 8, 4, 1]
        self.linear = nn.Sequential(*([j for input, output in zip([LSTM_output]+self.Linear_sizes[:-1], self.Linear_sizes) for j in [nn.Linear(input, output), nn.LeakyReLU()][:-1]]))
        torch.nn.Linear(16, 1)

    def forward(self, x:torch.Tensor, edge_index:torch.Tensor, edge_weight:Union[None, torch.Tensor]=None) -> torch.Tensor:
        x = self.recurrent_1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.recurrent_2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear(x)
        return x
    
    def configure_optimizers (self):
        optimizer = optim.Adam(self.parameters(), lr=(self.lr or 1e-5))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}
    
    def training_step (self, train_batch, batch_idx):
        sector_edge_lst, c_edge_lst, s_edge_lst, cur_hist_ret_df, cur_weekly_ret_df, mask = train_batch
        
        # Single element, remove batch argument
        sector_edge_lst = sector_edge_lst.squeeze(dim=0)
        c_edge_lst = c_edge_lst.squeeze(dim=0)
        s_edge_lst = s_edge_lst.squeeze(dim=0)
        cur_hist_ret_df = torch.nan_to_num(cur_hist_ret_df).squeeze(dim=0)
        cur_weekly_ret_df = cur_weekly_ret_df.squeeze(dim=0)
        mask = mask.squeeze(dim=0)[:, None] > 0.5
        
        #cur_hist_ret_df = torch.nan_to_num(cur_hist_ret_df, nan=-float('inf'))
        predicted_rets = self.forward({'stock': cur_hist_ret_df}, {('stock', 'sector', 'stock'): sector_edge_lst, ('stock', 'supplier', 'stock'): s_edge_lst, ('stock', 'customer', 'stock'): c_edge_lst})
        time = predicted_rets.shape[1]
        predicted_rets = torch.masked_select(predicted_rets[:, (time//2):]   , mask)
        actual_rets    = torch.masked_select(cur_weekly_ret_df[:, (time//2):], mask)
        loss = (predicted_rets - actual_rets).square().mean().sqrt()
        self.log('train_loss', loss)
        r2   = 1 - (predicted_rets - actual_rets).square().mean()/(self.train_avg - actual_rets).square().mean()
        self.log('train_r2', r2)
        final = loss + (0 if torch.isnan(r2).item() else (1-r2))
        return final
    
    def validation_step(self, val_batch, batch_idx): 
        sector_edge_lst, c_edge_lst, s_edge_lst, cur_hist_ret_df, cur_weekly_ret_df, mask = val_batch
        
        # Single element, remove batch argument
        sector_edge_lst = sector_edge_lst.squeeze(dim=0)
        c_edge_lst = c_edge_lst.squeeze(dim=0)
        s_edge_lst = s_edge_lst.squeeze(dim=0)
        cur_hist_ret_df = torch.nan_to_num(cur_hist_ret_df).squeeze(dim=0)
        cur_weekly_ret_df = cur_weekly_ret_df.squeeze(dim=0)
        mask = mask.squeeze(dim=0)[:, None] > 0.5
        
        #cur_hist_ret_df = torch.nan_to_num(cur_hist_ret_df, nan=-float('inf'))
        predicted_rets = self.forward(cur_hist_ret_df, sector_edge_lst, s_edge_lst, c_edge_lst)
        time = predicted_rets.shape[1]
        predicted_rets = torch.masked_select(predicted_rets[:, (time//2):]   , mask)
        actual_rets    = torch.masked_select(cur_weekly_ret_df[:, (time//2):], mask)
        loss = (predicted_rets - actual_rets).square().mean().sqrt()
        self.log('val_loss', loss)
        r2   = 1 - (predicted_rets - actual_rets).square().mean()/(self.test_avg - actual_rets).square().mean()
        self.log('val_r2', r2)
        return loss
    
    # model = to_hetero(model, data.metadata(), aggr='sum')