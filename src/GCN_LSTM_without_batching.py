import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
from torchmetrics.functional import r2_score

from typing import Union, List

class GCN_LSTM(pl.LightningModule):#(nn.Module):
    def __init__(self, LSTM_input_size:int, LSTM_output_size:int=16, LSTM_num_layers:int=2, GCN_sizes:Union[None, List[int]]=None,
                 Linear_Sizes:Union[None, List[int]]=None, dropout=0.1, lr=5e-4, debug=lambda x: print(x),
                 train_avg=None, test_avg=None) -> None:
        super().__init__()
        self.lr = lr
        self.train_avg=train_avg
        self.test_avg=test_avg
        self.save_hyperparameters()
        
        if GCN_sizes is None:
            GCN_sizes = [16, 16]
        if Linear_Sizes is None:
            Linear_Sizes = [16, 8, 4, 1]
        
        self.LSTM_input_size = LSTM_input_size
        self.LSTM_output_size = LSTM_output_size
        self.GCN_sizes = GCN_sizes
        self.Linear_sizes = Linear_Sizes
        
        self.lstm   = nn.LSTM(self.LSTM_input_size, self.LSTM_output_size, LSTM_num_layers, batch_first=True, dropout=dropout)
        self.conv_stock   = nn.ModuleList([GCNConv(input, output, improved=True) for input, output in zip([self.LSTM_output_size]+self.GCN_sizes[:-1], self.GCN_sizes)])
        self.conv_supplies_to   = nn.ModuleList([GCNConv(input, output, improved=True) for input, output in zip([self.LSTM_output_size]+self.GCN_sizes[:-1], self.GCN_sizes)])
        self.conv_supplies_from   = nn.ModuleList([GCNConv(input, output, improved=True) for input, output in zip([self.LSTM_output_size]+self.GCN_sizes[:-1], self.GCN_sizes)])
        self.linear = nn.Sequential(*([j for input, output in zip([self.GCN_sizes[-1]*3+self.LSTM_output_size]+self.Linear_sizes[:-1], self.Linear_sizes) for j in [nn.Linear(input, output), nn.LeakyReLU()]][:-1]))
        self.debug = debug

    def change_edges(self, edge, time, n_nodes):
        shape = edge.shape[1]
        #return edge.repeat(time, 1, 1).view(time, 2, shape) + total_edges*torch.arange(total_edges).repeat_interleave(total_edges)
        a = edge.repeat(time, 1, 1).view(time, 2, shape).permute(1, 0, 2)
        b = torch.arange(time, device=self.device)[None, :, None]*n_nodes
        return (a + b).reshape(2, -1)
    
    def forward(self, x:torch.Tensor, edge_index_stock:torch.IntTensor, edge_index_supplies_to:torch.IntTensor, edge_index_supplies_from:torch.IntTensor) -> torch.Tensor:
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
        #print('a1', (torch.isnan(x).sum(dim=(1, 2)) > 0.5).sum())
        x, _ = self.lstm(x) #Also outputs (final hidden states, the final cell state) https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        #print('a2', (torch.isnan(x).sum(dim=(1, 2)) > 0.5).sum())
        node_features = F.leaky_relu(x)
        #print('a3', (torch.isnan(node_features).sum(dim=(1, 2)) > 0.5).sum())
        
        # For conv, we have batch*time batches, each with n_stocks nodesx1
        self.debug('b')
        node_features  = node_features.permute(1, 0, 2).view(time*n_stocks, self.LSTM_output_size)
        #batch_vector   = torch.arange(time).repeat_interleave(n_stocks)
        new_edge_index_stock = self.change_edges(edge_index_stock, time, n_stocks)
        new_edge_index_supplies_to = self.change_edges(edge_index_supplies_to, time, n_stocks)
        new_edge_index_supplies_from = self.change_edges(edge_index_supplies_from, time, n_stocks)
        
        self.debug('c')
        node_features_stock = node_features
        for i in self.conv_stock:
            #node_features_stock = F.leaky_relu(i(node_features_stock, new_edge_index_stock, batch=batch_vector))
            #self.debug((node_features_stock.shape, new_edge_index_stock.shape, new_edge_index_stock.max().item()))
            node_features_stock = F.leaky_relu(i(node_features_stock, new_edge_index_stock))
            #print('c', (torch.isnan(node_features_stock.view(time,n_stocks,-1)).sum(dim=(0, 2)) > 0.5).sum())
        
        self.debug('d')
        node_features_supplies_to = node_features
        for i in self.conv_supplies_to:
            #node_features_supplies_to = F.leaky_relu(i(node_features_supplies_to, new_edge_index_supplies_to, batch=batch_vector))
            #self.debug((node_features_supplies_to.shape, new_edge_index_supplies_to.shape))
            node_features_supplies_to = F.leaky_relu(i(node_features_supplies_to, new_edge_index_supplies_to))
            #print('d', (torch.isnan(node_features_supplies_to.view(time,n_stocks,-1)).sum(dim=(0, 2)) > 0.5).sum())
        
        self.debug('e')
        node_features_supplies_from = node_features
        for i in self.conv_supplies_from:
            #node_features_supplies_from = F.leaky_relu(i(node_features_supplies_from, new_edge_index_supplies_from, batch=batch_vector))
            node_features_supplies_from = F.leaky_relu(i(node_features_supplies_from, new_edge_index_supplies_from))
            #print('e', (torch.isnan(node_features_supplies_from.view(time,n_stocks,-1)).sum(dim=(0, 2)) > 0.5).sum())
        
        self.debug('f')
        out = torch.cat([node_features_stock, node_features_supplies_to, node_features_supplies_from, node_features], 1)
        
        self.debug('g')
        out = torch.squeeze(self.linear(out))
        #print('g', (torch.isnan(out.view(time,n_stocks,-1)).sum(dim=(0, 2)) > 0.5).sum())
        
        #currently, shape batch*time*n_stocks. convert to batch, n_stocks, time
        return out.reshape(time, n_stocks).T
    
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
        predicted_rets = self.forward(cur_hist_ret_df, sector_edge_lst, s_edge_lst, c_edge_lst)
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
        sector_edge_lst, c_edge_lst, s_edge_lst, cur_hist_ret_df, _, mask = val_batch
        
        # Single element, remove batch argument
        sector_edge_lst = sector_edge_lst.squeeze(dim=0)
        c_edge_lst = c_edge_lst.squeeze(dim=0)
        s_edge_lst = s_edge_lst.squeeze(dim=0)
        cur_hist_ret_df = torch.nan_to_num(cur_hist_ret_df).squeeze(dim=0)
        mask = mask.squeeze(dim=0)[:, None] > 0.5
        
        #cur_hist_ret_df = torch.nan_to_num(cur_hist_ret_df, nan=-float('inf'))
        predicted_rets = self.forward(cur_hist_ret_df, sector_edge_lst, s_edge_lst, c_edge_lst)
        time = predicted_rets.shape[1]
        predicted_rets = torch.masked_select(predicted_rets[:, (time//2):]   , mask)
        return predicted_rets