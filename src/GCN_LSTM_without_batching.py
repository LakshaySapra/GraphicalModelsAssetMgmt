import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv

from typing import Union, List

class GCN_LSTM(nn.Module):
    def __init__(self, LSTM_input_size:int, LSTM_output_size:int=16, LSTM_num_layers:int=2, GCN_sizes:Union[None, List[int]]=None,
                 Linear_Sizes:Union[None, List[int]]=None, dropout=0.1) -> None:
        super().__init__()
        
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
        self.linear = nn.Sequential(*([j for input, output in zip([self.GCN_sizes[-1]*3]+self.Linear_sizes[:-1], self.Linear_sizes) for j in [nn.Linear(input, output), nn.LeakyReLU()][:-1]]))

    def change_edges(self, edge, time):
        shape = edge.shape[1]
        total_edges    = edge.max().item() + 1 #ensure this is int
        return edge.repeat(time, 1, 1).view(time, 2, shape) + total_edges*torch.arange(total_edges).repeat_interleave(total_edges)
    
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
        n_stocks, time, features = x.shape
        node_features = F.leaky_relu(self.lstm(x))
        
        # For conv, we have batch*time batches, each with n_stocks nodesx1
        node_features  = node_features.permute(1, 0, 2).view(time*n_stocks, self.LSTM_output_size)
        batch_vector   = torch.arange(time).repeat_interleave(n_stocks)
        new_edge_index_stock = self.change_edges(edge_index_stock, time)
        new_edge_index_supplies_to = self.change_edges(edge_index_supplies_to, time)
        new_edge_index_supplies_from = self.change_edges(edge_index_supplies_from, time)
        
        node_features_stock = node_features
        for i in self.conv_stock:
            node_features_stock = F.leaky_relu(i(node_features_stock, new_edge_index_stock, batch=batch_vector))
        
        node_features_supplies_to = node_features
        for i in self.conv_supplies_to:
            node_features_supplies_to = F.leaky_relu(i(node_features_supplies_to, new_edge_index_supplies_to, batch=batch_vector))
        
        node_features_supplies_from = node_features
        for i in self.conv_supplies_from:
            node_features_supplies_from = F.leaky_relu(i(node_features_supplies_from, new_edge_index_supplies_from, batch=batch_vector))
        
        out = torch.cat([node_features_stock, node_features_supplies_to, node_features_supplies_from], 1)
        
        out = torch.squeeze(self.linear(node_features))
        
        #currently, shape batch*time*n_stocks. convert to batch, n_stocks, time
        return out.reshape(time, n_stocks).T
        
        