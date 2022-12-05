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
        self.conv   = nn.ModuleList([GCNConv(input, output, improved=True) for input, output in zip([self.LSTM_output_size]+self.GCN_sizes[:-1], self.GCN_sizes)])
        self.linear = nn.Sequential(*[nn.Linear(input, output) for input, output in zip([self.GCN_sizes[-1]]+self.Linear_sizes[:-1], self.Linear_sizes)])

    def forward(self, x:torch.Tensor, edge_index:torch.IntTensor) -> torch.Tensor:
        """forward function

        Args:
            x (torch.Tensor): Tensor of dimension batch*n_stocks*time*features
            edge_index (_type_): Tensor of dimension batch*2*edges*features
        Returns:
            returns (torch.Tensor): Expected returns, dimension batch*n_stocks*time
        """
        batch, n_stocks, time, features = x.shape
        node_features = self.lstm(x.view(batch*n_stocks, time, features)).view(batch, n_stocks, time, self.LSTM_output_size)
        
        # For conv, we have batch*time batches, each with n_stocks nodes
        node_features  = node_features.permute(0, 2, 1, 3).view(batch*time*n_stocks, self.LSTM_output_size)
        batch_vector   = torch.arange(batch*time).repeat_interleave(n_stocks)
        total_edges    = edge_index.max().item() + 1 #ensure this is int
        new_edge_index = edge_index.repeat(batch*time) + total_edges*torch.arange(total_edges).repeat_interleave(total_edges)
        
        for i in self.conv:
            node_features = i(node_features, new_edge_index, batch=batch_vector)
        
        out=torch.squeeze(self.linear(node_features))
        
        #currently, shape batch*time*n_stocks. convert to batch, n_stocks, time
        return out.reshape(batch, time, n_stocks).permute(0, 2, 1)
        
        