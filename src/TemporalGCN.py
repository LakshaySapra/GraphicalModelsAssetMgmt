import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvLSTM

from typing import Union

class RecurrentGCN(torch.nn.Module):

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
    
    # model = to_hetero(model, data.metadata(), aggr='sum')