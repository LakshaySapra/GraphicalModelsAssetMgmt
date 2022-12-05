from dataset import GNNDataset
from torch.utils.data import DataLoader
from GCN_LSTM_without_batching import GCN_LSTM
import torch

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GNNDataset(device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False, persistent_workers=False) #takes care of shuffling
    print([i.shape for i in next(iter(dataloader))])
    #model = GCN_LSTM()

#https://stackoverflow.com/questions/69952475/how-to-solve-the-pytorch-geometric-install-error-undefined-symbol-zn5torch3ji worked