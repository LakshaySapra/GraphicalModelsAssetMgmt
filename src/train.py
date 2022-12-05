from dataset import GNNDataset
from torch.utils.data import DataLoader
from GCN_LSTM_without_batching import GCN_LSTM

if __name__ == '__main__':
    dataset = GNNDataset(device='cuda')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False, persistent_workers=False) #takes care of shuffling
    print([i for i in next(iter(dataloader))])

#https://stackoverflow.com/questions/69952475/how-to-solve-the-pytorch-geometric-install-error-undefined-symbol-zn5torch3ji worked