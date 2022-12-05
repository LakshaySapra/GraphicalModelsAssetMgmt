from dataset import GNNDataset
from torch.utils.data import DataLoader
from GCN_LSTM_without_batching import GCN_LSTM

if __name__ == '__main__':
    dataset = GNNDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False, persistent_workers=False)
    print(next(iter(dataloader)))