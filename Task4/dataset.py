import numpy
import torch
from torch.utils.data import Dataset, DataLoader

import Task4.config as config


class SeqDataset(Dataset):


    def __init__(self, X: numpy.ndarray, M: numpy.ndarray, y: numpy.ndarray, ids: numpy.ndarray):
        self.X = torch.from_numpy(X).float() # (N, T, F)
        self.M = torch.from_numpy(M).float() # (N, T, F)
        self.y = torch.from_numpy(y).long() # (N,)
        self.ids = ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.ids[idx], self.X[idx], self.M[idx], self.y[idx]


class FlatDataset:

    def __init__(self, X: numpy.ndarray, M: numpy.ndarray, y: numpy.ndarray, ids: numpy.ndarray):
        self.y = y
        self.ids = ids
        self.X_flat = self._summarise(X, M)

    @staticmethod
    def _summarise(X: numpy.ndarray, M: numpy.ndarray) -> numpy.ndarray:
        N, T, F = X.shape
        out = numpy.zeros((N, F * 7), dtype=numpy.float32)
        for f in range(F):
            col = X[:, :, f]    # (N, T)
            msk = M[:, :, f]    # (N, T)
            out[:, f * 7 + 0] = col.mean(axis=1)
            out[:, f * 7 + 1] = col.std(axis=1)
            out[:, f * 7 + 2] = col.min(axis=1)
            out[:, f * 7 + 3] = col.max(axis=1)
            out[:, f * 7 + 4] = col[:, 0] # first value
            out[:, f * 7 + 5] = col[:, -1] # last value
            out[:, f * 7 + 6] = msk.mean(axis=1) # % observed
        return out

    def as_arrays(self):
        return self.X_flat, self.y


def make_seq_loaders(splits: dict, batch_size: int = None,
                     num_workers: int = 0) -> dict:
    
    #Convenience: turns split dict -> DataLoader dict.

    bs = batch_size or config.TrainConfig.batch
    loaders = {}
    for name in ["train", "val", "test"]:
        ds = SeqDataset(splits[name]["X"], splits[name]["M"], splits[name]["y"], splits[name]["ids"])
        loaders[name] = DataLoader(
            ds,
            batch_size=bs,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
    return loaders


def make_flat_data(splits: dict) -> dict:
    
    #Convenience: returns dict of FlatDataset objects.
    flat = {}
    for name in ["train", "val", "test"]:
        flat[name] = FlatDataset(
            splits[name]["X"], splits[name]["M"], splits[name]["y"], splits[name]["ids"]
        )
    return flat
