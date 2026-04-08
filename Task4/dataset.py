from pathlib import Path

import numpy
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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


def _load_race_lookup():
    repo_root = Path(__file__).resolve().parents[1]
    demo_path = repo_root / "Tests" / "outputs" / "Task2Demographics.csv"
    demo = pd.read_csv(demo_path)
    demo.columns = demo.columns.str.strip()
    return dict(zip(demo["stay_id"], demo["race"]))


def _make_race_sample_weights(ids, max_weight: float = 10.0):
    race_lookup = _load_race_lookup()
    races = [race_lookup.get(int(sid), None) for sid in ids]

    counts = {}
    for race in races:
        if race is None or race == "Multipule":
            continue
        counts[race] = counts.get(race, 0) + 1

    if not counts:
        return torch.ones(len(ids), dtype=torch.double)

    total = sum(counts.values())
    raw_weights = {race: total / count for race, count in counts.items()}
    capped_weights = {race: min(weight, max_weight) for race, weight in raw_weights.items()}

    weights = []
    for race in races:
        if race in capped_weights:
            weights.append(capped_weights[race])
        else:
            weights.append(1.0)

    return torch.tensor(weights, dtype=torch.double)


def make_seq_loaders(splits: dict, batch_size: int = None,
                     num_workers: int = 0,
                     sampling_strategy: str = None,
                     max_sampling_weight: float = 10.0) -> dict:
    
    #Convenience: turns split dict -> DataLoader dict.

    bs = batch_size or config.TrainConfig.batch
    loaders = {}
    for name in ["train", "val", "test"]:
        ds = SeqDataset(splits[name]["X"], splits[name]["M"], splits[name]["y"], splits[name]["ids"])
        sampler = None
        shuffle = (name == "train")

        if name == "train" and sampling_strategy == "race_balanced":
            sample_weights = _make_race_sample_weights(
                splits[name]["ids"],
                max_weight=max_sampling_weight,
            )
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False

        loaders[name] = DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            sampler=sampler,
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
