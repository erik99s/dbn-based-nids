import pandas as pd

import torch
from torch.utils.data.dataset import Dataset

from utils import utils


class CICIDSDataset(Dataset):

    def __init__(self, features_file, target_file, transform=None, target_transform=None):
        """
        Args:
            features_file (string): Path to the csv file with features.
            target_file (string): Path to the csv file with labels.
            transform (callable, optional): Optional transform to be applied on features.
            target_transform (callable, optional): Optional transform to be applied on labels.
        """
        self.features = pd.read_pickle(features_file)
        self.labels = pd.read_pickle(target_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features.iloc[idx, :]
        label = self.labels.iloc[idx]
        if self.transform:
            feature = self.transform(feature.values, dtype=torch.float32)
        if self.target_transform:
            label = self.target_transform(label, dtype=torch.int64)
        return feature, label

class FilteredDataset(Dataset):
    def __init__(self, feature_file, target_file, transform=None, target_transform=None):

        """
        Args:
        Same as CICIDS2017Dataset, but feature file and target file isn't a read file
        sel
        """
        self.features = feature_file
        self.labels = target_file
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]   # list indexing
        label = self.labels[idx]

        if self.transform:
            feature = self.transform(feature, dtype=torch.float32)
        if self.target_transform:
            label = self.target_transform(label, dtype=torch.int64)
        
        return feature, label


def get_dataset(data_path: str, index: int):

    if index == 0:
        train_data = CICIDSDataset(
            features_file=f"{data_path}/processed0/train/train_features.pkl",
            target_file=f"{data_path}/processed0/train/train_labels.pkl",
            transform=torch.tensor,
            target_transform=torch.tensor
        )

        val_data = CICIDSDataset(
            features_file=f"{data_path}/processed0/val/val_features.pkl",
            target_file=f"{data_path}/processed0/val/val_labels.pkl",
            transform=torch.tensor,
            target_transform=torch.tensor
        )

        test_data = CICIDSDataset(
            features_file=f"{data_path}/processed0/test/test_features.pkl",
            target_file=f"{data_path}/processed0/test/test_labels.pkl",
            transform=torch.tensor,
            target_transform=torch.tensor
        )

        print(train_data.labels.value_counts())
        print(val_data.labels.value_counts())
        print(test_data.labels.value_counts())
        return train_data, val_data, test_data

    if index == 1:

        train_AE_data = CICIDSDataset(
            features_file=f"{data_path}/processed3/trainAE/train_AE_features.pkl",
            target_file=f"{data_path}/processed3/trainAE/train_AE_labels.pkl",
            transform=torch.tensor,
            target_transform=torch.tensor
        )

        val_AE_data = CICIDSDataset(
            features_file=f"{data_path}/processed3/valAE/val_AE_features.pkl",
            target_file=f"{data_path}/processed3/valAE/val_AE_labels.pkl",
            transform=torch.tensor,
            target_transform=torch.tensor
        )

        train_DBN_data = CICIDSDataset(
            features_file=f"{data_path}/processed3/trainDBN/train_DBN_features.pkl",
            target_file=f"{data_path}/processed3/trainDBN/train_DBN_labels.pkl",
            transform=torch.tensor,
            target_transform=torch.tensor
        )

        val_DBN_data = CICIDSDataset(
            features_file=f"{data_path}/processed3/valDBN/val_DBN_features.pkl",
            target_file=f"{data_path}/processed3/valDBN/val_DBN_labels.pkl",
            transform=torch.tensor,
            target_transform=torch.tensor
        )

        test_data = CICIDSDataset(
            features_file=f"{data_path}/processed3/test/test_features.pkl",
            target_file=f"{data_path}/processed3/test/test_labels.pkl",
            transform=torch.tensor,
            target_transform=torch.tensor
        )

        print(train_AE_data.labels.value_counts())
        print(val_AE_data.labels.value_counts())
        print(train_DBN_data.labels.value_counts())
        print(val_DBN_data.labels.value_counts())
        print(test_data.labels.value_counts())

        return train_AE_data, val_AE_data, train_DBN_data, val_DBN_data, test_data


def load_data(data_path: str, batch_size: int, index: int):

    if index == 0:
        train_data, val_data, test_data = get_dataset(data_path=data_path, index=index)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=val_data,
            batch_size=batch_size,
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=1,
            shuffle=True
        )

        return train_loader, valid_loader, test_loader

    if index == 1:
        train_AE_data, val_AE_data, train_DBN_data, val_DBN_data, test_data = get_dataset(data_path=data_path, index=index)

        train_AE_loader = torch.utils.data.DataLoader(
            dataset=train_AE_data,
            batch_size=batch_size,
            shuffle=True
        )
        val_AE_loader = torch.utils.data.DataLoader(
            dataset=val_AE_data,
            batch_size=batch_size,
            shuffle=True
        )
        train_DBN_loader = torch.utils.data.DataLoader(
            dataset=train_DBN_data,
            batch_size=batch_size,
            shuffle=True
        )
        val_DBN_loader = torch.utils.data.DataLoader(
            dataset=val_DBN_data,
            batch_size=batch_size,
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=1,
            shuffle=True
        )

        return train_AE_loader, val_AE_loader, train_DBN_loader, val_DBN_loader, test_data
