# Copyright (c) 2020 Data Privacy and Trustworthy Machine Learning Research Lab
# Licensed under the MIT License. See LICENSE file for details.

"""This file contains functions for loading the dataset"""

import math
import os
import pickle
import subprocess
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer

from dataset import TabularDataset, UTKDataset, GroupLabelDataset
from dataset.preprocess import *
from trainers.fast_train import get_batches, load_cifar10_data


class InfinitelyIndexableDataset(Dataset):
    """
    A PyTorch Dataset that is able to index the given dataset infinitely.
    This is a helper class to allow easier and more efficient computation later when repeatedly indexing the dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to be indexed repeatedly.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        # If the index is out of range, wrap it around
        return self.dataset[idx % len(self.dataset)]


def get_dataset(dataset_name: str, data_dir: str, logger: Any, **kwargs: Any) -> Any:
    """
    Function to load the dataset from the pickle file or download it from the internet.

    Args:
        dataset_name (str): Dataset name.
        data_dir (str): Indicate the log directory for loading the dataset.
        logger (logging.Logger): Logger object for the current run.

    Raises:
        NotImplementedError: If the dataset is not implemented.

    Returns:
        Any: Loaded dataset.
    """
    path = f"{data_dir}/{dataset_name}"
    if os.path.exists(f"{path}.pkl"):
        with open(f"{path}.pkl", "rb") as file:
            all_data = pickle.load(file)
        logger.info(f"Load data from {path}.pkl")
    elif data_dir == "data/tabular":
        if dataset_name == 'adult':
            target = "Class-label"
            df = pd.read_csv(os.path.join(data_dir, 'adult-clean.csv'))
            all_data = preprocess_adult(df, "sex", target)
        elif dataset_name == 'dutch':
            target = "occupation"
            df = pd.read_csv(os.path.join(data_dir, 'dutch.csv'))
            all_data = preprocess_dutch(df, "sex", target)
        elif dataset_name == 'bank':
            target = "y"
            df = pd.read_csv(os.path.join(data_dir, 'bank-full.csv'))
            all_data = preprocess_bank(df, "marital", target)
        elif dataset_name == 'credit':
            target = "default payment"
            df = pd.read_csv(os.path.join(data_dir, 'credit-card-clients.csv'))
            all_data = preprocess_credit(df, "SEX", target)
        elif dataset_name == 'compas':
            target = "two_year_recid"
            df = pd.read_csv(os.path.join(data_dir, 'compas-scores-two-years_clean.csv'))
            all_data = preprocess_compas(df, "race", target)
        elif dataset_name == 'law':
            target = "pass_bar"
            df = pd.read_csv(os.path.join(data_dir, 'law_school_clean.csv'))
            all_data = preprocess_law(df, "race", target)
        
        feature_columns = all_data.columns.to_list()
        feature_columns.remove(target)
        feature_columns.remove("protected_group")
        all_data = GroupLabelDataset(torch.tensor(all_data[feature_columns].values.astype(np.float32), dtype=torch.get_default_dtype()),
                                     torch.tensor(all_data[target].to_list(), dtype=torch.long),
                                     torch.tensor(all_data["protected_group"].values.tolist(), dtype=torch.long))
        with open(f"{path}.pkl", "wb") as file:
            pickle.dump(all_data, file)
        logger.info(f"Save data to {path}.pkl")
    else:
        if dataset_name == "mnist":
            # load MNIST dataset
            MNIST_MEAN = (0.1307,)
            MNIST_STD_DEV = (0.3081,)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(MNIST_MEAN, MNIST_STD_DEV)
            ])
            all_data = torchvision.datasets.MNIST(
                root=path, train=True, download=True, transform=transform
            )
            test_data = torchvision.datasets.MNIST(
                root=path, train=False, download=True, transform=transform
            )
            all_features = np.concatenate([all_data.data, test_data.data], axis=0)
            all_targets = np.concatenate([all_data.targets, test_data.targets], axis=0)
            all_data.data = all_features
            all_data.targets = all_targets
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            logger.info(f"Save data to {path}.pkl")
        elif dataset_name == "cifar10":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            all_data = torchvision.datasets.CIFAR10(
                root=path, train=True, download=True, transform=transform
            )
            test_data = torchvision.datasets.CIFAR10(
                root=path, train=False, download=True, transform=transform
            )
            all_features = np.concatenate([all_data.data, test_data.data], axis=0)
            all_targets = np.concatenate([all_data.targets, test_data.targets], axis=0)
            all_data.data = all_features
            all_data.targets = all_targets
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            logger.info(f"Save data to {path}.pkl")
        elif dataset_name == "utkface":
            dataFrame = pd.read_csv(f'{data_dir}/age_gender.gz', compression='gzip')
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.49,), (0.23,))
            ])
            all_data = UTKDataset(dataFrame, label_name="gender", transform=transform)
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            logger.info(f"Save data to {path}.pkl")
        elif dataset_name == "raceface":
            dataFrame = pd.read_csv(f'{data_dir}/age_gender.gz', compression='gzip')
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.49,), (0.23,))
            ])
            all_data = UTKDataset(dataFrame, label_name="ethnicity", transform=transform)
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            logger.info(f"Save data to {path}.pkl")
        elif dataset_name == "texas100":
            if not os.path.exists(f"{data_dir}/texas/100/feats"):
                logger.info(
                    f"{dataset_name} not found in {data_dir}/dataset_purchase. Downloading to /data..."
                )
                try:
                    # Download the dataset to /data
                    subprocess.run(
                        [
                            "wget",
                            "https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz",
                            "-P",
                            f"./{data_dir}",
                        ],
                        check=True,
                    )
                    # Extract the dataset to /data
                    subprocess.run(
                        [
                            "tar",
                            "-xf",
                            f"./{data_dir}/dataset_texas.tgz",
                            "-C",
                            "./data",
                        ],
                        check=True,
                    )
                    logger.info(
                        "Dataset downloaded and extracted to /data successfully."
                    )
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error during download or extraction: {e}")
                    raise RuntimeError("Failed to download or extract the dataset.")
                
            X = (
                pd.read_csv(
                    f"{data_dir}/texas/100/feats", header=None, encoding="utf-8"
                )
                .to_numpy()
                .astype(np.float32)
            )
            y = (
                pd.read_csv(
                    f"{data_dir}/texas/100/labels",
                    header=None,
                    encoding="utf-8",
                )
                .to_numpy()
                .reshape(-1)
                - 1
            )
            all_data = TabularDataset(X, y)
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            logger.info(f"Save data to {path}.pkl")
        else:
            raise NotImplementedError(f"{dataset_name} is not implemented")

    logger.info(f"The whole dataset size: {len(all_data)}")
    return all_data


def get_subset_indices(dataset, data_size, class_value, class_dim):
    # #class_size = (len(dataset) - data_size) // class_dim
    # ratio = data_size / len(dataset)
    class_value = np.array(class_value)
    # # Calculate the number of samples per class based on the ratio
    # class_counts = np.bincount(class_value, minlength=class_dim)
    # # Calculate how many samples to take from each class
    # samples_per_class = np.ceil(class_counts * ratio).astype(int)
    samples_per_class = int(data_size / class_dim)

    indices_to_keep = np.full(len(dataset), False, dtype=bool)
    for class_label in range(class_dim):
        class_indices = np.where(class_value == class_label)[0]

        # Sample randomly from the class_indices
        # sampled_indices = np.random.choice(class_indices, samples_per_class[class_label], replace=False)
        sampled_indices = np.random.choice(class_indices, samples_per_class, replace=False)

        indices_to_keep[sampled_indices] = True
    
    return indices_to_keep


def get_subset_dataset(dataset_name: str, data_size: int, data_dir: str, dataset, logger):
    path = data_dir + dataset_name
    if os.path.exists(f"{path}.pkl"):
        with open(f"{path}.pkl", "rb") as file:
            sub_dataset = pickle.load(file)
        logger.info(f"Load data from {path}.pkl")
        return sub_dataset

    if dataset_name == "mnist" or dataset_name == "cifar10" or dataset_name == "utkface":
        # Extract the class labels from the dataset
        label_value = []
        for _, label in dataset:
            label_value.append(label)
        class_dim = len(set(label_value))
        indices_to_keep = get_subset_indices(dataset, data_size, label_value, class_dim=class_dim)
        indices = np.where(indices_to_keep)[0]
        sub_dataset = Subset(dataset, indices)
    else:
        attribute_value = []
        for _, _, z in dataset:
            attribute_value.append(z)
        indices_to_keep = get_subset_indices(dataset, data_size, attribute_value, class_dim=2)
        
    indices = np.where(indices_to_keep)[0]
    sub_dataset = Subset(dataset, indices)
    logger.info(f"The whole dataset size: {len(sub_dataset)}")

    with open(f"{path}.pkl", "wb") as file:
        pickle.dump(sub_dataset, file)
    logger.info(f"Save data to {path}.pkl")

    return sub_dataset


def load_dataset_subsets(
    dataset,
    index: List[int],
    model_type: str,
    batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function to divide dataset into subsets and load them into device (GPU).

    Args:
        dataset (torchvision.datasets): The whole dataset.
        index (List[int]): List of sample indices.
        model_type (str): Type of the model.
        batch_size (int): Batch size for getting signals.
        device (str): Device used for loading models.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Loaded samples and their labels.
    """
    assert max(index) < len(dataset) and min(index) >= 0, "Index out of range"
    input_list = []
    targets_list = []
    if model_type != "speedyresnet":
        data_loader = get_dataloader(
            torch.utils.data.Subset(dataset, index),
            batch_size=batch_size,
            shuffle=False,
        )
        for batch in data_loader:
            if len(batch) == 3:  
                inputs, targets, groups = batch
            elif len(batch) == 2: 
                inputs, targets = batch
            input_list.append(inputs)
            targets_list.append(targets)
        inputs = torch.cat(input_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
    else:
        data = load_cifar10_data(dataset, index[:1], index, device=device)
        size = len(index)
        list_divisors = list(
            set(
                factor
                for i in range(1, int(math.sqrt(size)) + 1)
                if size % i == 0
                for factor in (i, size // i)
                if factor < batch_size
            )
        )
        batch_size = max(list_divisors)

        for inputs, targets in get_batches(
            data, key="eval", batchsize=batch_size, shuffle=False, device=device
        ):
            input_list.append(inputs)
            targets_list.append(targets)
        inputs = torch.cat(input_list, dim=0)
        targets = torch.cat(targets_list, dim=0).max(dim=1)[1]
    return inputs, targets


def get_dataloader(
    dataset,
    batch_size: int,
    loader_type: str = "torch",
    shuffle: bool = True,
) -> DataLoader:
    """
    Function to get DataLoader.

    Args:
        dataset (torchvision.datasets or GroupLabelDataset): The whole dataset.
        batch_size (int): Batch size for getting signals.
        loader_type (str): Loader type.
        shuffle (bool): Whether to shuffle dataset or not.

    Returns:
        DataLoader: DataLoader object.
    """
    if loader_type == "torch":
        repeated_data = InfinitelyIndexableDataset(dataset)
        return DataLoader(
            repeated_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=16,
        )
    else:
        raise NotImplementedError(f"{loader_type} is not supported")
