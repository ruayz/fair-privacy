
import copy
import json
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from transformers import AutoModelForCausalLM

from dataset.utils import get_dataloader
from trainers.utils import fairness_metrics_per_group
from models import AlexNet, CNN, MLP, WideResNet, LR
from torchvision import models
from trainers.default_trainer import train, inference
from trainers import create_trainer
from trainers.fast_train import (
    load_cifar10_data,
    NetworkEMA,
    make_net,
    print_training_details,
    logging_columns_list,
    fast_train_fun,
)
from trainers.train_transformers import *

INPUT_OUTPUT_SHAPE = {
    "mnist": [1, 10],
    "cifar10": [3, 10],
    # "cifar100": [3, 100],
    # "purchase100": [600, 100],
    "texas100": [6169, 100],
    "utkface": [1, 5],
    "raceface": [1, 5],
}


def get_model(model_type: str, dataset_name: str, dataset):
    """
    Instantiate and return a model based on the given model type and dataset name.

    Args:
        model_type (str): Type of the model to be instantiated.
        dataset_name (str): Name of the dataset the model will be used for.

    Returns:
        torch.nn.Module or PreTrainedModel: An instance of the specified model, ready for training or inference.
    """
    if model_type == "gpt2":
        return AutoModelForCausalLM.from_pretrained("gpt2")

    if dataset_name == "mnist" or dataset_name == "cifar10" or dataset_name == "utkface" or dataset_name == "raceface":
        num_classes = INPUT_OUTPUT_SHAPE[dataset_name][1]
        in_shape = INPUT_OUTPUT_SHAPE[dataset_name][0]
    else:
        num_classes = 2
        in_shape = len(dataset[0][0])

    if model_type == "MLP":  
        return MLP(in_shape=in_shape, num_classes=num_classes)
    elif model_type == "resnet":
        return models.resnet18(num_classes=num_classes)
    elif model_type == "CNN":
        return CNN(in_shape=in_shape, num_classes=num_classes)
    elif model_type == "LR":
        return LR(in_shape=in_shape, num_classes=num_classes)
    elif model_type == "alexnet":
        return AlexNet(num_classes=num_classes)
    elif model_type == "wrn16-4":
        return WideResNet(nin=in_shape, nclass=num_classes, depth=16, width=4)
    elif model_type == "wrn28-1":
        return WideResNet(nin=in_shape, nclass=num_classes, depth=28, width=1)
    elif model_type == "wrn28-2":
        return WideResNet(nin=in_shape, nclass=num_classes, depth=28, width=2)
    elif model_type == "wrn28-10":
        return WideResNet(nin=in_shape, nclass=num_classes, depth=28, width=10)
    elif model_type == "vgg16":
        return torchvision.models.vgg16(pretrained=False)
    else:
        raise NotImplementedError(f"{model_type} is not implemented")


def load_existing_model(
    model_metadata: dict, dataset, device: str
):
    """Load an existing model from disk based on the provided metadata.

    Args:
        model_metadata (dict): Metadata dictionary containing information about the model.
        dataset (torchvision.datasets): Dataset object used to instantiate the model.
        device (str): The device on which to load the model, such as 'cpu' or 'cuda'.

    Returns:
        model (torch.nn.Module): Loaded model object with weights restored from disk.
    """
    model_name = model_metadata["model_name"]
    dataset_name = model_metadata["dataset"]

    if model_name != "speedyresnet":
        model = get_model(model_name, dataset_name, dataset)
    else:
        data = load_cifar10_data(dataset, [0], [0], device=device)
        model = NetworkEMA(make_net(data, device=device))

    model_checkpoint_extension = os.path.splitext(model_metadata["model_path"])[1]
    if model_checkpoint_extension == "pkl":
        with open(model_metadata["model_path"], "rb") as file:
            model_weight = pickle.load(file)
        model.load_state_dict(model_weight)
    elif model_checkpoint_extension == "pt" or model_checkpoint_extension == "pth":
        model.load_state_dict(torch.load(model_metadata["model_path"]))
    elif model_checkpoint_extension == "":
        if isinstance(model, PreTrainedModel):
            model = model.from_pretrained(model_metadata["model_path"])
        else:
            raise ValueError(f"Model path is invalid.")
    return model


def load_models(log_dir, dataset, num_models, configs, logger):
    """
    Load trained models from disk if available.

    Args:
        log_dir (str): Path to the directory containing model logs and metadata.
        dataset (torchvision.datasets or df): Dataset object used for model training.
        num_models (int): Number of models to be loaded from disk.
        configs (dict): Dictionary of configuration settings, including device information.
        logger (logging.Logger): Logger object for logging the model loading process.

    Returns:
        model_list (list of nn.Module): List of loaded model objects.
        all_memberships (np.array): Membership matrix for all loaded models, indicating training set membership.
    """
    experiment_dir = f"{log_dir}/models"
    if os.path.exists(f"{experiment_dir}/models_metadata.json"):
        with open(f"{experiment_dir}/models_metadata.json", "r") as f:
            model_metadata_dict = json.load(f)
        all_memberships = np.load(f"{experiment_dir}/memberships.npy")
        if len(model_metadata_dict) < num_models:
            return None, None
    else:
        return None, None

    # model_list = []
    # for model_idx in range(len(model_metadata_dict)):
    #     logger.info(f"Loading model {model_idx}")
    #     model_obj = load_existing_model(
    #         model_metadata_dict[str(model_idx)],
    #         dataset,
    #         configs["audit"]["device"],
    #     )
    #     model_list.append(model_obj)
    #     if len(model_list) == num_models:
    #         break
    return False, all_memberships


def train_models(log_dir, dataset, data_split_info, all_memberships, configs, logger, outcome_fairness=False):
    """
    Train models based on the dataset split information.

    Args:
        log_dir (str): Path to the directory where models and logs will be saved.
        dataset (torchvision.datasets or GroupLabelDataset): Dataset object used for training the models.
        data_split_info (list): List of dictionaries containing training and test split information for each model.
        all_memberships (np.array): Membership matrix indicating which samples were used in training each model.
        configs (dict): Configuration dictionary containing training settings.
        logger (logging.Logger): Logger object for logging the training process.

    Returns:
        model_list (list of nn.Module): List of trained model objects.
    """
    experiment_dir = f"{log_dir}/models"
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Training {len(data_split_info)} models")

    model_list = prepare_models(
        experiment_dir, dataset, data_split_info, all_memberships, configs, logger, outcome_fairness
    )
    return model_list


def split_dataset_for_training(dataset_size, num_model_pairs, ratio):
    """
    Split dataset into training and test partitions for model pairs.

    Args:
        dataset_size (int): Total number of samples in the dataset.
        num_model_pairs (int): Number of model pairs to be trained, with each pair trained on different halves of the dataset.

    Returns:
        data_split (list): List of dictionaries containing training and test split indices for each model.
        master_keep (np.array): D boolean array indicating the membership of samples in each model's training set.
    """
    data_splits = []
    indices = np.arange(dataset_size)
    #split_index = len(indices) // 2
    split_index = int(len(indices) * ratio)
    master_keep = np.full((2 * num_model_pairs, dataset_size), True, dtype=bool)

    for i in range(num_model_pairs):
        np.random.shuffle(indices)
        master_keep[i * 2, indices[split_index:]] = False
        master_keep[i * 2 + 1, indices[:split_index]] = False
        keep = master_keep[i * 2, :]
        train_indices = np.where(keep)[0]
        test_indices = np.where(~keep)[0]
        data_splits.append(
            {
                "train": train_indices,
                "test": test_indices,
            }
        )
        data_splits.append(
            {
                "train": test_indices,
                "test": train_indices,
            }
        )

    return data_splits, master_keep

def split_samples_for_training(dataset_size, num_model_pairs, sample_idxs):
    data_splits = []
    indices = sample_idxs
    split_index = int(len(sample_idxs) * 0.5)
    master_keep = np.full((2 * num_model_pairs, dataset_size), True, dtype=bool)

    for i in range(num_model_pairs):
        np.random.shuffle(indices)
        master_keep[i * 2, indices[split_index:]] = False
        master_keep[i * 2 + 1, indices[:split_index]] = False
        train_keep = master_keep[i * 2, :]
        test_keep = master_keep[i * 2 + 1, :]
        train_indices = np.where(train_keep)[0]
        test_indices = np.where(test_keep)[0]
        data_splits.append(
            {
                "train": train_indices,
                "test": test_indices,
            }
        )
        data_splits.append(
            {
                "train": test_indices,
                "test": train_indices,
            }
        )

    return data_splits, master_keep


def split_one_data_for_training(dataset_size, num_model_pairs, dataset_idx):
    data_splits = []
    indices = np.arange(dataset_size)
    master_keep = np.full((2 * num_model_pairs, dataset_size), True, dtype=bool)

    for i in range(num_model_pairs):
        master_keep[i * 2, dataset_idx] = False
        keep = master_keep[i * 2, :]
        train_indices = np.where(keep)[0]
        keep = master_keep[i*2+1]
        test_indices = np.where(keep)[0]
        data_splits.append(
            {
                "train": train_indices,
                "test": test_indices,
            }
        )
        data_splits.append(
            {
                "train": test_indices,
                "test": train_indices,
            }
        )

    return data_splits, master_keep


def prepare_models(
    log_dir: str,
    dataset,
    data_split_info: list,
    all_memberships: np.array,
    configs: dict,
    logger,
    outcome_fairness=False
):
    """
    Train models based on the dataset split information and save their metadata.

    Args:
        log_dir (str): Path to the directory where model logs and metadata will be saved.
        dataset (torchvision.datasets or GroupLabelDataset): Dataset object used for training.
        data_split_info (list): List of dictionaries containing training and test split indices for each model.
        all_memberships (np.array): Membership matrix indicating which samples were used in training each model.
        configs (dict): Configuration dictionary containing training settings.
        logger (logging.Logger): Logger object for logging the training process.

    Returns:
        list: List of trained model objects.
    """
    np.save(f"{log_dir}/memberships.npy", all_memberships)

    model_metadata_dict = {}
    model_list = []

    # for split, split_info in enumerate(data_split_info):
    for split in range(len(data_split_info)):
        split_info = data_split_info[split]
        baseline_time = time.time()
        logger.info(50 * "-")
        logger.info(
            f"Training model {split}: Train size {len(split_info['train'])}, Test size {len(split_info['test'])}"
        )

        model_name, dataset_name, batch_size, device = (
            configs["train"]["model_name"],
            configs["data"]["dataset"],
            configs["train"]["batch_size"],
            configs["train"]["device"],
        )

        if model_name == "gpt2":
            hf_dataset = dataset.hf_dataset
            model, train_loss, test_loss = train_transformer(
                hf_dataset.select(split_info["train"]),
                get_model(model_name, dataset_name),
                configs,
                hf_dataset.select(split_info["test"]),
            )
            train_acc, test_acc = None, None

        ##########train here###########
        elif model_name != "speedyresnet":
            train_loader = get_dataloader(
                torch.utils.data.Subset(dataset, split_info["train"]),
                batch_size=batch_size,
                shuffle=True,
            )
            test_loader = get_dataloader(
                torch.utils.data.Subset(dataset, split_info["test"]),
                batch_size=batch_size,
            )

            train_method = configs["train"].get("method", None)
            if train_method == "regular" or train_method == "dpsgd":
            # if train_method is None:
                model = train(
                    get_model(model_name, dataset_name, dataset),
                    train_loader,
                    configs["train"],
                    test_loader,
                )
            else:
                trainer = create_trainer(
                    train_loader,
                    get_model(model_name, dataset_name, dataset),
                    configs["train"],
                    log_dir,
                )
                model = trainer.train()

            test_loss, test_acc = inference(model, test_loader, device)
            train_loss, train_acc = inference(model, train_loader, device)
            logger.info(f"Train accuracy {train_acc}, Train Loss {train_loss}")
            logger.info(f"Test accuracy {test_acc}, Test Loss {test_loss}")

            if outcome_fairness:
                acc_per_group = []
                num_groups = configs["train"]["num_groups"]
                fairness_metrics = fairness_metrics_per_group(model, test_loader, device, num_groups)
                acc_per_group = fairness_metrics["acc_per_group"]
                dop_per_group = fairness_metrics["dop_per_group"]
                eop_per_group = fairness_metrics["eop_per_group"]
                eod_per_group = fairness_metrics["eod_per_group"]

        ###############################

        elif model_name == "speedyresnet" and dataset_name == "cifar10":
            data = load_cifar10_data(
                dataset,
                split_info["train"],
                split_info["test"],
                device=device,
            )
            eval_batch_size, test_size = batch_size, len(split_info["test"])
            divisors = [
                factor
                for i in range(1, int(np.sqrt(test_size)) + 1)
                if test_size % i == 0
                for factor in (i, test_size // i)
                if factor <= eval_batch_size
            ]
            eval_batch_size = max(divisors)  # to support smaller GPUs
            print_training_details(logging_columns_list, column_heads_only=True)
            model, train_acc, train_loss, test_acc, test_loss = fast_train_fun(
                data,
                make_net(data, device=device),
                eval_batchsize=eval_batch_size,
                device=device,
            )
        else:
            raise ValueError(
                f"The {model_name} is not supported for the {dataset_name}"
            )

        model_list.append(copy.deepcopy(model))
        # del model  # 删除模型对象
        # torch.cuda.empty_cache()  # 清空 GPU 上未使用的内存

        logger.info(
            "Training model %s took %s seconds",
            split,
            time.time() - baseline_time,
        )

        model_idx = split

        # with open(f"{log_dir}/model_{model_idx}.pkl", "wb") as f:
        #     pickle.dump(model.state_dict(), f)

        model_metadata_dict[model_idx] = {
            "num_train": len(split_info["train"]),
            "method": configs["train"]["method"],
            "optimizer": configs["train"]["optimizer"],
            "batch_size": batch_size,
            "epochs": configs["train"]["epochs"],
            "model_name": model_name,
            "learning_rate": configs["train"]["learning_rate"],
            "weight_decay": configs["train"]["weight_decay"],
            # "clipping_bound": configs["train"]["clip_norm"],
            "model_path": f"{log_dir}/model_{model_idx}.pkl",
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "dataset": dataset_name,
        }

        if outcome_fairness:
            model_metadata_dict[model_idx].update({
                "accuracy_per_group": acc_per_group,
                "accuracy_parity": max(acc_per_group) - min(acc_per_group),
                "demographic_parity": max(dop_per_group) - min(dop_per_group),
                "equal_opportunity": max(eop_per_group) - min(eop_per_group),
                "equalized_odds": max(eod_per_group) - min(eod_per_group),
            })

    with open(f"{log_dir}/models_metadata.json", "w") as f:
        json.dump(model_metadata_dict, f, indent=4)

    return model_list