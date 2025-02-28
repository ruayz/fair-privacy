# Copyright (c) 2020 Data Privacy and Trustworthy Machine Learning Research Lab
# Licensed under the MIT License. See LICENSE file for details.

import argparse
import math
import time

import torch
import yaml
import numpy as np
import random

from audit import audit_models, audit_records, sample_auditing_dataset, exp_estimated_epsilon
from get_signals import get_model_signals
from models.utils import load_models, train_models, split_dataset_for_training
from util import (
    check_configs,
    setup_log,
    initialize_seeds,
    create_directories,
    load_dataset,
    load_subset_dataset,
)

# Enable benchmark mode in cudnn to improve performance when input sizes are consistent
torch.backends.cudnn.benchmark = True

def main(device, method, num_rep=1, scale=None):
    # configs = f"configs/raceface/raceface_{method}.yaml"
    # configs = f"configs/mnist/mnist_{method}.yaml"
    configs = f"configs/tabular/bank/bank_{method}.yaml"
    with open(configs, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    # Validate configurations
    check_configs(configs)
    configs["train"]["device"] = device
    configs["audit"]["device"] = device
    if scale is not None:
        method = f"{method}_{scale}"

    # Initialize seeds for reproducibility
    np.random.seed(configs["run"]["random_seed"])
    random.seed(configs["run"]["random_seed"])

    # Create necessary directories
    subdata_dir = configs["train"]["log_dir"] 
    log_dir = subdata_dir + configs["train"]["model_name"] + "/" + method
    log_dir += f"/eps{int(configs["train"]["epsilon"])}"
    configs["train"]["log_dir"] = log_dir

    directories = {
        "log_dir": log_dir,
        "report_dir": f"{log_dir}/report",
        "data_dir": configs["data"]["data_dir"],
        "subdata_dir": subdata_dir,
    }
    create_directories(directories)

    # Set up logger
    logger = setup_log(
        directories["report_dir"], "time_analysis", configs["run"]["time_log"]
    )

    baseline_time = time.time()
    dataset = load_dataset(configs, directories["data_dir"], logger)
    logger.info("Loading dataset took %0.5f seconds", time.time() - baseline_time)
    # subset of dataset
    if configs["train"]["data_size"] < len(dataset):
        dataset = load_subset_dataset(configs, dataset, f"{directories["subdata_dir"]}", logger)
        logger.info("Loading sub-dataset took %0.5f seconds", time.time() - baseline_time)

    # Define experiment parameters
    model_num = num_rep
    num_model_pairs = 1

    # Load or train models
    baseline_time = time.time()
    models_list, memberships = load_models(
        log_dir, dataset, model_num, configs, logger
    )

    # TODO: add accuracy parity and demo parity
    outcome_fairness = True
    if models_list is None:
        data_splits, memberships = split_dataset_for_training(
            len(dataset), num_model_pairs, ratio=0.8
        )
        data_splits.pop()
        memberships = memberships[0]

        data_splits = data_splits * num_rep
        memberships = memberships * num_rep
        models_list = train_models(
            log_dir, dataset, data_splits, memberships, configs, logger, outcome_fairness
        )
    logger.info(
        "Model loading/training took %0.1f seconds", time.time() - baseline_time
    )

if __name__ == "__main__":
    method = "dpsgdf"
    device = "cuda:6"
    num_rep = 1 
    # scale = 4
    main(device, method, num_rep)