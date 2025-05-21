# Copyright (c) 2020 Data Privacy and Trustworthy Machine Learning Research Lab
# Licensed under the MIT License. See LICENSE file for details.

"""This file is the main entry point for running the privacy auditing tool."""

import argparse
import math
import time

import torch
import yaml
import numpy as np

from audit import get_average_audit_results, audit_models, audit_records, sample_auditing_dataset, exp_all_acc, exp_all_avg_acc, exp_all_group_avg_acc
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


def main(device, method, seed="", scale=None):
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run individual privacy audit.")
    parser.add_argument(
        "--cf",
        type=str,
        # default=f"configs/mnist/mnist_{method}.yaml",
        default=f"configs/raceface/raceface_{method}.yaml",
        # default=f"configs/tabular/adult/adult_{method}.yaml",
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()

    # Load configuration file
    with open(args.cf, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    # Validate configurations
    check_configs(configs)
    configs["train"]["device"] = device
    configs["audit"]["device"] = device
    if seed != "" :
        configs["run"]["random_seed"] = seed
    if scale is not None:
        method = f"{method}_{scale}"
        configs["train"]["scale"] = scale

    # Initialize seeds for reproducibility
    initialize_seeds(configs["run"]["random_seed"])

    # Create necessary directories
    subdata_dir = configs["run"]["log_dir"] 
    log_dir = f'{configs["run"]["log_dir"]}{configs["train"]["model_name"]}/{method}'
    log_dir += f'/eps{int(configs["train"]["epsilon"])}_{seed}'
    configs["run"]["log_dir"] = log_dir
    directories = {
        "log_dir": log_dir,
        "report_dir": f"{log_dir}/report",
        "signal_dir": f"{log_dir}/signals",
        "data_dir": configs["data"]["data_dir"],
        "subdata_dir": subdata_dir,
    }
    create_directories(directories)

    # Set up logger
    logger = setup_log(
        directories["report_dir"], "time_analysis", configs["run"]["time_log"]
    )

    start_time = time.time()

    # Load the dataset
    baseline_time = time.time()
    dataset = load_dataset(configs, directories["data_dir"], logger)
    logger.info("Loading dataset took %0.5f seconds", time.time() - baseline_time)
    # subset of dataset
    if configs["train"]["data_size"] < len(dataset):
        dataset = load_subset_dataset(configs, dataset, f'{directories["subdata_dir"]}', logger)
        logger.info("Loading sub-dataset took %0.5f seconds", time.time() - baseline_time)

    # Define experiment parameters
    num_experiments = configs["run"]["num_experiments"]
    num_models = configs["audit"]["num_models"]
    num_reference_models = int(num_models/2 - 1)
    num_model_pairs = max(math.ceil(num_experiments / 2.0), num_reference_models + 1)

    # Load or train models
    baseline_time = time.time()
    models_list, memberships = load_models(
        log_dir, dataset, num_model_pairs * 2, configs, logger
    )
    if models_list is None:
        # Split dataset for training two models per pair
        data_splits, memberships = split_dataset_for_training(
            len(dataset), num_model_pairs, ratio=0.5
        )
        models_list = train_models(
            log_dir, dataset, data_splits, memberships, configs, logger
        )
    logger.info(
        "Model loading/training took %0.1f seconds", time.time() - baseline_time
    )

    # auditing_dataset, auditing_membership = sample_auditing_dataset(
    #     configs, dataset, logger, memberships
    # )
    auditing_dataset, auditing_membership = dataset, memberships
    baseline_time = time.time()
    signals = get_model_signals(models_list, auditing_dataset, configs, logger)
    logger.info("Preparing signals took %0.5f seconds", time.time() - baseline_time)

    # Perform the privacy audit
    #target_model_indices = list(range(num_experiments))
    target_model_indices = list(range(num_models)) # for all pair models

    # shape of mia_score_list: (n, m) n=(num_models, m=len(auditing_dataset)
    mia_score_list, membership_list = audit_records(
        f"{directories['report_dir']}",
        target_model_indices,
        signals,
        auditing_membership,
        num_reference_models,
        logger,
        attack_algorithm=configs["audit"]["algorithm"],
    )

    # if configs["train"]["num_groups"] == 2:
    #     pro_attributes = np.array([(item[1], item[2]) for item in dataset])
    #     np.save(f"{directories['report_dir']}/pro_attributes.npy", pro_attributes)
    # else:
    #     sample_labels = np.array([(i, item[1]) for i, item in enumerate(dataset)])
    #     np.save(f"{directories["subdata_dir"]}sample_labels.npy", sample_labels)

    loss_scores = mia_score_list 
    memberships = membership_list
    num_group = configs["train"]["num_groups"]
    exp_all_acc(f"{directories['report_dir']}/loss", loss_scores, memberships, dataset, num_group)
    exp_all_avg_acc(f"{directories['report_dir']}/loss", loss_scores, memberships, dataset, num_group)
    exp_all_group_avg_acc(f"{directories['report_dir']}/loss", loss_scores, memberships, dataset, num_group)

if __name__ == "__main__":
    method = "dpsgd"
    device = "cuda:6"
    scale = 5
    # main(device, method)

    random_seeds = [1, 12, 123, 1234]
    # for seed in random_seeds:
    main(device, method, seed=random_seeds[3])