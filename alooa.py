
import argparse
import math
import time

import torch
import yaml
import numpy as np
import pandas as pd

from audit import get_average_audit_results, audit_models, audit_records, sample_auditing_dataset
from get_signals import get_model_signals
from models.utils import load_models, train_models, split_dataset_for_training, split_samples_for_training
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


def main(device, method, sample_idxs=None, scale=None):
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run individual privacy audit.")
    parser.add_argument(
        "--cf",
        type=str,
        default=f"configs/mnist/mnist_{method}.yaml",
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
    if scale is not None:
        method = f"{method}_{scale}"

    # Initialize seeds for reproducibility
    initialize_seeds(configs["run"]["random_seed"])

    # Create necessary directories
    subdata_dir = configs["run"]["log_dir"] 
    log_dir = f"{configs["run"]["log_dir"]}{configs["train"]["model_name"]}/{method}"
    log_dir += f"/eps{int(configs["train"]["epsilon"])}"
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
        dataset = load_subset_dataset(configs, dataset, f"{directories["subdata_dir"]}", logger)
        logger.info("Loading sub-dataset took %0.5f seconds", time.time() - baseline_time)

    # Define experiment parameters
    num_experiments = configs["run"]["num_experiments"]
    num_models = configs["audit"]["num_models"]
    num_model_pairs = max(math.ceil(num_experiments / 2.0), int(num_models/2))

    # Load or train models
    baseline_time = time.time()
    models_list, memberships = load_models(
        log_dir, dataset, num_model_pairs * 2, configs, logger
    )
    if models_list is None:
        # Split dataset for training two models per pair
        # data_splits, memberships = split_dataset_for_training(
        #     len(dataset), num_model_pairs, ratio=0.5
        # )
        data_splits, memberships = split_samples_for_training(
            len(dataset), num_model_pairs, sample_idxs
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
    num_reference_models = int(num_models/2 - 1)
    mia_score_list, membership_list = audit_records(
        f"{directories['report_dir']}",
        target_model_indices,
        signals,
        auditing_membership,
        num_reference_models,
        logger,
        attack_algorithm=configs["audit"]["algorithm"],
    )


if __name__ == "__main__":
    log_dir = 'exp/demo_mnist_compare/'
    samples = pd.read_csv(f"{log_dir}sample_idx.txt", sep=',')
    method = "regular"
    device = "cuda:6"
    scale = 5
    main(device, method, sample_idxs=samples["idx"].to_numpy())
