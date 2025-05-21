
import argparse
import math
import time

import torch
import yaml
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from audit import audit_models, audit_records, sample_auditing_dataset, exp_estimated_epsilon
from get_signals import get_model_signals
from models.utils import load_models, train_models, split_dataset_for_training, split_one_data_for_training
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

def main(device, num_ref_models, dataset_idx=None, wish_model_nums=None):
    dataset_idx = dataset_idx
    configs = "configs/mnist/mnist_regular.yaml"
    with open(configs, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    # Validate configurations
    check_configs(configs)

    # Initialize seeds for reproducibility
    # initialize_seeds(configs["run"]["random_seed"])

    configs["train"]["device"] = device
    configs["audit"]["device"] = device
    configs["audit"]["num_ref_models"] = num_ref_models

    # Create necessary directories
    subdata_dir = configs["run"]["log_dir"] 
    if dataset_idx is None:
        log_dir = f"{configs["run"]["log_dir"]}{configs["train"]["model_name"]}/{configs["train"]["method"]}_{configs["audit"]["num_ref_models"]}"
    else:
        log_dir = f"{configs["run"]["log_dir"]}{configs["train"]["model_name"]}/{configs["train"]["method"]}_{configs["audit"]["num_ref_models"]}/data/data_{dataset_idx}"
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

    path = f"{directories["subdata_dir"]}{configs['data']['dataset']}.pkl"
    with open(path, "rb") as file:
        dataset = pickle.load(file)

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
            # data_splits, memberships = split_one_data_for_training(
            #     len(dataset), num_model_pairs, dataset_idx
            # )
            data_splits, memberships = split_dataset_for_training(
                len(dataset), num_model_pairs, ratio=0.5
            )
            models_list = train_models(
                log_dir, dataset, data_splits, memberships, configs, logger
            )
    logger.info(
            "Model loading/training took %0.1f seconds", time.time() - baseline_time
        )
    
    # get softmax or get loss
    auditing_dataset, auditing_membership = dataset, memberships
    baseline_time = time.time()
    signals = get_model_signals(models_list, auditing_dataset, configs, logger)
    logger.info("Preparing signals took %0.5f seconds", time.time() - baseline_time)

    if wish_model_nums is not None:
        for model_num in wish_model_nums:
            num_reference_models = (model_num // 2)-1
            target_model_indices = list(range(model_num)) 
            local_path =  f"{directories["subdata_dir"]}data/data_{dataset_idx}/model_{model_num}"
            Path(local_path).mkdir(parents=True, exist_ok=True)

            # shape of mia_score_list: (n, m) n=(num_models, m=len(auditing_dataset)
            mia_score_list, membership_list = audit_records(
                    local_path,
                    target_model_indices,
                    signals,
                    auditing_membership,
                    num_reference_models,
                    logger,
                    attack_algorithm=configs["audit"]["algorithm"],
                )
    else:
        # Perform the privacy audit
        target_model_indices = list(range((num_reference_models+1)*2)) # for all pair models

        # shape of mia_score_list: (n, m) n=(num_reference_models+1)*2, m=len(auditing_dataset)
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
    dataset_idxs = pd.read_csv(f"{log_dir}sample_idx.txt", sep=',')
    
    device = "cuda:2"
    num_ref_models = 299
    wish_model_nums = [100]
    for i in range(len(dataset_idxs)):
        main(device, num_ref_models, dataset_idx[i])
