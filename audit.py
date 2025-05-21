# Copyright (c) 2020 Data Privacy and Trustworthy Machine Learning Research Lab
# Licensed under the MIT License. See LICENSE file for details.

import time
from pathlib import Path

import os
import pickle
import numpy as np
import torch.utils.data
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Subset

from attacks import tune_offline_a, run_rmia, run_record_rmia, run_loss
from visual.visualize import plot_roc, plot_roc_log

from scipy.stats import binomtest, norm
from scipy.optimize import root_scalar
from privacy_estimates import AttackResults, compute_eps_lo
import concurrent.futures
from tqdm import tqdm
import math


def compute_attack_results(mia_scores, target_memberships):
    """
    Compute attack results (TPR-FPR curve, AUC, etc.) based on MIA scores and membership of samples.

    Args:
        mia_scores (np.array): MIA score computed by the attack.
        target_memberships (np.array): Membership of samples in the training set of target model.

    Returns:
        dict: Dictionary of results, including fpr and tpr list, AUC, TPR at 1%, 0.1% and 0% FPR.
    """
    fpr_list, tpr_list, _ = roc_curve(target_memberships.ravel(), mia_scores.ravel())
    roc_auc = auc(fpr_list, tpr_list)
    one_fpr = tpr_list[np.where(fpr_list <= 0.01)[0][-1]]
    one_tenth_fpr = tpr_list[np.where(fpr_list <= 0.001)[0][-1]]
    zero_fpr = tpr_list[np.where(fpr_list <= 0.0)[0][-1]]

    return {
        "fpr": fpr_list,
        "tpr": tpr_list,
        "auc": roc_auc,
        "one_fpr": one_fpr,
        "one_tenth_fpr": one_tenth_fpr,
        "zero_fpr": zero_fpr,
    }


def get_audit_results(report_dir, model_idx, mia_scores, target_memberships, logger):
    """
    Generate and save ROC plots for attacking a single model.

    Args:
        report_dir (str): Folder for saving the ROC plots.
        model_idx (int): Index of model subjected to the attack.
        mia_scores (np.array): MIA score computed by the attack.
        target_memberships (np.array): Membership of samples in the training set of target model.
        logger (logging.Logger): Logger object for the current run.

    Returns:
        dict: Dictionary of results, including fpr and tpr list, AUC, TPR at 1%, 0.1% and 0% FPR.
    """
    attack_result = compute_attack_results(mia_scores, target_memberships)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        "Target Model %d: AUC %.4f, TPR@0.1%%FPR of %.4f, TPR@0.0%%FPR of %.4f",
        model_idx,
        attack_result["auc"],
        attack_result["one_tenth_fpr"],
        attack_result["zero_fpr"],
    )

    plot_roc(
        attack_result["fpr"],
        attack_result["tpr"],
        attack_result["auc"],
        f"{report_dir}/ROC_{model_idx}.png",
    )
    plot_roc_log(
        attack_result["fpr"],
        attack_result["tpr"],
        attack_result["auc"],
        f"{report_dir}/ROC_log_{model_idx}.png",
    )

    np.savez(
        f"{report_dir}/attack_result_{model_idx}",
        attack_result=attack_result,
        fpr=attack_result["fpr"],
        tpr=attack_result["tpr"],
        auc=attack_result["auc"],
        one_tenth_fpr=attack_result["one_tenth_fpr"],
        zero_fpr=attack_result["zero_fpr"],
        scores=mia_scores.ravel(),
        memberships=target_memberships.ravel(),
    )
    return attack_result


def get_average_audit_results(report_dir, mia_score_list, membership_list, logger):
    """
    Generate and save ROC plots for attacking multiple models by aggregating all scores and membership labels.

    Args:
        report_dir (str): Folder for saving the ROC plots.
        mia_score_list (list): List of MIA scores for each target model.
        membership_list (list): List of membership labels of each target model.
        logger (logging.Logger): Logger object for the current run.
    """

    mia_scores = np.concatenate(mia_score_list)
    target_memberships = np.concatenate(membership_list)

    attack_result = compute_attack_results(mia_scores, target_memberships)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        "Average result: AUC %.4f, TPR@0.1%%FPR of %.4f, TPR@0.0%%FPR of %.4f",
        attack_result["auc"],
        attack_result["one_tenth_fpr"],
        attack_result["zero_fpr"],
    )

    plot_roc(
        attack_result["fpr"],
        attack_result["tpr"],
        attack_result["auc"],
        f"{report_dir}/ROC_average.png",
    )
    plot_roc_log(
        attack_result["fpr"],
        attack_result["tpr"],
        attack_result["auc"],
        f"{report_dir}/ROC_log_average.png",
    )

    np.savez(
        f"{report_dir}/attack_result_average",
        fpr=attack_result["fpr"],
        tpr=attack_result["tpr"],
        auc=attack_result["auc"],
        one_tenth_fpr=attack_result["one_tenth_fpr"],
        zero_fpr=attack_result["zero_fpr"],
        scores=mia_scores.ravel(),
        memberships=target_memberships.ravel(),
    )


def audit_models(
    report_dir,
    target_model_indices,
    all_signals,
    all_memberships,
    num_reference_models,
    logger,
    configs,
):
    """
    Audit target model(s) using a Membership Inference Attack algorithm.

    Args:
        report_dir (str): Folder to save attack result.
        target_model_indices (list): List of the target model indices.
        all_signals (np.array): Signal value of all samples in all models (target and reference models).
        all_memberships (np.array): Membership matrix for all models.
        num_reference_models (int): Number of reference models used for performing the attack.
        logger (logging.Logger): Logger object for the current run.
        configs (dict): Configs provided by the user.

    Returns:
        list: List of MIA score arrays for all audited target models.
        list: List of membership labels for all target models.
    """
    all_memberships = np.transpose(all_memberships)

    mia_score_list = []
    membership_list = []

    for target_model_idx in target_model_indices:
        baseline_time = time.time()
        if configs["audit"]["algorithm"] == "RMIA":
            offline_a = tune_offline_a(
                target_model_idx, all_signals, all_memberships, logger
            )
            logger.info(f"The best offline_a is %0.1f", offline_a)
            mia_scores = run_rmia(
                target_model_idx,
                all_signals,
                all_memberships,
                num_reference_models,
                offline_a,
            )
        elif configs["audit"]["algorithm"] == "LOSS":
            mia_scores = run_loss(all_signals[:, target_model_idx])
        else:
            raise NotImplementedError(
                f"{configs['audit']['algorithm']} is not implemented"
            )

        target_memberships = all_memberships[:, target_model_idx]

        mia_score_list.append(mia_scores.copy())
        membership_list.append(target_memberships.copy())

        _ = get_audit_results(
            report_dir, target_model_idx, mia_scores, target_memberships, logger
        )

        logger.info(
            "Auditing the privacy risks of target model %d costs %0.1f seconds",
            target_model_idx,
            time.time() - baseline_time,
        )

    return mia_score_list, membership_list


def audit_records(
    report_dir,
    target_model_indices,
    all_signals,
    all_memberships,
    num_reference_models,
    logger,
    attack_algorithm,
):
    """
    Audit target record(s) using a Membership Inference Attack algorithm.

    Args:
        report_dir (str): Folder to save attack result.
        target_model_indices (list): List of the target model indices.
        all_signals (np.array): Signal value of all samples in all models (target and reference models).
        all_memberships (np.array): Membership matrix for all models.
        num_reference_models (int): Number of reference models used for performing the attack.
        logger (logging.Logger): Logger object for the current run.
        configs (dict): Configs provided by the user.

    Returns:
        list: List of MIA score arrays for all audited target models.
        list: List of membership labels for all target models.
    """
    path_scores = f"{report_dir}/{attack_algorithm.lower()}_scores.npy"
    path_member = f"{report_dir}/memberships.npy"
    if os.path.exists(path_scores):
        mia_score_list  = np.load(path_scores)
        membership_list = np.load(path_member)
        logger.info("mia_score_list and membership_list loaded to disk.")
    else:
        all_memberships = np.transpose(all_memberships) 

        mia_score_list = [] # all samples in all models
        membership_list = [] # all samples in all models

        for target_model_idx in target_model_indices:
            if attack_algorithm == "RMIA":
                # offline_a = tune_offline_a(
                #     target_model_idx, all_signals, all_memberships, logger
                # )
                #logger.info(f"The best offline_a is %0.1f", offline_a)
                offline_a = 0.5
                mia_scores = run_rmia(
                    target_model_idx,
                    all_signals,
                    all_memberships,
                    num_reference_models,
                    offline_a,
                )
            elif attack_algorithm == "LOSS":
                mia_scores = run_loss(all_signals[:, target_model_idx])
            else:
                raise NotImplementedError(
                    f"{attack_algorithm} is not implemented"
                )

            target_memberships = all_memberships[:, target_model_idx]

            mia_score_list.append(mia_scores.copy()) # (n models, m samples)
            membership_list.append(target_memberships.copy())

            # _ = get_audit_results(
            #     report_dir, target_model_idx, mia_scores, target_memberships, logger
            # )
            # logger.info(
            #     "Auditing the privacy risks of target model %d costs %0.1f seconds",
            #     target_model_idx,
            #     time.time() - baseline_time,
            # )
        # Save scores and membership (m samples, n models)
        mia_score_list = np.transpose(mia_score_list)
        membership_list = np.transpose(membership_list)
        np.save(
            f"{report_dir}/{attack_algorithm.lower()}_scores.npy",
            mia_score_list,
        )
        np.save(
            f"{report_dir}/memberships.npy",
            membership_list,
        )
        logger.info("mia_score_list and membership_list saved to disk.")

    return mia_score_list, membership_list


def sample_auditing_dataset(
    configs, dataset: torch.utils.data.Dataset, logger, memberships: np.ndarray
):
    """
    Downsample the dataset in auditing if specified.

    Args:
        configs (Dict[str, Any]): Configuration dictionary
        dataset (Any): The full dataset from which the audit subset will be sampled.
        logger (Any): Logger object used to log information during downsampling.
        memberships (np.ndarray): A 2D boolean numpy array where each row corresponds to a model and
                                  each column corresponds to whether the corresponding sample is a member (True)
                                  or non-member (False).

    Returns:
        Tuple[torch.utils.data.Subset, np.ndarray]: A tuple containing:
            - The downsampled dataset or the full dataset if downsampling is not applied.
            - The corresponding membership labels for the samples in the downsampled dataset.

    Raises:
        ValueError: If the requested audit data size is larger than the full dataset or not an even number.
    """
    if configs["run"]["num_experiments"] > 1:
        logger.warning(
            "Auditing multiple models. Balanced downsampling is only based on the data membership of the FIRST target model!"
        )

    audit_data_size = configs["audit"].get("data_size", len(dataset))
    if audit_data_size < len(dataset):
        if audit_data_size % 2 != 0:
            raise ValueError("Audit data size must be an even number.")

        logger.info(
            "Downsampling the dataset for auditing to %d samples. The numbers of members and non-members are only "
            "guaranteed to be equal for the first target model, if more than one are used.",
            audit_data_size,
        )
        # Sample equal numbers of members and non-members according to the first target model randomly
        members_idx = np.random.choice(
            np.where(memberships[0, :])[0], audit_data_size // 2, replace=False
        )
        non_members_idx = np.random.choice(
            np.where(~memberships[0, :])[0], audit_data_size // 2, replace=False
        )

        # Randomly sample members and non-members
        auditing_dataset = Subset(
            dataset, np.concatenate([members_idx, non_members_idx])
        )
        auditing_membership = memberships[
            :, np.concatenate([members_idx, non_members_idx])
        ].reshape((memberships.shape[0], audit_data_size))
    elif audit_data_size == len(dataset):
        auditing_dataset = dataset
        auditing_membership = memberships
    else:
        raise ValueError("Audit data size cannot be larger than the dataset.")
    return auditing_dataset, auditing_membership


def sample_auditing_dataset_and_group(
    configs, dataset: torch.utils.data.Dataset, logger, memberships: np.ndarray
):
    if configs["run"]["num_experiments"] > 1:
        logger.warning(
            "Auditing multiple models. Balanced downsampling is only based on the data membership of the FIRST target model!"
        )

    audit_data_size = configs["audit"].get("data_size", len(dataset))
    if audit_data_size < len(dataset):
        if audit_data_size % 2 != 0:
            raise ValueError("Audit data size must be an even number.")

        logger.info(
            "Downsampling the dataset for auditing to %d samples. The numbers of members and non-members are only "
            "guaranteed to be equal for the first target model, if more than one are used.",
            audit_data_size,
        )
        # Sample equal numbers of members and non-members according to the first target model randomly
        members_idx = np.random.choice(
            np.where(memberships[0, :])[0], audit_data_size // 2, replace=False
        )
        non_members_idx = np.random.choice(
            np.where(~memberships[0, :])[0], audit_data_size // 2, replace=False
        )

        # Randomly sample members and non-members
        auditing_dataset = Subset(
            dataset, np.concatenate([members_idx, non_members_idx])
        )
        auditing_membership = memberships[
            :, np.concatenate([members_idx, non_members_idx])
        ].reshape((memberships.shape[0], audit_data_size))
        auditing_group = dataset.z[np.concatenate([members_idx, non_members_idx])]
    elif audit_data_size == len(dataset):
        auditing_dataset = dataset
        auditing_membership = memberships
        auditing_group = dataset.z
    else:
        raise ValueError("Audit data size cannot be larger than the dataset.")
    return auditing_dataset, auditing_membership, auditing_group


# def group_auditing_dataset(
#         configs, dataset: torch.utils.data.Dataset, group: int, logger, memberships: np.ndarray
# ):
    
#     if configs["run"]["num_experiments"] > 1:
#         logger.warning(
#             "Auditing multiple models. group audit is only based on the data membership of the FIRST target model!"
#         )

#     group_idx = np.where(dataset.z == group)[0]

#     # Randomly sample members and non-members
#     auditing_dataset = Subset(dataset, group_idx)
#     data_size = len(group_idx)
#     auditing_membership = memberships[:, group_idx].reshape((memberships.shape[0], data_size))
    
#     return auditing_dataset, auditing_membership



"""Utility functions to audit DP algorithms using (eps, delta)-DP or GDP"""
def compute_eps_for_data(s, m, alpha, delta):
    m = m.astype(int)
    _, emp_eps_loss = compute_eps_lower_from_mia(s, m, alpha, delta)
    return emp_eps_loss


def exp_estimated_epsilon(report_dir, scores, memberships, dataset, configs):
    path = report_dir + "_eps.pkl"
    alpha = 0.05
    delta = 1e-5

    if os.path.exists(path):
        with open(path, 'rb') as f: 
            all_eps = pickle.load(f)
    else:
        all_eps = [0] * len(scores)
        with concurrent.futures.ProcessPoolExecutor(max_workers=80) as executor, \
         tqdm(total=len(scores), leave=False) as pbar:
            
            futures = {}
            for idx, (s, m) in enumerate(zip(scores, memberships)):
                futures[executor.submit(compute_eps_for_data, s, m, alpha, delta)] = idx
            
            for future in concurrent.futures.as_completed(futures):
                exp_epsilon = future.result()
                idx = futures[future]
                # group = dataset[idx][1] if num_group == 10 else dataset[idx][2]
                all_eps[idx] = exp_epsilon
                pbar.update(1)
        # with tqdm(total=len(scores), leave=False) as pbar:
        #     for idx, (s, m) in enumerate(zip(scores, memberships)):
        #         exp_epsilon = compute_eps_for_data(s, m, alpha, delta)
        #         all_eps[dataset[idx][1]].append(exp_epsilon)
        #         pbar.update(1)
        
        with open(path, 'wb') as f:
            pickle.dump(all_eps, f)

    return all_eps


def compute_eps_lower_gdp(results, alpha, delta):
    """Convert FPR and FNR to eps, delta using GDP at significance level alpha"""
    # Step 1: calculate CP upper bound on FPR and FNR at significance level alpha
    _, fpr_r = binomtest(int(results.FP), int(results.N)).proportion_ci(confidence_level=1 - 2 * alpha)
    _, fnr_r = binomtest(int(results.FN), int(results.P)).proportion_ci(confidence_level=1 - 2 * alpha)

    # Step 2: calculate lower bound on mu-GDP
    mu_l = norm.ppf(1 - fpr_r) - norm.ppf(fnr_r)

    if mu_l < 0:
        # GDP is not defined for mu < 0
        return 0

    try:
        # Step 3: convert mu-GDP to (eps, delta)-DP using Equation (6) from Tight Auditing DPML paper
        def eq6(epsilon):
            return norm.cdf(-epsilon / mu_l + mu_l / 2) - np.exp(epsilon) * norm.cdf(-epsilon / mu_l - mu_l / 2) - delta

        sol = root_scalar(eq6, bracket=[0, 50], method='brentq')
        eps_l = sol.root
    except Exception:
        eps_l = 0

    return eps_l


def compute_eps_lower_single(results, alpha, delta, method='GDP'):
    """Given FPR and FNR estimate epsilon lower bound using different methods at a given significance level alpha and delta
    For (eps, delta)-DP use method(s) described in https://proceedings.mlr.press/v202/zanella-beguelin23a/zanella-beguelin23a.pdf
    For GDP use method described in https://arxiv.org/pdf/2302.07956
    """
    method = "GDP"
    if method == 'GDP':
        return compute_eps_lower_gdp(results, alpha, delta)
    
    method_map = {
        'zb': 'joint-beta',
        # 'cp': 'beta',
        # 'jeff': 'jeffreys'
    }

    max_eps_lo = -1
    for curr_method, curr_method_full in method_map.items():
        if method == 'all' or curr_method == method:
            # try:
            curr_eps_lo = compute_eps_lo(count=results, delta=delta, alpha=alpha, method=curr_method_full)
            max_eps_lo = max(curr_eps_lo, max_eps_lo)
            # except Exception:
            #     pass

    return max_eps_lo


def best_acc_indices(scores, labels, threshs):
    acc_list = []
    for t in threshs:
        tp = np.sum(scores[labels == 1] >= t)
        fp = np.sum(scores[labels == 0] >= t)
        fn = np.sum(scores[labels == 1] < t)
        tn = np.sum(scores[labels == 0] < t)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        acc_list.append(accuracy)
    max_acc  = np.max(acc_list)
    
    return np.where(acc_list == max_acc)[0]


def compute_eps_lower_from_mia(scores, labels, alpha, delta, method='all', n_procs=1):
    """Compute lower bound for epsilon using privacy estimation procedure
    Step 1: For each threshold, calculate TP, FP, TN, FN and estimate epsilon lower bound using different methods at a given significance level alpha and delta
    Step 2: Output the maximum epsilon lower bound 
    """
    scores, labels = np.array(scores), np.array(labels)
    threshs = np.sort(np.unique(scores))

    indices = best_acc_indices(scores, labels, threshs)
    threshs = threshs[indices]

    resultss = []
    #for t in threshs[: len(threshs)//2+1]:
    for t in threshs:
        tp = np.sum(scores[labels == 1] >= t)
        fp = np.sum(scores[labels == 0] >= t)
        fn = np.sum(scores[labels == 1] < t)
        tn = np.sum(scores[labels == 0] < t)

        results = AttackResults(FN=fn, FP=fp, TN=tn, TP=tp)
        resultss.append((t, results))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as executor: 
        # tqdm(total=len(resultss), leave=False) as pbar:

        futures = {}
        for (t, curr_results) in resultss:
            futures[executor.submit(compute_eps_lower_single, curr_results, alpha, delta, method)] = t
        
        max_eps_lo, max_t = None, None
        for future in concurrent.futures.as_completed(futures):
            curr_max_eps_lo = future.result()
            t = futures[future]
            if not math.isnan(curr_max_eps_lo) and (max_eps_lo is None or curr_max_eps_lo > max_eps_lo):
                max_eps_lo = curr_max_eps_lo
                max_t = t
            # pbar.update(1)
    
    return max_t, max_eps_lo

def exp_worst_eps(scores, memberships):
    all_acc = [0] * len(scores)
    with concurrent.futures.ProcessPoolExecutor(max_workers=64) as executor, \
                tqdm(total=len(scores), leave=False) as pbar:

            futures = {}
            for idx, (s, m) in enumerate(zip(scores, memberships)):
                futures[executor.submit(get_best_acc, s, m)] = idx

            for future in concurrent.futures.as_completed(futures):
                exp_acc = future.result()
                idx = futures[future]
                # if num_group == 2:
                #     index = dataset[idx][1]*2 + dataset[idx][2]
                #     all_acc[dataset[idx][2]].append(exp_acc)
                # else:
                all_acc[idx] = exp_acc
                pbar.update(1)
    index = np.argmax(all_acc)
    print(max(all_acc))
    eps = compute_eps_for_data(scores[index], memberships[index], alpha=0.05, delta=1e-5)
    return eps


"""attack acc performance"""
def get_best_acc(scores, labels):
    scores, labels = np.array(scores), np.array(labels)
    threshs = np.sort(np.unique(scores))
    acc_list = []
    for t in threshs:
        tp = np.sum(scores[labels == 1] >= t)
        fp = np.sum(scores[labels == 0] >= t)
        fn = np.sum(scores[labels == 1] < t)
        tn = np.sum(scores[labels == 0] < t)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        acc_list.append(accuracy)
    max_acc = np.max(acc_list)
    max_idx = np.argmax(acc_list)

    return max_acc

def get_acc(scores, labels, t):
    tp = np.sum(scores[labels == 1] >= t)
    fp = np.sum(scores[labels == 0] >= t)
    fn = np.sum(scores[labels == 1] < t)
    tn = np.sum(scores[labels == 0] < t)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy

def get_avg_mia_acc(scores, labels, threshs):
    guesses = []
    for s, t in zip(scores, threshs):
        if s >= t:
            guesses.append(1)
        else:
            guesses.append(0)
    guesses = np.array(guesses)
    accuracy = np.mean(guesses==labels)
    return accuracy

def get_model_best_acc(scores, labels):
    scores, labels = np.array(scores), np.array(labels)
    threshs = np.sort(np.unique(scores))
    acc_list = [0] * len(threshs)

    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor, \
                tqdm(total=len(threshs), leave=False) as pbar:

            futures = {}
            for idx, t in enumerate(zip(threshs)):
                futures[executor.submit(get_acc, scores, labels, t)] = idx

            for future in concurrent.futures.as_completed(futures):
                exp_acc = future.result()
                idx = futures[future]
                acc_list[idx] = exp_acc
                pbar.update(1)

    max_acc = np.max(acc_list)
    max_idx = np.argmax(acc_list)

    return threshs[max_idx]


def get_model_group_best_acc(scores, labels, num_group, data_class):
    group_threshs = []
    for i in range(num_group):
        s = scores[data_class == i]
        l = labels[data_class == i]
        threshs = np.sort(np.unique(s))
        acc_list = [0] * len(threshs)

        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor, \
            tqdm(total=len(threshs), leave=False) as pbar:

            futures = {}
            for idx, t in enumerate(zip(threshs)):
                futures[executor.submit(get_acc, s, l, t)] = idx

            for future in concurrent.futures.as_completed(futures):
                exp_acc = future.result()
                idx = futures[future]
                acc_list[idx] = exp_acc
                pbar.update(1)

        # max_acc = np.max(acc_list)
        max_idx = np.argmax(acc_list)
        group_threshs.append(threshs[max_idx])

    return group_threshs


def exp_one_acc(scores, memberships, data_idx):
    for idx, (s, m) in enumerate(zip(scores, memberships)):
        if idx == data_idx:
            acc = get_best_acc(s, m)
    return acc


def exp_all_acc(report_dir, scores, memberships, dataset, num_group=10):
    path = report_dir + "_acc.pkl"

    if os.path.exists(path):
        with open(path, 'rb') as f:
            all_acc = pickle.load(f)
    else:
        all_acc = [[] for _ in range(num_group)]
        # for idx, (s, m) in enumerate(zip(scores, memberships)):
        #     acc = get_best_acc(s, m)
        #     all_acc[dataset[idx][1]].append([dataset[idx][0], acc])
        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor, \
                tqdm(total=len(scores), leave=False) as pbar:

            futures = {}
            for idx, (s, m) in enumerate(zip(scores, memberships)):
                futures[executor.submit(get_best_acc, s, m)] = idx

            for future in concurrent.futures.as_completed(futures):
                exp_acc = future.result()
                idx = futures[future]
                # if num_group == 2:
                #     index = dataset[idx][1]*2 + dataset[idx][2]
                #     all_acc[index].append(exp_acc)
                # else:
                if num_group == 2 or num_group == 5:
                    all_acc[dataset[idx][2]].append(exp_acc)
                else:
                    all_acc[dataset[idx][1]].append(exp_acc)
                pbar.update(1)

        with open(path, 'wb') as f:
            pickle.dump(all_acc, f)
    return all_acc


def exp_all_avg_acc(report_dir, scores, memberships, dataset, num_group=10):
    path = report_dir + "_avg_acc.pkl"
    scores = scores.T
    memberships = memberships.T

    if os.path.exists(path):
        with open(path, 'rb') as f:
            all_avg_acc = pickle.load(f)
    else:
        best_threshs = [0] * len(scores)
        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor, \
            tqdm(total=len(scores), leave=False) as pbar:

            futures = {}
            for idx, (s, m) in enumerate(zip(scores, memberships)):
                futures[executor.submit(get_model_best_acc, s, m)] = idx
            for future in concurrent.futures.as_completed(futures):
                b_t = future.result()
                idx = futures[future]
                best_threshs[idx] =  b_t

        ## audit ind here
        scores = scores.T
        memberships = memberships.T
        all_avg_acc = [[] for _ in range(num_group)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor, \
            tqdm(total=len(scores), leave=False) as pbar:

            futures = {}
            for idx, (s, m) in enumerate(zip(scores, memberships)):
                futures[executor.submit(get_avg_mia_acc, s, m, best_threshs)] = idx
            for future in concurrent.futures.as_completed(futures):
                exp_acc = future.result()
                idx = futures[future]
                # all_avg_acc[dataset[idx][1]].append(exp_acc)
                if num_group == 2 or num_group == 5:
                    all_avg_acc[dataset[idx][2]].append(exp_acc)
                else:
                    all_avg_acc[dataset[idx][1]].append(exp_acc)
        with open(path, 'wb') as f:
            pickle.dump(all_avg_acc, f)
    return all_avg_acc


def exp_all_group_avg_acc(report_dir, scores, memberships, dataset, num_group=10):
    path = report_dir + "_group_avg_acc.pkl"
    scores = scores.T
    memberships = memberships.T
    
    data_class = []
    for data in dataset:
        if num_group == 2 or num_group == 5:
            data_class.append(data[2])
        else:
            data_class.append(data[1])
    data_class = np.array(data_class)

    if os.path.exists(path):
        with open(path, 'rb') as f:
            all_avg_acc = pickle.load(f)
    else:
        best_group_threshs = [0] * len(scores)
        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor, \
            tqdm(total=len(scores), leave=False) as pbar:

            futures = {}
            for idx, (s, m) in enumerate(zip(scores, memberships)):
                futures[executor.submit(get_model_group_best_acc, s, m, num_group, data_class)] = idx
            for future in concurrent.futures.as_completed(futures):
                group_threshs = future.result()
                idx = futures[future]
                best_group_threshs[idx] =  group_threshs
        best_group_threshs = np.array(best_group_threshs)

        ## audit ind here
        scores = scores.T
        memberships = memberships.T
        all_avg_acc = [[] for _ in range(num_group)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor, \
            tqdm(total=len(scores), leave=False) as pbar:

            futures = {}
            for idx, (s, m) in enumerate(zip(scores, memberships)):
                ind_class = data_class[idx]
                futures[executor.submit(get_avg_mia_acc, s, m, best_group_threshs[:,ind_class])] = idx
            for future in concurrent.futures.as_completed(futures):
                exp_acc = future.result()
                idx = futures[future]
                # all_avg_acc[dataset[idx][1]].append(exp_acc)
                if num_group == 2 or num_group == 5:
                    all_avg_acc[dataset[idx][2]].append(exp_acc)
                else:
                    all_avg_acc[dataset[idx][1]].append(exp_acc)
        with open(path, 'wb') as f:
            pickle.dump(all_avg_acc, f)
    return all_avg_acc

