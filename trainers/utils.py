# Copyright (c) 2020 Data Privacy and Trustworthy Machine Learning Research Lab
# Licensed under the MIT License. See LICENSE file for details.

import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import copy
import torch
import pandas as pd
from torch import nn
import torch.nn.functional as F
from opacus.accountants.analysis.gdp import compute_eps_poisson
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
from opacus.grad_sample.grad_sample_module import GradSampleModule


def privacy_checker(sample_rate, config):
    assert sample_rate <= 1.0
    steps = config["max_epochs"] * math.ceil(1 / sample_rate)

    if config["accountant"] == 'rdp':
        orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
        rdp = compute_rdp(
            q=sample_rate,
            noise_multiplier=config["noise_multiplier"],
            steps=steps,
            orders=orders)
        epsilon, alpha = get_privacy_spent(
            orders=orders,
            rdp=rdp,
            delta=config["delta"])
        print(
            "-----------privacy------------"
            f"\nDP-SGD (RDP) with\n\tsampling rate = {100 * sample_rate:.3g}%,"
            f"\n\tnoise_multiplier = {config['noise_multiplier']},"
            f"\n\titerated over {steps} steps,\nsatisfies "
            f"differential privacy with\n\tepsilon = {epsilon:.3g},"
            f"\n\tdelta = {config['delta']}."
            f"\nThe optimal alpha is {alpha}."
        )
    elif config["accountant"] == 'gdp':
        eps = compute_eps_poisson(
            steps=steps,
            noise_multiplier=config["noise_multiplier"],
            sample_rate=sample_rate,
            delta=config["delta"],
        )
        print(
            "-----------privacy------------"
            f"\nDP-SGD (GDP) with\n\tsampling rate = {100 * sample_rate:.3g}%,"
            f"\n\tnoise_multiplier = {config['noise_multiplier']},"
            f"\n\titerated over {steps} steps,\nsatisfies "
            f"differential privacy with\n\tepsilon = {eps:.3g},"
            f"\n\tdelta = {config['delta']}."
        )
    else:
        raise ValueError(f"Unknown accountant {config['accountant']}. Try 'rdp' or 'gdp'.")


def get_grads(named_parameters, group, num_groups):
    ave_grads = [[] for _ in range(num_groups)]
    max_grads = [[] for _ in range(num_groups)]
    name_grads = list(named_parameters)
    for batch_idx in range(group.shape[0]):
        grads_per_sample = []
        for layer_idx in range(len(name_grads)):
            if name_grads[layer_idx][1].requires_grad:
                grads_per_sample.append(name_grads[layer_idx][1].grad_sample[batch_idx].abs().reshape(-1))
        ave_grads[group[batch_idx]].append(torch.mean(torch.cat(grads_per_sample, 0)))
        max_grads[group[batch_idx]].append(torch.max(torch.cat(grads_per_sample, 0)))
    return ave_grads, max_grads


def get_grad_norms_clip(per_sample_grads, group, num_groups, clipping_scale_fn, **kwargs):
    grad_norms = [[] for _ in range(num_groups)]
    clip_grad_norms = [[] for _ in range(num_groups)]
    sum_grad_vec = []
    sum_clip_grad_vec = []
    for sample_idx in range(group.shape[0]):
        grad_vec = per_sample_grads[sample_idx]
        grad_norm = torch.linalg.norm(grad_vec).item()
        clipping_scale = clipping_scale_fn(grad_norm, sample_idx, **kwargs)
        clip_grad_vec = clipping_scale * grad_vec
        clip_grad_norm = torch.linalg.norm(clip_grad_vec).item()
        grad_norms[group[sample_idx]].append(grad_norm)
        clip_grad_norms[group[sample_idx]].append(clip_grad_norm)
        if sample_idx == 0:
            for _ in range(num_groups):
                sum_grad_vec.append(torch.zeros(grad_vec.shape[0], device=grad_vec.device, requires_grad=False))
                sum_clip_grad_vec.append(torch.zeros(grad_vec.shape[0], device=grad_vec.device, requires_grad=False))
        sum_grad_vec[group[sample_idx]] += grad_vec
        sum_clip_grad_vec[group[sample_idx]] += clip_grad_vec

    return grad_norms, clip_grad_norms, sum_grad_vec, sum_clip_grad_vec


def get_grad_norms(per_sample_grads, group, num_groups):
    grad_norms = [[] for _ in range(num_groups)]
    sum_grad_vec = []
    for sample_idx in range(group.shape[0]):
        grad_vec = per_sample_grads[sample_idx]
        grad_norms[group[sample_idx]].append(torch.linalg.norm(grad_vec).item())
        if sample_idx == 0:
            for _ in range(num_groups):
                sum_grad_vec.append(torch.zeros(grad_vec.shape[0], device=grad_vec.device, requires_grad=False))
        sum_grad_vec[group[sample_idx]] += grad_vec
    return grad_norms, sum_grad_vec


def get_num_layers(model):
    num_layers = 0
    for n, p in model.named_parameters():
        if (p.requires_grad) and ("bias" not in n):
            num_layers += 1

    return num_layers


# splits data, labels according to group of data point
# returns tensor of size num_groups, each element is (subset of data, subset of labels)  
# corresponding to specific group given by index
def split_by_group(data, labels, group, num_groups, return_counts=False):
    sorter = torch.argsort(group)
    unique, counts = torch.unique(group, return_counts=True)
    unique = unique.tolist()
    counts = counts.tolist()

    complete_unique = [0] * num_groups
    complete_counts = [0] * num_groups
    for i in range(num_groups):
        complete_unique[i] = i
        if i in unique:
            j = unique.index(i)
            complete_counts[i] = counts[j]

    sorted_data = torch.split(data[sorter], complete_counts)
    sorted_labels = torch.split(labels[sorter], complete_counts)

    if not return_counts:
        return list(zip(sorted_data, sorted_labels))
    return list(zip(sorted_data, sorted_labels)), complete_counts


def plot_by_group(data_by_group, writer, data_title=None, scale_to_01=False):
    fig = plt.figure()
    plt.bar(range(len(data_by_group)), data_by_group, width=0.9)
    plt.xlabel("Groups")
    if data_title is not None:
        plt.ylabel(data_title)
    plt.title(data_title)
    plt.xticks(range(len(data_by_group)))
    if scale_to_01:
        plt.ylim(0, 1)
    #plt.show()
    #plt.close()
    writer.write_figure(data_title, fig)


def unflatten_grads(params, grad_vec):
    grad_size = []
    for layer in params:
        grad_size.append(layer.reshape(-1).shape[0])
    grad_list = list(torch.split(grad_vec, grad_size))
    for layer_idx in range(len(grad_list)):
        grad_list[layer_idx] = grad_list[layer_idx].reshape(params[layer_idx].shape)
    return tuple(grad_list)


def rademacher(tens):
    """Draws a random tensor of size [tens.shape] from the Rademacher distribution (P(x=1) == P(x=-1) == 0.5)"""
    x = torch.empty_like(tens)
    x.random_(0, 2)
    x[x == 0] = -1
    return x


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def FedWeightAvg(w, size):
    totalSize = sum(size)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w[0][k]*size[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * size[i]
        # print(w_avg[k])
        w_avg[k] = torch.div(w_avg[k], totalSize)
    return w_avg


def clip_weights(weights, max_norm):
    norm = torch.norm(weights, p=2)
    scale = min(1.0, max_norm / norm)
    return weights * scale


def accuracy(model, dataloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        #device = model._module.device if isinstance(model, GradSampleModule) else model.device
        for _batch_idx, batch in enumerate(dataloader):
            if len(batch) == 3:  
                data, target, group = batch
            elif len(batch) == 2: 
                data, target = batch
                group = target
            # Ensure that both data and target are moved to the same device as the model
            data, target = (
                data.to(device, non_blocking=True),
                target.to(device, non_blocking=True).long(),
            )
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum()
    return (correct / total).item()


def accuracy_per_group(model, dataloader, device, num_groups):
    correct_per_group = [0] * num_groups
    total_per_group = [0] * num_groups
    with torch.no_grad():
        #device = model._module.device if isinstance(model, GradSampleModule) else model.device
        for _batch_idx, batch in enumerate(dataloader):
            if len(batch) == 3:  
                data, target, group = batch
            elif len(batch) == 2: 
                data, target = batch
                group = target
            # Ensure that both data and target are moved to the same device as the model
            data, target = (
                data.to(device, non_blocking=True),
                target.to(device, non_blocking=True).long(),
            )
            per_group = split_by_group(data, target, group, num_groups)
            for i, group in enumerate(per_group):
                data_group, labels_group = group
                outputs = model(data_group)
                _, predicted = torch.max(outputs, 1)
                total_per_group[i] += labels_group.size(0)
                correct_per_group[i] += (predicted == labels_group).sum()
    return [float(correct_per_group[i] / total_per_group[i]) for i in range(num_groups)]


def fairness_metrics_per_group(model, dataloader, device, num_groups):
    """ 
    Compute accuracy per group, Demographic Parity, Equal Opportunity, 和 Equalized Odds
    """
    model.eval().to(device)  # Make sure the model is on the correct device
    correct_per_group = [0] * num_groups
    total_per_group = [0] * num_groups

    total_pred_positive_per_group = [0] * num_groups  # 预测为正例的总数 (用于 DP)
    total_actual_positive_per_group = [0] * num_groups  # 真实正例的总数 (用于 EOpp)
    total_actual_negative_per_group = [0] * num_groups  # 真实负例的总数 (用于 EOdds)
    
    pred_positive_given_actual_positive = [0] * num_groups  # P(ŷ=1 | Y=1, G=g) (EOpp)
    pred_positive_given_actual_negative = [0] * num_groups  # P(ŷ=1 | Y=0, G=g) (EOdds)

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:  
                data, target, group = batch
            elif len(batch) == 2: 
                data, target = batch
                group = target  
            
            data, target = data.to(device), target.to(device).long()
            outputs = model(data)
            predicted = torch.argmax(outputs, dim=1)  

            for i in range(num_groups):
                mask = (group == i).to(device)
                total_per_group[i] += mask.sum().item()
                correct_per_group[i] += (predicted[mask] == target[mask]).sum().item()

                total_pred_positive_per_group[i] += (predicted[mask] == 1).sum().item()
                
                # Equal Opportunity and Equalized Odds
                actual_positive_mask = (target == 1) & mask
                actual_negative_mask = (target == 0) & mask
                total_actual_positive_per_group[i] += actual_positive_mask.sum().item()
                total_actual_negative_per_group[i] += actual_negative_mask.sum().item()
                pred_positive_given_actual_positive[i] += (predicted[actual_positive_mask] == 1).sum().item()
                pred_positive_given_actual_negative[i] += (predicted[actual_negative_mask] == 1).sum().item()

    acc_per_group = [correct_per_group[i] / total_per_group[i] if total_per_group[i] > 0 else 0 for i in range(num_groups)]

    dop_per_group = [total_pred_positive_per_group[i] / total_per_group[i] if total_per_group[i] > 0 else 0 for i in range(num_groups)]

    eop_per_group = [pred_positive_given_actual_positive[i] / total_actual_positive_per_group[i] if total_actual_positive_per_group[i] > 0 else 0 for i in range(num_groups)]

    neg_per_group = [pred_positive_given_actual_negative[i] / total_actual_negative_per_group[i] if total_actual_negative_per_group[i] > 0 else 0 for i in range(num_groups)]
    eod_per_group = [eop_per_group[i] + neg_per_group [i] for i in range(len(neg_per_group))]

    return {
        "acc_per_group": acc_per_group,
        "dop_per_group": dop_per_group,
        "eop_per_group": eop_per_group,
        "eod_per_group": eod_per_group
    }

