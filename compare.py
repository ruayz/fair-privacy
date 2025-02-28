import numpy as np
import pandas as pd
import pickle
import yaml
import sys
from audit import exp_one_acc
from visual import *
from scipy.stats import kruskal

log_dir = 'exp/demo_mnist_compare/'
all_samples = pd.read_csv(f"{log_dir}sample_idx.txt", sep=',')
all_idxs = all_samples["idx"]

def get_all_signle_acc(num):
    path_random = f'{log_dir}data/group_idx_acc_r{num}.txt'
    df = pd.read_csv(path_random, sep=',')
    all_single = df#["acc"]
    return all_single

def get_all_random_acc(num, method=None):
    path_random = f'{log_dir}CNN/regular_{num}/eps0/idx_acc.txt'
    if method == "AVG":
        path_random = f'{log_dir}CNN/regular_{num}/eps0/idx_avg_acc.txt'
    if method == "g_AVG":
        path_random = f'{log_dir}CNN/regular_{num}/eps0/idx_group_avg_acc.txt'
    df = pd.read_csv(path_random, sep=',')
    all_random = df#["acc"]
    return all_random

def exp_compare():
    path = log_dir + "figs/model_compare.pdf"
    random_model_num = np.array([49, 99, 149, 199, 249, 299])
    single_model_num = np.array([100, 200, 300, 400, 500, 600])
    # random_model_num = np.array([49, 99])
    # single_model_num = np.array([100, 200])
    
    all_err = []
    
    for i in range(len(random_model_num)):
        random_exp = get_all_random_acc(random_model_num[i])
        single_exp = get_all_signle_acc(single_model_num[i])
        err = abs(random_exp["acc"] - single_exp["acc"])
        single_exp["err"] = err
        all_err.append(err.values)
        # print(single_exp[err>=0.1][["idx","group", "err"]])

        # print(np.mean(err), np.std(err))

    plot_box_model(single_model_num, np.array(all_err), path)


def exp_compare_group():
    path = log_dir + "figs/group_compare.pdf"
    
    random_exp = get_all_random_acc(299)
    single_exp = get_all_signle_acc(600)
    err = random_exp["acc"] - single_exp["acc"] 
    single_exp["err"] = err
    single_exp["abs_err"] = abs(err)
    # print(single_exp[abs(err)>=0.07][["idx","group", "err"]])
    # print(single_acc.groupby("group")["err"].mean())

    group_errs = []
    for i in range(10):
        err = abs(np.mean(single_exp[single_exp["group"]==i]["acc"]) - np.mean(random_exp[random_exp["group"]==i]["acc"]))
        group_errs.append(err)

    stat, p = kruskal(*group_errs)
    print(f"统计量: {stat}, p值: {p}")

    plot_scatter(600, single_exp, group_errs, path)

def exp_compare_method():
    path = log_dir + "figs/method_compare.pdf"
    
    random_exp = get_all_random_acc(299)
    single_exp = get_all_signle_acc(600)
    err = abs(single_exp["acc"] - random_exp["acc"])
    single_exp["err"] = err
    print(np.mean(err))

    plot_two_method_scatter(random_exp["acc"], single_exp["acc"], path)


if __name__ == "__main__":
    # exp_compare_group()
    exp_compare_method()
    # exp_compare()
