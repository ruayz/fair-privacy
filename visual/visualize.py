import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

def plot_roc(fpr_list, tpr_list, roc_auc, path):
    """Function to get the ROC plot using FPR and TPR results

    Args:
        fpr_list (list or ndarray): List of FPR values
        tpr_list (list or ndarray): List of TPR values
        roc_auc (float or floating): Area Under the ROC Curve
        path (str): Folder for saving the ROC plot
    """
    range01 = np.linspace(0, 1)
    plt.fill_between(fpr_list, tpr_list, alpha=0.15)
    plt.plot(fpr_list, tpr_list, label="ROC curve")
    plt.plot(range01, range01, "--", label="Random guess")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.legend()
    plt.xlabel("False positive rate (FPR)")
    plt.ylabel("True positive rate (TPR)")
    plt.title("ROC curve")
    plt.text(
        0.7,
        0.3,
        f"AUC = {roc_auc:.03f}",
        horizontalalignment="center",
        verticalalignment="center",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.savefig(
        fname=path,
        dpi=1000,
    )
    plt.clf()


def plot_roc_log(fpr_list, tpr_list, roc_auc, path):
    """Function to get the log-scale ROC plot using FPR and TPR results

    Args:
        fpr_list (list or ndarray): List of False Positive Rate values
        tpr_list (list or ndarray): List of True Positive Rate values
        roc_auc (float or floating): Area Under the ROC Curve
        path (str): Folder for saving the ROC plot
    """
    range01 = np.linspace(0, 1)
    plt.fill_between(fpr_list, tpr_list, alpha=0.15)
    plt.plot(fpr_list, tpr_list, label="ROC curve")
    plt.plot(range01, range01, "--", label="Random guess")
    plt.xlim([10e-6, 1])
    plt.ylim([10e-6, 1])
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.xlabel("False positive rate (FPR)")
    plt.ylabel("True positive rate (TPR)")
    plt.title("ROC curve")
    plt.text(
        0.7,
        0.3,
        f"AUC = {roc_auc:.03f}",
        horizontalalignment="center",
        verticalalignment="center",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.savefig(
        fname=path,
        dpi=1000,
    )
    plt.clf()


def plot_single_line(nums, delta, path):
    plt.figure(figsize=(10, 6))
    # 全局字体大小设置
    plt.rcParams.update({
        "font.size": 20,      # 默认字体大小
        "axes.labelsize": 20, # 坐标轴标签字体大小
        "xtick.labelsize": 16, # x轴刻度字体大小
        "ytick.labelsize": 16  # y轴刻度字体大小
    })

    plt.plot(nums, delta, marker='x', linestyle='-', color='#71b7ed', label='Error')
    # plt.fill_between(nums, errs_mean - errs_std, errs_mean + errs_std, color='#71b7ed', alpha=0.2)
    # 添加标题和标签
    # plt.xlabel('nums')
    plt.ylabel('Mean of Diff.')
    # # 设置严格的横坐标刻度
    # plt.xticks(nums)
    r = [f"2R={i}" for i in nums]
    plt.xticks(nums, r)  # nums，r是标签

    plt.grid()

    # 显示图例
    plt.legend()
    plt.tight_layout()  # 自动调整布局以适应图形区域
    plt.savefig(path, dpi=300, bbox_inches='tight')


def plot_box_model(nums, all_err, path):
    plt.figure(figsize=(10, 6))
    # 全局字体大小设置
    plt.rcParams.update({
        "font.size": 22,      # 默认字体大小
        "axes.labelsize": 22, # 坐标轴标签字体大小
        "xtick.labelsize": 20, # x轴刻度字体大小
        "ytick.labelsize": 16  # y轴刻度字体大小
    })

    plt.boxplot(all_err.T, positions=nums, showfliers=False, whis=1, widths=15)

    means = np.mean(all_err, axis=1)
    plt.plot(nums, means, marker='x', linestyle='-', color='#71b7ed', label='Mean of Diff.') # 绘制折线图（均值）
    r = [f"2R={i}" for i in nums]
    # 设置 x 轴范围
    plt.xticks(nums, r, rotation=45)
    plt.xlim([nums[0] - 20, nums[-1] + 20])
    plt.ylabel('Difference')
    # plt.grid(True)

    # 显示网格
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # 显示图例
    plt.legend()

    plt.tight_layout()  # 自动调整布局以适应图形区域
    plt.savefig(path, dpi=300, bbox_inches='tight')


def plot_box_samples(sample_sizes, all_err, path):
        # 绘图
    plt.figure(figsize=(10, 8))

    # 绘制箱线图
    positions = [size * 10 for size in sample_sizes]  # 横坐标放大便于展示
    plt.boxplot(all_err, positions=positions, showfliers=False, widths=50, whis=1.5)

    # 计算均值并绘制折线图
    means = [np.mean(err) for err in all_err]
    plt.plot(positions, means, marker="o", color="red", label="Mean of Err")

    # 设置图表参数
    plt.xticks(positions, [str(size * 10) for size in sample_sizes], rotation=45)
    plt.ylabel("Error")
    plt.title("Error Distribution by Sample Size")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()

    plt.tight_layout()  # 自动调整布局以适应图形区域
    plt.savefig(path, dpi=300, bbox_inches='tight')


def plot_scatter(nums, df, group_errs, path):
    num_group = 10
    plt.figure(figsize=(10, 6))

    # 全局字体大小设置
    plt.rcParams.update({
        "font.size": 22,      # 默认字体大小
        "axes.labelsize": 22, # 坐标轴标签字体大小
        "xtick.labelsize": 16, # x轴刻度字体大小
        "ytick.labelsize": 16  # y轴刻度字体大小
    })

    # 绘制箱线图
    positions = range(0, num_group)
    box_data = [df[df["group"] == group]["err"] for group in range(num_group)]
    # plt.boxplot(box_data, positions=positions, showfliers=False, widths=0.6, whis=1.5)

    # 绘制散点图
    for i, group in enumerate(range(num_group)):
        group_data = df[df["group"] == group]["err"]
        x_jitter = np.random.uniform(-0.15, 0.15, size=len(group_data))  # 添加水平抖动
        plt.scatter(i + x_jitter, group_data, alpha=0.7, s=20, color="#6dbf70")

    plt.plot(positions, group_errs, marker='x', linestyle='-', color='#71b7ed', linewidth=2.5, label='Mean of Diff.') # 绘制折线图（均值）

    plt.legend()
    plt.xticks(positions, range(num_group))
    plt.xlabel("Group")
    plt.ylabel('Difference')
    plt.ylim(-0.1, 0.1)
    # plt.grid(True)

    # 显示网格
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # 显示图例

    plt.tight_layout()  # 自动调整布局以适应图形区域
    plt.savefig(path, dpi=300, bbox_inches='tight')


def plot_two_method_scatter(alooa_acc, looa_acc, path, method="PA-ALOOA"):
    num_group = 10
    plt.figure(figsize=(10, 8))

    # 全局字体大小设置
    plt.rcParams.update({
        "font.size": 22,      # 默认字体大小
        "axes.labelsize": 22, # 坐标轴标签字体大小
        "xtick.labelsize": 16, # x轴刻度字体大小
        "ytick.labelsize": 16  # y轴刻度字体大小
    })


    # 添加 y = x 的虚线
    plt.plot([0.44, 0.85], [0.44, 0.85], linestyle='--', color='#afb1ae', label="y = x")
    plt.scatter(alooa_acc, looa_acc, alpha=0.8, s=20, color="#2ca02c")
    plt.xlabel(method)
    plt.ylabel('PA-LOOA')
    plt.xlim(0.49, 0.85)
    plt.ylim(0.49, 0.85)
    plt.grid(True)

    # 显示网格
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # 显示图例

    plt.tight_layout()  # 自动调整布局以适应图形区域
    plt.savefig(path, dpi=300, bbox_inches='tight')


