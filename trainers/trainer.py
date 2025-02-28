# Copyright (c) 2020 Data Privacy and Trustworthy Machine Learning Research Lab
# Licensed under the MIT License. See LICENSE file for details.

import collections
import os
import time

import numpy as np
import pandas as pd
import torch

#from datasets.loaders import get_loader
#from functorch import make_functional, vjp, grad
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from trainers.utils import *
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule
import pickle


class BaseTrainer:
    """Base class for various training methods"""

    def __init__(self,
                 model,
                 optimizer,
                 train_loader,
                 device,
                 logdir,
                 #dataset,
                 method="regular",
                 max_epochs=100,
                 num_groups=None,
                 selected_groups=[0, 1],
                 evaluate_angles=False,
                 evaluate_hessian=False,
                 angle_comp_step=10,
                 lr=0.01,
                 seed=0,
                 num_hutchinson_estimates=100,
                 sampled_expected_loss=False
                 ):

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        #self.evaluator = evaluator
        self.device = device
        self.logdir = logdir

        #self.dataset = dataset
        self.method = method
        self.max_epochs = max_epochs
        self.num_groups = num_groups
        self.num_batch = len(self.train_loader)
        self.selected_groups = selected_groups
        self.epoch = 0
        self.num_layers = get_num_layers(self.model)

        self.evaluate_angles = evaluate_angles
        self.evaluate_hessian = evaluate_hessian
        self.angle_comp_step = angle_comp_step
        self.lr = lr
        self.seed = seed
        self.num_hutchinson_estimates = num_hutchinson_estimates
        self.sampled_expected_loss = sampled_expected_loss

    def _train_epoch(self, cosine_sim_per_epoch, expected_loss, train_loder, param_for_step=None):
        # methods: regular, dpsgd, dpsgd-global, dpsgd-f, dpsgd-global-adapt
        criterion = torch.nn.CrossEntropyLoss()
        losses = []
        losses_per_group = np.zeros(self.num_groups)
        all_grad_norms = [[] for _ in range(self.num_groups)]
        all_cilp_grad_norms = [[] for _ in range(self.num_groups)]
        group_max_grads = [0] * self.num_groups
        g_B_norms = []
        bar_g_B_norms = []
        g_B_k_norms = [[] for _ in range(self.num_groups)]
        bar_g_B_k_norms = [[] for _ in range(self.num_groups)]
        sum_g_D_k = [0] * self.num_groups
        sum_bar_g_D_k = [0] * self.num_groups

        for _batch_idx, batch in enumerate(train_loder):
            if len(batch) == 3:  
                data, target, group = batch
            elif len(batch) == 2: 
                data, target = batch
                group = target
            # Ensure that both data and target are moved to the same device as the model
            data, target = (
                data.to(self.device, non_blocking=True),
                target.to(self.device, non_blocking=True).long(),
            )
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(data)
            loss = criterion(output, target)
            losses_per_group = self.get_losses_per_group(criterion, data, target, group, losses_per_group)
            #print(torch.mean(loss))
            loss.backward()
            per_sample_grads = self.flatten_all_layer_params(self.model)

            # get sum of grads over groups over current batch
            if self.method == "regular":
                grad_norms, clip_grad_norms, sum_grad_vec_batch, sum_clip_grad_vec_batch = self.get_sum_grad_batch_from_vec(
                    per_sample_grads, group)
            elif self.method in ["dpsgd", "dpsgdg", "dpsgdga"]:
                grad_norms, clip_grad_norms, sum_grad_vec_batch, sum_clip_grad_vec_batch = self.get_sum_grad_batch_from_vec(
                    per_sample_grads, group, clipping_bound=self.optimizer.max_grad_norm)
            elif self.method in ["dpsgdf", "dpsgdfg", "dpsgdfga"]:
                C = self.compute_clipping_bound_per_sample(per_sample_grads, group)
                grad_norms, clip_grad_norms, sum_grad_vec_batch, sum_clip_grad_vec_batch = self.get_sum_grad_batch_from_vec(
                    per_sample_grads, group, clipping_bounds=C)
                
            _, group_counts_batch = split_by_group(data, target, group, self.num_groups, return_counts=1)
            '''
            g_B: torch.Tensor=所有样本梯度相加/样本量
            g_B_k: list, 每个元素是 torch.Tensor=每个组的累加梯度/该组的样本数
            bar_g_B: torch.Tensor=裁剪后的样本梯度相加/样本量 
            bar_g_B_k: list, 每个元素是 torch.Tensor=每个组裁剪后的累加梯度/该组的样本数
            '''
            g_B, g_B_k, bar_g_B, bar_g_B_k = self.mean_grads_over(group_counts_batch, sum_grad_vec_batch,
                                                                  sum_clip_grad_vec_batch)
            ########entire training dataset##########
            if (self.evaluate_angles or self.evaluate_hessian) and (
                    self.epoch * self.num_batch + _batch_idx) % self.angle_comp_step == 0:
                # compute sum of gradients over groups over entire training dataset
                if self.method == "regular":
                    sum_grad_vec_all, sum_clip_grad_vec_all, group_counts = self.get_sum_grad(
                        self.train_loader.dataset, criterion, g_B, bar_g_B, expected_loss, _batch_idx)
                elif self.method in ["dpsgd", "dpsgdf", "dpsgdg", "dpsgdga", "dpsgdfga"]:
                    sum_grad_vec_all, sum_clip_grad_vec_all, group_counts = self.get_sum_grad(self.train_loader.dataset,
                                                                                              criterion,
                                                                                              g_B,
                                                                                              bar_g_B,
                                                                                              expected_loss,
                                                                                              _batch_idx,
                                                                                              clipping_bound=self.optimizer.max_grad_norm)

                # average sum of gradients per group over entire training dataset
                _, g_D_k, _, _ = self.mean_grads_over(group_counts, sum_grad_vec_all, sum_clip_grad_vec_all)
                cosine_sim_per_epoch.append(self.evaluate_cosine_sim(_batch_idx, g_D_k, g_B, bar_g_B, g_B_k, bar_g_B_k))
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
            ###########################################

            for i in range(self.num_groups):
                if len(grad_norms[i]) != 0:
                    all_grad_norms[i] = all_grad_norms[i] + grad_norms[i]
                    all_cilp_grad_norms[i] = all_cilp_grad_norms[i] + clip_grad_norms[i]
                    group_max_grads[i] = max(group_max_grads[i], max(grad_norms[i]))
                    g_B_k_norms[i].append(torch.linalg.norm(g_B_k[i]).item())
                    bar_g_B_k_norms[i].append(torch.linalg.norm(bar_g_B_k[i]).item())
                    # if isinstance(sum_g_D_k[i], int):
                    #     sum_g_D_k[i] = g_B_k[i] * group_counts_batch[i]
                    #     sum_bar_g_D_k[i] = bar_g_B_k[i] * group_counts_batch[i]
                    # else:
                    #     sum_g_D_k[i] += g_B_k[i] * group_counts_batch[i]
                    #     sum_bar_g_D_k[i] += bar_g_B_k[i] * group_counts_batch[i]
            g_B_norms.append(torch.linalg.norm(g_B).item())
            bar_g_B_norms.append(torch.linalg.norm(bar_g_B).item())
            # if _batch_idx == 0:
            #     g_D = g_B
            #     bar_g_D = bar_g_B
            # else:
            #     g_D += g_B
            #     bar_g_D += bar_g_B

            if self.method == "dpsgdf":
                self.optimizer.step(C)
            elif self.method == "dpsgdg":
                self.optimizer.step(self.strict_max_grad_norm)
            elif self.method == "dpsgdfg":
                self.optimizer.step(C, self.strict_max_grad_norm)
            elif self.method == "dpsgdga":
                next_Z = self._update_Z(per_sample_grads, self.strict_max_grad_norm)
                self.optimizer.step(self.strict_max_grad_norm)
                self.strict_max_grad_norm = next_Z
            elif self.method == "dpsgdfga":
                next_Z = self._update_Z(per_sample_grads, self.strict_max_grad_norm)
                self.optimizer.step(C, self.strict_max_grad_norm)
                self.strict_max_grad_norm = next_Z
            else:
                self.optimizer.step()
            losses.append(loss.item())
        if self.method != "regular":
            if self.method in ["dpsgdf", "dpsgdfg", "dpsgdga", "dpsgdfga"]:
                self._update_privacy_accountant()
            epsilon = self.privacy_engine.get_epsilon(delta=self.delta)
            #print(f"(ε = {epsilon:.2f}, δ = {self.delta})")
            #privacy_dict = {"epsilon": epsilon, "delta": self.delta}
        group_ave_grad_norms = [np.mean(all_grad_norms[i]) for i in range(self.num_groups)]
        group_ave_clip_grad_norms = [np.mean(all_cilp_grad_norms[i]) for i in range(self.num_groups)]
        #print(f"group_ave_grad_norms:{group_ave_grad_norms}")
        group_norm_grad_ave = [np.mean(g_B_norms)] + [np.mean(g_B_k_norms[i]) for i in range(self.num_groups)]
        group_norm_clip_grad_ave = [np.mean(bar_g_B_norms)] + [np.mean(bar_g_B_k_norms[i]) for i in range(self.num_groups)]
        # 计算夹角（单位：度）
        # group_angle = [cosine_similarity(sum_g_D_k[i].unsqueeze(0), g_D.unsqueeze(0)).item() for i in range(self.num_groups)]
        # group_clip_angle = [cosine_similarity(sum_bar_g_D_k[i].unsqueeze(0), bar_g_D.unsqueeze(0)).item() for i in range(self.num_groups)]
        group_angle = []
        group_clip_angle = []
        return group_ave_grad_norms, group_ave_clip_grad_norms, group_max_grads, \
                group_norm_grad_ave, group_norm_clip_grad_ave, group_angle, group_clip_angle, losses, losses_per_group / self.num_batch


    def train(self):
        training_time = 0
        group_acc_epochs = []
        group_loss_epochs = []
        cos_sim_per_epoch = []
        expected_loss = []
        avg_grad_norms_epochs = []
        avg_clip_grad_norms_epochs = []
        max_grads_epochs = []
        norm_avg_grad_epochs = []
        norm_clip_avg_grad_epochs = []
        group_angle_epochs = []
        group_clip_angle_epochs = []

        while self.epoch < self.max_epochs:
            epoch_start_time = time.time()
            self.model.train()

            # # compute initial weights
            # if self.epoch == 0:
            #     acc_per_epoch = accuracy(self.model, self.train_loader, method=self.method)
            #     group_acc_per_epoch = accuracy_per_group(self.model, self.train_loader,
            #                                               num_groups=self.num_groups, method=self.method)
            #     #print(f"group_acc_per_epochs: {group_acc_per_epochs}")
            #     group_acc_epochs.append([-1, acc_per_epoch] + list(group_acc_per_epoch))

            avg_grad_norms, avg_clip_grad_norms, max_grads, norm_avg_grad, norm_clip_avg_grad, \
                group_angle, group_clip_angle, losses, group_losses = self._train_epoch(cos_sim_per_epoch,
                                                                                        expected_loss,
                                                                                        self.train_loader)
            #compute acc
            acc_per_epoch = accuracy(self.model, self.train_loader, self.device)
            group_acc_per_epoch = accuracy_per_group(self.model, self.train_loader,
                                                    self.device, num_groups=self.num_groups)
            #print(f"group_acc_per_epochs: {group_acc_per_epochs}")

            group_acc_epochs.append([self.epoch, acc_per_epoch] + list(group_acc_per_epoch))
            group_loss_epochs.append([self.epoch, np.mean(losses)] + list(group_losses))
            avg_grad_norms_epochs.append([self.epoch] + list(avg_grad_norms)) # diff group avg gradient norm
            avg_clip_grad_norms_epochs.append([self.epoch] + list(avg_clip_grad_norms))
            max_grads_epochs.append([self.epoch] + list(max_grads))
            norm_avg_grad_epochs.append([self.epoch] + list(norm_avg_grad))
            norm_clip_avg_grad_epochs.append([self.epoch] + list(norm_clip_avg_grad))
            # group_angle_epochs.append([self.epoch] + list(group_angle))
            # group_clip_angle_epochs.append([self.epoch] + list(group_clip_angle))

            epoch_training_time = time.time() - epoch_start_time
            training_time += epoch_training_time

            if (self.epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{self.epoch + 1}/{self.max_epochs}] | Train Loss: {np.mean(losses):.4f} | Train Acc: {acc_per_epoch:.4f}"
                )

            self.epoch += 1

            if self.epoch == self.max_epochs:
                loss_dict = dict()

                loss_dict["final_loss"] = np.mean(losses)
                loss_dict["final_loss_per_group"] = group_losses

        # K = self.num_groups
        # # write group_loss to csv
        # columns = ["epoch", "train_loss"] + [f"train_loss_{k}" for k in range(K)]
        # self.create_csv(group_loss_epochs, columns, "train_loss_per_epochs")
        # # write group_acc to csv
        # columns = ["epoch", "train_acc"] + [f"train_acc_{k}" for k in range(K)]
        # self.create_csv(group_acc_epochs, columns, "train_acc_per_epochs")
        # # write avg_grad_norms to csv
        # columns = ["epoch"] + [f"ave_grads_{k}" for k in range(K)]
        # self.create_csv(avg_grad_norms_epochs, columns, "avg_grad_norms_per_epochs")
        # columns = ["epoch"] + [f"ave_clip_grads_{k}" for k in range(K)]
        # self.create_csv(avg_clip_grad_norms_epochs, columns, "avg_clip_grad_norms_per_epochs")
        # # write nrom_avg_grad to csv
        # columns = ["epoch", "norm_avg_grad"] + [f"norm_avg_grad_{k}" for k in range(K)]
        # self.create_csv(norm_avg_grad_epochs, columns, "norm_avg_grad_per_epochs")
        # columns = ["epoch", "norm_clip_avg_grad"] + [f"norm_clip_avg_grad_{k}" for k in range(K)]
        # self.create_csv(norm_clip_avg_grad_epochs, columns, "norm_clip_avg_grad_per_epochs")
        # # write angle to csv
        # columns = ["epoch"] + [f"group_angle_{k}" for k in range(K)]
        # self.create_csv(group_angle_epochs, columns, "group_angle_per_epochs")
        # columns = ["epoch"] + [f"group_clip_angle_{k}" for k in range(K)]
        # self.create_csv(group_clip_angle_epochs, columns, "group_clip_angle_per_epochs")

        self.model.to("cpu")
        return self.model
        

    def create_csv(self, data, columns, title):
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join(self.logdir, f"{title}.csv"), index=False)

    def flatten_all_layer_params(self, net):
        """
        Flatten the parameters of all layers in a modelv

        Args:
            model: a pytorch model

        Returns:
            a tensor of shape num_samples in a batch * num_params
        """
        per_sample_grad = None
        for n, p in net.named_parameters():
            if p.requires_grad:
                if per_sample_grad is None:
                    per_sample_grad = torch.flatten(p.grad_sample, 1, -1)
                else:
                    per_sample_grad = torch.cat((per_sample_grad, torch.flatten(p.grad_sample, 1, -1)), 1)
        return per_sample_grad

    # def record_expected_loss(self, R_non_private, R_clip, R_noise, R_clip_dir_inner_prod_term, R_clip_dir_hess_term,
    #                          R_clip_dir, R_clip_mag_inner_prod_term, R_clip_mag_hess_term, R_clip_mag, batch_idx):
    #     step = self.epoch * self.num_batch + batch_idx
    #     self.writer.write_scalars("R_non_private", {'group' + str(k): v for k, v in enumerate(R_non_private)}, step)
    #     self.writer.write_scalars("R_clip", {'group' + str(k): v for k, v in enumerate(R_clip)}, step)
    #     self.writer.write_scalars("R_noise", {'group' + str(k): v for k, v in enumerate(R_noise)}, step)
    #     self.writer.write_scalars("R_clip_dir_inner_prod_term",
    #                               {'group' + str(k): v for k, v in enumerate(R_clip_dir_inner_prod_term)}, step)
    #     self.writer.write_scalars("R_clip_dir_hess_term",
    #                               {'group' + str(k): v for k, v in enumerate(R_clip_dir_hess_term)}, step)
    #     self.writer.write_scalars("R_clip_dir", {'group' + str(k): v for k, v in enumerate(R_clip_dir)}, step)
    #     self.writer.write_scalars("R_clip_mag_inner_prod_term",
    #                               {'group' + str(k): v for k, v in enumerate(R_clip_mag_inner_prod_term)}, step)
    #     self.writer.write_scalars("R_clip_mag_hess_term",
    #                               {'group' + str(k): v for k, v in enumerate(R_clip_mag_hess_term)}, step)
    #     self.writer.write_scalars("R_clip_mag", {'group' + str(k): v for k, v in enumerate(R_clip_mag)}, step)

    def expected_loss_batch_terms(self, data, target, group, g_B, bar_g_B, C, criterion):
        def create_hvp_fn(data, target):
            func_model, params = make_functional(self.model)

            def compute_loss(params):
                preds = func_model(params, data)
                loss = criterion(preds, target)
                return loss

            _, hvp_fn = vjp(grad(compute_loss), params)
            return hvp_fn

        per_group, counts = split_by_group(data, target, group, self.num_groups, True)
        per_slct_group = [per_group[i] for i in self.selected_groups]
        slct_counts = [counts[i] for i in self.selected_groups]
        groups_len = len(self.selected_groups)
        grad_hess_grad = np.zeros(groups_len)
        clip_grad_hess_clip_grad = np.zeros(groups_len)
        R_noise = np.zeros(groups_len)
        loss = np.zeros(groups_len)
        self.model.disable_hooks()
        _, params = make_functional(self.model)
        unflattened_g_B = unflatten_grads(params, g_B)
        unflattened_bar_g_B = unflatten_grads(params, bar_g_B)
        for group_idx, (data_group, target_group) in enumerate(per_slct_group):
            with torch.no_grad():
                hvp_fn = create_hvp_fn(data_group, target_group)
                self.optimizer.zero_grad()
                preds = self.model(data_group)
                loss[group_idx] = criterion(preds, target_group) * slct_counts[group_idx]
                result = 0
                for i in range(self.num_hutchinson_estimates):
                    rand_z = tuple(rademacher(el) for el in params)
                    hess_z = hvp_fn(rand_z)[0]
                    z_hess_z = torch.sum(
                        torch.stack([torch.dot(x.flatten(), y.flatten()) for (x, y) in zip(rand_z, hess_z)]))
                    result += z_hess_z.item()
                # combine results taking into account different batch sizes
                hessian_trace = result * slct_counts[group_idx] / self.num_hutchinson_estimates
                grad_hess = hvp_fn(unflattened_g_B)[0]
                flat_grad_hess = torch.cat([torch.flatten(t) for t in grad_hess])
                grad_hess_grad[group_idx] = torch.dot(flat_grad_hess, g_B) * slct_counts[group_idx]
                clip_grad_hess = hvp_fn(unflattened_bar_g_B)[0]
                flat_clip_grad_hess = torch.cat([torch.flatten(t) for t in clip_grad_hess])
                clip_grad_hess_clip_grad[group_idx] = torch.dot(flat_clip_grad_hess, bar_g_B) * slct_counts[group_idx]
                R_noise[group_idx] = self.lr ** 2 / 2 * hessian_trace * C ** 2 * self.optimizer.noise_multiplier ** 2
        self.model.enable_hooks()
        return grad_hess_grad, clip_grad_hess_clip_grad, R_noise, loss

    def expected_loss(self, g_B, bar_g_B, sum_grad_vec, grad_hess_grad, clip_grad_hess_clip_grad,
                      R_noise, loss, group_counts, expected_loss_terms, batch_indx):
        norm_g_B = torch.linalg.norm(g_B).item()
        norm_bar_g_B = torch.linalg.norm(bar_g_B).item()
        groups_len = len(self.selected_groups)
        R_non_private = np.zeros(groups_len)
        R_clip = np.zeros(groups_len)
        new_R_clip_dir = np.zeros(groups_len)
        new_R_clip_dir_inner_prod_term = np.zeros(groups_len)
        new_R_clip_dir_hess_term = np.zeros(groups_len)
        new_R_clip_mag = np.zeros(groups_len)
        new_R_clip_mag_inner_prod_term = np.zeros(groups_len)
        new_R_clip_mag_hess_term = np.zeros(groups_len)
        for group_idx in range(groups_len):
            g_D_a = sum_grad_vec[group_idx] / group_counts[group_idx]
            group_grad_dot_grad = torch.dot(g_D_a, g_B)
            R_non_private[group_idx] = loss[group_idx] - self.lr * group_grad_dot_grad + self.lr ** 2 / 2 * \
                                       grad_hess_grad[group_idx]
            R_clip[group_idx] = self.lr * (
                    group_grad_dot_grad - torch.dot(g_D_a, bar_g_B)) \
                                + self.lr ** 2 / 2 * (clip_grad_hess_clip_grad[group_idx] - grad_hess_grad[group_idx])

            new_R_clip_dir_inner_prod_term[group_idx] = self.lr * torch.dot(g_D_a,
                                                                            norm_bar_g_B / norm_g_B * g_B - bar_g_B)
            new_R_clip_dir_hess_term[group_idx] = self.lr ** 2 / 2 * (
                    clip_grad_hess_clip_grad[group_idx] - (norm_bar_g_B / norm_g_B) ** 2 * grad_hess_grad[
                group_idx])
            new_R_clip_dir[group_idx] = new_R_clip_dir_inner_prod_term[group_idx] + new_R_clip_dir_hess_term[group_idx]
            new_R_clip_mag_inner_prod_term[group_idx] = self.lr * torch.dot(g_D_a, g_B - norm_bar_g_B / norm_g_B * g_B)
            new_R_clip_mag_hess_term[group_idx] = self.lr ** 2 / 2 * ((norm_bar_g_B / norm_g_B) ** 2 - 1) * \
                                                  grad_hess_grad[group_idx]
            new_R_clip_mag[group_idx] = new_R_clip_mag_inner_prod_term[group_idx] + new_R_clip_mag_hess_term[group_idx]

        self.record_expected_loss(R_non_private, R_clip, R_noise, new_R_clip_dir_inner_prod_term,
                                  new_R_clip_dir_hess_term,
                                  new_R_clip_dir, new_R_clip_mag_inner_prod_term, new_R_clip_mag_hess_term,
                                  new_R_clip_mag, batch_indx)
        row = [self.epoch,
               batch_indx] + R_non_private.tolist() + R_clip.tolist() + new_R_clip_dir_inner_prod_term.tolist() + \
              new_R_clip_dir_hess_term.tolist() + new_R_clip_dir.tolist() + new_R_clip_mag_inner_prod_term.tolist() + \
              new_R_clip_mag_hess_term.tolist() + new_R_clip_mag.tolist() + R_noise.tolist()
        expected_loss_terms.append(row)

    def get_losses_per_group(self, criterion, data, target, group, group_losses):
        '''
        Given subset of GroupLabelDataset (data, target, group), computes
        loss of model on each subset (data, target, group=k) and returns
        np array of length num_groups = group_losses + group losses over given data
        '''
        per_group = split_by_group(data, target, group, self.num_groups)
        group_loss_batch = np.zeros(self.num_groups)
        for group_idx, (data_group, target_group) in enumerate(per_group):
            with torch.no_grad():
                if data_group.shape[0] == 0:  # if batch does not contain samples of group i
                    group_loss_batch[group_idx] = 0
                else:
                    group_output = self.model(data_group)
                    group_loss_batch[group_idx] = criterion(group_output, target_group).item()
        group_losses = group_loss_batch + group_losses
        return group_losses

    def get_sum_grad_batch(self, data, targets, groups, criterion, **kwargs):
        data = data.to(self.device)
        targets = targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        per_sample_grads = self.flatten_all_layer_params()

        return self.get_sum_grad_batch_from_vec(per_sample_grads, groups, **kwargs)

    def get_sum_grad_batch_from_vec(self, per_sample_grads, groups, **kwargs):
        # if self.method == "dpsgdf":
        #     #clipping_bounds = self.compute_clipping_bound_per_sample(per_sample_grads, groups)
        #     grad_norms, clip_grad_norms, sum_grad_vec, sum_clip_grad_vec = get_grad_norms_clip(per_sample_grads, groups,
        #                                                                                        self.num_groups,
        #                                                                                        self.clipping_scale_fn,
        #                                                                                        clipping_bounds=kwargs["clipping_bounds"])
        grad_norms, clip_grad_norms, sum_grad_vec, sum_clip_grad_vec = get_grad_norms_clip(per_sample_grads, groups,
                                                                                            self.num_groups,
                                                                                            self.clipping_scale_fn,
                                                                                            **kwargs)
        return grad_norms, clip_grad_norms, sum_grad_vec, sum_clip_grad_vec

    def get_sum_grad(self, dataset, criterion, g_B, bar_g_B, expected_loss_terms, batch_idx, **kwargs):
        #loader = get_loader(self.train_loader.dataset, self.device, 1000, drop_last=False)
        loader = torch.utils.data.DataLoader(
            self.train_loader.dataset.to(self.device),
            batch_size=1000,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            pin_memory=False
        )
        groups_len = len(self.selected_groups)
        running_sum_grad_vec = None
        running_sum_clip_grad_vec = None
        sum_grad_hess_grad = np.zeros(groups_len)
        sum_clip_grad_hess_clip_grad = np.zeros(groups_len)
        sum_R_noise = np.zeros(groups_len)
        sum_loss = np.zeros(groups_len)
        # First argument is a dummy
        _, group_counts = split_by_group(dataset.y, dataset.y, dataset.z, self.num_groups, return_counts=True)
        for data, target, group in loader:
            if self.method == "dpsgdf":
                _, _, sum_grad_vec_batch, sum_clip_grad_vec_batch = self.get_sum_grad_batch(
                    data, target, group, criterion, **kwargs)
            else:
                _, _, sum_grad_vec_batch, sum_clip_grad_vec_batch = self.get_sum_grad_batch(
                    data, target, group, criterion, **kwargs)
            if running_sum_grad_vec is None:
                running_sum_grad_vec = sum_grad_vec_batch
            else:
                running_sum_grad_vec = [a + b for a, b in zip(running_sum_grad_vec, sum_grad_vec_batch)]
            if running_sum_clip_grad_vec is None:
                running_sum_clip_grad_vec = sum_clip_grad_vec_batch
            else:
                running_sum_clip_grad_vec = [a + b for a, b in zip(running_sum_clip_grad_vec, sum_clip_grad_vec_batch)]
            if self.evaluate_hessian and self.method != "regular":
                clipping_bound = kwargs['clipping_bound']
                grad_hess_grad, clip_grad_hess_clip_grad, R_noise, loss = self.expected_loss_batch_terms(
                    data, target, group, g_B, bar_g_B, clipping_bound, criterion)
                sum_grad_hess_grad += grad_hess_grad
                sum_clip_grad_hess_clip_grad += clip_grad_hess_clip_grad
                sum_R_noise += R_noise
                sum_loss += loss
            if self.sampled_expected_loss:
                _, group_counts = split_by_group(data, target, group, self.num_groups, return_counts=True)
                break


        if self.evaluate_hessian:
            final_sum_grad_vec_batch = [running_sum_grad_vec[i] for i in self.selected_groups]
            group_counts_vec = np.array([group_counts[i] for i in self.selected_groups])
            final_grad_hess_grad = sum_grad_hess_grad / group_counts_vec
            final_clip_grad_hess_clip_grad = sum_clip_grad_hess_clip_grad / group_counts_vec
            final_R_noise = sum_R_noise / group_counts_vec
            final_loss = sum_loss / group_counts_vec
            self.expected_loss(g_B, bar_g_B, final_sum_grad_vec_batch, final_grad_hess_grad,
                               final_clip_grad_hess_clip_grad, final_R_noise, final_loss,
                               group_counts_vec, expected_loss_terms, batch_idx)
        return running_sum_grad_vec, running_sum_clip_grad_vec, group_counts

    def mean_grads_over(self, group_counts, sum_grad_vec, clip_sum_grad_vec):
        g_D = torch.stack(sum_grad_vec, dim=0).sum(dim=0) / sum(group_counts)
        g_D_k = [sum_grad_vec[i] / group_counts[i] for i in range(self.num_groups)]

        bar_g_D = torch.stack(clip_sum_grad_vec, dim=0).sum(dim=0) / sum(group_counts)
        bar_g_D_k = [clip_sum_grad_vec[i] / group_counts[i] for i in range(self.num_groups)]
        return g_D, g_D_k, bar_g_D, bar_g_D_k

    def evaluate_cosine_sim(self, batch_idx, g_D_k, g_B, bar_g_B, g_B_k, bar_g_B_k):
        cos_g_D_k_g_B_k = []
        cos_g_D_k_bar_g_B_k = []
        cos_g_D_k_g_B = []
        cos_g_D_k_bar_g_B = []
        norm_g_D_k = []
        norm_g_B_k = []
        norm_bar_g_B_k = []

        cos_g_B_bar_g_B = cosine_similarity(g_B, bar_g_B, dim=0).item()
        norm_g_B = torch.linalg.norm(g_B).item()
        norm_bar_g_B = torch.linalg.norm(bar_g_B).item()

        for k in self.selected_groups:
            cos_g_D_k_g_B_k.append(cosine_similarity(g_D_k[k], g_B_k[k], dim=0).item())
            cos_g_D_k_bar_g_B_k.append(cosine_similarity(g_D_k[k], bar_g_B_k[k], dim=0).item())
            cos_g_D_k_g_B.append(cosine_similarity(g_D_k[k], g_B, dim=0).item())
            cos_g_D_k_bar_g_B.append(cosine_similarity(g_D_k[k], bar_g_B, dim=0).item())

            norm_g_D_k.append(torch.linalg.norm(g_D_k[k]).item())
            norm_g_B_k.append(torch.linalg.norm(g_B_k[k]).item())
            norm_bar_g_B_k.append(torch.linalg.norm(bar_g_B_k[k]).item())

        row = [self.epoch, batch_idx] + cos_g_D_k_g_B_k + cos_g_D_k_bar_g_B_k + cos_g_D_k_g_B + cos_g_D_k_bar_g_B + [
            cos_g_B_bar_g_B, norm_g_B, norm_bar_g_B] + norm_g_D_k + norm_g_B_k + norm_bar_g_B_k
        return row


class RegularTrainer(BaseTrainer):
    """Class for non-private training"""

    # given norm of gradient, computes S such that clipped gradient = S * gradient
    def clipping_scale_fn(self, grad_norm, idx):
        return 1


class DpsgdTrainer(BaseTrainer):
    """Class for DPSGD training"""

    # given norm of gradient, computes S such that clipped gradient = S * gradient
    def clipping_scale_fn(self, grad_norm, idx, clipping_bound):
        return min(1, clipping_bound / (grad_norm + 1e-6))

    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            device,
            logdir,
            delta=1e-5,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            train_loader,
            device,
            logdir,
            **kwargs
        )

        self.privacy_engine = privacy_engine
        self.delta = delta


class DpsgdFTrainer(BaseTrainer):
    """Class for DPSGD-F training"""

    # given norm of gradient, computes S such that clipped gradient = S * gradient
    def clipping_scale_fn(self, grad_norm, idx, clipping_bounds):
        #clipping_bounds = kwargs["clipping_bounds"]
        return min((clipping_bounds[idx] / (grad_norm + 1e-6)).item(), 1)

    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            device,
            logdir,
            delta=1e-5,
            base_max_grad_norm=1,  # C0
            counts_noise_multiplier=10,  # noise multiplier applied on mk and ok
            **kwargs
    ):
        """
        Initialization function. Initialize parent class while adding new parameter clipping_bound and noise_scale.

        Args:
            model: model from privacy_engine.make_private()
            optimizer: a DPSGDF_Optimizer
            privacy_engine: DPSGDF_Engine
            train_loader: train_loader from privacy_engine.make_private()
            valid_loader: normal pytorch data loader for validation set
            test_loader: normal pytorch data loader for test set
            writer: writer to tensorboard
            evaluator: evaluate for model performance
            device: device to train the model
            delta: definition in privacy budget
            clipping_bound: C0 in the original paper, defines the threshold of gradients
            counts_noise_multiplier: sigma1 in the original paper, defines noise added to the number of samples with gradient bigger than clipping_bound C0
        """
        super().__init__(
            model,
            optimizer,
            train_loader,
            device,
            logdir,
            **kwargs
        )

        self.privacy_engine = privacy_engine
        self.delta = delta
        # new parameters for DPSGDF
        self.base_max_grad_norm = base_max_grad_norm  # C0
        self.counts_noise_multiplier = counts_noise_multiplier  # noise scale applied on mk and ok
        self.sample_rate = 1 / self.num_batch
        self.privacy_step_history = []

    def _update_privacy_accountant(self):
        """
        The Opacus RDP accountant minimizes computation when many SGM steps are taken in a row with the same parameters.
        We alternate between privatizing counts, and gradients with different parameters.
        Accounting is sped up by tracking steps in groups rather than alternating.
        The order of accounting does not affect the privacy guarantee.
        """
        for step in self.privacy_step_history:
            self.privacy_engine.accountant.step(noise_multiplier=step[0], sample_rate=step[1])
        self.privacy_step_history = []

    def compute_clipping_bound_per_sample(self, per_sample_grads, group):
        """compute clipping bound for each group """
        # Calculate group-specific scaling factor
        group_norms = {}  # Store group-wise L2 norms
        for group_idx in range(self.num_groups):
            group_grads = per_sample_grads[group == group_idx]  # Get gradients for each group
            # group_norms[group_idx] = torch.mean(torch.norm(group_grads, p=2, dim=1))  # Average L2 norm
            if group_grads.size(0) > 0: 
                summed_group_grads = group_grads.sum(dim=0) 
                avg_group_grads = summed_group_grads / group_grads.size(0)  
                group_norms[group_idx] = torch.norm(avg_group_grads, p=2)  # Average L2 norm
            else:
                group_norms[group_idx] = 1e-6

        # Normalize clipping bounds based on the group's gradient norms
        Ck = {}
        # avg_norm = torch.mean(torch.norm(per_sample_grads, p=2, dim=1)) 
        avg_grads = torch.mean(group_grads, dim=0)
        avg_norm = torch.norm(avg_grads, p=2)

        for group_idx in range(self.num_groups):
            # Scale clipping bound for each group according to its gradient norm
            max_scale = 2
            scale = min(max_scale, avg_norm / group_norms[group_idx])
            # Ck[group_idx] = min(0.5, self.base_max_grad_norm * scale)
            Ck[group_idx] = self.base_max_grad_norm * scale
            #Ck[group_idx] = self.base_max_grad_norm * (1 + (avg_group_norm / group_norms[group_idx]) ** alpha)

        # Calculate final clipping bounds for each sample based on group
        per_sample_clipping_bound = []
        for i in range(len(group)):  # Looping over batch
            group_idx = group[i].item()
            # Adjust the clipping bound for each sample
            per_sample_clipping_bound.append(Ck[group_idx])

        # Return per-sample clipping bounds
        return torch.Tensor(per_sample_clipping_bound).to(device=self.device)

class DpsgdGlobalTrainer(DpsgdTrainer):

    # given norm of gradient, computes S such that clipped gradient = S * gradient
    def clipping_scale_fn(self, grad_norm, idx, clipping_bound):
        if grad_norm > self.strict_max_grad_norm:
            return 0
        else:
            return clipping_bound / self.strict_max_grad_norm

    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            device,
            logdir,
            delta=1e-5,
            strict_max_grad_norm=100,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            device,
            logdir,
            **kwargs
        )
        self.strict_max_grad_norm = strict_max_grad_norm

class DpsgdFGlobalTrainer(BaseTrainer):
    """Class for DPSGD-FGlobal training"""

    # given norm of gradient, computes S such that clipped gradient = S * gradient
    def clipping_scale_fn(self, grad_norm, idx, clipping_bounds):
        if grad_norm > self.strict_max_grad_norm:
            return 0
        else:
            return clipping_bounds[idx] / self.strict_max_grad_norm

    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            device,
            logdir,
            delta=1e-5,
            strict_max_grad_norm=100,
            base_max_grad_norm=1,  # C0
            counts_noise_multiplier=10,  # noise multiplier applied on mk and ok
            **kwargs
    ):
        """
        Initialization function. Initialize parent class while adding new parameter clipping_bound and noise_scale.

        Args:
            model: model from privacy_engine.make_private()
            optimizer: a DPSGDF_Optimizer
            privacy_engine: DPSGDF_Engine
            train_loader: train_loader from privacy_engine.make_private()
            valid_loader: normal pytorch data loader for validation set
            test_loader: normal pytorch data loader for test set
            writer: writer to tensorboard
            evaluator: evaluate for model performance
            device: device to train the model
            delta: definition in privacy budget
            clipping_bound: C0 in the original paper, defines the threshold of gradients
            counts_noise_multiplier: sigma1 in the original paper, defines noise added to the number of samples with gradient bigger than clipping_bound C0
        """
        super().__init__(
            model,
            optimizer,
            train_loader,
            device,
            logdir,
            **kwargs
        )

        self.privacy_engine = privacy_engine
        self.delta = delta
        # new parameters for DPSGDFGlobal
        self.strict_max_grad_norm = strict_max_grad_norm
        self.base_max_grad_norm = base_max_grad_norm  # C0
        self.counts_noise_multiplier = counts_noise_multiplier  # noise scale applied on mk and ok
        self.sample_rate = 1 / self.num_batch
        self.privacy_step_history = []

    def _update_privacy_accountant(self):
        """
        The Opacus RDP accountant minimizes computation when many SGM steps are taken in a row with the same parameters.
        We alternate between privatizing counts, and gradients with different parameters.
        Accounting is sped up by tracking steps in groups rather than alternating.
        The order of accounting does not affect the privacy guarantee.
        """
        for step in self.privacy_step_history:
            self.privacy_engine.accountant.step(noise_multiplier=step[0], sample_rate=step[1])
        self.privacy_step_history = []

    def compute_clipping_bound_per_sample(self, per_sample_grads, group):
        """compute clipping bound for each group """
        # Calculate group-specific scaling factor
        group_norms = {}  # Store group-wise L2 norms
        for group_idx in range(self.num_groups):
            group_grads = per_sample_grads[group == group_idx]  # Get gradients for each group
            if group_grads.size(0) > 0: 
                summed_group_grads = group_grads.sum(dim=0) 
                avg_group_grads = summed_group_grads / group_grads.size(0)  
                group_norms[group_idx] = torch.norm(avg_group_grads, p=2)  # Average L2 norm
            else:
                group_norms[group_idx] = 1e-8

        # Normalize clipping bounds based on the group's gradient norms
        Ck = {}
        avg_grads = torch.mean(group_grads, dim=0)
        avg_norm = torch.norm(avg_grads, p=2)

        for group_idx in range(self.num_groups):
            # Scale clipping bound for each group according to its gradient norm
            scale = min(1, avg_norm / group_norms[group_idx])
            Ck[group_idx] = self.base_max_grad_norm * scale
            #Ck[group_idx] = self.base_max_grad_norm * (1 + (avg_group_norm / group_norms[group_idx]) ** alpha)

        # Calculate final clipping bounds for each sample based on group
        per_sample_clipping_bound = []
        for i in range(len(group)):  # Looping over batch
            group_idx = group[i].item()
            # Adjust the clipping bound for each sample
            per_sample_clipping_bound.append(Ck[group_idx])

        # Return per-sample clipping bounds
        return torch.Tensor(per_sample_clipping_bound).to(device=self.device)


class DpsgdGlobalAdaptiveTrainer(BaseTrainer):

    # given norm of gradient, computes S such that clipped gradient = S * gradient
    def clipping_scale_fn(self, grad_norm, idx, clipping_bound):
        if grad_norm > self.strict_max_grad_norm:
            return min(1, clipping_bound / (grad_norm + 1e-6))
        else:
            return clipping_bound / self.strict_max_grad_norm

    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            device,
            logdir,
            delta=1e-5,
            strict_max_grad_norm=100,
            bits_noise_multiplier=10,
            lr_Z=0.01,
            threshold=1.0,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            train_loader,
            device,
            logdir,
            **kwargs
        )
        self.privacy_engine = privacy_engine
        self.delta = delta
        self.strict_max_grad_norm = strict_max_grad_norm  # Z
        self.bits_noise_multiplier = bits_noise_multiplier
        self.lr_Z = lr_Z
        self.sample_rate = 1 / self.num_batch
        self.privacy_step_history = []
        self.threshold = threshold

    def _update_privacy_accountant(self):
        """
        The Opacus RDP accountant minimizes computation when many SGM steps are taken in a row with the same parameters.
        We alternate between privatizing counts, and gradients with different parameters.
        Accounting is sped up by tracking steps in groups rather than alternating.
        The order of accounting does not affect the privacy guarantee.
        """
        for step in self.privacy_step_history:
            self.privacy_engine.accountant.step(noise_multiplier=step[0], sample_rate=step[1])
        self.privacy_step_history = []

    def _update_Z(self, per_sample_grads, Z):
        # get the l2 norm of gradients of all parameters for each sample, in shape of (batch_size, )
        l2_norm_grad_per_sample = torch.norm(per_sample_grads, p=2, dim=1)
        batch_size = len(l2_norm_grad_per_sample)

        dt = 0  # sample count in a batch exceeding Z * threshold
        for i in range(batch_size):  # looping over batch
            if l2_norm_grad_per_sample[i].item() > self.threshold * Z:
                dt += 1

        dt = dt * 1.0 / batch_size  # percentage of samples in a batch that's bigger than the threshold * Z
        noisy_dt = dt + torch.normal(0, self.bits_noise_multiplier, (1,)).item() * 1.0 / batch_size

        factor = math.exp(- self.lr_Z + noisy_dt)

        next_Z = Z * factor

        self.privacy_step_history.append([self.bits_noise_multiplier, self.sample_rate])
        return next_Z

class DpsgdF_GlobalAdaptiveTrainer(BaseTrainer):

    # given norm of gradient, computes S such that clipped gradient = S * gradient
    def clipping_scale_fn(self, grad_norm, idx, clipping_bounds):
        if grad_norm > self.strict_max_grad_norm:
            return min(1, clipping_bounds[idx] / (grad_norm + 1e-6))
        else:
            return clipping_bounds[idx] / self.strict_max_grad_norm

    def __init__(
            self,
            model,
            optimizer,
            privacy_engine,
            train_loader,
            device,
            logdir,
            delta=1e-5,
            base_max_grad_norm=1,  # C0
            strict_max_grad_norm=100,
            bits_noise_multiplier=10,
            counts_noise_multiplier=10,  # noise multiplier applied on mk and ok
            lr_Z=0.01,
            threshold=1.0,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            train_loader,
            device,
            logdir,
            **kwargs
        )
        self.privacy_engine = privacy_engine
        self.delta = delta
        self.base_max_grad_norm = base_max_grad_norm  # Z
        self.strict_max_grad_norm = strict_max_grad_norm  # Z
        self.bits_noise_multiplier = bits_noise_multiplier
        self.counts_noise_multiplier = counts_noise_multiplier  # noise scale applied on mk and ok
        self.lr_Z = lr_Z
        self.sample_rate = 1 / self.num_batch
        self.privacy_step_history = []
        self.threshold = threshold

    def _update_privacy_accountant(self):
        """
        The Opacus RDP accountant minimizes computation when many SGM steps are taken in a row with the same parameters.
        We alternate between privatizing counts, and gradients with different parameters.
        Accounting is sped up by tracking steps in groups rather than alternating.
        The order of accounting does not affect the privacy guarantee.
        """
        for step in self.privacy_step_history:
            self.privacy_engine.accountant.step(noise_multiplier=step[0], sample_rate=step[1])
        self.privacy_step_history = []

    def _update_Z(self, per_sample_grads, Z):
        # get the l2 norm of gradients of all parameters for each sample, in shape of (batch_size, )
        l2_norm_grad_per_sample = torch.norm(per_sample_grads, p=2, dim=1)
        batch_size = len(l2_norm_grad_per_sample)

        dt = 0  # sample count in a batch exceeding Z * threshold
        for i in range(batch_size):  # looping over batch
            if l2_norm_grad_per_sample[i].item() > self.threshold * Z:
                dt += 1

        dt = dt * 1.0 / batch_size  # percentage of samples in a batch that's bigger than the threshold * Z
        noisy_dt = dt + torch.normal(0, self.bits_noise_multiplier, (1,)).item() * 1.0 / batch_size

        factor = math.exp(- self.lr_Z + noisy_dt)

        next_Z = Z * factor

        self.privacy_step_history.append([self.bits_noise_multiplier, self.sample_rate])
        return next_Z

    def compute_clipping_bound_per_sample(self, per_sample_grads, group):
        """compute clipping bound for each group """
        # Calculate group-specific scaling factor
        group_norms = {}  # Store group-wise L2 norms
        for group_idx in range(self.num_groups):
            group_grads = per_sample_grads[group == group_idx]  # Get gradients for each group
            if group_grads.size(0) > 0: 
                summed_group_grads = group_grads.sum(dim=0) 
                avg_group_grads = summed_group_grads / group_grads.size(0)  
                group_norms[group_idx] = torch.norm(avg_group_grads, p=2)  # Average L2 norm
            else:
                group_norms[group_idx] = 1e-8

        # Normalize clipping bounds based on the group's gradient norms
        Ck = {}
        avg_grads = torch.mean(group_grads, dim=0)
        avg_norm = torch.norm(avg_grads, p=2)

        for group_idx in range(self.num_groups):
            # Scale clipping bound for each group according to its gradient norm
            scale = min(1, avg_norm / group_norms[group_idx])
            Ck[group_idx] = self.base_max_grad_norm * scale
            #Ck[group_idx] = self.base_max_grad_norm * (1 + (avg_group_norm / group_norms[group_idx]) ** alpha)

        # Calculate final clipping bounds for each sample based on group
        per_sample_clipping_bound = []
        for i in range(len(group)):  # Looping over batch
            group_idx = group[i].item()
            # Adjust the clipping bound for each sample
            per_sample_clipping_bound.append(Ck[group_idx])

        # Return per-sample clipping bounds
        return torch.Tensor(per_sample_clipping_bound).to(device=self.device)

