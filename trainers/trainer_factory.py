# Copyright (c) 2020 Data Privacy and Trustworthy Machine Learning Research Lab
# Licensed under the MIT License. See LICENSE file for details.

from .trainer import RegularTrainer, DpsgdTrainer, DpsgdSTrainer, DpsgdFTrainer, DpsgdGlobalTrainer, DpsgdGlobalAdaptiveTrainer
from .default_trainer import get_optimizer
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import sys

from privacy_engines.dpsgd_s_engine import DPSGDS_PrivacyEngine
from privacy_engines.dpsgd_f_engine import DPSGDF_PrivacyEngine
from privacy_engines.dpsgd_global_engine import DPSGDGlobalPrivacyEngine
from privacy_engines.dpsgd_global_adaptive_engine import DPSGDGlobalAdaptivePrivacyEngine
# from privacy_engines.dpsgd_f_global_engine import DPSGDFGlobal_PrivacyEngine
# from privacy_engines.dpsgd_f_global_adaptive_engine import DPSGDFGlobal_Adaptive_PrivacyEngine


def create_trainer(
        train_loader,
        model,
        configs,
        log_dir
):

    kwargs = {
        'method': configs['method'],
        'max_epochs': configs['epochs'],
        'lr': configs['learning_rate'],
        #'seed': configs['seed'],
        'num_groups': configs['num_groups'],
        'selected_groups': range(configs['num_groups']),
        'evaluate_angles': configs['evaluate_angles'],
        'evaluate_hessian': configs['evaluate_hessian'],
        'angle_comp_step': configs['angle_comp_step'],
        'num_hutchinson_estimates': configs['num_hutchinson_estimates'],
        'sampled_expected_loss': configs['sampled_expected_loss']
    }

    device = configs.get("device", "cpu")
    optimizer = get_optimizer(model, configs)
    logdir = log_dir
    epsilon = configs.get("epsilon", 0)
    model = model.to(device)  # Make sure the model is on the correct device
    if epsilon >= 0:
        errors = ModuleValidator.validate(model, strict=False)
        if len(errors) > 0 :
            model = ModuleValidator.fix(model)
            optimizer = get_optimizer(model, configs)

    if configs["method"] == "regular":
        # doing regular training
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=0,
            max_grad_norm=sys.float_info.max,
            poisson_sampling=False
        )
        trainer = RegularTrainer(
            model,
            optimizer,
            train_loader,
            device,
            logdir,
            **kwargs
        )
    elif configs["method"] == "dpsgd":
        privacy_engine = PrivacyEngine(accountant="rdp")
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=configs['epsilon'],
            target_delta=float(configs["delta"]),
            epochs=configs["epochs"],
            max_grad_norm=configs['clip_norm']  
        )
        trainer = DpsgdTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            device,
            logdir,
            **kwargs
        )
    elif configs["method"] == "dpsgds":
        privacy_engine = DPSGDS_PrivacyEngine(accountant="rdp")
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=configs['epsilon'],
            target_delta=float(configs["delta"]),
            epochs=configs["epochs"],
            max_grad_norm=0  # this parameter is not applicable for DPSGD-S
        )
        scale = configs.get("scale", 2)
        trainer = DpsgdSTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            device,
            logdir,
            base_max_grad_norm=configs["base_max_grad_norm"],  # C0
            counts_noise_multiplier=configs["counts_noise_multiplier"],  # noise scale 
            scale=scale,
            **kwargs
        )
    elif configs["method"] == "dpsgdf":
        privacy_engine = DPSGDF_PrivacyEngine(accountant="rdp")
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=configs['epsilon'],
            target_delta=float(configs["delta"]),
            epochs=configs["epochs"],
            max_grad_norm=0  # this parameter is not applicable for DPSGD-S
        )
        trainer = DpsgdFTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            device,
            logdir,
            base_max_grad_norm=configs["base_max_grad_norm"],  # C0
            counts_noise_multiplier=configs["counts_noise_multiplier"],  # noise scale 
            **kwargs
        )
    elif configs["method"] == "dpsgdg":
        privacy_engine = DPSGDGlobalPrivacyEngine(accountant="rdp")
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=configs['epsilon'],
            target_delta=float(configs["delta"]),
            epochs=configs["epochs"],
            max_grad_norm=configs['clip_norm'],
        )
        trainer = DpsgdGlobalTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            device,
            logdir,
            strict_max_grad_norm=configs["strict_max_grad_norm"],
            **kwargs
        )
    elif configs["method"] == "dpsgdga":
        privacy_engine = DPSGDGlobalAdaptivePrivacyEngine(accountant="rdp")
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=configs['epsilon'],
            target_delta=float(configs["delta"]),
            epochs=configs["epochs"],
            max_grad_norm=configs['clip_norm'],
        )
        trainer = DpsgdGlobalAdaptiveTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            device,
            logdir,
            strict_max_grad_norm=configs["strict_max_grad_norm"],
            bits_noise_multiplier=configs["bits_noise_multiplier"],
            lr_Z=configs["lr_Z"],
            threshold=configs["threshold"],
            **kwargs
        )
    else:
        raise ValueError("Training method not implemented")

    return trainer
