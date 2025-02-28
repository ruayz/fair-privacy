# Copyright 2023 Layer 6 AI
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
# Full license in LICENSE file.

from typing import Optional, Callable

import torch
import opacus
from opacus.optimizers.optimizer import DPOptimizer, _generate_noise, _check_processed_flag, _mark_as_processed
from torch.optim import Optimizer

class DPSGDFGLOBAL_Adaptive_Optimizer(DPOptimizer):
    """
    Customized optimizer for DPSGD-F, inherited from DPOptimizer and overwriting the following

    - clip_and_accumulate(self, per_sample_clip_bound) now takes an extra tensor list parameter indicating the clipping bound per sample
    - add_noise(self, max_grad_clip:float) takes an extra paramter ``max_grad_clip``,
        which is the maximum clipping factor among all the groups, i.e. max(per_sample_clip_bound)
    - pre_step() and step() are overwritten by taking this extra parameter
    """

    def __init__(
            self,
            optimizer: Optimizer,
            *,
            noise_multiplier: float,
            expected_batch_size: Optional[int],
            loss_reduction: str = "mean",
            generator=None,
            secure_mode: bool = False,
    ):
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=0,  # not applicable for DPSGDF_Optimizer
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )

    
    def clip_and_accumulate(self, per_sample_clip_bound, strict_max_grad_norm):
        """
        Clips gradient according to per sample clipping bounds and accumulates gradient for a given batch
        Args:
        per_sample_clip_bound: a tensor list of clip bound per sample
        """

        per_param_norms = [
            g.view(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        per_sample_clip_factor = (per_sample_clip_bound / (per_sample_norms + 1e-6)).clamp(
            max=1.0
        )

        # C = max_grad_norm
        # Z = strict_max_grad_norm
        # condition is equivalent to norm[i] <= Z
        # when condition holds, scale gradient by C/Z
        # otherwise, clip to 0
        per_sample_global_clip_factor = torch.where(per_sample_clip_factor >= per_sample_clip_bound / strict_max_grad_norm,
                                                    # scale by C/Z
                                                    torch.ones_like(
                                                        per_sample_clip_factor) * per_sample_clip_bound / strict_max_grad_norm,
                                                    #torch.zeros_like(per_sample_clip_factor))  # clip to 0
                                                    per_sample_clip_factor) # clip to Ck
        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)
            
             # refer to lines 197-199 in
            # https://github.com/pytorch/opacus/blob/ee6867e6364781e67529664261243c16c3046b0b/opacus/per_sample_gradient_clip.py
            # as well as https://github.com/woodyx218/opacus_global_clipping README
            grad = torch.einsum("i,i...", per_sample_global_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)


    def add_noise(self, max_grad_clip: float):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        Args:
            max_grad_clip: C = max(C_k), for all group k
        """

        for p in self.params:
            _check_processed_flag(p.summed_grad)

            noise = _generate_noise(
                std=self.noise_multiplier * max_grad_clip,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            p.grad = (p.summed_grad + noise).view_as(p.grad)
            _mark_as_processed(p.summed_grad)


    def pre_step(
            self, per_sample_clip_bound, strict_max_grad_norm, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``
        Args:
            per_sample_clip_bound: Defines the clipping bound for each sample.
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        self.clip_and_accumulate(per_sample_clip_bound, strict_max_grad_norm)
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        self.add_noise(torch.max(per_sample_clip_bound).item())
        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True

    def step(self, per_sample_clip_bound, strict_max_grad_norm, closure: Optional[Callable[[], float]] = None) -> Optional[
        float]:
        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step(per_sample_clip_bound, strict_max_grad_norm):
            return self.original_optimizer.step()
        else:
            return None
