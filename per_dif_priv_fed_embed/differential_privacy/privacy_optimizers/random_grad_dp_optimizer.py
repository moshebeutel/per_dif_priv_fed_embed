import torch
from opacus.optimizers.optimizer import DPOptimizer, _check_processed_flag, _mark_as_processed
from torch.optim import Optimizer
from typing import Optional, Callable
from functorch import vmap
from math import ceil, floor


class RandomGradDPOptimizer(DPOptimizer):
    SENSITIVE_RATIO = 0.0
    NUM_SAMPLES = 0.0

    def __init__(
            self,
            optimizer: Optimizer,
            expected_batch_size: int = 64):
        self._batch_size = expected_batch_size
        super(RandomGradDPOptimizer, self).__init__(optimizer,
                                                    noise_multiplier=0,
                                                    max_grad_norm=float("inf"),
                                                    loss_reduction='mean',
                                                    expected_batch_size=expected_batch_size)

    def scale_grad(self):
        """
        Applies given ``loss_reduction`` to ``p.grad``.

        Does nothing if ``loss_reduction="sum"``. Divides gradients by
        ``self.expected_batch_size`` if ``loss_reduction="mean"``
        """
        if self.loss_reduction == "mean":
            for p in self.params:
                grad_sample = self._get_flat_grad_sample(p)
                grad_sample_batch = grad_sample.shape[0]
                # grad_sample_batch = min(self._batch_size, grad_sample_batch)
                grad_sample_batch = grad_sample_batch if grad_sample_batch % 2 == 1 else grad_sample_batch - 1
                # print(p.grad_sample.shape[0], grad_sample_batch)
                grad_sample = grad_sample[-grad_sample_batch:]  # self._get_flat_grad_sample(p)
                # sign_grad = torch.sign(grad_sample)
                # p.grad = apply_along_axis(func1d=weighted_sum, axis=0, arr=grad_sample)
                with torch.no_grad():
                    grad_sample_mean = grad_sample.mean(dim=0, keepdims=True)
                    grad_sample_std = grad_sample.std(dim=0, keepdims=True)
                    #
                    #     # The following does not work because it requires grad_sample_std.flatten().diag()
                    #     # which is a huge matrix
                    #     # with torch.no_grad():
                    #     #     grads_gaussian = MultivariateNormal(loc=grad_sample_mean.flatten(),
                    #     #                                         scale_tril=grad_sample_std.flatten().diag())
                    #     #
                    #     #
                    #     #     random_grad = grads_gaussian.sample().reshape(grad_sample_mean.shape)
                    #
                    mean_vector = grad_sample_mean.flatten()
                    std_vector = grad_sample_std.flatten()
                    random_grad = torch.randn(mean_vector.shape, device=mean_vector.device)
                    random_grad *= std_vector
                    random_grad += mean_vector
                    random_grad = random_grad.reshape(p.grad.shape)

                # # With probability p, flip the sign of grad
                # # ******************************
                # prob = 0.01
                # # create tensor containing the probabilities NOT to flip at each entry
                # flip_probs = torch.ones_like(sign_grad) * (1 - prob)
                # # Bernoulli puts 1 with prob 1-p and the linear transform (0,1) => (-1,1)
                # flip_factors = torch.bernoulli(flip_probs) * 2 - 1
                # # Flipping
                # sign_grad *= flip_factors

                # sensitive_mask = apply_along_axis(func1d=is_sensitive, arr=sign_grad, axis=0)

                # not_sensitive_mask = torch.ones_like(sensitive_mask) - sensitive_mask

                # sensitives = float(torch.sum(sensitive_mask))
                # non_sensitives = float(torch.sum(not_sensitive_mask))

                # num_samples = sensitives + non_sensitives
                # SignDPOptimizer.NUM_SAMPLES += num_samples
                # SignDPOptimizer.SENSITIVE_RATIO += (sensitives/num_samples)

                # p.grad = torch.mul(sensitive_mask, torch.sign(torch.rand_like(sensitive_mask) - 0.5)) + \
                #          torch.mul(not_sensitive_mask, torch.median(sign_grad, dim=0).values)

                # p.grad = torch.sum(sign_grad, dim=0)
                # p.grad = torch.median(sign_grad, dim=0).values
                # p.grad = torch.sum(grad_sample, dim=0)
                # print(p.grad.shape, random_grad.shape)
                p.grad = random_grad
                # p.grad /= self.expected_batch_size * self.accumulated_iterations

    def pre_step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``

        Args:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        self.clip_and_accumulate()
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        # self.add_noise()
        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True
