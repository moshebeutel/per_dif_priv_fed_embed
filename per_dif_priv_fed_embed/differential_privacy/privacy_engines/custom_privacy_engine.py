import torch
from opacus.privacy_engine import PrivacyEngine
from opacus.optimizers.optimizer import DPOptimizer
from typing import Union, List


class CustomPrivacyEngine(PrivacyEngine):
    def __init__(self, *, accountant: str = "prv", secure_mode: bool = False, optim_class: DPOptimizer = DPOptimizer,
                 expected_batch_size=128):
        super(CustomPrivacyEngine, self).__init__(accountant=accountant, secure_mode=secure_mode)
        self._optim_class = optim_class
        self._expected_batch_size = expected_batch_size

    def _prepare_optimizer(
            self,
            optimizer: torch.optim.Optimizer,
            *,
            noise_multiplier: float,
            max_grad_norm: Union[float, List[float]],
            expected_batch_size: int,
            loss_reduction: str = "mean",
            distributed: bool = False,
            clipping: str = "flat",
            noise_generator=None,
            grad_sample_mode="hooks",
    ) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        # optim_class = self._optim_class(
        #     clipping=clipping,
        #     distributed=distributed,
        #     grad_sample_mode=grad_sample_mode,
        # )

        return self._optim_class(
            optimizer=optimizer,
            expected_batch_size=self._expected_batch_size
            # noise_multiplier=noise_multiplier,
            # max_grad_norm=max_grad_norm,
            # expected_batch_size=expected_batch_size,
            # loss_reduction=loss_reduction,
            # generator=generator,
            # secure_mode=self.secure_mode,
        )
