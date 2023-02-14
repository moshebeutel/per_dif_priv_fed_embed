from opacus import PrivacyEngine
from differential_privacy.privacy_engines.custom_privacy_engine import CustomPrivacyEngine
from differential_privacy.privacy_optimizers.random_grad_dp_optimizer import RandomGradDPOptimizer


def get_private_model(model, optimizer, loader, batch_size):
    privacy_engine = PrivacyEngine()
    # privacy_engine = CustomPrivacyEngine(optim_class=RandomGradDPOptimizer, expected_batch_size=batch_size)
    priv_model, priv_optimizer, priv_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
    )
    return priv_model, priv_optimizer, priv_loader