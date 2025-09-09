"""
expertsim.train.hooks
====================
Enhanced callback classes for training events.
"""

import wandb
import torch
import os
import logging
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
from expertsim.train.training_utils import save_models_and_architectures
logger = logging.getLogger(__name__)

ENTITY = "bedkowski-patrick"
PROJECT = "Generative-DNN-for-CERN-Fast-Simulations"


class Callback:
    """Base callback class."""
    def on_epoch_start(self, epoch: int, metrics: Dict, *args, **kwargs): pass
    def on_epoch_end(self, epoch: int, metrics: Dict, *args, **kwargs): pass
    def on_train_start(self, cfg): pass
    def on_train_end(self, history): pass


class WandBLogger(Callback):
    """Enhanced WandB logger with comprehensive metrics tracking."""
    date_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f")[:-3]

    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = getattr(cfg.wandb, 'log_experiments', False)
        self.run_name = cfg.wandb.run_name
        if self.enabled:
            wandb.init(
                entity=ENTITY,
                project=PROJECT,
                name=self.run_name,
                config=self.create_config(cfg),
                tags=[cfg.model.architecture],
                settings=wandb.Settings(
                    _disable_stats=True,
                    _disable_meta=True,
                    _disable_service=True,
                    code_dir=None,  # don’t snapshot code dir
                    start_method="thread",  # avoid spawning a separate process
                    _noop=False,  # ensure it actually logs online
                ),
            )
            logger.info("WandB logging initialized")
        if cfg.wandb.api_key:
            wandb.login(cfg.wandb.api_key, relogin=True)
            logger.info("WandB login successful!")

    def create_config(self, cfg: Any) -> Dict[str, Any]:
        return {
        "disc_type_features": 'conv',
        "dataset": f"{cfg.dataset.zdc_type}_data",
        "n_experts": cfg.model.n_experts,
        "epochs": cfg.train.epochs,
        "alpha": cfg.model.router.alpha,
        "min_weight": cfg.model.router.min_weight,
        "Date": self.cfg.config.date,
        "Proton_min": cfg.photon_sum_min,
        "Proton_max": cfg.photon_sum_max,
        "generator_architecture": cfg.generator_name,
        "discriminator_architecture": cfg.discriminator_name,
        "router_arch": cfg.router_name,
        "stop_router_training_epoch": cfg.model.router.stop_router_training_epoch,
        "diversity_strength": cfg.model.generator.di_strength,
        "intensity_strength": cfg.model.generator.in_strength,
        "auxiliary_strength": cfg.model.aux_reg.strength,
        "Generator_strength": cfg.model.router.gan_strength,
        "Utilization_strength": cfg.model.router.util_strength,
        "differentiation_strength": cfg.model.router.diff_strength,
        "alb_strength": cfg.model.router.alb_strength,
        "expert_distribution_strength": cfg.model.router.ed_strength,
        "Learning rate_generator": cfg.model.generator.lr_g,
        "Learning rate_discriminator": cfg.model.discriminator.lr_d,
        "Learning rate_router": cfg.model.router.lr_r,
        "Learning rate_aux_reg": cfg.model.aux_reg.lr_a,
        "Experiment_dir_name": cfg.train.save_experiments_dir,
        "Batch_size": cfg.train.batch_size,
        "Batch_size_aggregate": cfg.train.batch_size_aggregate,
        "Noise_dim": cfg.model.noise_dim,
        "intensity_loss_type": "mae"
    }

    def on_epoch_end(self, epoch: int, metrics: Dict, *args, **kwargs):
        if not self.enabled:
            return
        try:
            wandb.log(metrics)

        except Exception as e:
            logger.warning(f"WandB logging failed: {e}")


class CheckpointSaver(Callback):
    """Enhanced checkpoint saver with best model + EMA tracking."""

    def __init__(self, dir_path: str, ema_helper=None,
                 monitor: str = "ws_mean", ws_threshold: float = 2.0):
        self.dir_path = Path(dir_path)
        self.monitor = monitor
        self.threshold = ws_threshold
        self.ema_helper = ema_helper

        self.dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoints will be saved to {self.dir_path}")

    def on_epoch_end(self, epoch: int, metrics: Dict, moe, gen_optims,
                     disc_optims, aux_reg_optim, router_optim):

        current = metrics.get(self.monitor, float("inf"))

        should_save = current < self.threshold

        if should_save:
            # Save normal training weights
            save_models_and_architectures(
                self.dir_path, moe.n_experts, moe.aux_regs, aux_reg_optim,
                moe.generators, gen_optims, moe.discriminators,
                disc_optims, moe.router, router_optim, epoch,
                multiple_aux_regs=True
            )
            logger.info(f"New best {self.monitor}: {current:.4f} at epoch {epoch}")

            # Save EMA weights if available
            if self.ema_helper is not None:
                ema_path = self.dir_path / f"ema_generators_epoch_{epoch}.pt"
                self.save_ema_weights(ema_path, epoch=epoch)
                logger.info(f"EMA weights saved to {ema_path}")

    def save_ema_weights(self, save_path, epoch=None):
        """Save EMA weights for all experts' generators."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "ema_shadow": {
                i: {k: v.cpu() for k, v in state.items()}
                for i, state in self.ema_helper.shadow.items()
            },
            "decay": self.ema_helper.decay,
            "epoch": epoch
        }

        torch.save(checkpoint, save_path)

    def load_ema_weights(self, load_path, moe):
        """Load EMA weights into the model generators."""
        checkpoint = torch.load(load_path, map_location="cpu")
        ema_shadow = checkpoint["ema_shadow"]

        # Apply shadow weights to each generator
        for i, gen in enumerate(moe.generators):
            for name, param in gen.named_parameters():
                if param.requires_grad:
                    param.data = ema_shadow[i][name].clone()

        logger.info(f"[EMA] Loaded EMA weights from {load_path} (epoch {checkpoint.get('epoch')})")


class MetricsTracker(Callback):
    """Track and analyze training metrics."""

    def __init__(self):
        self.history = []

    def on_epoch_end(self, epoch: int, metrics: Dict):
        self.history.append({**metrics, 'epoch': epoch})

    def get_best_metric(self, metric_name: str, mode: str = 'min') -> Dict:
        """Get best value for a specific metric."""
        if not self.history:
            return {}

        values = [h.get(metric_name, float('inf')) for h in self.history]
        best_idx = min(range(len(values)), key=lambda i: values[i]) if mode == 'min' else max(range(len(values)), key=lambda i: values[i])

        return self.history[best_idx]
