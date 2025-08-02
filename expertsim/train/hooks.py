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

logger = logging.getLogger(__name__)


class Callback:
    """Base callback class."""
    def on_epoch_start(self, epoch: int, metrics: Dict): pass
    def on_epoch_end(self, epoch: int, metrics: Dict): pass
    def on_train_start(self, cfg): pass
    def on_train_end(self, history): pass


class WandBLogger(Callback):
    """Enhanced WandB logger with comprehensive metrics tracking."""
    date_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f")[:-3]
    wandb.login(key="d53387a3b34fda2a3caaf861b5fad88cb4ec99ef")

    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = getattr(cfg.wandb, 'log_experiments', False)
        self.run_name = f"{getattr(cfg.config, 'run_name', 'experiment')}_{self.date_str}"
        if self.enabled:
            wandb.init(
                entity="bedkowski-patrick",
                project="Generative-DNN-for-CERN-Fast-Simulations",
                name=self.run_name,
                config=dict(cfg),
                tags=[cfg.model.architecture]
            )
            logger.info("WandB logging initialized")

    def on_epoch_end(self, epoch: int, metrics: Dict):
        if not self.enabled:
            return

        print("metrics for WANDB")
        print(metrics)

        try:
            # Prepare logging data
            log_data = {'epoch': epoch}

            # Log scalar metrics
            scalar_metrics = ['router_loss', 'gan_loss', 'entropy_loss',
                              'distribution_loss', 'differentiation_loss', 'alb_loss',
                              'ws_mean', 'epoch_time', "ws_mean", "ws_std"]

            for metric in scalar_metrics:
                if metric in metrics:
                    log_data[metric] = metrics[metric]

            # Log per-expert metrics
            expert_metrics = ['gen_losses', 'disc_losses', 'div_losses',
                              'intensity_losses', 'aux_losses']

            for metric in expert_metrics:
                if metric in metrics and isinstance(metrics[metric], list):
                    for i, value in enumerate(metrics[metric]):
                        log_data[f'{metric}_expert_{i}'] = value
                    log_data[f'{metric}_mean'] = sum(metrics[metric]) / len(metrics[metric])

            wandb.log(log_data)

        except Exception as e:
            logger.warning(f"WandB logging failed: {e}")


class CheckpointSaver(Callback):
    """Enhanced checkpoint saver with best model tracking."""

    def __init__(self, dir_path: str, monitor: str = "ws_mean", mode: str = "min", threshold: float = 2.0):
        self.dir_path = Path(dir_path)
        self.monitor = monitor
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.threshold = threshold

        # Create directory
        self.dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoints will be saved to {self.dir_path}")

    def on_epoch_end(self, epoch: int, metrics: Dict):
        current = metrics.get(self.monitor, float("inf"))

        should_save = False
        if self.mode == "min":
            should_save = current < self.best and current < self.threshold
        else:
            should_save = current > self.best

        if should_save:
            self.best = current
            self.save_checkpoint(epoch, metrics, is_best=True)
            logger.info(f"New best {self.monitor}: {current:.4f} at epoch {epoch}")

        # Save regular checkpoint every N epochs
        if epoch % 10 == 0:
            self.save_checkpoint(epoch, metrics, is_best=False)

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save checkpoint with metadata."""
        checkpoint_data = {
            'epoch': epoch,
            'metrics': metrics,
            'best_metric': self.best
        }

        # Save checkpoint
        filename = 'best.ckpt' if is_best else f'epoch_{epoch}.ckpt'
        filepath = self.dir_path / filename

        try:
            torch.save(checkpoint_data, filepath)
            logger.info(f"Saved checkpoint: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")


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
