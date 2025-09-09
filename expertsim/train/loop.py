"""
expertsim.train.loop
====================
Main training orchestration: epochs, batches, and high-level flow.
"""
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
import logging
from typing import Dict, List
import matplotlib.pyplot as plt

from expertsim.models import build_model
from expertsim.models.moe import MoEWrapper
from expertsim.train.utils import plot_expert_specialization
from expertsim.utils.utils_eval import plot_proton_photonsum_histograms_shared
from expertsim.train.utils import get_predictions_from_generator_results, generate_and_save_images_from_generations
import wandb

from .training_setup import setup_optimizers
from .hooks import WandBLogger, CheckpointSaver

logger = logging.getLogger(__name__)


def train(cfg, train_loader: DataLoader, test_loader: DataLoader = None) -> List[Dict]:
    """
    Main training function with proper error handling and logging.

    Args:
        cfg: Configuration object
        train_loader: Training data loader
        test_loader: Optional test data loader

    Returns:
        List of epoch metrics dictionaries
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    try:
        # Setup MoE system
        moe = setup_moe_system(cfg, device)
        ema_helper = EMAHelper(moe, decay=0.99)

        # Setup optimizers
        gen_optims, disc_optims, aux_reg_optim, router_optim = setup_optimizers(moe, cfg)

        # Setup callbacks
        callbacks = setup_callbacks(cfg, moe)

        # Training loop
        history = []
        start_epoch = 0 if cfg.train.epoch_to_load is None else cfg.train.epoch_to_load

        logger.info(f"Starting training from epoch {start_epoch} to {cfg.train.epochs}")

        for epoch in range(start_epoch, cfg.train.epochs):
            epoch_start_time = time.time()

            # Training phase
            epoch_metrics = train_epoch(
                moe, train_loader, gen_optims, disc_optims,
                aux_reg_optim, router_optim, cfg, device, epoch, ema_helper
            )
            epoch_time_train = time.time() - epoch_start_time

            # Evaluation phase
            if test_loader is not None:
                eval_metrics = evaluate_epoch(moe, test_loader, epoch, cfg, device)
                epoch_metrics.update(eval_metrics)

            # Add timing information
            epoch_time = time.time() - epoch_start_time
            epoch_metrics['epoch_time'] = epoch_time
            epoch_metrics['epoch'] = epoch

            # Run callbacks
            for callback in callbacks:
                try:
                    callback.on_epoch_end(epoch, epoch_metrics, moe, gen_optims, disc_optims,
                                          aux_reg_optim, router_optim)
                except Exception as e:
                    logger.warning(f"Callback {callback.__class__.__name__} failed: {e}")

            history.append(epoch_metrics)
            logger.info(f"Epoch {epoch} completed in {epoch_time_train:.2f}, {epoch_time:.2f}s. WS {epoch_metrics['ws_mean']:.2f}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    logger.info("Training completed successfully")
    return history


def train_epoch(
    moe: MoEWrapper,
    train_loader: DataLoader,
    gen_optims: List[torch.optim.Optimizer],
    disc_optims: List[torch.optim.Optimizer],
    aux_reg_optims: torch.optim.Optimizer,
    router_optim: torch.optim.Optimizer,
    cfg,
    device: torch.device,
    epoch: int,
    ema_helper
) -> Dict:
    """Execute one training epoch."""

    moe.train()

    # Metrics accumulators
    metrics = {
        'gen_loss': [], 'disc_loss': [], 'router_loss': [],
        'div_loss': [], 'intensity_loss': [], 'aux_reg_loss': [],
        'expert_entropy_loss': [], 'expert_distribution_loss': [],
        'differentiation_loss': [], 'adaptive_load_balancing_loss': [], 'gan_loss': [],
        **{f"std_intensities_experts_{i}": [] for i in range(cfg.model.n_experts)},
        **{f"mean_intensities_experts_{i}": [] for i in range(cfg.model.n_experts)},
        **{f"gen_loss_{i}": [] for i in range(cfg.model.n_experts)},
        **{f"disc_loss_{i}": [] for i in range(cfg.model.n_experts)},
        **{f"div_loss_experts_{i}": [] for i in range(cfg.model.n_experts)},
        **{f"intensity_loss_experts_{i}": [] for i in range(cfg.model.n_experts)},
        **{f"aux_reg_loss_experts_{i}": [] for i in range(cfg.model.n_experts)},
        **{f"n_choosen_experts_mean_epoch_{i}": [] for i in range(cfg.model.n_experts)},
    }

    for batch_idx, batch in enumerate(train_loader):
        batch_metrics = train_step(
            batch, moe, gen_optims, disc_optims, aux_reg_optims,
            router_optim, cfg, device, epoch, ema_helper
        )

        # Accumulate metrics - convert tensors to CPU/Python scalars only once here
        for key, value in batch_metrics.items():
            # Convert tensor to Python scalar if needed
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:  # Scalar tensor
                    value = value.detach().cpu().item()
                else:  # Multi-element tensor (shouldn't happen in logging, but safe)
                    value = value.detach().cpu().numpy()

            # Accumulate in metrics dict
            if key in metrics:
                metrics[key].append(value)
            else:
                metrics[key] = [value]  # Initialize list if first occurrence

    averaged_metrics = {}
    for i in range(moe.n_experts):
        averaged_metrics[f"G_steps_{i}"] = moe.g_steps[i]
        averaged_metrics[f"D_steps_{i}"] = moe.d_steps[i]

    # Average metrics across batches
    for key, values in metrics.items():
        if values:  # Only if we have values
            averaged_metrics[key] = np.mean(values)
        else:
            averaged_metrics[key] = 0.0
    return averaged_metrics


def train_step(
    batch, moe: MoEWrapper, gen_optims, disc_optims, aux_reg_optim, router_optim,
    cfg, device: torch.device, epoch: int, ema_helper
) -> Dict:
    """Execute one training step (batch)."""

    # Unpack batch
    real_images, _, cond, std, intensity, true_positions = batch

    # Move to device
    real_images = real_images.unsqueeze(1).to(device)
    cond = cond.to(device)
    std = std.to(device)
    intensity = intensity.to(device)
    true_positions = true_positions.to(device)

    output_losses = moe.train_step(epoch, cond, real_images, true_positions, std, intensity, aux_reg_optim, gen_optims,
                                   disc_optims, router_optim, ema_helper, device)
    return output_losses


def evaluate_epoch(moe: MoEWrapper, test_loader: DataLoader, epoch: int, cfg, device: torch.device) -> Dict:
    """Evaluate model on test set."""
    moe.eval()

    # Metrics accumulators
    metrics = {
        'ws_mean': [],
        **{f"ws_mean_{i}": [] for i in range(cfg.model.n_experts)},
        'ws_std': [],
        **{f"ws_std_{i}": [] for i in range(cfg.model.n_experts)}
    }
    # Keep all samples for plotting
    all_real_images = []
    all_cond = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            real_images, _, cond, std, intensity, true_positions = batch
            cond = cond.to(device)

            # Save for later plotting
            all_real_images.append(real_images)
            all_cond.append(cond)

            batch_metrics = moe.evaluate(epoch, cond, real_images, true_positions, std, intensity, cfg, device)
            for key, value in batch_metrics.items():
                if key in metrics.keys():
                    metrics[key].append(value)

    # Concatenate all stored batches
    all_real_images = np.concatenate(all_real_images, axis=0)
    all_cond = torch.cat(all_cond, dim=0)


    # generate images here given the indices list
    averaged_metrics = {}
    for key, values in metrics.items():
        if values:
            if isinstance(values[0], list):
                # Handle lists of values (e.g., per-expert losses)
                averaged_metrics[key] = [sum(expert_vals) / len(expert_vals)
                                         for expert_vals in zip(*values)]
            else:
                averaged_metrics[key] = sum(values) / len(values)
        else:
            averaged_metrics[key] = 0.0

    #
    # Plot images
    #
    if cfg.wandb.plot_images:
        generated_imgs_log = generate_images_from_conditioning(
            moe,
            all_cond,
            all_real_images,
            cfg,
            epoch,
            device,
        )

        # generate specialization plots
        expert_specialization_log = generate_specialization_plots(moe,
                                                                  all_cond,
                                                                  all_real_images,
                                                                  cfg,
                                                                  epoch,
                                                                  device)

        averaged_metrics.update(generated_imgs_log)
        averaged_metrics.update(expert_specialization_log)

    return averaged_metrics


@torch.no_grad()
def generate_images_from_conditioning(moe, cond, x_test, cfg, epoch, device):

    log_data = {}
    IDX_GENERATE = [1, 2, 3, 4, 5, 6]  # for each expert select the samples to be generated
    y_test_temp = cond.to(device)

    gates, logits = moe.router(y_test_temp)
    _, predicted_expert = torch.max(gates, 1)
    indices_experts = [np.where(predicted_expert.cpu().numpy() == i)[0] for i in range(moe.n_experts)]

    # Generate Plots
    plot_experts = [None] * moe.n_experts
    noise_cond_experts = [cond[indices_experts[i]] for i in range(moe.n_experts)]

    results_generators = []
    for i in range(moe.n_experts):
        results_generator, pred_no_transform = get_predictions_from_generator_results(batch_size=cfg.train.batch_size,
                                                                   num_samples=len(noise_cond_experts[i]),
                                                                   noise_dim=cfg.model.noise_dim,
                                                                   device=device,
                                                                   y_test=noise_cond_experts[i],
                                                                   generator=moe.generators[i],
                                                                   shape_images=cfg.dataset.input_image_shape)
        results_generators.append(results_generator)

        # available counts
        n_idx = len(IDX_GENERATE)  # n of samples to generate by default
        n_pred = len(pred_no_transform)  # predictions made by the models
        n_xtest = len(x_test[indices_experts[i]])  # or x_test[indices_experts[i]].shape
        k = min(n_idx, n_pred, n_xtest)
        plot_generations = generate_and_save_images_from_generations(
            pred_no_transform[:k],
            epoch,
            x_test[indices_experts[i]][:k],
            f'Expert {i}',
            k,
            shape_images=cfg.dataset.input_image_shape
        )

        plot_experts[i] = plot_generations
        log_data[f"plot_expert_{i}"] = wandb.Image(plot_experts[i]) if not plot_experts[i] is None else None
    # generate plot for photonsum distributions
    photonsum_mean_generated_images_experts = []
    for i in range(moe.n_experts):
        results_generator = results_generators[i]
        photonsum_on_all_generated_images = np.sum(results_generator, axis=(1, 2))
        photonsum_mean_generated_images_experts.append(photonsum_on_all_generated_images)


    photonsum_distr_experts = plot_proton_photonsum_histograms_shared(
        photonsum_mean_generated_images_experts, labels=None, save_path=None)
    log_data[f"photonsum_distr_experts"] = wandb.Image(photonsum_distr_experts)
    plt.close(photonsum_distr_experts)
    return log_data


@torch.no_grad()
def generate_specialization_plots(moe, cond, x_test, cfg, epoch, device):
    log_data = {}
    IDX_GENERATE = [1, 2, 3, 4, 5, 6]  # for each expert select the samples to be generated
    y_test_temp = cond.to(device)

    gates, logits = moe.router(y_test_temp)
    _, predicted_expert = torch.max(gates, 1)
    indices_experts = [np.where(predicted_expert.cpu().numpy() == i)[0] for i in range(moe.n_experts)]

    # Generate Plots
    plot = plot_expert_specialization(cond, indices_experts, epoch, cfg.data_cond_names)

    log_data[f"cond_expert_specialization"] = wandb.Image(plot)
    return log_data


def setup_moe_system(cfg, device: torch.device) -> MoEWrapper:
    """Setup the MoE system with proper model instantiation."""

    # Inject shared parameters into sub-configs
    cfg.model.generator.noise_dim = cfg.model.noise_dim
    cfg.model.generator.cond_dim = cfg.model.cond_dim
    cfg.model.generator.n_experts = cfg.model.n_experts
    cfg.model.discriminator.cond_dim = cfg.model.cond_dim
    cfg.model.discriminator.n_experts = cfg.model.n_experts
    cfg.model.router.cond_dim = cfg.model.cond_dim
    cfg.model.router.n_experts = cfg.model.n_experts

    # Build model components
    generator = build_model(f"{cfg.model.architecture}.generator", cfg.model.generator, device)
    discriminator = build_model(f"{cfg.model.architecture}.discriminator", cfg.model.discriminator, device)
    aux_reg = build_model(f"{cfg.model.architecture}.aux_reg", cfg.model.aux_reg, device)
    router = build_model(f"{cfg.model.router.version}", cfg.model.router, device)

    # # Create MoE wrapper
    moe = MoEWrapper(generator, discriminator, aux_reg, router, cfg.model.n_experts,
                     cfg, image_shape=cfg.dataset.input_image_shape).to(device)

    return moe


def setup_callbacks(cfg, moe) -> List:
    """Setup training callbacks."""
    callbacks = []

    cfg.generator_name = moe.generators[0].name
    cfg.discriminator_name = moe.discriminators[0].name
    cfg.router_name = moe.router.name

    if getattr(cfg, 'wandb', {}).get('log_experiments', False):
        callbacks.append(WandBLogger(cfg))

    if getattr(cfg, 'train', {}).get('save_experiment_data', False):
        callbacks.append(CheckpointSaver(
            dir_path=cfg.train.dir_models,
            monitor='ws_mean',
            ws_threshold=cfg.train.ws_threshold_model_save,
        ))

    return callbacks


import torch

class EMAHelper:
    def __init__(self, moe, decay=0.999):
        self.decay = decay
        self.shadow = {i: {} for i in range(len(moe.generators))}
        self.backup = {}

        # Initialize shadow with generator weights
        for i, gen in enumerate(moe.generators):
            for name, param in gen.named_parameters():
                if param.requires_grad:
                    self.shadow[i][name] = param.data.clone()

    def update(self, moe, updated_indices):
        """Update EMA only for generators that got an optimizer step."""
        for i in updated_indices:
            gen = moe.generators[i]
            for name, param in gen.named_parameters():
                if param.requires_grad:
                    new_average = (self.decay * self.shadow[i][name]
                                   + (1.0 - self.decay) * param.data)
                    self.shadow[i][name] = new_average.clone()

    def apply_shadow(self, moe):
        """Swap all gens to EMA weights (for eval)."""
        self.backup = {}
        for i, gen in enumerate(moe.generators):
            self.backup[i] = {}
            for name, param in gen.named_parameters():
                if param.requires_grad:
                    self.backup[i][name] = param.data.clone()
                    param.data = self.shadow[i][name].clone()

    def restore(self, moe):
        """Restore original gen weights after eval."""
        for i, gen in enumerate(moe.generators):
            for name, param in gen.named_parameters():
                if param.requires_grad and i in self.backup:
                    param.data = self.backup[i][name].clone()
        self.backup = {}
