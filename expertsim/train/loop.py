"""
expertsim.train.loop
====================
Main training orchestration: epochs, batches, and high-level flow.
"""

import time
import torch
from torch.utils.data import DataLoader
import logging
from typing import Dict, List

from expertsim.models.moe import MoEWrapper, MoEWrapperUnified
from .training_setup import setup_optimizers, setup_optimizers_unified
from .training_utils import (
    generator_train_step,
    discriminator_train_step,
    compute_router_losses
)
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
    print("Size of train_loader:", len(train_loader))

    try:
        # Setup MoE system
        moe = setup_moe_system(cfg, device)

        # Setup optimizers
        # gen_optims, disc_optims, aux_reg_optim, router_optim = setup_optimizers_unified(moe, cfg)
        gen_optims, disc_optims, aux_reg_optim, router_optim = setup_optimizers(moe, cfg)

        # Setup callbacks
        callbacks = setup_callbacks(cfg)

        # Training loop
        history = []
        start_epoch = 0 if cfg.train.epoch_to_load is None else cfg.train.epoch_to_load

        logger.info(f"Starting training from epoch {start_epoch} to {cfg.train.epochs}")

        for epoch in range(start_epoch, cfg.train.epochs):
            epoch_start_time = time.time()

            # Training phase
            epoch_metrics = train_epoch(
                moe, train_loader, gen_optims, disc_optims,
                aux_reg_optim, router_optim, cfg, device, epoch
            )

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
                    callback.on_epoch_end(epoch, epoch_metrics)
                except Exception as e:
                    logger.warning(f"Callback {callback.__class__.__name__} failed: {e}")

            history.append(epoch_metrics)
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")

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
    aux_reg_optim: torch.optim.Optimizer,
    router_optim: torch.optim.Optimizer,
    cfg,
    device: torch.device,
    epoch: int
) -> Dict:
    """Execute one training epoch."""

    moe.train()

    # Metrics accumulators
    metrics = {
        'gen_loss': [], 'disc_loss': [], 'router_loss': [],
        'div_loss': [], 'intensity_loss': [], 'aux_loss': [],
        'entropy_losses': [], 'distribution_losses': [],
        'differentiation_losses': [], 'alb_losses': [], 'gan_losses': []
    }

    for batch_idx, batch in enumerate(train_loader):
        # try:
        batch_metrics = train_step(
            batch, moe, gen_optims, disc_optims, aux_reg_optim,
            router_optim, cfg, device, epoch
        )

        # Accumulate metrics
        for key, value in batch_metrics.items():
            if key in metrics:
                metrics[key].append(value)
        #
        # except Exception as e:
        #     logger.warning(f"Batch {batch_idx} failed with error: {e}")
        #     raise

    # Average metrics across batches
    averaged_metrics = {}
    for key, values in metrics.items():
        if values:  # Only if we have values
            if isinstance(values[0], list):
                # Handle lists of values (e.g., per-expert losses)
                averaged_metrics[key] = [sum(expert_vals) / len(expert_vals)
                                         for expert_vals in zip(*values)]
            else:
                averaged_metrics[key] = sum(values) / len(values)
        else:
            averaged_metrics[key] = 0.0

    return averaged_metrics

#
# batch, moe, gen_optims, disc_optims, aux_reg_optim,
# router_optim, cfg, device, epoch
#
# def train_step(batch, moe, generator_optimizer, discriminator_optimizer, aux_reg_optim, router_optim, cfg, device,
#                epoch):
#     real_images, _, cond, std, intensity, true_positions = batch
#
#     # Move to device
#     real_images = real_images.unsqueeze(1).to(device)
#     cond = cond.to(device)
#
#     # need to be expanded to n_experts, then mask applied to mask the entries
#     # during the calculations, the difference between the predicted std and the true should be modified
#
#     std = std.to(device)
#
#     intensity = intensity.to(device)
#     true_positions = true_positions.to(device)
#     batch_size = real_images.shape[0]
#
#     noise_1 = torch.randn(batch_size, cfg.model.noise_dim, device=device)
#     noise_2 = torch.randn(batch_size, cfg.model.noise_dim, device=device)
#
#     # Single forward pass for generation
#     start_time = time.time()
#
#     gen_outputs = moe(noise_1, noise_2, cond)
#     moe_forward_time = time.time() - start_time
#
#     generated_1 = gen_outputs['generated_1']
#     generated_2 = gen_outputs['generated_2']
#     gumbel_softmax = gen_outputs['gates']
#
#     # Prepare real images
#     real_images_exp = real_images.unsqueeze(1).expand(-1, moe.n_experts, -1, -1, -1).contiguous()
#
#     # ======================
#     # DISCRIMINATOR TRAINING
#     # ======================
#
#     # Freeze generator parameters
#     for param in moe.generator.parameters():
#         param.requires_grad_(False)
#
#     discriminator_optimizer.zero_grad()
#
#     # Batched discriminator forward with DETACHED inputs (no generator gradients)
#     disc_inputs = torch.cat([generated_1.detach(), generated_2.detach(), real_images_exp], dim=0)
#     disc_cond = torch.cat([cond.detach(), cond.detach(), cond.detach()], dim=0)
#     disc_gumbel = torch.cat([gumbel_softmax.detach(), gumbel_softmax.detach(), gumbel_softmax.detach()], dim=0)
#     disc_out, disc_latent = moe.discriminator(disc_inputs, disc_cond, disc_gumbel)
#
#     # Split outputs
#     disc_fake_out = disc_out[:batch_size]
#     disc_fake_out_2 = disc_out[batch_size:2 * batch_size]
#     disc_real_out = disc_out[2 * batch_size:]
#
#     # mask the outputs of the discriminator, for samples chosen by the router
#     # meaning that if we get the tensor fake_output[0, :, :]. That would show whether the 3 samples of images
#     # were chosen by the 3 sub-discriminators. Thus, we want to mask the outputs of the sub-discriminator that were not
#     # chosen, so that the single discriminator will have updated parameters
#
#     # Discriminator loss and backward
#     start_time = time.time()
#     disc_loss = discriminator_train_step(
#         disc_real_out,
#         disc_fake_out,
#         discriminator_optimizer,
#         device)
#     discriminator_train_step_time = time.time() - start_time
#
#
#     # ======================
#     # GENERATOR TRAINING
#     # ======================
#
#     # Unfreeze generator parameters
#     for param in moe.generator.parameters():
#         param.requires_grad_(True)
#
#     # Freeze discriminator parameters
#     for param in moe.discriminator.parameters():
#         param.requires_grad_(False)
#
#     generator_optimizer.zero_grad()
#
#     start_time = time.time()
#     # Batched discriminator forward with ORIGINAL tensors (keeps generator gradients)
#     gen_inputs = torch.cat([generated_1, generated_2], dim=0)
#     gen_cond = torch.cat([cond, cond], dim=0)
#     gen_gumbel = torch.cat([gumbel_softmax, gumbel_softmax], dim=0)
#
#     gen_disc_out, gen_disc_latent = moe.discriminator(gen_inputs, gen_cond, gen_gumbel)
#
#     # Split outputs
#     disc_fake_out_gen = gen_disc_out[:batch_size]
#     disc_fake_latent_gen = gen_disc_latent[:batch_size]
#     disc_fake_out_2_gen = gen_disc_out[batch_size:]
#     disc_fake_latent_2_gen = gen_disc_latent[batch_size:]
#     process_discriminator_time = time.time() - start_time
#
#
#     # Generator loss and backward
#     # div_loss, int_loss, aux_loss, std_int, mean_int, mean_ints, aux_features
#     start_time = time.time()
#     gen_loss = generator_train_step(
#         noise_1, noise_2,
#         generated_1,
#         disc_fake_out_gen,
#         disc_fake_latent_gen,
#         disc_fake_latent_2_gen,
#         generator_optimizer,
#         std, intensity,
#         cfg.model.generator.di_strength,
#         cfg.model.generator.in_strength,
#         cfg,
#         device
#     )
#     gen_ts_end_time = time.time() - start_time
#
#
#     print("moe_forward_time:", f"{moe_forward_time}:.2f", "seconds")
#     print("discriminator_train_step_time:", f"{discriminator_train_step_time}:.2f", "seconds")
#     print("Proces discriminator time:", f"{process_discriminator_time}:.2f", "seconds")
#     print("Generator trains step time:", f"{gen_ts_end_time}:.2f", "seconds")
#
#
#     # Unfreeze discriminator parameters
#     for param in moe.discriminator.parameters():
#         param.requires_grad_(True)
#
#     # Return metrics
#     return {
#         'gen_loss': gen_loss,
#         'disc_loss': disc_loss,
#         # 'div_loss': div_loss.item(),
#         # 'int_loss': int_loss.item(),
#         # 'aux_loss': aux_loss.item(),
#         # 'std_int': std_int,
#         # 'mean_int': mean_int,
#         # 'mean_ints': mean_ints
#     }

#
# def train_step(
#     batch, moe: MoEWrapper, gen_optims, disc_optims, aux_reg_optim, router_optim,
#     cfg, device: torch.device, epoch: int
# ) -> Dict:
#     """Execute one training step (batch)."""
#
#     # Unpack batch
#     real_images, _, cond, std, intensity, true_positions = batch
#
#     # Move to device
#     real_images = real_images.unsqueeze(1).to(device)
#     cond = cond.to(device)
#     std = std.to(device)
#     intensity = intensity.to(device)
#     true_positions = true_positions.to(device)
#     batch_size = real_images.shape[0]
#
#     noise_1 = torch.randn(batch_size, cfg.model.noise_dim, device=device)
#     noise_2 = torch.randn(batch_size, cfg.model.noise_dim, device=device)
#
#     # Get expert assignments from router
#     router_optim.zero_grad()
#     outputs = moe(noise_1, noise_2, cond, real_images)
#
#     gates = outputs['gates']
#     expert_assignments = outputs['expert_assignments']
#     generated_1 = outputs['generated_1']
#     disc_real_outputs = outputs['disc_real_out']
#     disc_fake_outputs = outputs['disc_fake_out']
#     disc_fake_outputs_2 = outputs['disc_fake_out_2']
#     disc_fake_latents = outputs['disc_fake_latent']
#     disc_fake_latents_2 = outputs['disc_fake_latent_2']
#     # aux_reg_outputs = outputs['aux_reg_outputs']
#     # aux_reg_outputs_features = outputs['aux_reg_outputs_features']
#
#     # Get class counts for learning rate adjustment
#     # class_counts = moe.get_expert_assignment_counts(expert_assignments)
#
#     # Train experts
#     gen_losses = []
#     disc_losses = []
#     div_losses = []
#     intensity_losses = []
#     aux_losses = []
#     aux_reg_features_experts = []
#     mean_intensities_batch = torch.zeros(batch_size, device=device)
#
#     # if len(expert_indices) <= 1:
#     #     # Skip if too few samples
#     #     gen_losses.append(0.0)
#     #     disc_losses.append(0.0)
#     #     div_losses.append(0.0)
#     #     intensity_losses.append(0.0)
#     #     aux_losses.append(0.0)
#     #     aux_reg_features_experts.append(torch.zeros(1, moe.aux_reg.feature_shape_conv_channels, device=device))
#     #     continue
#
#     # Train discriminator
#     disc_loss = discriminator_train_step(
#         real_output=disc_real_outputs,
#         fake_output=disc_fake_outputs,
#         d_optimizer=disc_optims,
#         device=device
#     )
#     disc_losses.append(disc_loss)
#     # expand and repeat the channels to the number of experts
#     intensity_experts = intensity.expand(-1, 3)
#     std_experts = std.expand(-1, 3)
#
#     # Train generator
#     gen_loss, div_loss, int_loss, aux_loss, std_int, mean_int, mean_ints, aux_features = generator_train_step(
#         noise_1, noise_2,
#         generated_1,
#         disc_fake_outputs_2,
#         disc_fake_latents,
#         disc_fake_latents_2,
#         gen_optims,
#         # class_counts,
#         #std_experts, intensity_experts,
#         #aux_reg_optim, aux_reg_outputs, true_positions,
#         # cfg.model.aux_reg.strength,
#         cfg.model.generator.di_strength, cfg.model.generator.in_strength, cfg, device
#     )
#
#     gen_losses.append(gen_loss)
#     div_losses.append(div_loss)
#     intensity_losses.append(int_loss)
#     aux_losses.append(aux_loss)
#     aux_reg_features_experts.append(aux_features)
#
#     # Update mean intensities for router loss
#     mean_intensities_batch = mean_ints.clone().detach().squeeze()
#
#
#     # Compute router losses
#     if moe.n_experts > 1 and epoch < getattr(cfg.train, 'stop_router_training_epoch', float('inf')):
#         router_losses = compute_router_losses(
#             gen_losses, disc_losses, gates, aux_reg_features_experts,
#             mean_intensities_batch, cfg, device
#         )
#
#         # Backprop router loss
#         router_losses['router_loss'].backward()
#         router_optim.step()
#     else:
#         router_losses = {key: 0.0 for key in ['router_loss', 'gan_loss', 'entropy_loss',
#                                               'distribution_loss', 'differentiation_loss', 'alb_loss']}
#
#     # Return metrics
#     return {
#         'gen_losses': gen_losses,
#         'disc_losses': disc_losses,
#         'div_losses': div_losses,
#         'intensity_losses': intensity_losses,
#         'aux_losses': aux_losses,
#         **{k: v.item() if hasattr(v, 'item') else v for k, v in router_losses.items()}
#     }


def train_step(
    batch, moe: MoEWrapper, gen_optims, disc_optims, aux_reg_optim, router_optim,
    cfg, device: torch.device, epoch: int
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
                                   disc_optims, router_optim, device)


    # Return metrics
    return output_losses


def evaluate_epoch(moe: MoEWrapper, test_loader: DataLoader, epoch: int, cfg, device: torch.device) -> Dict:
    """Evaluate model on test set."""
    moe.eval()

    # Metrics accumulators
    metrics = {
        'gen_loss': [], 'disc_loss': [], 'router_loss': [],
        'div_loss': [], 'intensity_loss': [], 'aux_loss': [],
        'entropy_losses': [], 'distribution_losses': [],
        'differentiation_losses': [], 'alb_losses': [], 'gan_losses': [],
        "ws_mean": [], "ws_std": []
    }
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            real_images, _, cond, std, intensity, true_positions = batch

            # epoch, y_test, x_test, device, ch_org, cfg
            batch_metrics = moe.evaluate(epoch, cond, real_images, true_positions, std, intensity, cfg, device)

            # Accumulate metrics
            for key, value in batch_metrics.items():
                if key in metrics:
                    metrics[key].append(value)

    # Average metrics across batches
    averaged_metrics = {}
    for key, values in metrics.items():
        if values:  # Only if we have values
            if isinstance(values[0], list):
                # Handle lists of values (e.g., per-expert losses)
                averaged_metrics[key] = [sum(expert_vals) / len(expert_vals)
                                         for expert_vals in zip(*values)]
            else:
                averaged_metrics[key] = sum(values) / len(values)
        else:
            averaged_metrics[key] = 0.0

    return averaged_metrics


def setup_moe_system(cfg, device: torch.device) -> MoEWrapper:
    """Setup the MoE system with proper model instantiation."""
    from expertsim.models import build_model

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
    # generator = build_model(f"{cfg.model.architecture}.generator_unified", cfg.model.generator, device)
    discriminator = build_model(f"{cfg.model.architecture}.discriminator", cfg.model.discriminator, device)
    # discriminator = build_model(f"{cfg.model.architecture}.discriminator_unified", cfg.model.discriminator, device)
    aux_reg = build_model(f"{cfg.model.architecture}.aux_reg", cfg.model.aux_reg, device)
    router = build_model(f"{cfg.model.router.version}", cfg.model.router, device)

    # # Create MoE wrapper
    moe = MoEWrapper(generator, discriminator, aux_reg, router, cfg.model.n_experts,
                     cfg, image_shape=cfg.dataset.input_image_shape).to(device)
    # moe = MoEWrapperUnified(generator, discriminator, router, cfg.model.n_experts).to(device)

    return moe


def setup_callbacks(cfg) -> List:
    """Setup training callbacks."""
    callbacks = []

    if getattr(cfg, 'wandb', {}).get('log_experiments', False):
        callbacks.append(WandBLogger(cfg))

    if getattr(cfg, 'save_experiment_data', False):
        callbacks.append(CheckpointSaver(
            dir_path=cfg.paths.checkpoint_dir,
            monitor='ws_mean',
            mode='min'
        ))

    return callbacks
