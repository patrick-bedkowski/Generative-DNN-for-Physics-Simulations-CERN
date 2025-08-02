from expertsim.models.moe import MoEWrapper, MoEWrapperUnified
import torch
import torch.optim as optim
import os
from typing import Tuple, List


def count_model_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_optimizers(wrapper: MoEWrapper, cfg) -> Tuple[List, List, torch.optim.Optimizer, torch.optim.Optimizer]:
    """
    Create optimizers for all model components.

    Returns:
        Tuple of (generator_optimizers, discriminator_optimizers, aux_reg_optimizer, router_optimizer)
    """
    # Generator optimizers
    gen_optims = [
        torch.optim.Adam(gen.parameters(), lr=cfg.model.generator.lr_g)
        for gen in wrapper.generators
    ]

    # Discriminator optimizers
    disc_optims = [
        torch.optim.Adam(disc.parameters(), lr=cfg.model.discriminator.lr_d)
        for disc in wrapper.discriminators
    ]

    # Auxiliary regressor optimizer
    aux_reg_optim = torch.optim.Adam(wrapper.aux_reg.parameters(), lr=cfg.model.aux_reg.lr_a)

    # Router optimizer
    router_optim = torch.optim.Adam(wrapper.router.parameters(), lr=cfg.model.router.lr_r)

    return gen_optims, disc_optims, aux_reg_optim, router_optim


def setup_optimizers_unified(wrapper: MoEWrapperUnified, cfg) -> Tuple[List, List, torch.optim.Optimizer, torch.optim.Optimizer]:
    """
    Create optimizers for all model components.

    Returns:
        Tuple of (generator_optimizers, discriminator_optimizers, aux_reg_optimizer, router_optimizer)
    """
    # Generator optimizers
    gen_optims = torch.optim.Adam(wrapper.generator.parameters(), lr=cfg.model.generator.lr_g)

    # Discriminator optimizers
    disc_optims = torch.optim.Adam(wrapper.discriminator.parameters(), lr=cfg.model.discriminator.lr_d)

    # Auxiliary regressor optimizer
    # aux_reg_optim = torch.optim.Adam(wrapper.aux_reg.parameters(), lr=cfg.model.aux_reg.lr_a)
    aux_reg_optim = None
    # Router optimizer
    router_optim = torch.optim.Adam(wrapper.router.parameters(), lr=cfg.model.router.lr_r)

    return gen_optims, disc_optims, aux_reg_optim, router_optim


def print_model_info(wrapper: MoEWrapper):
    """Print information about the MoE system."""
    print(f"=== MoE System Information ===")
    print(f"Number of experts: {wrapper.n_experts}")

    total_params = 0
    for i, gen in enumerate(wrapper.generators):
        params = count_model_parameters(gen)
        print(f"Generator {i}: {params:,} parameters")
        total_params += params

    for i, disc in enumerate(wrapper.discriminators):
        params = count_model_parameters(disc)
        print(f"Discriminator {i}: {params:,} parameters")
        total_params += params

    router_params = count_model_parameters(wrapper.router)
    aux_params = count_model_parameters(wrapper.aux_reg)

    print(f"Router: {router_params:,} parameters")
    print(f"Aux Regressor: {aux_params:,} parameters")

    total_params += router_params + aux_params
    print(f"Total parameters: {total_params:,}")


def load_checkpoint_weights(checkpoint_dir,
                            epoch,
                            generators,
                            generator_optimizers,
                            discriminators,
                            discriminator_optimizers,
                            aux_regs,
                            aux_reg_optimizers,
                            router_network,
                            router_optimizer,
                            device="cuda"):
    """
    Load weights for generators, discriminators, auxiliary regularizers, router network,
    and their respective optimizers from a specific checkpoint directory and epoch.

    Args:
        checkpoint_dir (str): Path to the checkpoint directory.
        epoch (int): Epoch for which the weights should be loaded.
        generators (list): List of generator models.
        generator_optimizers (list): List of generator optimizers.
        discriminators (list): List of discriminator models.
        discriminator_optimizers (list): List of discriminator optimizers.
        aux_regs (list): List of auxiliary regularizers.
        aux_reg_optimizers (list): List of auxiliary regularizer optimizers.
        router_network (nn.Module): Router network model.
        router_optimizer (torch.optim.Optimizer): Router network optimizer.
        device (str): Device to load the weights onto.
    """
    # --------- Load Generators (full models) and update optimizers ---------
    for i, gen_opt in enumerate(generator_optimizers):
        gen_file = os.path.join(checkpoint_dir, f"gen_{i}_{epoch}.pth")
        gen_opt_file = os.path.join(checkpoint_dir, f"gen_optim_{i}_{epoch}.pth")

        if os.path.exists(gen_file):
            print(f"Loading generator {i} model from {gen_file}")
            try:
                # Load the entire generator object (not just a state_dict)
                loaded_generator = torch.load(gen_file, map_location=device)
                generators[i] = loaded_generator  # replace with loaded model

                # Since the optimizer was referencing the old model’s parameters,
                # update every parameter group to use the new ones.
                for group in gen_opt.param_groups:
                    group["params"] = list(loaded_generator.parameters())
            except Exception as e:
                print(f"Error loading generator {i} model from {gen_file}: {e}")
        else:
            print(f"Generator {i} model not found for epoch {epoch}")

        if os.path.exists(gen_opt_file):
            print(f"Loading generator {i} optimizer state from {gen_opt_file}")
            try:
                # Open the file in binary mode to avoid zip archive issues.
                with open(gen_opt_file, "rb") as f:
                    optimizer_state = torch.load(f, map_location=device)
                gen_opt.load_state_dict(optimizer_state)
            except Exception as e:
                print(f"Error loading generator {i} optimizer state from {gen_opt_file}: {e}")
        else:
            print(f"Generator {i} optimizer state not found for epoch {epoch}")

    # --------- Load Discriminators (full models) and update optimizers ---------
    for i, disc_opt in enumerate(discriminator_optimizers):
        disc_file = os.path.join(checkpoint_dir, f"disc_{i}_{epoch}.pth")
        disc_opt_file = os.path.join(checkpoint_dir, f"disc_optim_{i}_{epoch}.pth")

        if os.path.exists(disc_file):
            print(f"Loading discriminator {i} model from {disc_file}")
            try:
                loaded_disc = torch.load(disc_file, map_location=device)
                discriminators[i] = loaded_disc
                for group in disc_opt.param_groups:
                    group["params"] = list(loaded_disc.parameters())
            except Exception as e:
                print(f"Error loading discriminator {i} model from {disc_file}: {e}")
        else:
            print(f"Discriminator {i} model not found for epoch {epoch}")

        if os.path.exists(disc_opt_file):
            print(f"Loading discriminator {i} optimizer state from {disc_opt_file}")
            try:
                with open(disc_opt_file, "rb") as f:
                    optimizer_state = torch.load(f, map_location=device)
                disc_opt.load_state_dict(optimizer_state)
            except Exception as e:
                print(f"Error loading discriminator {i} optimizer state from {disc_opt_file}: {e}")
        else:
            print(f"Discriminator {i} optimizer state not found for epoch {epoch}")

    # --------- Load Auxiliary Regularizers (full models) and update optimizers ---------
    for i, aux_opt in enumerate(aux_reg_optimizers):
        aux_file = os.path.join(checkpoint_dir, f"aux_reg_{i}_{epoch}.pth")
        aux_opt_file = os.path.join(checkpoint_dir, f"aux_reg_optim_{i}_{epoch}.pth")

        if os.path.exists(aux_file):
            print(f"Loading auxiliary regularizer {i} model from {aux_file}")
            try:
                loaded_aux = torch.load(aux_file, map_location=device)
                aux_regs[i] = loaded_aux
                for group in aux_opt.param_groups:
                    group["params"] = list(loaded_aux.parameters())
            except Exception as e:
                print(f"Error loading auxiliary regularizer {i} model from {aux_file}: {e}")
        else:
            print(f"Auxiliary regularizer {i} model not found for epoch {epoch}")

        if os.path.exists(aux_opt_file):
            print(f"Loading auxiliary regularizer {i} optimizer state from {aux_opt_file}")
            try:
                with open(aux_opt_file, "rb") as f:
                    optimizer_state = torch.load(f, map_location=device)
                aux_opt.load_state_dict(optimizer_state)
            except Exception as e:
                print(f"Error loading auxiliary regularizer {i} optimizer state from {aux_opt_file}: {e}")
        else:
            print(f"Auxiliary regularizer {i} optimizer state not found for epoch {epoch}")

    # --------- Load Router Network (full model) and update its optimizer ---------
    router_file = os.path.join(checkpoint_dir, f"router_network_{epoch}.pth")
    router_opt_file = os.path.join(checkpoint_dir, f"router_network_optim_{epoch}.pth")

    if os.path.exists(router_file):
        print(f"Loading router network model from {router_file}")
        try:
            loaded_router = torch.load(router_file, map_location=device)
            # Update in-place so that references to router_network remain valid.
            router_network.__dict__.update(loaded_router.__dict__)
        except Exception as e:
            print(f"Error loading router network model from {router_file}: {e}")
    else:
        print(f"Router network model not found for epoch {epoch}")

    if os.path.exists(router_opt_file):
        print(f"Loading router optimizer state from {router_opt_file}")
        try:
            with open(router_opt_file, "rb") as f:
                optimizer_state = torch.load(f, map_location=device)
            # Update the optimizer’s parameter groups to refer to the (updated) router_network.
            for group in router_optimizer.param_groups:
                group["params"] = list(router_network.parameters())
            router_optimizer.load_state_dict(optimizer_state)
        except Exception as e:
            print(f"Error loading router optimizer state from {router_opt_file}: {e}")
    else:
        print(f"Router optimizer state not found for epoch {epoch}")