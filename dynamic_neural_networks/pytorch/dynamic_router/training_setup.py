from models_pytorch import Generator, Discriminator, RouterNetwork, AuxReg,\
    GeneratorNeutron, DiscriminatorNeutron, AuxRegNeutron, count_model_parameters

import torch
import torch.optim as optim
import os


def setup_experts(N_EXPERTS, N_COND, NOISE_DIM, LR_G, LR_D, LR_A, DI_STRENGTH, IN_STRENGTH, device=torch.device("cuda:0")):
    # Define experts
    generators = []
    for generator_idx in range(N_EXPERTS):
        generator = Generator(NOISE_DIM, N_COND, DI_STRENGTH, IN_STRENGTH).to(device)
        # expert_weights = f"/net/tscratch/people/plgpbedkowski/data/weights/gen_{generator_idx}_80.h5"
        # print(f'Loading weights for {generator_idx} GEN')
        # generator.load_state_dict(torch.load(expert_weights, map_location='cpu'))
        generator = generator.to(device)  # or whichever CUDA device you're using
        generators.append(generator)

    num_params = count_model_parameters(generator)
    print(f"Generator model has {num_params} trainable parameters.")
    generator_optimizers = [optim.Adam(gen.parameters(), lr=LR_G) for gen in generators]

    # Define discriminators
    discriminators = []
    for generator_idx in range(N_EXPERTS):
        discriminator = Discriminator(N_COND).to(device)
        # # load previous weights
        # expert_weights = f"/net/tscratch/people/plgpbedkowski/data/weights/disc_{generator_idx}_80.h5"
        # print(f'weights loaded for {generator_idx} DISC')
        # discriminator.load_state_dict(torch.load(expert_weights, map_location='cpu'))
        discriminator = discriminator.to("cuda:0")
        discriminators.append(discriminator)
    num_params = count_model_parameters(discriminator)
    print(f"Discriminator model has {num_params} trainable parameters.")
    discriminator_optimizers = [optim.Adam(disc.parameters(), lr=LR_D) for disc in discriminators]

    # Define aux reg
    aux_regs = []
    for generator_idx in range(N_EXPERTS):
        aux_reg = AuxReg().to(device)
        aux_regs.append(aux_reg)
    num_params = count_model_parameters(aux_reg)
    print(f"Aux Reg model has {num_params} trainable parameters.")
    aux_reg_optimizers = [optim.Adam(aux_reg.parameters(), lr=LR_A) for aux_reg in aux_regs]

    return generators, generator_optimizers, discriminators, discriminator_optimizers, aux_regs, aux_reg_optimizers


def setup_experts_neutron(N_EXPERTS, N_COND, NOISE_DIM, LR_G, LR_D, LR_A, DI_STRENGTH, IN_STRENGTH,
                          device=torch.device("cuda:0")):
    # Define experts
    generators = []
    for generator_idx in range(N_EXPERTS):
        generator = GeneratorNeutron(NOISE_DIM, N_COND, DI_STRENGTH, IN_STRENGTH).to(device)
        # expert_weights = f"/net/tscratch/people/plgpbedkowski/data/weights/gen_{generator_idx}_80.h5"
        # print(f'Loading weights for {generator_idx} GEN')
        # generator.load_state_dict(torch.load(expert_weights, map_location='cpu'))
        generator = generator.to(device)  # or whichever CUDA device you're using
        generators.append(generator)

    num_params = count_model_parameters(generator)
    print(f"Generator model has {num_params} trainable parameters.")
    generator_optimizers = [optim.Adam(gen.parameters(), lr=LR_G) for gen in generators]

    # Define discriminators
    discriminators = []
    for generator_idx in range(N_EXPERTS):
        discriminator = DiscriminatorNeutron(N_COND).to(device)
        # # load previous weights
        # expert_weights = f"/net/tscratch/people/plgpbedkowski/data/weights/disc_{generator_idx}_80.h5"
        # print(f'weights loaded for {generator_idx} DISC')
        # discriminator.load_state_dict(torch.load(expert_weights, map_location='cpu'))
        discriminator = discriminator.to("cuda:0")
        discriminators.append(discriminator)
    num_params = count_model_parameters(discriminator)
    print(f"Discriminator model has {num_params} trainable parameters.")
    discriminator_optimizers = [optim.Adam(disc.parameters(), lr=LR_D) for disc in discriminators]

    # Define aux reg
    aux_regs = []
    for generator_idx in range(N_EXPERTS):
        aux_reg = AuxRegNeutron().to(device)
        aux_regs.append(aux_reg)
    num_params = count_model_parameters(aux_reg)
    print(f"Aux Reg model has {num_params} trainable parameters.")
    aux_reg_optimizers = [optim.Adam(aux_reg.parameters(), lr=LR_A) for aux_reg in aux_regs]

    return generators, generator_optimizers, discriminators, discriminator_optimizers, aux_regs, aux_reg_optimizers


def setup_router(N_COND, N_EXPERTS, LR_R, device=torch.device("cuda:0")):
    router_network = RouterNetwork(N_COND, N_EXPERTS)
    # load previous weights
    # expert_weights = f"/net/tscratch/people/plgpbedkowski/data/weights/router_network_epoch_80.pth"
    # print(f'weights loaded for ROUTER')
    # router_network.load_state_dict(torch.load(expert_weights, map_location='cpu'))
    router_network = router_network.to(device)
    router_optimizer = optim.Adam(router_network.parameters(), lr=LR_R)
    num_params = count_model_parameters(router_network)
    print(f"Router model has {num_params} trainable parameters.")
    # Define the learning rate scheduler
    # router_scheduler = lr_scheduler.ReduceLROnPlateau(router_optimizer, mode='min', patience=3, factor=0.1, verbose=True)
    return router_network, router_optimizer


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
    # Load generator weights and optimizers
    for i, (generator, gen_opt) in enumerate(zip(generators, generator_optimizers)):
        gen_file = os.path.join(checkpoint_dir, f"gen_{i}_{epoch}.pth")
        gen_opt_file = os.path.join(checkpoint_dir, f"gen_optim_{i}_{epoch}.pth")

        if os.path.exists(gen_file):
            print(f"Loading generator {i} weights from {gen_file}")
            generator.load_state_dict(torch.load(gen_file, map_location=device))
        else:
            print(f"Generator {i} weights not found for epoch {epoch}")

        if os.path.exists(gen_opt_file):
            print(f"Loading generator {i} optimizer state from {gen_opt_file}")
            gen_opt.load_state_dict(torch.load(gen_opt_file, map_location=device))
        else:
            print(f"Generator {i} optimizer state not found for epoch {epoch}")

    # Load discriminator weights and optimizers
    for i, (discriminator, disc_opt) in enumerate(zip(discriminators, discriminator_optimizers)):
        disc_file = os.path.join(checkpoint_dir, f"disc_{i}_{epoch}.pth")
        disc_opt_file = os.path.join(checkpoint_dir, f"disc_optim_{i}_{epoch}.pth")

        if os.path.exists(disc_file):
            print(f"Loading discriminator {i} weights from {disc_file}")
            discriminator.load_state_dict(torch.load(disc_file, map_location=device))
        else:
            print(f"Discriminator {i} weights not found for epoch {epoch}")

        if os.path.exists(disc_opt_file):
            print(f"Loading discriminator {i} optimizer state from {disc_opt_file}")
            disc_opt.load_state_dict(torch.load(disc_opt_file, map_location=device))
        else:
            print(f"Discriminator {i} optimizer state not found for epoch {epoch}")

    # Load auxiliary regularizers and optimizers
    for i, (aux_reg, aux_opt) in enumerate(zip(aux_regs, aux_reg_optimizers)):
        aux_file = os.path.join(checkpoint_dir, f"aux_reg_{i}_{epoch}.pth")
        aux_opt_file = os.path.join(checkpoint_dir, f"aux_reg_optim_{i}_{epoch}.pth")

        if os.path.exists(aux_file):
            print(f"Loading Aux Reg {i} weights from {aux_file}")
            aux_reg.load_state_dict(torch.load(aux_file, map_location=device))
        else:
            print(f"Aux Reg {i} weights not found for epoch {epoch}")

        if os.path.exists(aux_opt_file):
            print(f"Loading Aux Reg {i} optimizer state from {aux_opt_file}")
            aux_opt.load_state_dict(torch.load(aux_opt_file, map_location=device))
        else:
            print(f"Aux Reg {i} optimizer state not found for epoch {epoch}")

    # Load router network weights and optimizer
    router_file = os.path.join(checkpoint_dir, f"router_network_{epoch}.pth")
    router_opt_file = os.path.join(checkpoint_dir, f"router_network_optim_{epoch}.pth")

    if os.path.exists(router_file):
        print(f"Loading router network weights from {router_file}")
        router_network.load_state_dict(torch.load(router_file, map_location=device))
    else:
        print(f"Router network weights not found for epoch {epoch}")

    if os.path.exists(router_opt_file):
        print(f"Loading router optimizer state from {router_opt_file}")
        router_optimizer.load_state_dict(torch.load(router_opt_file, map_location=device))
    else:
        print(f"Router optimizer state not found for epoch {epoch}")
