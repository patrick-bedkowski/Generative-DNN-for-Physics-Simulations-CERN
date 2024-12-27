from models_pytorch import Generator, Discriminator, RouterNetwork, AuxReg

import torch
import torch.optim as optim


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
    discriminator_optimizers = [optim.Adam(disc.parameters(), lr=LR_D) for disc in discriminators]

    # Define aux reg
    aux_regs = []
    for generator_idx in range(N_EXPERTS):
        aux_reg = AuxReg().to(device)
        aux_regs.append(aux_reg)
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
    # Define the learning rate scheduler
    # router_scheduler = lr_scheduler.ReduceLROnPlateau(router_optimizer, mode='min', patience=3, factor=0.1, verbose=True)
    return router_network, router_optimizer
