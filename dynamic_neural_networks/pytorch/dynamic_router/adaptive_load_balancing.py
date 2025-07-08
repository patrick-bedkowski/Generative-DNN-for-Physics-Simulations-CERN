import time
import os
import wandb

import pandas as pd
import numpy as np
from datetime import datetime

from itertools import combinations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset

from utils import (sum_channels_parallel, calculate_ws_ch_proton_model,
                   calculate_joint_ws_across_experts,
                   create_dir, save_scales, evaluate_router,
                   intensity_regularization, sdi_gan_regularization,
                   generate_and_save_images,
                   calculate_expert_distribution_loss,
                   regressor_loss, calculate_expert_utilization_entropy,
                   StratifiedBatchSampler, plot_cond_pca_tsne, plot_expert_heatmap,
                   calculate_adaptive_load_balancing_loss)
from data_transformations import transform_data_for_training, ZDCType, SCRATCH_PATH
from training_setup import setup_experts, setup_router, load_checkpoint_weights
from training_utils import save_models_and_architectures


print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)           # Check which CUDA version PyTorch was built with
print(f"Number of CUDA devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA Device 0: {torch.cuda.get_device_name(0)}")


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.autograd.set_detect_anomaly(True)

SAVE_EXPERIMENT_DATA = False
PLOT_IMAGES = True

# SETTINGS & PARAMETERS
WS_MEAN_SAVE_THRESHOLD = 2
DI_STRENGTH = 0.1
IN_STRENGTH = 1e-3  # 0.001
IN_STRENGTH_LOWER_VAL = 0.001  # 0.000001
AUX_STRENGTH = 0.001

N_RUNS = 1
N_EXPERTS = 3
# N_HEADS = 4
# HIDDEN_DIM = 128
BATCH_SIZE = 256
NOISE_DIM = 10
N_COND = 9  # number of conditional features
EPOCHS = 250
LR_G = 1e-4
LR_D = 1e-5
LR_A = 1e-4
LR_R = 1e-3

# Router loss parameters
ED_STRENGTH = 0 #0.01  # Strength on the expert distribution loss in the router loss calculation
GEN_STRENGTH = 1e-1  # Strength on the generator loss in the router loss calculation
DIFF_STRENGTH = 1e-5  # Differentation on the generator loss in the router loss calculation
UTIL_STRENGTH = 1e-3  #  1e-4  # Strength on the expert utilization entropy in the router loss calculation
ALB_STRENGTH = 0 #1e-7  # 1e-3  #1e-4  # Strength on the adaptive load balancing in the router loss calculation
STOP_ROUTER_TRAINING_EPOCH = EPOCHS
CLIP_DIFF_LOSS = "No-clip" #-1.0

DATA_IMAGES_PATH = "/net/tscratch/people/plgpbedkowski/data/data_proton_photonsum_proton_1_2312.pkl"
DATA_COND_PATH = "/net/tscratch/people/plgpbedkowski/data/data_cond_photonsum_proton_1_2312.pkl"
# data of coordinates of maximum value of pixel on the images
DATA_POSITIONS_PATH = "/net/tscratch/people/plgpbedkowski/data/data_coord_proton_photonsum_proton_1_2312.pkl"
INPUT_IMAGE_SHAPE = (56, 30)
MIN_PROTON_THRESHOLD = 1

CHECKPOINT_DATA_INDICES_FILE = None
CHECKPOINT_RUN_MODELS = None  #r"/net/tscratch/people/plgpbedkowski/dynamic_neural_networks/experiments/test_single_disc_scaled_LR_disc_alb_loss_0.0001_1e-05_0.001_22_03_2025_07_20_19_762_1_2312_22_03_2025_07_20_19_762/models"
epoch_to_load = None  #

NAME = f"test_entropy_old_router"
# NAME = f"test_entropy_st_gumbel_softmax_v2_intensity_without_detach"
# NAME = f"test_continue_adaptive_load_balancing_detach_gumbel_softmax"
# NAME = f"test_single_disc_alb_loss_continue_the_best"

# NAME = f"no_entropy_min_{MIN_PROTON_THRESHOLD}_ijcai2025_proton_{ED_STRENGTH}_{GEN_STRENGTH}_{UTIL_STRENGTH}"

# NAME = f"diff_with_intensitiy_and_std_ijcai2025_proton_{ED_STRENGTH}_{GEN_STRENGTH}_{UTIL_STRENGTH}"
# NAME = f"Sanitycheck-intenisty-clamp-max-1-expert"
# NAME = f"removed-detach-from-intensity-Sanitycheck-intenisty-with-scaler-on-the-input-and-output-expert-without-clamp-1"
# NAME = f"test_modified_intensity_refactoring_test_3_disc_expertsim"

N = 10000
for _ in range(N_RUNS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_pickle(DATA_IMAGES_PATH)[:N]
    data_cond = pd.read_pickle(DATA_COND_PATH)[:N]
    data_posi = pd.read_pickle(DATA_POSITIONS_PATH)[:N]

    data_cond_idx = data_cond[data_cond.proton_photon_sum >= MIN_PROTON_THRESHOLD].index
    data_cond = data_cond.loc[data_cond_idx].reset_index(drop=True)
    data = data[data_cond_idx]
    data_posi = data_posi.loc[data_cond_idx].reset_index(drop=True)

    photon_sum_proton_min, photon_sum_proton_max = data_cond.proton_photon_sum.min(), data_cond.proton_photon_sum.max()
    print(data_posi)
    print('Loaded positions: ', data_posi.shape, "max:", data_posi.values.max(), "min:", data_posi.values.min())

    DATE_STR = datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f")[:-3]
    wandb_run_name = f"{NAME}_{DATE_STR}"
    EXPERIMENT_DIR_NAME = f"experiments/{wandb_run_name}"
    EXPERIMENT_DIR_NAME = os.path.join(SCRATCH_PATH, EXPERIMENT_DIR_NAME)

    ### TRANSFORM DATA FOR TRAINING ###
    x_train, x_test, x_train_2, x_test_2, y_train, y_test, std_train, std_test, \
    intensity_train, intensity_test, positions_train, positions_test, expert_number_train, \
    expert_number_test, scaler_poz, data_cond_names, filepath_models = transform_data_for_training(
        data_cond, data,
        data_posi,
        EXPERIMENT_DIR_NAME,
        ZDCType.PROTON,
        SAVE_EXPERIMENT_DATA,
        load_data_file_from_checkpoint=CHECKPOINT_DATA_INDICES_FILE)

    # CALCULATE DISTRIBUTION OF CHANNELS IN ORIGINAL TEST DATA #
    org = np.exp(x_test) - 1
    ch_org = np.array(org).reshape(-1, *INPUT_IMAGE_SHAPE)
    del org
    ch_org = pd.DataFrame(sum_channels_parallel(ch_org)).values

    # Loss and optimizer
    binary_cross_entropy_criterion = nn.BCELoss()
    aux_reg_criterion = regressor_loss

    generators, generator_optimizers, discriminators,\
    discriminator_optimizers, aux_regs, aux_reg_optimizers = setup_experts(N_EXPERTS, N_COND, NOISE_DIM, LR_G, LR_D,
                                                                           LR_A, DI_STRENGTH, IN_STRENGTH, device)
    router_network, router_optimizer = setup_router(N_COND, N_EXPERTS, LR_R, device)
    # router_network, router_optimizer = setup_router_attention(N_COND, N_EXPERTS, N_HEADS, HIDDEN_DIM, LR_R, device)

    if CHECKPOINT_RUN_MODELS:
        load_checkpoint_weights(
            CHECKPOINT_RUN_MODELS,
            epoch_to_load,
            generators,
            generator_optimizers,
            discriminators,
            discriminator_optimizers,
            aux_regs,
            aux_reg_optimizers,
            router_network,
            router_optimizer,
            device=device
        )

    def generator_train_step(generator, discriminator, a_reg, cond, g_optimizer, a_optimizer, criterion,
                             true_positions, std, intensity, class_counts, batch_size):
        # Train Generator
        g_optimizer.zero_grad()

        noise = torch.randn(batch_size, NOISE_DIM, device=device)
        noise_2 = torch.randn(batch_size, NOISE_DIM, device=device)

        # generate fake images
        fake_images = generator(noise, cond)
        fake_images_2 = generator(noise_2, cond)

        # validate two images
        fake_output, fake_latent = discriminator(fake_images, cond)
        fake_output_2, fake_latent_2 = discriminator(fake_images_2, cond)

        gen_loss = criterion(fake_output, torch.ones_like(fake_output))

        div_loss = sdi_gan_regularization(fake_latent, fake_latent_2,
                                          noise, noise_2,
                                          std, generator.di_strength)

        intensity_loss, mean_intenisties, std_intensity, mean_intensity = intensity_regularization(fake_images,
                                                                                                   intensity,
                                                                                                   generator.in_strength)

        gen_loss = gen_loss + div_loss + intensity_loss

        # Train auxiliary regressor
        a_optimizer.zero_grad()
        generated_positions = a_reg(fake_images)

        aux_reg_loss = aux_reg_criterion(true_positions, generated_positions, scaler_poz)*AUX_STRENGTH

        gen_loss += aux_reg_loss

        gen_loss.backward()
        g_optimizer.param_groups[0]['lr'] = LR_G * class_counts.clone().detach()
        g_optimizer.step()
        a_optimizer.step()

        return gen_loss.data, div_loss.data, intensity_loss.data, aux_reg_loss.data, std_intensity, mean_intensity,\
               mean_intenisties


    def discriminator_train_step(disc, generator, d_optimizer, criterion, real_images, cond, class_counts, batch_size) -> np.float32:
        """Returns Python float of disc_loss value"""
        # Train discriminator

        d_optimizer.zero_grad()

        # calculate loss for real images
        real_output, real_latent = disc(real_images, cond)
        real_labels = torch.ones_like(real_output)
        loss_real_disc = criterion(real_output, real_labels)

        # calculate loss for generated images
        noise = torch.randn(batch_size, NOISE_DIM, device=device)
        fake_images = generator(noise, cond)
        fake_output, fake_latent = disc(fake_images.detach(), cond)
        fake_labels = torch.zeros_like(fake_output)
        loss_fake_disc = criterion(fake_output, fake_labels)

        # Accumulate and compute discriminator loss
        disc_loss = loss_real_disc + loss_fake_disc
        disc_loss.backward()  # call backward computations on accumulated gradients for efficiency

        # d_optimizer.param_groups[0]['lr'] = LR_D * class_counts
        d_optimizer.step()
        return disc_loss.item()

    def train_step(batch, epoch):
        real_images, real_images_2, cond, std, intensity, true_positions = batch

        real_images = real_images.unsqueeze(1).to(device).clone()
        cond = cond.to(device).clone()
        std = std.to(device).clone()
        intensity = intensity.to(device).clone()
        true_positions = true_positions.to(device).clone()
        BATCH_SIZE = real_images.shape[0]

        # Train router network
        router_optimizer.zero_grad()
        predicted_expert_one_hot = router_network(cond)  # Get predicted experts assignments for samples. Outputs are the probabilities of each expert for each sample. Shape: (batch, N_EXPERTS)
        _, predicted_expert = torch.max(predicted_expert_one_hot, 1)  # (BATCH_SIZE, 1)

        # calculate the class counts for each expert
        class_counts = torch.zeros(N_EXPERTS, dtype=torch.float).to(device)
        for class_label in range(N_EXPERTS):
            class_counts[class_label] = (predicted_expert == class_label).sum().item()
        class_counts_adjusted = class_counts / predicted_expert.size(0)

        # train experts
        gen_losses = torch.zeros(N_EXPERTS).to(device)
        mean_intensities_experts = np.zeros(N_EXPERTS)  # mean intensities for each expert for each batch
        std_intensities_experts = np.zeros(N_EXPERTS)  # std intensities for each expert for each batch
        mean_intensities_in_batch_expert = torch.zeros(BATCH_SIZE, device=device)

        # Train each discriminator independently
        disc_losses = torch.zeros(N_EXPERTS)
        for i in range(N_EXPERTS):
            selected_indices = (predicted_expert == i).nonzero(as_tuple=True)[0]
            BATCH_SIZE = len(selected_indices)
            if selected_indices.numel() <= 1:
                disc_losses[i] = torch.tensor(0.0, requires_grad=True).to(device)
                continue

            # Clone or detach tensors to avoid in-place modifications
            selected_cond = cond[selected_indices]
            selected_generator = generators[i]
            selected_discriminator = discriminators[i]  # always take first discriminator
            selected_discriminator_optimizer = discriminator_optimizers[i]  # always take first discriminator
            selected_real_images = real_images[selected_indices]
            selected_class_counts = class_counts_adjusted[i]

            disc_loss = discriminator_train_step(selected_discriminator, selected_generator,
                                                 selected_discriminator_optimizer,
                                                 binary_cross_entropy_criterion, selected_real_images,
                                                 selected_cond, selected_class_counts, BATCH_SIZE)
            disc_losses[i] = disc_loss

        #
        # Train each generator
        #
        for i in range(N_EXPERTS):
            selected_indices = (predicted_expert == i).nonzero(as_tuple=True)[0]
            if selected_indices.numel() <= 1:
                gen_losses[i] = torch.tensor(0.0, requires_grad=True).to(device)
                continue

            selected_cond = cond[selected_indices]
            selected_true_positions = true_positions[selected_indices]
            selected_intensity = intensity[selected_indices]
            selected_std = std[selected_indices]
            selected_generator = generators[i]
            selected_generator_optimizer = generator_optimizers[i]
            selected_discriminator = discriminators[i]
            selected_aux_reg = aux_regs[i]
            selected_aux_reg_optimizer = aux_reg_optimizers[i]
            selected_class_counts = class_counts_adjusted[i]

            gen_loss, div_loss, intensity_loss, \
            aux_reg_loss, std_intensity, mean_intensity, mean_intensities = generator_train_step(selected_generator,
                                                                                                 selected_discriminator,
                                                                                                 selected_aux_reg,
                                                                                                 selected_cond,
                                                                                                 selected_generator_optimizer,
                                                                                                 selected_aux_reg_optimizer,
                                                                                                 binary_cross_entropy_criterion,
                                                                                                 selected_true_positions,
                                                                                                 selected_std,
                                                                                                 selected_intensity,
                                                                                                 selected_class_counts,
                                                                                                 len(selected_indices))

            mean_intensities_in_batch_expert[
                selected_indices] = mean_intensities.clone().detach().squeeze()  # input the mean intensities for calculated samples

            # Save statistics
            mean_intensities_experts[i] = mean_intensity
            std_intensities_experts[i] = std_intensity
            gen_losses[i] = gen_loss  # do the detach so when calculating loss for the router, the gradient doesnt flow back to the generators

        #
        # Calculate router loss
        #
        if N_EXPERTS > 1:
            gan_loss_scaled = (gen_losses.mean()+disc_losses.mean()) * GEN_STRENGTH  #  Added that on 06.01.25
            expert_entropy_loss = calculate_expert_utilization_entropy(predicted_expert_one_hot.clone(), UTIL_STRENGTH) if UTIL_STRENGTH != 0 else torch.tensor(0.0,
                                                                                                                                                                requires_grad=False,
                                                                                                                                                                device=predicted_expert_one_hot.device)

            expert_distribution_loss = calculate_expert_distribution_loss(predicted_expert_one_hot.clone(),
                                                                          mean_intensities_in_batch_expert.reshape(-1, 1),
                                                                          ED_STRENGTH) if ED_STRENGTH != 0. else torch.tensor(0.0, requires_grad=False)


            # Compute differentiation loss for all experts
            differentiation_loss_intensities = sum(
                np.abs((mean_intensities_experts[i] - mean_intensities_experts[j]))
                for i, j in combinations(range(N_EXPERTS), 2)  # Generate all unique pairs of experts
            ) if DIFF_STRENGTH != 0. else torch.tensor(0.0)
            # differentiation_loss_stds = sum(
            #     np.abs((std_intensities_experts[i] - std_intensities_experts[j]))
            #     for i, j in combinations(range(N_EXPERTS), 2)  # Generate all unique pairs of experts
            # ) if DIFF_STRENGTH != 0. else torch.tensor(0.0)

            # differentiation_loss = torch.clamp(torch.tensor(differentiation_loss, device=device), min=CLIP_DIFF_LOSS)
            differentiation_loss = differentiation_loss_intensities * DIFF_STRENGTH

            # print("logits router for every sample")
            # print(predicted_expert_one_hot.shape)
            routing_scores = predicted_expert_one_hot.sum(dim=0)
            # print("Routing scores:")
            # print(routing_scores)
            # print("-------------------------")
            alb_loss = calculate_adaptive_load_balancing_loss(routing_scores, ALB_STRENGTH) if ALB_STRENGTH != 0. else torch.tensor(0.0)

            router_loss = gan_loss_scaled + expert_distribution_loss - expert_entropy_loss - differentiation_loss + alb_loss

            if epoch < STOP_ROUTER_TRAINING_EPOCH:
                # Train Router Network
                router_loss.backward()
                router_optimizer.step()
            else:
                router_loss = torch.tensor(0.0)
        else:
            gan_loss_scaled = torch.tensor(0.0)
            router_loss = torch.tensor(0.0)
            expert_distribution_loss = torch.tensor(0.0)
            differentiation_loss = torch.tensor(0.0)
            expert_entropy_loss = torch.tensor(0.0)

        gen_losses = [gen_loss.item() for gen_loss in gen_losses]
        disc_losses = [disc_loss.item() for disc_loss in disc_losses]
        return gen_losses, \
               disc_losses, router_loss.item(), div_loss.cpu().item(), \
               intensity_loss.cpu().item(), aux_reg_loss.cpu().item(), class_counts.cpu().numpy(), \
               std_intensities_experts, mean_intensities_experts, expert_distribution_loss.item(), \
               differentiation_loss.item(), expert_entropy_loss.item(), gan_loss_scaled.item(), alb_loss.item()

    # Training loop
    def train(train_loader, epochs, y_test):
        history = []
        if epoch_to_load is None:
            start_epoch = 0
        else:
            start_epoch = epoch_to_load + 1
        for epoch in range(start_epoch, epochs):
            start = time.time()
            gen_losses_epoch = []
            disc_losses_epoch = []
            router_loss_epoch = []
            div_loss_epoch = []
            intensity_loss_epoch = []
            aux_reg_loss_epoch = []
            expert_distribution_loss_epoch = []
            differentiation_loss_epoch = []
            expert_entropy_loss_epoch = []
            alb_loss_epoch = []
            gan_loss_epoch = []

            n_chosen_experts = [[] for _ in range(N_EXPERTS)]
            mean_intensities_experts = [[] for _ in range(N_EXPERTS)]
            std_intensities_experts = [0] * N_EXPERTS

            # Iterate through both data loaders
            for batch in train_loader:
                gen_losses, disc_losses, router_loss, div_loss, intensity_loss, \
                aux_reg_loss, n_chosen_experts_batch, std_intensities_experts_batch, \
                mean_intensities_experts_batch, expert_distribution_loss, differentiation_loss,\
                expert_entropy_loss, gan_loss, alb_loss = train_step(batch, epoch)


                print("--------------n chocen in batch----------------")
                print(n_chosen_experts_batch)
                print(n_chosen_experts_batch.shape)
                gen_losses_epoch.append(gen_losses)
                disc_losses_epoch.append(disc_losses)
                router_loss_epoch.append(router_loss)
                div_loss_epoch.append(div_loss)
                intensity_loss_epoch.append(intensity_loss)
                aux_reg_loss_epoch.append(aux_reg_loss)
                expert_distribution_loss_epoch.append(expert_distribution_loss)
                differentiation_loss_epoch.append(differentiation_loss)
                expert_entropy_loss_epoch.append(expert_entropy_loss)
                alb_loss_epoch.append(alb_loss)
                gan_loss_epoch.append(gan_loss)

                for i in range(N_EXPERTS):
                    n_chosen_experts[i].append(n_chosen_experts_batch[i])
                    mean_intensities_experts[i].append(mean_intensities_experts_batch[i])
                    std_intensities_experts[i] = np.mean(std_intensities_experts_batch[i])

            print("================================n_chosen_experts================================")
            print(len(n_chosen_experts[0]))
            print(len(n_chosen_experts[1]))
            print(len(n_chosen_experts[2]))

            print(np.mean(n_chosen_experts[0]))
            print(np.mean(n_chosen_experts[1]))
            print(np.mean(n_chosen_experts[2]))
            print('+++++++++++++++++++')
            # when router stops training, choose the expert with the highest mean intensity
            # if epoch == STOP_ROUTER_TRAINING_EPOCH:
            #     mean_intensities_experts = [np.mean(mean_intensities_experts_0),
            #                                 np.mean(mean_intensities_experts_1),
            #                                 np.mean(mean_intensities_experts_2),
            #                                 np.mean(mean_intensities_experts_3)]
            #     print(mean_intensities_experts)
            #     index_of_max = mean_intensities_experts.index(max(mean_intensities_experts))
            #     # set the intensity strength of that expert to lower value
            #     generators[index_of_max].in_strength = IN_STRENGTH_LOWER_VAL
            #     print(f'Intensity strength of expert {index_of_max} set to lower value: {IN_STRENGTH_LOWER_VAL}')

            #
            # TEST GENERATION
            #

            # ws_values = []
            # for batch in test_loader:
            #     real_images, real_images_2, cond, std, intensity, true_positions, _ = batch
            #     # use router to calculate choose the expert assignments for samples
            #     cond = cond.to(device)
            #     # std = std.to(device)
            #     # intensity = intensity.to(device)
            #     # true_positions = true_positions.to(device)
            y_test_temp = torch.tensor(y_test, device=device)

            # Test Router Network
            with torch.no_grad():
                router_network.eval()
                predicted_expert_one_hot = router_network(y_test_temp)

            _, predicted_expert = torch.max(predicted_expert_one_hot, 1)

            indices_experts = [np.where(predicted_expert.cpu().numpy() == i)[0] for i in range(N_EXPERTS)]

            # Process expert indices dynamically
            ch_org_experts = []
            for i in range(N_EXPERTS):
                if len(indices_experts[i]) > 0:
                    org = np.exp(x_test[indices_experts[i]]) - 1
                    ch_org_exp = np.array(org).reshape(-1, *INPUT_IMAGE_SHAPE)
                    del org
                    ch_org_exp = pd.DataFrame(sum_channels_parallel(ch_org_exp)).values
                else:
                    ch_org_exp = np.zeros((len(indices_experts[i]), 5))
                ch_org_experts.append(ch_org_exp)

            y_test_experts = [y_test[indices_experts[i]] for i in range(N_EXPERTS)]
            # Calculate WS distance across all distribution
            ws_mean, ws_std, ws_mean_exp, ws_std_exp = calculate_joint_ws_across_experts(
                min(epoch // 5 + 1, 5),
                [x_test[indices_experts[i]] for i in range(N_EXPERTS)],
                y_test_experts,
                generators, ch_org,
                ch_org_experts,
                NOISE_DIM, device,
                n_experts=N_EXPERTS)

            # Generate Plots
            plot_experts = [None] * N_EXPERTS
            IDX_GENERATE = [1, 2, 3, 4, 5, 6]
            noise_cond_experts = [
                y_test[indices_experts[i]][IDX_GENERATE]
                if len(y_test[indices_experts[i]]) > len(IDX_GENERATE)
                else None
                for i in range(N_EXPERTS)
            ]
            epoch_time = time.time() - start

            # Log to WandB tool
            log_data = {
                'ws_mean': ws_mean,
                'ws_std': ws_std,
                'div_loss': np.mean(div_loss_epoch),
                'intensity_loss': np.mean(intensity_loss_epoch),
                'router_loss': np.mean(router_loss_epoch),
                'expert_distribution_loss': np.mean(expert_distribution_loss_epoch),
                'differentiation_loss': np.mean(differentiation_loss_epoch),
                'expert_entropy_loss': np.mean(expert_entropy_loss_epoch),
                'adaptive_load_balancing_loss': np.mean(alb_loss_epoch),
                'gan_loss': np.mean(gan_loss_epoch),
                'aux_reg_loss': np.mean(aux_reg_loss_epoch),
                'epoch_time': epoch_time,
                'epoch': epoch
            }
            gen_losses_epoch = np.mean(np.array(gen_losses_epoch), axis=0)
            disc_losses_epoch = np.mean(np.array(disc_losses_epoch), axis=0)
            for i in range(N_EXPERTS):
                log_data[f"ws_mean_{i}"] = ws_mean_exp[i]
                log_data[f"ws_std_{i}"] = ws_std_exp[i]
                log_data[f"gen_loss_{i}"] = gen_losses_epoch[i]
                log_data[f"disc_loss_{i}"] = disc_losses_epoch[i]
                log_data[f"std_intensities_experts_{i}"] = np.mean(std_intensities_experts[i])
                log_data[f"mean_intensities_experts_{i}"] = np.mean(mean_intensities_experts[i])
                log_data[f"n_choosen_experts_mean_epoch_{i}"] = np.mean(n_chosen_experts[i])
                if PLOT_IMAGES:
                    gen_image_start = time.time()
                    noise = torch.randn(len(IDX_GENERATE), NOISE_DIM,
                                        device=device)  # same noise vector for each expert
                    for i in range(N_EXPERTS):
                        plot = generate_and_save_images(generators[i], epoch, noise, noise_cond_experts[i],
                                                        x_test[indices_experts[i]],
                                                        photon_sum_proton_min, photon_sum_proton_max, device,
                                                        f'Expert {i}')
                        plot_experts[i] = plot

                    log_data[f"plot_expert_{i}"] = wandb.Image(plot_experts[i]) if not plot_experts[i] is None else None
                    log_data[f"generate_images_time"] = time.time() - gen_image_start

            # Project data to TSNE
            # cond_projections = plot_cond_pca_tsne(y_test, indices_experts, epochs)
            # log_data[f"cond_projection"] = wandb.Image(cond_projections)

            # Plot the cond expert specialization
            # cond_expert_specialization = plot_expert_heatmap(y_test, indices_experts, epoch, data_cond_names)
            # log_data[f"cond_expert_specialization"] = wandb.Image(cond_expert_specialization)

            wandb.log(log_data)

            # save models if all generators pass threshold
            if ws_mean < WS_MEAN_SAVE_THRESHOLD:
                save_models_and_architectures(filepath_models, N_EXPERTS, aux_regs, aux_reg_optimizers,
                                              generators, generator_optimizers, discriminators,
                                              discriminator_optimizers,
                                              router_network, router_optimizer, epoch)

            print(f'Time for epoch {epoch} is {epoch_time:.2f} sec')


    config_wandb = {
        "Model": NAME,
        "dataset": "proton_data",
        "n_experts": N_EXPERTS,
        "epochs": EPOCHS,
        "Date": DATE_STR,
        "Proton_min": photon_sum_proton_min,
        "Proton_max": photon_sum_proton_max,
        "generator_architecture": generators[0].name,
        "discriminator_architecture": discriminators[0].name,
        'stop_router_training_epoch': STOP_ROUTER_TRAINING_EPOCH,
        # 'num_heads_router': N_HEADS,
        # 'hidden_dim_router': HIDDEN_DIM,
        "diversity_strength": DI_STRENGTH,
        "intensity_strength": IN_STRENGTH,
        "intensity_strength_after_router_stops": IN_STRENGTH_LOWER_VAL,
        "auxiliary_strength": AUX_STRENGTH,
        "Generator_strength": GEN_STRENGTH,
        "Utilization_strength": UTIL_STRENGTH,
        "differentiation_strength": DIFF_STRENGTH,
        "alb_strength": ALB_STRENGTH,
        "expert_distribution_strength": ED_STRENGTH,
        'clip_diff_loss_value': CLIP_DIFF_LOSS,
        "Learning rate_generator": LR_G,
        "Learning rate_discriminator": LR_D,
        "Learning rate_router": LR_R,
        "Learning rate_aux_reg": LR_A,
        "Experiment_dir_name": EXPERIMENT_DIR_NAME,
        "Batch_size": BATCH_SIZE,
        "Noise_dim": NOISE_DIM,
        "router_arch": router_network.name,
        "intensity_loss_type": "mae"
    }


    # Separate datasets for each expert
    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(x_train_2),
                                  torch.tensor(y_train), torch.tensor(std_train),
                                  torch.tensor(intensity_train), torch.tensor(positions_train))

    # limited_train_dataset = Subset(train_dataset, range(10000))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
                              # batch_sampler=StratifiedBatchSampler(expert_number_train,
                              #                                      batch_size=BATCH_SIZE))

    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(x_test_2),
                                 torch.tensor(y_test), torch.tensor(std_test),
                                 torch.tensor(intensity_test), torch.tensor(positions_test))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
                             # batch_sampler=StratifiedBatchSampler(expert_number_test,
                             #                                      batch_size=BATCH_SIZE))

    wandb.finish()
    run = wandb.init(
        project="Generative-DNN-for-CERN-Fast-Simulations",
        entity="bedkowski-patrick",
        name=wandb_run_name,
        config=config_wandb,
        tags=["stratified_batch_samples",
              "param_sweep",
              f"proton_min_{photon_sum_proton_min}",
              f"proton_max_{photon_sum_proton_max}",
              "sdi_gan_intensity"]
    )
    run.log_code(name="adaptive_load_balancing.py")

    history = train(train_loader, EPOCHS, y_test)
