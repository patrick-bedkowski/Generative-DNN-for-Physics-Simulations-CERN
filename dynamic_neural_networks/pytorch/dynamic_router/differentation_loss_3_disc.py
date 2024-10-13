# MOE APPEAOCH
# General training of experts: Learning rate is scaled according to number of samples in the batch

# Genrator architecture: sdigan-intensity-aux-reg-1-disc-3-experts

# ROUTER LOSS: generator's loss * Expert Distribution Regularization
# 1. Generator's Loss: Combined loss of all generators, focusing on how well they reconstruct the target output.
# 2. Expert Distribution Regularization:
#    This regularization term aims to enforce a balanced and effective task distribution among experts.
#    It is designed to minimize imbalances in how the router assigns tasks (or images) to different experts.
#    The regularization is calculated by considering the gating probabilities and the feature distances (e.g., sum of pixels),
#    encouraging similar tasks to be routed to the same expert and dissimilar tasks to different experts.
#    This results in improved specialization of experts and better overall model performance.
#
#    - Gating Probabilities: The likelihoods assigned by the router to each expert for each task.
#    - Feature Distances: Measures such as the Euclidean distance between features (e.g., sum of pixels),
#      used to determine the similarity between tasks.
#    - Regularization Loss: Encourages routing similar tasks to the same expert and different tasks to different experts,
#      promoting balanced expert utilization.
#    - Goal: Minimize the regularization loss to ensure that each expert specializes in a distinct subset of tasks,
#      thereby improving the diversity and effectiveness of the generated outputs.

# Modification: 20.08.24
# Added calculation of WS ofr the whole distribution. Each generator generates samples, then join all of them and calculate.

# Modification: 29.08.24
# Modified the calculation of the WS. Previously it used the selected indices of samples corresponding to the experts
# from the 1-5, 6-17, 18+ photons sums intervals. Now the samples are selected by the router.

# Modification: 03.10.24
# Added 3 discriminators approach


import time
import os
import wandb

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.optim.lr_scheduler as lr_scheduler

from utils import (sum_channels_parallel, calculate_ws_ch_proton_model,
                   calculate_joint_ws_across_experts,
                   create_dir, save_scales, evaluate_router,
                   intensity_regularization, sdi_gan_regularization,
                   generate_and_save_images,
                   calculate_expert_distribution_loss,
                   regressor_loss, calculate_expert_utilization_entropy,
                   StratifiedBatchSampler)

from models_pytorch import Generator, Discriminator, RouterNetwork, AuxReg

print(torch.cuda.is_available())
print(torch.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.autograd.set_detect_anomaly(True)

# SETTINGS & PARAMETERS
SAVE_EXPERIMENT_DATA = True
WS_MEAN_SAVE_THRESHOLD = 5
DI_STRENGTH = 0.1
IN_STRENGTH = 0.001
IN_STRENGTH_LOWER_VAL = 0.000001
AUX_STRENGTH = 0.001

# Router loss parameters
ED_STRENGTH = 0.01  # Strength on the expert distribution loss in the router loss calculation
GEN_STRENGTH = 0.1  # Strength on the generator loss in the router loss calculation
UTIL_STRENGTH = 0.0001  # Strength on the expert utilization entropy in the router loss calculation
DIFF_STRENGTH = 0.00001  # Differentation on the generator loss in the router loss calculation
STOP_ROUTER_TRAINING_EPOCH = 80

N_RUNS = 1
N_EXPERTS = 3
BATCH_SIZE = 256
NOISE_DIM = 10
N_COND = 9  # number of conditional features
EPOCHS = 250
LR_G = 1e-4
LR_D = 1e-5
LR_A = 1e-4
LR_R = 1e-3

# NAME = f"strat_batch_sampler_diff_expert_distribution_utilization_{ED_STRENGTH}_{GEN_STRENGTH}_{ENT_STRENGTH}"
NAME = f"5_proton_3_exp_diff_expert_distribution_utilization_{ED_STRENGTH}_{GEN_STRENGTH}_{UTIL_STRENGTH}_{STOP_ROUTER_TRAINING_EPOCH}"

for _ in range(N_RUNS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_photonsum_proton_1_2312.pkl')
    data_cond = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_cond_photonsum_proton_1_2312.pkl')
    photon_sum_proton_min, photon_sum_proton_max = data_cond.proton_photon_sum.min(), data_cond.proton_photon_sum.max()

    # data of coordinates of maximum value of pixel on the images
    data_posi = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_coord_proton_photonsum_proton_1_2312.pkl')
    print('Loaded positions: ', data_posi.shape, "max:", data_posi.values.max(), "min:", data_posi.values.min())

    DATE_STR = datetime.now().strftime("%d_%m_%Y_%H_%M")
    wandb_run_name = f"{NAME}_{LR_G}_{LR_D}_{LR_R}_{DATE_STR}"
    EXPERIMENT_DIR_NAME = f"experiments/{wandb_run_name}_{int(photon_sum_proton_min)}_{int(photon_sum_proton_max)}_{DATE_STR}"

    expert_number = data_cond.expert_number  # number 0,1,2

    # group conditional data
    data_cond["cond"] = data_cond["Energy"].astype(str) +"|"+ data_cond["Vx"].astype(str) +"|"+ data_cond["Vy"].astype(str) +"|"+ data_cond["Vz"].astype(str) +"|"+  data_cond["Px"].astype(str) +"|"+  data_cond["Py"].astype(str) +"|"+ data_cond["Pz"].astype(str) +"|"+  data_cond["mass"].astype(str) +"|"+  data_cond["charge"].astype(str)
    data_cond_id = data_cond[["cond"]].reset_index()
    ids = data_cond_id.merge(data_cond_id.sample(frac=1), on=["cond"], how="inner").groupby("index_x").first()
    ids = ids["index_y"]

    data = np.log(data + 1).astype(np.float32)

    data_2 = data[ids]
    data_cond = data_cond.drop(columns="cond")

    # Diversity regularization
    scaler = MinMaxScaler()
    std = data_cond["std_proton"].values.reshape(-1, 1)
    std = np.float32(std)
    std = scaler.fit_transform(std)
    print("std max", std.max(), "min", std.min())

    # Intensity regularization
    scaler_intensity = MinMaxScaler()
    intensity = data_cond["proton_photon_sum"].values.reshape(-1, 1)
    intensity = np.float32(intensity)
    intensity = scaler_intensity.fit_transform(intensity)
    print("intensity max", intensity.max(), "min", intensity.min())

    # Auxiliary regressor
    scaler_poz = StandardScaler()
    data_xy = np.float32(data_posi.copy()[["max_x", "max_y"]])
    data_xy = scaler_poz.fit_transform(data_xy)
    print('Load positions:', data_xy.shape, "cond max", data_xy.max(), "min", data_xy.min())

    scaler_cond = StandardScaler()
    data_cond = scaler_cond.fit_transform(data_cond.drop(columns=["std_proton", "proton_photon_sum",
                                                                  'group_number_proton',
                                                                  'expert_number'])).astype(np.float32)

    x_train, x_test, x_train_2, x_test_2, y_train, y_test, std_train, \
    std_test, intensity_train, intensity_test, positions_train, positions_test, \
    expert_number_train, expert_number_test = train_test_split(
        data, data_2, data_cond, std, intensity, data_xy, expert_number.values, test_size=0.2, shuffle=True)

    print("Data shapes:", x_train.shape, x_test.shape,
          y_train.shape, y_test.shape,
          expert_number_train.shape, expert_number_test.shape)

    # Save scales
    if SAVE_EXPERIMENT_DATA:
        filepath = f"{EXPERIMENT_DIR_NAME}/scales/"
        create_dir(filepath, SAVE_EXPERIMENT_DATA)
        save_scales("Proton", scaler_cond.mean_, scaler_cond.scale_, filepath)
        filepath_mod = f"{EXPERIMENT_DIR_NAME}/models/"
        create_dir(filepath_mod, SAVE_EXPERIMENT_DATA)

    # CALCULATE DISTRIBUTION OF CHANNELS IN ORIGINAL TEST DATA #
    org = np.exp(x_test) - 1
    ch_org = np.array(org).reshape(-1, 56, 30)
    del org
    ch_org = pd.DataFrame(sum_channels_parallel(ch_org)).values

    # Loss and optimizer
    binary_cross_entropy_criterion = nn.BCELoss()
    router_criterion = nn.CrossEntropyLoss()
    aux_reg_criterion = regressor_loss

    # Define experts
    generators = []
    for generator_idx in range(N_EXPERTS):
        generator = Generator(NOISE_DIM, N_COND, DI_STRENGTH, IN_STRENGTH).to(device)
        generators.append(generator)

    generator_optimizers = [optim.Adam(gen.parameters(), lr=LR_G) for gen in generators]

    # Define discriminators
    discriminators = []
    for generator_idx in range(N_EXPERTS):
        discriminators.append(Discriminator(N_COND).to(device))
    discriminator_optimizers = [optim.Adam(disc.parameters(), lr=LR_D) for disc in discriminators]

    router_network = RouterNetwork(N_COND, N_EXPERTS).to(device)
    router_optimizer = optim.Adam(router_network.parameters(), lr=LR_R)
    # Define the learning rate scheduler
    #router_scheduler = lr_scheduler.ReduceLROnPlateau(router_optimizer, mode='min', patience=3, factor=0.1, verbose=True)

    # Define Auxiliary regressor
    aux_reg = AuxReg().to(device)
    aux_reg_optimizer = optim.Adam(aux_reg.parameters(), lr=LR_A)

    def generator_train_step(gene, disc, a_reg, cond, g_optimizer, a_optimizer, criterion,
                             true_positions, std, intensity, class_counts, BATCH_SIZE):
        # Train Generator
        g_optimizer.zero_grad()

        noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
        noise.requires_grad = False
        noise_2 = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
        noise_2.requires_grad = False

        # generate fake images
        fake_images = gene(noise, cond)
        fake_images_2 = gene(noise_2, cond)

        # validate two images
        fake_output, fake_latent = disc(fake_images, cond)
        fake_output_2, fake_latent_2 = disc(fake_images_2, cond)

        gen_loss = criterion(fake_output, torch.ones_like(fake_output))

        div_loss = sdi_gan_regularization(fake_latent, fake_latent_2,
                                          noise, noise_2,
                                          std, gene.di_strength)

        intensity_loss, mean_intenisties, std_intensity, mean_intensity = intensity_regularization(fake_images,
                                                                                                   intensity,
                                                                                                   scaler_intensity,
                                                                                                   gene.in_strength)

        gen_loss = gen_loss + div_loss + intensity_loss

        # Train auxiliary regressor
        aux_reg_optimizer.zero_grad()
        generated_positions = a_reg(fake_images)

        aux_reg_loss = aux_reg_criterion(true_positions, generated_positions, scaler_poz, AUX_STRENGTH)

        gen_loss += aux_reg_loss

        gen_loss.backward()
        g_optimizer.param_groups[0]['lr'] = LR_G * class_counts.clone().detach()
        g_optimizer.step()
        a_optimizer.step()

        return gen_loss.data, div_loss.data, intensity_loss.data, aux_reg_loss.data, std_intensity, mean_intensity,\
               mean_intenisties


    def discriminator_train_step(disc, generator, d_optimizer, criterion, real_images, cond, BATCH_SIZE):
        # Train discriminator
        d_optimizer.zero_grad()

        real_output, real_latent = disc(real_images, cond)

        # calculate loss for real images
        real_labels = torch.ones_like(real_output)
        loss_real_disc = criterion(real_output, real_labels)

        noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
        fake_images = generator(noise, cond)
        fake_output, fake_latent = disc(fake_images, cond)
        loss_fake_disc = criterion(fake_output, torch.zeros_like(fake_output))
        disc_loss = loss_real_disc + loss_fake_disc

        # Accumulate and compute discriminator loss outside the loop
        disc_loss.backward()
        d_optimizer.step()

        return disc_loss.data

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
        gen_losses = torch.zeros(N_EXPERTS)
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
            selected_cond = cond[selected_indices].clone()
            selected_generator = generators[i]
            selected_discriminator = discriminators[i]
            selected_discriminator_optimizer = discriminator_optimizers[i]
            selected_real_images = real_images[selected_indices].clone()

            disc_loss = discriminator_train_step(selected_discriminator, selected_generator,
                                                 selected_discriminator_optimizer,
                                                 binary_cross_entropy_criterion, selected_real_images,
                                                 selected_cond, BATCH_SIZE)
            disc_losses[i] = disc_loss

        disc_loss = disc_losses.sum()

        #
        # Train each generator
        #
        for i in range(N_EXPERTS):
            selected_indices = (predicted_expert == i).nonzero(as_tuple=True)[0]
            if selected_indices.numel() <= 1:
                gen_losses[i] = torch.tensor(0.0, requires_grad=True).to(device)
                continue

            # Clone or detach tensors to avoid in-place modifications
            selected_cond = cond[selected_indices].clone()
            selected_true_positions = true_positions[selected_indices].clone()
            selected_intensity = intensity[selected_indices].clone()
            selected_std = std[selected_indices].clone()
            selected_generator = generators[i]
            selected_discriminator = discriminators[i]
            selected_class_counts = class_counts_adjusted[i]

            gen_loss, div_loss, intensity_loss, \
            aux_reg_loss, std_intensity, mean_intensity, mean_intensities = generator_train_step(selected_generator,
                                                                                                 selected_discriminator,
                                                                                                 aux_reg,
                                                                                                 selected_cond,
                                                                                                 generator_optimizers[
                                                                                                     i],
                                                                                                 aux_reg_optimizer,
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
            gen_losses[i] = gen_loss.clone().detach()  # do the detach so when calculating loss for the router, the gradient doesnt flow back to the generators

        #
        # Calculate router loss
        #
        gen_loss_scaled = gen_losses.mean() * GEN_STRENGTH
        expert_entropy_loss = calculate_expert_utilization_entropy(predicted_expert_one_hot.clone(), UTIL_STRENGTH)
        expert_distribution_loss = calculate_expert_distribution_loss(predicted_expert_one_hot.clone(),
                                                                      mean_intensities_in_batch_expert.reshape(-1, 1),
                                                                      ED_STRENGTH)

        differentiation_loss = - (mean_intensities_experts[0] - mean_intensities_experts[1]) ** 2 \
                               - (mean_intensities_experts[0] - mean_intensities_experts[2]) ** 2 \
                               - (mean_intensities_experts[1] - mean_intensities_experts[2]) ** 2

        differentiation_loss = differentiation_loss * DIFF_STRENGTH
        router_loss = gen_loss_scaled + expert_distribution_loss - expert_entropy_loss + differentiation_loss

        if epoch < STOP_ROUTER_TRAINING_EPOCH:
            # Train Router Network
            router_loss.backward()
            router_optimizer.step()
        else:
            router_loss = torch.tensor(0.0)

        gen_losses = [gen_loss.item() for gen_loss in gen_losses]
        return gen_losses, \
               disc_loss.item(), router_loss.item(), div_loss.cpu().item(), \
               intensity_loss.cpu().item(), aux_reg_loss.cpu().item(), class_counts.cpu().detach(), \
               std_intensities_experts, mean_intensities_experts, expert_distribution_loss.item(), \
               differentiation_loss.item(), expert_entropy_loss.item()

    # Training loop
    def train(train_loader, epochs, y_test):
        history = []
        for epoch in range(epochs):
            start = time.time()
            gen_losses_epoch = []
            disc_loss_epoch = []
            router_loss_epoch = []
            div_loss_epoch = []
            intensity_loss_epoch = []
            aux_reg_loss_epoch = []
            expert_distribution_loss_epoch = []
            differentiation_loss_epoch = []
            expert_entropy_loss_epoch = []
            n_chosen_experts_0 = []
            n_chosen_experts_1 = []
            n_chosen_experts_2 = []
            std_intensities_experts_0 = []
            std_intensities_experts_1 = []
            std_intensities_experts_2 = []
            mean_intensities_experts_0 = []
            mean_intensities_experts_1 = []
            mean_intensities_experts_2 = []

            if epoch == STOP_ROUTER_TRAINING_EPOCH:
                # Save the model's state_dict
                router_filename = f"router_network_epoch_{STOP_ROUTER_TRAINING_EPOCH}.pth"
                filepath_router = os.path.join(filepath_mod, router_filename)
                torch.save(router_network.state_dict(), filepath_router)

            # Iterate through both data loaders
            for batch in train_loader:
                gen_losses, disc_loss, router_loss, div_loss, intensity_loss, \
                aux_reg_loss, n_chosen_experts_batch, std_intensities_experts, \
                mean_intensities_experts, expert_distribution_loss, differentiation_loss,\
                expert_entropy_loss = train_step(batch, epoch)

                gen_losses_epoch.append(gen_losses)
                disc_loss_epoch.append(disc_loss)
                router_loss_epoch.append(router_loss)
                div_loss_epoch.append(div_loss)
                intensity_loss_epoch.append(intensity_loss)
                aux_reg_loss_epoch.append(aux_reg_loss)
                n_chosen_experts_0.append(n_chosen_experts_batch[0])
                n_chosen_experts_1.append(n_chosen_experts_batch[1])
                n_chosen_experts_2.append(n_chosen_experts_batch[2])
                mean_intensities_experts_0.append(mean_intensities_experts[0])  # [n_experts, BATCH_SIZE]
                mean_intensities_experts_1.append(mean_intensities_experts[1])  # [n_experts, BATCH_SIZE]
                mean_intensities_experts_2.append(mean_intensities_experts[2])  # [n_experts, BATCH_SIZE]
                std_intensities_experts_0 = np.sqrt(np.mean(std_intensities_experts[0] ** 2))
                std_intensities_experts_1 = np.sqrt(np.mean(std_intensities_experts[1] ** 2))
                std_intensities_experts_2 = np.sqrt(np.mean(std_intensities_experts[2] ** 2))
                expert_distribution_loss_epoch.append(expert_distribution_loss)
                differentiation_loss_epoch.append(differentiation_loss)
                expert_entropy_loss_epoch.append(expert_entropy_loss)

                # disc_loss_epoch.append((disc_loss_0 + disc_loss_1 + disc_loss_2) / NUM_GENERATORS)
                # router_loss_epoch.append((router_loss_0 + router_loss_1 + router_loss_2) / NUM_GENERATORS)
                # div_loss_epoch.append((div_loss_0 + div_loss_1 + div_loss_2) / NUM_GENERATORS)
                # intensity_loss_epoch.append((intensity_loss_0 + intensity_loss_1 + intensity_loss_2) / NUM_GENERATORS)
                # aux_reg_loss_epoch.append((aux_reg_loss_0 + aux_reg_loss_1 + aux_reg_loss_2) / NUM_GENERATORS)

            # when router stops training, choose the expert with the highest mean intensity
            if epoch == STOP_ROUTER_TRAINING_EPOCH:
                mean_intensities_experts = [np.mean(mean_intensities_experts_0),
                                            np.mean(mean_intensities_experts_1),
                                            np.mean(mean_intensities_experts_2)]
                print(mean_intensities_experts)
                index_of_max = mean_intensities_experts.index(max(mean_intensities_experts))
                # set the intensity strength of that expert to lower value
                generators[index_of_max].in_strength = IN_STRENGTH_LOWER_VAL
                print(f'Intensity strength of expert {index_of_max} set to lower value: {IN_STRENGTH_LOWER_VAL}')

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

            indices_expert_0 = np.where(predicted_expert.cpu().numpy() == 0)[0]
            indices_expert_1 = np.where(predicted_expert.cpu().numpy() == 1)[0]
            indices_expert_2 = np.where(predicted_expert.cpu().numpy() == 2)[0]
            print(len(indices_expert_0)+len(indices_expert_1)+len(indices_expert_2))

            # calculate the ch_org for the test dataset for each expert
            org_0 = np.exp(x_test[indices_expert_0]) - 1
            ch_org_0 = np.array(org_0).reshape(-1, 56, 30)
            del org_0
            ch_org_0 = pd.DataFrame(sum_channels_parallel(ch_org_0)).values

            org_1 = np.exp(x_test[indices_expert_1]) - 1
            ch_org_1 = np.array(org_1).reshape(-1, 56, 30)
            del org_1
            ch_org_1 = pd.DataFrame(sum_channels_parallel(ch_org_1)).values

            org_2 = np.exp(x_test[indices_expert_2]) - 1
            ch_org_2 = np.array(org_2).reshape(-1, 56, 30)
            del org_2
            ch_org_2 = pd.DataFrame(sum_channels_parallel(ch_org_2)).values

            # Calculate WS distance across all distribution
            ws_mean, ws_mean_0, ws_mean_1, ws_mean_2 = calculate_joint_ws_across_experts(min(epoch // 5 + 1, 5),
                                                                                         [x_test[indices_expert_0],
                                                                                          x_test[indices_expert_1],
                                                                                          x_test[indices_expert_2]],
                                                                                         [y_test[indices_expert_0],
                                                                                          y_test[indices_expert_1],
                                                                                          y_test[indices_expert_2]],
                                                                                         generators, ch_org,
                                                                                         [ch_org_0, ch_org_1, ch_org_2],
                                                                                         NOISE_DIM, device, n_experts=N_EXPERTS)

            # Generate Plots
            IDX_GENERATE = [1, 2, 3, 4, 5, 6]

            start_gen_images = time.time()
            noise_cond_0 = y_test[indices_expert_0][IDX_GENERATE]
            noise_cond_1 = y_test[indices_expert_1][IDX_GENERATE]
            noise_cond_2 = y_test[indices_expert_2][IDX_GENERATE]
            noise = torch.randn(len(IDX_GENERATE), NOISE_DIM, device=device)  # same noise vector for each expert

            plot_0 = generate_and_save_images(generators[0], epoch, noise, noise_cond_0, x_test[indices_expert_0],
                                              photon_sum_proton_min, photon_sum_proton_max, device, 'Expert 0')
            plot_1 = generate_and_save_images(generators[1], epoch, noise, noise_cond_1, x_test[indices_expert_1],
                                              photon_sum_proton_min, photon_sum_proton_max, device, 'Expert 1')
            plot_2 = generate_and_save_images(generators[2], epoch, noise, noise_cond_2, x_test[indices_expert_2],
                                              photon_sum_proton_min, photon_sum_proton_max, device, 'Expert 2')
            end_gen_images = time.time() - start_gen_images
            print(f'Time for generating images: {end_gen_images:.2f} sec')

            epoch_time = time.time() - start
            #router_scheduler.step(ws_mean)

            # Log to WandB tool
            log_data = {
                'ws_mean': ws_mean,
                'ws_mean_0': ws_mean_0,
                'ws_mean_1': ws_mean_1,
                'ws_mean_2': ws_mean_2,
                'gen_loss_0': np.mean([gen_loss[0] for gen_loss in gen_losses_epoch]),
                'gen_loss_1': np.mean([gen_loss[1] for gen_loss in gen_losses_epoch]),
                'gen_loss_2': np.mean([gen_loss[2] for gen_loss in gen_losses_epoch]),
                'div_loss': np.mean(div_loss_epoch),
                'intensity_loss': np.mean(intensity_loss_epoch),
                'router_loss': np.mean(router_loss_epoch),
                'std_intensities_experts_0': np.mean(std_intensities_experts_0),
                'std_intensities_experts_1': np.mean(std_intensities_experts_1),
                'std_intensities_experts_2': np.mean(std_intensities_experts_2),
                'mean_intensities_experts_0': np.mean(mean_intensities_experts_0),
                'mean_intensities_experts_1': np.mean(mean_intensities_experts_1),
                'mean_intensities_experts_2': np.mean(mean_intensities_experts_2),
                'expert_distribution_loss': np.mean(expert_distribution_loss_epoch),
                'differentiation_loss': np.mean(differentiation_loss_epoch),
                'expert_entropy_loss': np.mean(expert_entropy_loss_epoch),
                'disc_loss': np.mean(disc_loss_epoch),
                'aux_reg_loss': np.mean(aux_reg_loss_epoch),
                'n_choosen_experts_mean_epoch_0': np.mean(n_chosen_experts_0),
                'n_choosen_experts_mean_epoch_1': np.mean(n_chosen_experts_1),
                'n_choosen_experts_mean_epoch_2': np.mean(n_chosen_experts_2),
                'epoch_time': epoch_time,
                'epoch': epoch,
                'plot_expert_0': wandb.Image(plot_0),
                'plot_expert_1': wandb.Image(plot_1),
                'plot_expert_2': wandb.Image(plot_2)
            }

            wandb.log(log_data)

            # save models if all generators pass threshold
            if ws_mean < WS_MEAN_SAVE_THRESHOLD:
                for i, generator in enumerate(generators):
                    torch.save(generator.state_dict(), os.path.join(filepath_mod, "gen_" + str(i) + "_" + str(epoch) + ".h5"))
                    torch.save(discriminators[i].state_dict(), os.path.join(filepath_mod, "disc_" + str(i) + "_" + str(epoch) + ".h5"))

            print(f'Time for epoch {epoch} is {epoch_time:.2f} sec')


    config_wandb = {
        "Model": NAME,
        "dataset": "proton_data",
        "epochs": EPOCHS,
        "Date": DATE_STR,
        "Proton_min": photon_sum_proton_min,
        "Proton_max": photon_sum_proton_max,
        "generator_architecture": generators[0].name,
        "discriminator_architecture": discriminators[0].name,
        'stop_router_training_epoch': STOP_ROUTER_TRAINING_EPOCH,
        "diversity_strength": DI_STRENGTH,
        "intensity_strength": IN_STRENGTH,
        "intensity_strength_after_router_stops": IN_STRENGTH_LOWER_VAL,
        "auxiliary_strength": AUX_STRENGTH,
        "Generator_strength": GEN_STRENGTH,
        "utilization_strength": UTIL_STRENGTH,
        "differentiation_strength": DIFF_STRENGTH,
        "expert_distribution_strength": ED_STRENGTH,
        "Learning rate_generator": LR_G,
        "Learning rate_discriminator": LR_D,
        "Learning rate_router": LR_R,
        "Learning rate_aux_reg": LR_A,
        "Experiment_dir_name": EXPERIMENT_DIR_NAME,
        "Batch_size": BATCH_SIZE,
        "Noise_dim": NOISE_DIM,
        "router_arch": "128-64-32-16",
        "intensity_loss_type": "mae"
    }


    # Separate datasets for each expert
    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(x_train_2),
                                  torch.tensor(y_train), torch.tensor(std_train),
                                  torch.tensor(intensity_train), torch.tensor(positions_train))
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
    wandb.init(
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

    history = train(train_loader, EPOCHS, y_test)
