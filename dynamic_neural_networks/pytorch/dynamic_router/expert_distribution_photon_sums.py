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

# Modification: 27.08.24
# modified architecture of router from
# 128-64-32-3 to 256-128-64-32-3
# and LR_R from 1e-3 to 5e-4

# Modification: 29.08.24
# Modified the calculation of the WS. Previously it used the selected indices of samples corresponding to the experts
# from the 1-5, 6-17, 18+ photons sums intervals. Now the samples are selected by the router.


import time
import os
import wandb

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils import (sum_channels_parallel, calculate_ws_ch_proton_model,
                   calculate_joint_ws_across_experts,
                   create_dir, save_scales, evaluate_router,
                   intensity_regularization, sdi_gan_regularization,
                   generate_and_save_images,
                   calculate_expert_distribution_loss)
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score


print(torch.cuda.is_available())
print(torch.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.autograd.set_detect_anomaly(True)

# SETTINGS & PARAMETERS
SAVE_EXPERIMENT_DATA = True
WS_MEAN_SAVE_THRESHOLD = 5
DI_STRENGTH = 0.1
IN_STRENGTH = 0.001
AUX_STRENGTH = 0.1
ED_STRENGTH = 1e0  # Strength on the expert distribution loss in the router loss calculation

GEN_STRENGTH = 1e-1  # Strength on the generator loss in the router loss calculation
IN_ENTROPY_STRENGTH = 1e-1  # Strength on the intraentropy of photon sums in the router loss calculation
ENT_STRENGTH = 1e1  # Strength on the expert utilization entropy in the router loss calculation
N_RUNS = 2
N_EXPERTS = 3

for _ in range(N_RUNS):
    print("GEN_STRENGTH:", GEN_STRENGTH, "IN_ENTROPY_STRENGTH:", IN_ENTROPY_STRENGTH)
    BATCH_SIZE = 256
    NOISE_DIM = 10
    EPOCHS = 200
    LR_G = 1e-4
    LR_D = 1e-5
    LR_R = 5e-4
    LR_A = 5e-5
    NAME = f"expert_distribution_loss_test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_proton_photonsum_proton_1_2312.pkl')
    data_cond = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_cond_photonsum_proton_1_2312.pkl')
    photon_sum_proton_min, photon_sum_proton_max = data_cond.proton_photon_sum.min(), data_cond.proton_photon_sum.max()

    # data of coordinates of maximum value of pixel on the images
    data_posi = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_coord_proton_photonsum_proton_1_2312.pkl')
    print('Loaded positions: ', data_posi.shape, "max:", data_posi.values.max(), "min:", data_posi.values.min())

    DATE_STR = datetime.now().strftime("%d_%m_%Y_%H_%M")
    wandb_run_name = f"{NAME}_{LR_G}_{LR_D}_{LR_R}_{DATE_STR}"
    EXPERIMENT_DIR_NAME = f"experiments/{wandb_run_name}_{int(photon_sum_proton_min)}_{int(photon_sum_proton_max)}_{DATE_STR}"

    expert_number = data_cond.expert_number  # number 0,1,2
    print('Loaded expert number: ', expert_number.shape, "max:", expert_number.values.max(), "min:", expert_number.values.min())

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
    data_cond = scaler_cond.fit_transform(data_cond.drop(columns=["std_proton", "proton_photon_sum", 'group_number_proton',
                                                                  'expert_number'])).astype(np.float32)

    x_train, x_test, x_train_2, x_test_2, y_train, y_test, std_train, std_test,\
    intensity_train, intensity_test, positions_train, positions_test, expert_number_train, expert_number_test = train_test_split(
        data, data_2, data_cond, std, intensity, data_xy, expert_number.values, test_size=0.2, shuffle=False)

    print("Data shapes:", x_train.shape, x_test.shape, y_train.shape, y_test.shape,
          expert_number_train.shape, expert_number_test.shape)

    # Save scales
    if SAVE_EXPERIMENT_DATA:
        filepath = f"{EXPERIMENT_DIR_NAME}/scales/"
        create_dir(filepath, SAVE_EXPERIMENT_DATA)
        save_scales("Proton", scaler_cond.mean_, scaler_cond.scale_, filepath)

    if SAVE_EXPERIMENT_DATA:
        filepath_mod = f"{EXPERIMENT_DIR_NAME}/models/"
        create_dir(filepath_mod, SAVE_EXPERIMENT_DATA)

    # CALCULATE DISTRIBUTION OF CHANNELS IN ORIGINAL TEST DATA #
    org = np.exp(x_test) - 1
    ch_org = np.array(org).reshape(-1, 56, 30)
    del org

    ch_org = pd.DataFrame(sum_channels_parallel(ch_org)).values


    # Define the Generator model
    class Generator(nn.Module):
        def __init__(self, noise_dim, cond_dim):
            super(Generator, self).__init__()
            self.fc1 = nn.Sequential(
                nn.Linear(noise_dim + cond_dim, 256),  # This should be 19 (10 + 9)
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2)
            )
            self.fc2 = nn.Sequential(
                nn.Linear(256, 128 * 20 * 10),
                nn.BatchNorm1d(128 * 20 * 10),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2)
            )
            self.upsample = nn.Upsample(scale_factor=(3, 2))
            self.conv_layers = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=(2, 2)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2),
                nn.Upsample(scale_factor=(1, 2)),
                nn.Conv2d(256, 128, kernel_size=(2, 2)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2),
                nn.Conv2d(128, 64, kernel_size=(2, 2)),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2),
                nn.Conv2d(64, 1, kernel_size=(2, 7)),
                nn.ReLU()
            )

        def forward(self, noise, cond):
            x = torch.cat((noise, cond), dim=1)
            x = self.fc1(x)
            x = self.fc2(x)
            x = x.view(-1, 128, 20, 10)
            x = self.upsample(x)
            x = self.conv_layers(x)
            return x


    # Define the Discriminator model
    class Discriminator(nn.Module):
        def __init__(self, cond_dim, num_generators):
            super(Discriminator, self).__init__()
            self.num_generators = num_generators
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3, 3)),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Conv2d(32, 16, kernel_size=(3, 3)),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2),
                nn.MaxPool2d(kernel_size=(2, 1))
            )
            self.fc1 = nn.Sequential(
                nn.Linear(16 * 12 * 12 + cond_dim, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2)
            )
            self.fc2 = nn.Sequential(
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2)
            )
            self.fc3 = nn.Linear(64, self.num_generators)
            self.sigmoid = nn.Sigmoid()

        def forward(self, img, cond):
            x = self.conv_layers(img)
            x = x.view(x.size(0), -1)
            x = torch.cat((x, cond), dim=1)
            x = self.fc1(x)
            latent = self.fc2(x)
            out = self.fc3(latent)
            out = self.sigmoid(out)
            return out, latent

    # Define the Router Network
    class RouterNetwork(nn.Module):
        def __init__(self, cond_dim):
            super(RouterNetwork, self).__init__()
            self.fc_layers = nn.Sequential(
                nn.Linear(cond_dim, 128),
                nn.LeakyReLU(0.1),
                nn.Linear(128, 64),
                nn.LeakyReLU(0.1),
                nn.Linear(64, 32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 3),
                nn.Softmax(dim=1)
            )

        def forward(self, cond):
            return self.fc_layers(cond)


    class AuxReg(nn.Module):
        def __init__(self):
            super(AuxReg, self).__init__()

            self.conv3 = nn.Conv2d(1, 32, kernel_size=3)
            self.conv3_bd = nn.Sequential(
                nn.BatchNorm2d(32),
                nn.Dropout(0.2),
                nn.LeakyReLU(0.1)
            )
            self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

            self.conv4 = nn.Conv2d(32, 64, kernel_size=3)
            self.conv4_bd = nn.Sequential(
                nn.BatchNorm2d(64),
                nn.Dropout(0.2),
                nn.LeakyReLU(0.1)
            )
            self.pool4 = nn.MaxPool2d(kernel_size=(2, 1))

            self.conv5 = nn.Conv2d(64, 128, kernel_size=3)
            self.conv5_bd = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Dropout(0.2),
                nn.LeakyReLU(0.1)
            )
            self.pool5 = nn.MaxPool2d(kernel_size=(2, 1))

            self.conv6 = nn.Conv2d(128, 256, kernel_size=3)
            self.conv6_bd = nn.Sequential(
                nn.BatchNorm2d(256),
                nn.Dropout(0.2),
                nn.LeakyReLU(0.1)
            )

            self.flatten = nn.Flatten()
            self.dense = nn.Linear(256 * 3 * 8, 2)  # Adjust the input dimension based on your input image size

        def forward(self, x):
            x = self.conv3(x)
            x = self.conv3_bd(x)
            x = self.pool3(x)

            x = self.conv4(x)
            x = self.conv4_bd(x)
            x = self.pool4(x)

            x = self.conv5(x)
            x = self.conv5_bd(x)
            x = self.pool5(x)

            x = self.conv6(x)
            x = self.conv6_bd(x)

            x = self.flatten(x)
            x = self.dense(x)

            return x

    # Define the loss function
    def regressor_loss(real_coords, fake_coords, AUX_STRENGTH):
        # Move fake_coords to CPU and convert to NumPy array
        fake_coords_cpu = fake_coords.cpu().detach().numpy()

        # Transform using the scaler
        fake_coords_scaled = scaler_poz.transform(fake_coords_cpu)

        # Convert back to tensor and move to the appropriate device
        fake_coords_scaled = torch.tensor(fake_coords_scaled).to(real_coords.device)

        # Compute the MAE loss
        return F.l1_loss(fake_coords_scaled, real_coords) * AUX_STRENGTH


    # Loss and optimizer
    generator_criterion = nn.BCELoss()
    discriminator_criterion = nn.BCELoss()
    router_criterion = nn.CrossEntropyLoss()
    aux_reg_criterion = regressor_loss
    generator0 = Generator(NOISE_DIM, 9).to(device)
    generator1 = Generator(NOISE_DIM, 9).to(device)
    generator2 = Generator(NOISE_DIM, 9).to(device)
    router_network = RouterNetwork(9).to(device)

    generators = [generator0, generator1, generator2]
    generator_optimizers = [optim.Adam(gen.parameters(), lr=LR_G) for gen in generators]

    NUM_GENERATORS = len(generators)
    # Initialize single discriminator
    discriminator = Discriminator(9, NUM_GENERATORS).to(device)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LR_D)

    router_optimizer = optim.Adam(router_network.parameters(), lr=LR_R)

    # Example usage
    aux_reg = AuxReg().to(device)
    aux_reg_optimizer = optim.Adam(aux_reg.parameters(), lr=LR_A)

    # Initialize metrics
    accuracy_metric = MulticlassAccuracy(num_classes=NUM_GENERATORS).to(device)
    precision_metric = MulticlassPrecision(num_classes=NUM_GENERATORS, average='macro').to(device)
    recall_metric = MulticlassRecall(num_classes=NUM_GENERATORS, average='macro').to(device)
    f1_metric = MulticlassF1Score(num_classes=NUM_GENERATORS, average='macro').to(device)


    def calculate_entropy(p):
        """
        Calculate entropy of a probability distribution p.
        """
        return -torch.sum(p * torch.log(p + 1e-9), dim=-1)


    def calculate_expert_utilization_entropy(gating_probs):
        """
        Calculate the expert utilization entropy H_u.

        Parameters:
        gating_probs (torch.Tensor): A tensor of shape (N, M) where N is the number of samples
                                     and M is the number of experts. Each entry is the gating
                                     probability of an expert for a given sample.

        Returns:
        torch.Tensor: The entropy of the average gating probabilities.
        """
        avg_gating_probs = torch.mean(gating_probs, dim=0)  # Average over samples
        entropy = calculate_entropy(avg_gating_probs)
        return entropy


    # Adjust train_step function to work with the single discriminator
    def train_step(batch):
        real_images, real_images_2, cond, std, intensity, true_positions, _ = batch

        real_images = real_images.unsqueeze(1).to(device)
        # real_images_2 = real_images_2.unsqueeze(1).to(device)
        cond = cond.to(device)
        std = std.to(device)
        intensity = intensity.to(device)
        true_positions = true_positions.to(device)

        # Train Router Network
        router_optimizer.zero_grad()
        predicted_expert_one_hot = router_network(cond)

        expert_utilization_entropy = calculate_expert_utilization_entropy(predicted_expert_one_hot)

        _, predicted_expert = torch.max(predicted_expert_one_hot, 1)  # (BATCH_SIZE, N_EXPERTS)
        class_counts = torch.zeros(NUM_GENERATORS, dtype=torch.float).to(device)
        for class_label in range(NUM_GENERATORS):
            class_counts[class_label] = (predicted_expert == class_label).sum().item()
        class_counts_adjusted = class_counts / predicted_expert.size(0)

        noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
        noise_2 = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)

        gen_losses = torch.zeros(3)
        mean_intensities_experts = np.zeros(3)  # mean intensities for each expert for each batch
        std_intensities_experts = np.zeros(3)  # std intensities for each expert for each batch
        photonsum_entropy_experts = torch.zeros(3)  # entropy for each expert for each batch
        mean_intensities_experts_batch = torch.zeros(BATCH_SIZE)  # entropy for each expert for each batch
        aux_reg_losses = []
        disc_losses = []

        #
        # FOR EACH GENERATOR
        #

        for i in range(NUM_GENERATORS):
            generator_optimizers[i].zero_grad()
            selected_indices = (predicted_expert == i).nonzero(as_tuple=True)[0]
            if selected_indices.numel() <= 1:
                gen_losses[i] = torch.tensor(0.0, requires_grad=True).to(device)
                continue

            # Clone or detach tensors to avoid in-place modifications
            selected_noise = noise[selected_indices].clone().detach()
            selected_noise_2 = noise_2[selected_indices].clone().detach()
            selected_cond = cond[selected_indices].clone().detach()
            selected_real_images = real_images[selected_indices].clone().detach()

            fake_images = generators[i](selected_noise, selected_cond).to(device)
            fake_images_2 = generators[i](selected_noise_2, selected_cond).to(device)

            print(f'GEN {i}')
            print(f"fake images shape: {fake_images.shape}")
            print(f"fake images 2 shape: {fake_images_2.shape}")

            fake_output, fake_latent = discriminator(fake_images, selected_cond)
            fake_output_2, fake_latent_2 = discriminator(fake_images_2, selected_cond)

            gen_loss = generator_criterion(fake_output, torch.ones_like(fake_output))
            div_loss = sdi_gan_regularization(fake_latent, fake_latent_2,
                                              selected_noise, selected_noise_2,
                                              std[selected_indices].clone().detach(), DI_STRENGTH)

            intensity_loss, mean_intenisties, std_intensity, mean_intensity = intensity_regularization(fake_images,
                                                                                                       intensity[selected_indices].clone().detach().to(device),
                                                                                                       scaler_intensity, IN_STRENGTH)
            mean_intensities_experts_batch[selected_indices] = mean_intenisties.squeeze()  # input the mean intensities for calculated samples
            gen_loss = gen_loss + div_loss + intensity_loss

            #
            # Calculate Expert Specification
            #
            mean_intensities_experts[i] = mean_intensity
            std_intensities_experts[i] = std_intensity
            # calculate the cond values specification, mean, std, min, max for selected cond
            # mean_cond = selected_cond.mean()
            # std_cond = selected_cond.std()
            # min_cond = selected_cond.min()
            # max_cond = selected_cond.max()
            # calculate the min max values of pixels on the generated images
            # min_pixel, max_pixel = fake_images.min(), fake_images.max()

            # mean_intensity is a float value from the whole batch
            # mean_intenisties = torch.tensor(sum_all_axes_p, dtype=torch.float32).to(device)

            # calculate intraenropy inside expert generator based on the intensity photon sum

            # Normalizacja sumy pikseli do przedziału [0, 1]
            normalized_pixel_sums = (mean_intenisties - mean_intenisties.min()) / (mean_intenisties.max() - mean_intenisties.min() + 1e-8)

            # Oblicz histogram
            histogram = torch.histc(normalized_pixel_sums, bins=30, min=0, max=1)  # TODO: Parameter
            histogram = histogram / histogram.sum()  # Normalizacja histogramu

            # Obliczenie entropii
            entropy = -torch.sum(histogram * torch.log(histogram + 1e-8))
            photonsum_entropy_experts[i] = entropy.clone().detach().to(device)

            # scale learning rate:
            generator_optimizers[i].param_groups[0]['lr'] = LR_G * class_counts_adjusted[i].clone().detach()
            gen_loss.backward()
            generator_optimizers[i].step()
            gen_losses[i] = gen_loss.clone().detach().to(device)

            real_output, real_latent = discriminator(selected_real_images.clone().detach(), selected_cond.clone().detach())
            real_labels = torch.ones_like(real_output).clone().detach()
            loss_real_disc = discriminator_criterion(real_output, real_labels)

            fake_output, fake_latent = discriminator(fake_images.detach().clone(), selected_cond.detach().clone())
            fake_labels = torch.zeros_like(fake_output).clone().detach()
            loss_fake_disc = discriminator_criterion(fake_output, fake_labels)

            disc_loss = (loss_real_disc + loss_fake_disc) / 2
            disc_losses.append(disc_loss)

            generated_positions = aux_reg(fake_images.detach().clone())

            # Ensure true_positions and generated_positions require grad
            selected_true_positions = true_positions[selected_indices].clone().detach()
            selected_true_positions.requires_grad_(True)
            generated_positions.requires_grad_(True)

            aux_reg_loss = aux_reg_criterion(selected_true_positions, generated_positions, AUX_STRENGTH)
            aux_reg_loss = aux_reg_loss * class_counts_adjusted[i].clone().detach()
            aux_reg_losses.append(aux_reg_loss)

        aux_reg_loss = torch.stack(aux_reg_losses).mean()

        aux_reg_optimizer.zero_grad()
        aux_reg_loss.backward()
        aux_reg_optimizer.step()

        # Accumulate and compute discriminator loss outside the loop
        total_disc_loss = torch.stack(disc_losses).mean()
        discriminator_optimizer.zero_grad()
        total_disc_loss.backward()
        discriminator_optimizer.step()

        # calculate intra entropy
        intra_entropy_loss = torch.mean(photonsum_entropy_experts) * IN_ENTROPY_STRENGTH

        # Router loss is the sum of the losses from the generators
        gen_loss_scaled = gen_losses.mean().clone() * GEN_STRENGTH
        # expert_utilization_loss = expert_utilization_entropy * ENT_STRENGTH

        expert_distribution_loss = calculate_expert_distribution_loss(predicted_expert, mean_intensities_experts_batch, ED_STRENGTH)

        router_loss = gen_loss_scaled.clone().detach().to(device) + expert_distribution_loss
        router_loss.requires_grad_(True)

        # Train Router Network
        router_loss.backward()
        router_optimizer.step()

        return gen_losses[0].item(), gen_losses[1].item(), gen_losses[2].item(),\
               total_disc_loss.item(), router_loss.item(), div_loss.cpu().item(),\
               intensity_loss.cpu().item(), aux_reg_loss.cpu().item(), class_counts.cpu().detach(),\
               intra_entropy_loss.cpu().item(), std_intensities_experts, mean_intensities_experts, expert_distribution_loss

    # Settings for plotting
    START_GENERATING_IMG_FROM_IDX = 20
    # IDX_GENERATE = [23771, 18670, 23891, 23924, 23886, 32028]
    IDX_GENERATE = [1, 2, 3, 4, 5, 6]

    # Training loop
    def train(train_loader, epochs, y_test):
        history = []
        for epoch in range(epochs):
            start = time.time()
            gen_loss_epoch_0 = []
            gen_loss_epoch_1 = []
            gen_loss_epoch_2 = []
            disc_loss_epoch = []
            router_loss_epoch = []
            div_loss_epoch = []
            intensity_loss_epoch = []
            aux_reg_loss_epoch = []
            intra_entropy_loss_epoch = []
            n_chosen_experts_0 = []
            n_chosen_experts_1 = []
            n_chosen_experts_2 = []
            std_intensities_experts_0 = []
            std_intensities_experts_1 = []
            std_intensities_experts_2 = []
            mean_intensities_experts_0 = []
            mean_intensities_experts_1 = []
            mean_intensities_experts_2 = []

            # Iterate through both data loaders
            for batch in train_loader:
                gen_loss_0, gen_loss_1, gen_loss_2, disc_loss, router_loss, div_loss, intensity_loss, \
                aux_reg_loss, n_chosen_experts_batch, intra_entropy_loss, std_intensities_experts,\
                mean_intensities_experts = train_step(batch)

                gen_loss_epoch_0.append(gen_loss_0)
                gen_loss_epoch_1.append(gen_loss_1)
                gen_loss_epoch_2.append(gen_loss_2)
                disc_loss_epoch.append(disc_loss)
                router_loss_epoch.append(router_loss)
                div_loss_epoch.append(div_loss)
                intensity_loss_epoch.append(intensity_loss)
                aux_reg_loss_epoch.append(aux_reg_loss)
                n_chosen_experts_0.append(n_chosen_experts_batch[0])
                n_chosen_experts_1.append(n_chosen_experts_batch[1])
                n_chosen_experts_2.append(n_chosen_experts_batch[2])
                intra_entropy_loss_epoch.append(intra_entropy_loss)  # [n_experts, BATCH_SIZE]
                mean_intensities_experts_0.append(mean_intensities_experts[0])  # [n_experts, BATCH_SIZE]
                mean_intensities_experts_1.append(mean_intensities_experts[1])  # [n_experts, BATCH_SIZE]
                mean_intensities_experts_2.append(mean_intensities_experts[2])  # [n_experts, BATCH_SIZE]
                # Calculate the aggregated standard deviation in one line
                std_intensities_experts_0 = np.sqrt(np.mean(std_intensities_experts[0] ** 2))
                std_intensities_experts_1 = np.sqrt(np.mean(std_intensities_experts[1] ** 2))
                std_intensities_experts_2 = np.sqrt(np.mean(std_intensities_experts[2] ** 2))

                # disc_loss_epoch.append((disc_loss_0 + disc_loss_1 + disc_loss_2) / NUM_GENERATORS)
                # router_loss_epoch.append((router_loss_0 + router_loss_1 + router_loss_2) / NUM_GENERATORS)
                # div_loss_epoch.append((div_loss_0 + div_loss_1 + div_loss_2) / NUM_GENERATORS)
                # intensity_loss_epoch.append((intensity_loss_0 + intensity_loss_1 + intensity_loss_2) / NUM_GENERATORS)
                # aux_reg_loss_epoch.append((aux_reg_loss_0 + aux_reg_loss_1 + aux_reg_loss_2) / NUM_GENERATORS)
            #
            # TEST GENERATION
            #

            #
            # Plot the example results
            # choose random element from generators
            random_generator = np.random.randint(0, NUM_GENERATORS)
            model = generators[random_generator]
            noise_cond = y_test[IDX_GENERATE]
            noise = torch.randn(len(noise_cond), NOISE_DIM, device=device)
            if epoch % 5 == 0:  # plot image each 5 epoch from random generator
                plot = generate_and_save_images(model, epoch,
                                                noise,
                                                noise_cond,
                                                x_test[expert_number_test == random_generator],
                                                photon_sum_proton_min, photon_sum_proton_max,
                                                device, random_generator)
            else:
                plot = None

            # ws_values = []
            # for batch in test_loader:
            #     real_images, real_images_2, cond, std, intensity, true_positions, _ = batch
            #     # use router to calculate choose the expert assignments for samples
            #     cond = cond.to(device)
            #     # std = std.to(device)
            #     # intensity = intensity.to(device)
            #     # true_positions = true_positions.to(device)

            y_test = torch.tensor(y_test, device=device)

            # Test Router Network
            with torch.no_grad():
                router_network.eval()
                predicted_expert_one_hot = router_network(y_test)

            _, predicted_expert = torch.max(predicted_expert_one_hot, 1)

            indices_expert_0 = np.where(predicted_expert.cpu().numpy() == 0)[0]
            indices_expert_1 = np.where(predicted_expert.cpu().numpy() == 1)[0]
            indices_expert_2 = np.where(predicted_expert.cpu().numpy() == 2)[0]

            # Calculate WS distance across all distribution
            ws_mean = calculate_joint_ws_across_experts(min(epoch // 5 + 1, 5),
                                                        [x_test[indices_expert_0], x_test[indices_expert_1],
                                                         x_test[indices_expert_2]],
                                                        [y_test[indices_expert_0], y_test[indices_expert_1],
                                                         y_test[indices_expert_2]],
                                                        generators, ch_org,
                                                        NOISE_DIM, device)
            #
            # ws_mean = np.array(ws_values).mean()
            epoch_time = time.time() - start

            # Log to WandB tool
            log_data = {
                'ws_mean': ws_mean,
                'gen_loss_0': np.mean(gen_loss_epoch_0),
                'gen_loss_1': np.mean(gen_loss_epoch_1),
                'gen_loss_2': np.mean(gen_loss_epoch_2),
                'div_loss': np.mean(div_loss_epoch),
                'intensity_loss': np.mean(intensity_loss_epoch),
                'router_loss': np.mean(router_loss_epoch),
                'intra_entropy_loss_epoch': np.mean(intra_entropy_loss_epoch),
                'std_intensities_experts_0': np.mean(std_intensities_experts_0),
                'std_intensities_experts_1': np.mean(std_intensities_experts_1),
                'std_intensities_experts_2': np.mean(std_intensities_experts_2),
                'mean_intensities_experts_0': np.mean(mean_intensities_experts_0),
                'mean_intensities_experts_1': np.mean(mean_intensities_experts_1),
                'mean_intensities_experts_2': np.mean(mean_intensities_experts_2),
                'disc_loss': np.mean(disc_loss_epoch),
                'aux_reg_loss': np.mean(aux_reg_loss_epoch),
                'n_choosen_experts_mean_epoch_0': np.mean(n_chosen_experts_0),
                'n_choosen_experts_mean_epoch_1': np.mean(n_chosen_experts_1),
                'n_choosen_experts_mean_epoch_2': np.mean(n_chosen_experts_2),
                'epoch_time': epoch_time,
                'epoch': epoch,
                'plot': wandb.Image(plot) if plot else None
            }

            wandb.log(log_data)

            # save models if all generators pass threshold
            if ws_mean < WS_MEAN_SAVE_THRESHOLD:
                for i, generator in enumerate(generators):
                    torch.save(generator.state_dict(), os.path.join(filepath_mod, "gen_" + NAME + "_" + str(epoch) + ".h5"))

            print(f'Time for epoch {epoch + 1} is {epoch_time:.2f} sec')


    config_wandb = {
        "Model": NAME,
        "dataset": "proton_data",
        "epochs": EPOCHS,
        "Date": DATE_STR,
        "Proton_min": photon_sum_proton_min,
        "Proton_max": photon_sum_proton_max,
        "diversity_strength": DI_STRENGTH,
        "intensity_strength": IN_STRENGTH,
        "auxiliary_strength": AUX_STRENGTH,
        "Generator_strength": GEN_STRENGTH,
        "IntraEntropy_photon_sum_strength": IN_ENTROPY_STRENGTH,
        "Learning rate_generator": LR_G,
        "Learning rate_discriminator": LR_D,
        "Learning rate_router": LR_R,
        "Learning rate_aux_reg": LR_A,
        "Experiment_dir_name": EXPERIMENT_DIR_NAME,
        "Batch_size": BATCH_SIZE,
        "Noise_dim": NOISE_DIM,
        "router_arch": "64-64-32-16",
        "intensity_loss_type": "mae"
    }


    # Separate datasets for each expert
    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(x_train_2),
                                  torch.tensor(y_train), torch.tensor(std_train),
                                  torch.tensor(intensity_train), torch.tensor(positions_train),
                                  torch.tensor(expert_number_train))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(x_test_2),
                                 torch.tensor(y_test), torch.tensor(std_test),
                                 torch.tensor(intensity_test), torch.tensor(positions_test),
                                 torch.tensor(expert_number_test))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    wandb.finish()
    wandb.init(
        project="Generative-DNN-for-CERN-Fast-Simulations",
        entity="bedkowski-patrick",
        name=wandb_run_name,
        config=config_wandb,
        tags=[f"proton_min_{photon_sum_proton_min}", f"proton_max_{photon_sum_proton_max}", "sdi_gan_intensity"]
    )

    history = train(train_loader, EPOCHS, y_test)
