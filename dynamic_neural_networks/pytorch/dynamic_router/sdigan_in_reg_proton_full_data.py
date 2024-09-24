import time
import os
import wandb

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils import (sum_channels_parallel, calculate_ws_ch_proton_model,
                   calculate_joint_ws_across_experts,
                   create_dir, save_scales, evaluate_router,
                   intensity_regularization, sdi_gan_regularization,
                   generate_and_save_images,
                   calculate_expert_distribution_loss,
                   regressor_loss, calculate_expert_utilization_entropy)

from models_pytorch import Generator, Discriminator, RouterNetwork, AuxReg

print(torch.cuda.is_available())
print(torch.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.autograd.set_detect_anomaly(True)

# SETTINGS & PARAMETERS
SAVE_EXPERIMENT_DATA = True
WS_MEAN_SAVE_THRESHOLD = 5
DI_STRENGTH = 0.1
IN_STRENGTH = 1e-3  # 1e-3
AUX_STRENGTH = 1e-3

N_RUNS = 2
BATCH_SIZE = 128
NOISE_DIM = 10
N_COND = 9  # number of conditional features
EPOCHS = 250
LR_G = 1e-4
LR_D = 1e-5
LR_A = 1e-4
NAME = f"SDI-GAN-REG-PROTON-full-{DI_STRENGTH}-{IN_STRENGTH}-{AUX_STRENGTH}"


for _ in range(N_RUNS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_proton_photonsum_proton_1_2312.pkl')
    data_cond = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_cond_photonsum_proton_1_2312.pkl')
    photon_sum_proton_min, photon_sum_proton_max = data_cond.proton_photon_sum.min(), data_cond.proton_photon_sum.max()

    # data of coordinates of maximum value of pixel on the images
    data_posi = pd.read_pickle('/net/tscratch/people/plgpbedkowski/data/data_coord_proton_photonsum_proton_1_2312.pkl')
    print('Loaded positions: ', data_posi.shape, "max:", data_posi.values.max(), "min:", data_posi.values.min())

    DATE_STR = datetime.now().strftime("%d_%m_%Y_%H_%M")
    wandb_run_name = f"{NAME}_{LR_G}_{LR_D}_{DATE_STR}"
    EXPERIMENT_DIR_NAME = f"experiments/{wandb_run_name}_{int(photon_sum_proton_min)}_{int(photon_sum_proton_max)}_{DATE_STR}"

    print("Photon sum proton min:", photon_sum_proton_min, "max:", photon_sum_proton_max)

    # PRINT DATA SHAPE
    DATA_SHAPE_STR = {"data": data.shape, "data_cond": data_cond.shape, "data_posi": data_posi.shape}
    data_list = [[key, value] for key, value in DATA_SHAPE_STR.items()]
    print(tabulate(data_list, headers=['Data', 'Shape'], tablefmt='github', showindex="always"))

    print('Loaded positions: ', data_posi.shape, "max:", data_posi.values.max(), "min:", data_posi.values.min())

    DATE_STR = datetime.now().strftime("%d_%m_%Y_%H_%M")
    wandb_run_name = f"{NAME}_{LR_G}_{LR_D}_{DATE_STR}"
    EXPERIMENT_DIR_NAME = f"experiments/{wandb_run_name}_{int(photon_sum_proton_min)}_{int(photon_sum_proton_max)}_{DATE_STR}"

    # group conditional data
    data_cond["cond"] = data_cond["Energy"].astype(str) + "|" + data_cond["Vx"].astype(str) + "|" + data_cond[
        "Vy"].astype(str) + "|" + data_cond["Vz"].astype(str) + "|" + data_cond["Px"].astype(str) + "|" + data_cond[
                            "Py"].astype(str) + "|" + data_cond["Pz"].astype(str) + "|" + data_cond["mass"].astype(
        str) + "|" + data_cond["charge"].astype(str)
    data_cond_id = data_cond[["cond"]].reset_index()
    ids = data_cond_id.merge(data_cond_id.sample(frac=1), on=["cond"], how="inner").groupby("index_x").first()
    ids = ids["index_y"]

    data = np.log(data + 1).astype(np.float32)
    print("data max", data.max(), "min", data.min())

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

    print("Data_cond shape", data_cond.shape)

    x_train, x_test, x_train_2, x_test_2, \
    y_train, y_test, \
    std_train, std_test, \
    intensity_train, intensity_test, \
    positions_train, positions_test = train_test_split(data, data_2, data_cond, std,
                                                       intensity, data_xy,
                                                       test_size=0.2, shuffle=False)

    print("Data shapes after train_test_split:", x_train.shape, x_test.shape, y_train.shape, y_test.shape)

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

    # Loss and optimizer
    generator_discriminator_criterion = nn.BCELoss()
    aux_reg_criterion = regressor_loss

    # Define experts
    generator = Generator(NOISE_DIM, N_COND).to(device)
    generator_optimizer = optim.Adam(generator.parameters(), lr=LR_G)

    # Initialize single discriminator
    discriminator = Discriminator(N_COND, 1).to(device)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LR_D)

    # Define Auxiliary regressor
    aux_reg = AuxReg().to(device)
    aux_reg_optimizer = optim.Adam(aux_reg.parameters(), lr=LR_A)


    def generator_train_step(gene, disc, a_reg, cond, g_optimizer, a_optimizer, criterion, true_positions, std, intensity, BATCH_SIZE):
        # Train Generator
        generator_optimizer.zero_grad()

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
                                          std, DI_STRENGTH)

        intensity_loss, mean_intenisties, std_intensity, mean_intensity = intensity_regularization(fake_images,
                                                                                                   intensity,
                                                                                                   scaler_intensity,
                                                                                                   IN_STRENGTH)

        gen_loss = gen_loss + div_loss + intensity_loss

        # Train auxiliary regressor
        aux_reg_optimizer.zero_grad()
        generated_positions = a_reg(fake_images)

        aux_reg_loss = aux_reg_criterion(true_positions, generated_positions, scaler_poz, AUX_STRENGTH)

        gen_loss += aux_reg_loss

        gen_loss.backward()
        g_optimizer.step()
        a_optimizer.step()

        return gen_loss.data, div_loss.data, intensity_loss.data, aux_reg_loss.data, std_intensity, mean_intensity


    def discriminator_train_step(disc, gene, d_optimizer, criterion, real_images, cond, BATCH_SIZE):
        # Train discriminator
        d_optimizer.zero_grad()

        real_output, real_latent = disc(real_images, cond)

        # calculate loss for real images
        real_labels = torch.ones_like(real_output)
        loss_real_disc = criterion(real_output, real_labels)

        # calculate loss for fake images
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=device, requires_grad=False)
        fake_labels = torch.zeros_like(real_labels)
        fake_images = gene(noise, cond)
        fake_output, fake_latent = disc(fake_images, cond)

        loss_fake_disc = criterion(fake_output, torch.zeros_like(fake_labels))
        disc_loss = loss_real_disc + loss_fake_disc

        # Accumulate and compute discriminator loss outside the loop
        disc_loss.backward()
        discriminator_optimizer.step()

        return disc_loss.data

    # Adjust train_step function to work with the single discriminator
    def train_step(batch):
        real_images, real_images_2, cond, std, intensity, true_positions = batch

        # Clone or detach tensors to avoid in-place modifications
        real_images = real_images.unsqueeze(1).to(device).clone()
        # real_images.requires_grad = False
        cond = cond.to(device).clone()
        # cond.requires_grad = False
        std = std.to(device).clone()
        # std.requires_grad = False
        intensity = intensity.to(device).clone()
        # intensity.requires_grad = False
        true_positions = true_positions.to(device).clone()
        # true_positions.requires_grad = False
        BATCH_SIZE = real_images.shape[0]

        # Train discriminator
        disc_loss = discriminator_train_step(discriminator, generator, discriminator_optimizer,
                                             generator_discriminator_criterion, real_images, cond, BATCH_SIZE)

        # train generator
        gen_loss, div_loss, intensity_loss, \
        aux_reg_loss, std_intensity, mean_intensity = generator_train_step(generator,
                                                                           discriminator,
                                                                           aux_reg,
                                                                           cond,
                                                                           generator_optimizer,
                                                                           aux_reg_optimizer,
                                                                           generator_discriminator_criterion,
                                                                           true_positions,
                                                                           std,
                                                                           intensity,
                                                                           BATCH_SIZE)

        return gen_loss.item(), \
               disc_loss.item(), div_loss.cpu().item(), \
               intensity_loss.cpu().item(), aux_reg_loss.cpu().item(), \
               std_intensity.item(), mean_intensity.item()


    # Settings for plotting
    START_GENERATING_IMG_FROM_IDX = 20
    # IDX_GENERATE = [23771, 18670, 23891, 23924, 23886, 32028]
    IDX_GENERATE = [1, 2, 3, 4, 5, 6]


    # Training loop
    def train(train_loader, epochs, y_test):
        history = []
        for epoch in range(epochs):
            start = time.time()
            gen_loss_epoch = []
            disc_loss_epoch = []
            router_loss_epoch = []
            div_loss_epoch = []
            intensity_loss_epoch = []
            aux_reg_loss_epoch = []
            std_intensities_epoch = []
            mean_intensities_epoch = []

            # Iterate through both data loaders
            for batch in train_loader:
                gen_loss, disc_loss, div_loss, intensity_loss, \
                aux_reg_loss, std_intensities, \
                mean_intensities = train_step(batch)

                gen_loss_epoch.append(gen_loss)
                disc_loss_epoch.append(disc_loss)
                div_loss_epoch.append(div_loss)
                intensity_loss_epoch.append(intensity_loss)
                aux_reg_loss_epoch.append(aux_reg_loss)
                mean_intensities_epoch.append(mean_intensities)  # [n_experts, BATCH_SIZE]
                std_intensities_epoch = np.sqrt(np.mean(std_intensities ** 2))
            #
            # TEST GENERATION
            #

            # Plot the example results
            # choose random element from generators
            noise_cond = y_test[IDX_GENERATE]
            noise = torch.randn(len(noise_cond), NOISE_DIM, device=device)
            if epoch % 5 == 0:  # plot image each 5 epoch from random generator
                plot = generate_and_save_images(generator, epoch,
                                                noise,
                                                noise_cond,
                                                x_test,
                                                photon_sum_proton_min, photon_sum_proton_max,
                                                device, '0')
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

            # org_3 = np.exp(x_test[indices_expert_3]) - 1
            # ch_org_3 = np.array(org_3).reshape(-1, 56, 30)
            # del org_3
            # ch_org_3 = pd.DataFrame(sum_channels_parallel(ch_org_3)).values

            # Calculate WS distance across all distribution
            ws_mean = calculate_ws_ch_proton_model(min(epoch // 5 + 1, 5),
                                                   x_test, y_test,
                                                   generator, ch_org,
                                                   NOISE_DIM, device, batch_size=BATCH_SIZE)
            #
            # ws_mean = np.array(ws_values).mean()
            epoch_time = time.time() - start

            # Log to WandB tool
            log_data = {
                'ws_mean': ws_mean,
                'gen_loss': np.mean(gen_loss_epoch),
                'div_loss': np.mean(div_loss_epoch),
                'intensity_loss': np.mean(intensity_loss_epoch),
                'router_loss': np.mean(router_loss_epoch),
                'std_intensities_experts': np.mean(std_intensities_epoch),
                'mean_intensities_experts': np.mean(mean_intensities_epoch),
                'disc_loss': np.mean(disc_loss_epoch),
                'aux_reg_loss': np.mean(aux_reg_loss_epoch),
                # 'n_choosen_experts_mean_epoch_3': np.mean(n_chosen_experts_3),
                'epoch_time': epoch_time,
                'epoch': epoch,
                'plot': wandb.Image(plot) if plot else None
            }

            wandb.log(log_data)

            # save models if all generators pass threshold
            if ws_mean < WS_MEAN_SAVE_THRESHOLD:
                torch.save(generator.state_dict(), os.path.join(filepath_mod, "gen_" + NAME + "_" + str(epoch) + ".h5"))

            print(f'Time for epoch {epoch + 1} is {epoch_time:.2f} sec')


    config_wandb = {
        "Model": NAME,
        "dataset": "proton_data",
        "epochs": EPOCHS,
        "Date": DATE_STR,
        "Proton_min": photon_sum_proton_min,
        "Proton_max": photon_sum_proton_max,
        "generator_architecture": generator.name,
        "diversity_strength": DI_STRENGTH,
        "intensity_strength": IN_STRENGTH,
        "auxiliary_strength": AUX_STRENGTH,
        "Learning rate_generator": LR_G,
        "Learning rate_discriminator": LR_D,
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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(x_test_2),
                                 torch.tensor(y_test), torch.tensor(std_test),
                                 torch.tensor(intensity_test), torch.tensor(positions_test))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    wandb.finish()
    wandb.init(
        project="Generative-DNN-for-CERN-Fast-Simulations",
        entity="bedkowski-patrick",
        name=wandb_run_name,
        config=config_wandb,
        tags=["param_sweep", f"proton_min_{photon_sum_proton_min}", f"proton_max_{photon_sum_proton_max}",
              "sdi_gan_intensity"]
    )

    history = train(train_loader, EPOCHS, y_test)
