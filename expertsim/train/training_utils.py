import os
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.ndimage import center_of_mass
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List
from sklearn.model_selection import StratifiedKFold
from .utils import sum_channels_parallel
from omegaconf import DictConfig


"""
NEW PART
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict
from itertools import combinations


#
# def generator_train_step(noise, noise_2,
#                          fake_images,
#                          fake_output,
#                          fake_latent,
#                          fake_latent_2,
#                          g_optimizer,
#                          #class_counts,
#                          std, intensity,
#                          #a_optimizer, generated_positions, true_positions,
#                          #aux_strength,
#                          di_strength, in_strength,
#                          cfg, device):
#     # Train Generator
#     g_optimizer.zero_grad()
#     criterion = torch.nn.BCELoss()
#     fake_for_gen = fake_output
#     gen_loss = criterion(fake_for_gen, torch.ones_like(fake_for_gen))
#
#     # div_loss = sdi_gan_regularization(fake_latent, fake_latent_2,
#     #                                   noise, noise_2,
#     #                                   std, di_strength)
#     #
#     # intensity_loss, mean_intenisties, std_intensity, mean_intensity = intensity_regularization(fake_images,
#     #                                                                                            intensity,
#     #                                                                                            in_strength)
#
#     gen_loss = gen_loss  # + div_loss + intensity_loss
#
#     # Train auxiliary regressor
#     #a_optimizer.zero_grad()
#     # aux_reg_loss = regressor_loss(true_positions, generated_positions, aux_strength=aux_strength)
#
#     # gen_loss += aux_reg_loss
#
#
#     # # Check for invalid gradients
#     # for name, param in g_optimizer.param_groups[0]['params']:
#     #     if param.grad is not None:
#     #         if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
#     #             print(f"Invalid gradient in {name}")
#
#     gen_loss.backward()
#
#     # g_optimizer.param_groups[0]['lr'] = cfg.model.generator.lr_g * class_counts.clone().detach()
#     g_optimizer.step()
#     # a_optimizer.param_groups[0]['lr'] = cfg.model.generator.lr_a * class_counts.clone().detach()
#     # a_optimizer.step()
#
#     # # Auxiliary loss (Gradients flow back to the generator to update it)
#     # generated_positions = a_reg(fake_images)
#     # aux_reg_loss = aux_reg_criterion(true_positions, generated_positions, scaler_poz, AUX_STRENGTH)
#     #
#     # # Combined loss for generator
#     # total_gen_loss = gen_loss + aux_reg_loss.detach()  # Detach aux_loss from generator
#     #
#     # # Backpropagate generator losses
#     # total_gen_loss.backward()
#     # g_optimizer.param_groups[0]['lr'] = LR_G * class_counts.clone().detach()
#     # g_optimizer.step()
#     #
#     # # Train aux_reg separately
#     # a_optimizer.zero_grad()
#     # a_optimizer.param_groups[0]['lr'] = LR_A * class_counts.clone().detach()
#     # a_optimizer.step()
#
#     return gen_loss.item()#, div_loss.item(), intensity_loss.item(), aux_reg_loss.item(), std_intensity,\ mean_intensity, mean_intenisties


def generator_train_step(noise, noise_2,
                         fake_images,
                         fake_output,
                         fake_latent,
                         fake_latent_2,
                         g_optimizer,
                         class_counts,
                         std, intensity,
                         a_optimizer, generated_positions, true_positions,
                         aux_strength,
                         di_strength, in_strength,
                         cfg, device):
    # Train Generator
    g_optimizer.zero_grad()
    criterion = torch.nn.BCELoss()
    fake_for_gen = fake_output
    gen_loss = criterion(fake_for_gen, torch.ones_like(fake_for_gen))

    # div_loss = sdi_gan_regularization(fake_latent, fake_latent_2,
    #                                   noise, noise_2,
    #                                   std, di_strength)

    # intensity_loss, mean_intenisties, std_intensity, mean_intensity = intensity_regularization(fake_images,
    #                                                                                            intensity,
    #                                                                                            in_strength)

    gen_loss = gen_loss #+ div_loss + intensity_loss

    # Train auxiliary regressor
    #a_optimizer.zero_grad()
    # aux_reg_loss = regressor_loss(true_positions, generated_positions, aux_strength=aux_strength)

    # gen_loss += aux_reg_loss


    # # Check for invalid gradients
    # for name, param in g_optimizer.param_groups[0]['params']:
    #     if param.grad is not None:
    #         if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
    #             print(f"Invalid gradient in {name}")

    gen_loss.backward()

    # g_optimizer.param_groups[0]['lr'] = cfg.model.generator.lr_g * class_counts.clone().detach()
    g_optimizer.step()
    # a_optimizer.param_groups[0]['lr'] = cfg.model.generator.lr_a * class_counts.clone().detach()
    # a_optimizer.step()

    # # Auxiliary loss (Gradients flow back to the generator to update it)
    # generated_positions = a_reg(fake_images)
    # aux_reg_loss = aux_reg_criterion(true_positions, generated_positions, scaler_poz, AUX_STRENGTH)
    #
    # # Combined loss for generator
    # total_gen_loss = gen_loss + aux_reg_loss.detach()  # Detach aux_loss from generator
    #
    # # Backpropagate generator losses
    # total_gen_loss.backward()
    # g_optimizer.param_groups[0]['lr'] = LR_G * class_counts.clone().detach()
    # g_optimizer.step()
    #
    # # Train aux_reg separately
    # a_optimizer.zero_grad()
    # a_optimizer.param_groups[0]['lr'] = LR_A * class_counts.clone().detach()
    # a_optimizer.step()

    return gen_loss.item()#, div_loss.item(), intensity_loss.item(), aux_reg_loss.item(), std_intensity,\ mean_intensity, mean_intenisties


def discriminator_train_step(
        self,
        real_output: torch.tensor,
        fake_output: torch.tensor,
        d_optimizer: torch.optim.Optimizer,
        device: torch.device
) -> float:
    d_optimizer.zero_grad()
    criterion = torch.nn.BCELoss()

    # Clone to avoid graph issues
    real_output = real_output.clone()
    fake_output = fake_output.clone().detach()  # Detach fake to prevent gen gradient flow

    real_labels = torch.ones_like(real_output, device=device)
    loss_real = criterion(real_output, real_labels)

    fake_labels = torch.zeros_like(fake_output, device=device)
    loss_fake = criterion(fake_output, fake_labels)

    disc_loss = loss_real + loss_fake
    disc_loss.backward(retain_graph=True)
    d_optimizer.step()
    return disc_loss.item()


def compute_router_losses(
        gen_losses: List[float],
        disc_losses: List[float],
        predicted_expert_one_hot: torch.Tensor,
        aux_reg_features_experts: List[torch.Tensor],
        mean_intensities_batch: torch.Tensor,
        cfg: DictConfig,
        device: torch.device
) -> Dict[str, torch.tensor]:
    """Compute all router-related losses."""

    losses = {}

    # GAN loss
    if cfg.model.router.gen_strength > 0:
        gen_tensor = torch.tensor(gen_losses, device=device)
        disc_tensor = torch.tensor(disc_losses, device=device)
        losses['gan_loss'] = (gen_tensor.mean() + disc_tensor.mean()) * cfg.model.router.gen_strength
    else:
        losses['gan_loss'] = torch.tensor(0.0, device=device)

    # Expert entropy loss
    if cfg.model.router.util_strength > 0:
        avg_gating = torch.mean(predicted_expert_one_hot, dim=0)
        entropy = -torch.sum(avg_gating * torch.log(avg_gating + 1e-9))
        losses['entropy_loss'] = entropy * cfg.model.router.util_strength
    else:
        losses['entropy_loss'] = torch.tensor(0.0, device=device)

    # Expert distribution loss
    if cfg.model.router.ed_strength > 0:
        features = mean_intensities_batch.reshape(-1, 1)
        pairwise_distances = torch.cdist(features, features, p=2)
        gating_similarities = torch.matmul(predicted_expert_one_hot, predicted_expert_one_hot.T)
        losses['distribution_loss'] = (torch.sum(gating_similarities * pairwise_distances) /
                                       gating_similarities.size(0)) * cfg.model.router.ed_strength
    else:
        losses['distribution_loss'] = torch.tensor(0.0, device=device)

    # Differentiation loss
    if cfg.model.router.diff_strength > 0:
        diff_loss = torch.tensor(0.0, device=device)
        feature_means = [feat.mean(0, keepdim=True) for feat in aux_reg_features_experts if feat.numel() > 0]

        for i, j in combinations(range(len(feature_means)), 2):
            cosine_sim = F.cosine_similarity(feature_means[i], feature_means[j])
            diff_loss += torch.abs(cosine_sim)

        losses['differentiation_loss'] = diff_loss * cfg.model.router.diff_strength
    else:
        losses['differentiation_loss'] = torch.tensor(0.0, device=device)

    # Adaptive load balancing loss
    if cfg.model.router.alb_strength > 0:
        routing_scores = predicted_expert_one_hot.sum(dim=0)
        penalties = torch.exp(1.0 / (routing_scores + 1e-6))
        losses['alb_loss'] = penalties.mean() * cfg.model.router.alb_strength
    else:
        losses['alb_loss'] = torch.tensor(0.0, device=device)

    # Total router loss
    losses['router_loss'] = (losses['gan_loss'] + losses['distribution_loss'] -
                             losses['entropy_loss'] + losses['alb_loss'] -
                             losses['differentiation_loss'])

    return losses


def sdi_gan_regularization(fake_latent, fake_latent_2, noise, noise_2, std, di_strength):
    """ Calculate the absolute differences and their means along the batch dimension."""
    abs_diff_latent = torch.mean(torch.abs(fake_latent - fake_latent_2), dim=2)
    abs_diff_noise = torch.mean(torch.abs(noise - noise_2), dim=2)

    # Compute the division term
    div = abs_diff_latent / (abs_diff_noise + 1e-5)

    # Calculate the div_loss
    div_loss = std * di_strength / (div + 1e-5)

    # Calculate the final div_loss
    div_loss = torch.mean(std) * torch.mean(div_loss)

    return div_loss


"""
Old part
"""






def intensity_regularization(gen_im_proton, intensity_proton, IN_STRENGTH):
    """
    Computes the intensity regularization loss for generated images, returning the loss, the sum of intensities per image,
    and the mean and standard deviation of the intensity across the batch.

    Args:
        gen_im_proton (torch.Tensor): A tensor of generated images with shape [batch_size, channels, height, width].
        intensity_proton (torch.Tensor): A tensor representing the target intensity values for the batch, with shape [batch_size].
        IN_STRENGTH (float): A scalar that controls the strength of the intensity regularization in the final loss.

    Returns:
        torch.Tensor: The intensity regularization loss, calculated as the Mean Absolute Error (MAE) between the scaled
                      sum of the intensities in the generated images and the target intensities, multiplied by `IN_STRENGTH`.
        torch.Tensor: The sum of intensities in each generated image, with shape [n_samples, 1].
        torch.Tensor: The standard deviation of the scaled intensity values across the batch.
        torch.Tensor: The mean of the scaled intensity values across the batch.
    """

    # Sum the intensities in the generated images
    # gen_im_proton_rescaled = torch.exp(gen_im_proton.clone().detach()) - 1 #<- this fixed previous bad optimization
    gen_im_proton_rescaled = torch.exp(gen_im_proton) - 1
    # Gen shape from model torch.Size([B, 3, 56, 30])
    # After sum: torch.Size([B, 3, 1, 1])
    sum_all_axes_p_rescaled = torch.sum(gen_im_proton_rescaled, dim=[2, 3], keepdim=False)
    # Sum along the image dimensions

    # print(sum_all_axes_p_rescaled.shape)  # (batch_size_current, 1)
    # print(sum_all_axes_p_rescaled)
    # REMOVE THIS RESHAPE BECAUSE IT FLATTENS THE DATA FROM ALL EXPERTS
    # sum_all_axes_p_rescaled = sum_all_axes_p_rescaled.reshape(-1, 1)  # Scale and reshape back to (batch_size, 1)

    # Compute mean and std as PyTorch tensors
    std_intensity_scaled = sum_all_axes_p_rescaled.std()
    mean_intensity_scaled = sum_all_axes_p_rescaled.mean()  # scalar
    # print('---------------')
    # print(mean_intensity_scaled.shape)
    # print('---------------')
    # # Ensure intensity_proton is correctly shaped and on the same device
    intensity_proton = intensity_proton.view(-1, 1).to(gen_im_proton.device)  # Ensure it is of shape [batch_size, 1]

    # apply the MASK AS WELL FOR EXPERT COMPUTATIONS TO BOTH THE GENERATED AND REAL DATA
    # OR MAYBE CALCULATE THIS N_EXPERT times each for separate expert. TRY TO MAKE THIS PARALLEL

    # print('shape sum_all_axes_p_rescaled', sum_all_axes_p_rescaled.shape)
    # print('shape intensity_proton',intensity_proton.shape)
    assert sum_all_axes_p_rescaled.shape == intensity_proton.shape
    # Calculate MAE loss
    mae_value_p = F.l1_loss(sum_all_axes_p_rescaled, intensity_proton)*IN_STRENGTH

    return mae_value_p, sum_all_axes_p_rescaled, std_intensity_scaled, mean_intensity_scaled


def get_max_value_image_coordinates(img):
    return np.unravel_index(np.argmax(img), img.shape)


def calculate_joint_ws_across_experts(n_calc, x_tests: List, y_tests: List, generators: List,
                                      ch_org, ch_org_expert, noise_dim, device, batch_size=64,
                                      n_experts=3,
                                      shape_images=(56, 30)):
    """
    Calculates the Wasserstein (WS) distance across the whole distribution.
    """
    # if lengths of data are not the same, raise an error
    if len(x_tests) != len(y_tests) or len(x_tests) != len(generators):
        raise ValueError("Length of data is not the same")

    # Initialize WS distance arrays
    ws = np.zeros((n_calc, 5))  # Overall WS distances
    ws_exp = np.zeros((n_calc, n_experts, 5))  # WS distances for each expert

    for j in range(n_calc):  # Perform multiple calculations of the WS distance
        ch_gen_all = []  # For gathering the whole generated distribution of pixels
        ch_gen_expert = []  # For gathering expert-specific distributions

        for generator_idx in range(len(generators)):  # Iterate over all generators
            y_test_temp = torch.tensor(y_tests[generator_idx], device=device)
            num_samples = x_tests[generator_idx].shape[0]

            if num_samples == 0:
                ch_gen_expert.append(np.array([]))  # Append empty if no samples
                continue

            # Get predictions from generator
            results_all = get_predictions_from_generator_results(
                batch_size, num_samples, noise_dim,
                device, y_test_temp, generators[generator_idx],
                shape_images=shape_images
            )
            print(f"For generator {generator_idx}. Samples generated: {results_all.shape}, real_samples: {num_samples}")

            # Sum channels and store results
            ch_gen_smaller = pd.DataFrame(sum_channels_parallel(results_all)).values
            ch_gen_expert.append(ch_gen_smaller.copy())  # Expert-specific data
            ch_gen_all.extend(ch_gen_smaller.copy())  # Overall data

        ch_gen_all = np.array(ch_gen_all)  # Convert to numpy array
        print("Shape of all generated:", ch_gen_all.shape)

        # Calculate WS distances
        for i in range(5):
            ws[j][i] = wasserstein_distance(ch_org[:, i], ch_gen_all[:, i])  # Overall WS

            for exp_idx in range(len(generators)):  # Per expert
                if ch_gen_expert[exp_idx].shape[0] == 0 or ch_org_expert[exp_idx].shape[0] == 0:
                    continue
                ws_exp[j][exp_idx][i] = wasserstein_distance(
                    ch_org_expert[exp_idx][:, i], ch_gen_expert[exp_idx][:, i]
                )
    # Calculate the mean WS distances across runs
    ws_runs = ws.mean(axis=1)  # calculate mean of the all channels. WS for n_calc (n_calc, 1)
    ws_mean, ws_std = ws_runs.mean(), ws_runs.std()

    ws_exp_runs = ws_exp.mean(axis=2)  # (n_calc, n_experts, 1)
    ws_mean_exp = ws_exp_runs.mean(axis=0)  # calculate mean for each expert
    print("ws mean", f'{ws_mean:.2f}', end=" ")
    if n_experts == 4:
        return ws_mean, ws_std, ws_mean_exp[0], ws_mean_exp[1], ws_mean_exp[2], ws_mean_exp[3]
    elif n_experts == 5:
        return ws_mean, ws_std, ws_mean_exp[0], ws_mean_exp[1], ws_mean_exp[2], ws_mean_exp[3], ws_mean_exp[4]
    elif n_experts == 3:
        return ws_mean, ws_std, ws_mean_exp[0], ws_mean_exp[1], ws_mean_exp[2]
    elif n_experts == 2:
        return ws_mean, ws_std, ws_mean_exp[0], ws_mean_exp[1]
    elif n_experts == 1:
        return ws_mean, ws_std, ws_mean_exp[0]


def get_predictions_from_generator_results(generator, batch_size, num_samples, noise_dim,
                                           device, y_test, shape_images=(56, 30),
                                           input_noise=None):
    num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate number of batches
    results_all = np.zeros((num_samples, *shape_images))
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        if input_noise is not None:
            noise = input_noise[start_idx:end_idx,:]
        else:
            noise = torch.randn(end_idx - start_idx, noise_dim, device=device)
        noise_cond = y_test[start_idx:end_idx]

        # Generate results using the generator
        with torch.no_grad():
            generator.eval()
            results = generator(noise, noise_cond).cpu().numpy()

        results = np.exp(results) - 1
        # results = results*0.75
        results_all[start_idx:end_idx] = results.reshape(-1, *shape_images)
    return results_all


def get_predictions_from_experts_results(num_samples, noise_dim,
                                         device, y_test, router_network, experts):
    y_test_tensor = torch.tensor(y_test, device=device)
    results_all = np.zeros((num_samples, 56, 30))

    with torch.no_grad():
        router_network.eval()
        predicted_expert_one_hot = router_network(y_test_tensor).cpu().numpy()
        predicted_expert = np.argmax(predicted_expert_one_hot, axis=1)

    indx_0 = np.where(predicted_expert == 0)[0].tolist()
    indx_1 = np.where(predicted_expert == 1)[0].tolist()
    indx_2 = np.where(predicted_expert == 2)[0].tolist()
    data_cond_0 = y_test_tensor[indx_0]
    data_cond_1 = y_test_tensor[indx_1]
    data_cond_2 = y_test_tensor[indx_2]

    noise_0 = torch.randn(len(data_cond_0), noise_dim, device=device)
    with torch.no_grad():
        experts[0].eval()
        results_0 = experts[0](noise_0, data_cond_0).cpu().numpy()

    noise_1 = torch.randn(len(data_cond_1), noise_dim, device=device)
    with torch.no_grad():
        experts[1].eval()
        results_1 = experts[1](noise_1, data_cond_1).cpu().numpy()

    noise_2 = torch.randn(len(data_cond_2), noise_dim, device=device)
    with torch.no_grad():
        experts[2].eval()
        results_2 = experts[2](noise_2, data_cond_2).cpu().numpy()

    results_0 = np.exp(results_0) - 1 # 40
    results_1 = np.exp(results_1) - 1 # 40
    results_2 = np.exp(results_2) - 1 # 40

    results_all[indx_0] = results_0.reshape(-1, 56, 30)
    results_all[indx_1] = results_1.reshape(-1, 56, 30)
    results_all[indx_2] = results_2.reshape(-1, 56, 30)
    return results_all


# Define the loss function
def regressor_loss(real_coords, fake_coords, aux_strength):
    # Ensure real_coords and fake_coords are on the same device
    # real_coords = real_coords.to(fake_coords.device)

    # Use in-place scaling if the scaler provides the scale and mean attributes
    # scale = torch.tensor(scaler_poz.scale_, device=fake_coords.device, dtype=torch.float32)
    # mean = torch.tensor(scaler_poz.mean_, device=fake_coords.device, dtype=torch.float32)
    #
    # # Scale fake_coords directly using PyTorch operations
    # fake_coords_scaled = (fake_coords - mean) / scale

    # Compute the MAE loss
    return F.mse_loss(fake_coords, real_coords) * aux_strength


def calculate_ws_ch_proton_model(n_calc, x_test, y_test, generator,
                                 ch_org, noise_dim, device, batch_size=64):
    ws = np.zeros(5)

    # Ensure y_test is a PyTorch tensor
    y_test = torch.tensor(y_test, device=device)

    num_samples = x_test.shape[0]

    for j in range(n_calc):  # Perform few calculations of the ws distance
        # appends the generated image to the specific indices of the num_batches
        results_all = get_predictions_from_generator_results(batch_size, num_samples, noise_dim,
                                                             device, y_test, generator)
        # now results_all contains all images in batch
        try:
            ch_gen = pd.DataFrame(sum_channels_parallel(results_all)).values
            for i in range(5):
                if len(ch_org[:, i]) > 0 and len(ch_gen[:, i]) > 0:
                    ws[i] += wasserstein_distance(ch_org[:, i], ch_gen[:, i])
        except ValueError as e:
            print('Error occurred while generating')
            print(e)
    ws = ws / n_calc  # Average over the number of calculations
    ws_mean = ws.mean()
    print("ws mean", f'{ws_mean:.2f}', end=" ")
    for n, score in enumerate(ws):
        print("ch" + str(n + 1), f'{score:.2f}', end=" ")
    return ws_mean


def generate_and_save_images(model, epoch, noise, noise_cond, x_test,
                             photon_sum_proton_min, photon_sum_proton_max,
                             device, random_generator, shape_images=(56, 30)):
    if noise_cond is None:
        return None
    SUPTITLE_TXT = f"\nModel: SDI-GAN data from {random_generator}" \
                   f"\nPhotonsum interval: [{photon_sum_proton_min}, {photon_sum_proton_max}]" \
                   f"\nEPOCH: {epoch}"

    # Set the model to evaluation mode
    model.eval()

    # Ensure y_test is a PyTorch tensor
    noise_cond = torch.tensor(noise_cond, device=device)

    # Generate predictions
    with torch.no_grad():
        noise = noise.to(device)
        noise_cond = noise_cond.to(device)
        predictions = model(noise, noise_cond).cpu().numpy().reshape(-1, *shape_images)  # Move to CPU for numpy operations

    fig, axs = plt.subplots(2, 6, figsize=(15, 5))
    fig.suptitle(SUPTITLE_TXT, x=0.1, horizontalalignment='left')
    for i in range(0, 12):
        if i < 6:
            x = x_test[i].reshape(*shape_images)
        else:
            x = predictions[i - 6]
        im = axs[i // 6, i % 6].imshow(x, cmap='gnuplot')
        axs[i // 6, i % 6].axis('off')
        fig.colorbar(im, ax=axs[i // 6, i % 6])

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return fig




def calculate_entropy(p):
    """
    Calculate entropy of a probability distribution p.
    """
    return -torch.sum(p * torch.log(p + 1e-9), dim=-1)




class StratifiedBatchSampler:
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)


def save_models(filepath_models, n_experts, aux_regs, aux_reg_optimizers,
                generators, generator_optimizers, discriminators, discriminator_optimizers,
                router_network, router_optimizer, epoch):
    try:
        for i in range(n_experts):
            torch.save(generators[i].state_dict(),
                       os.path.join(filepath_models, "gen_" + str(i) + "_" + str(epoch) + ".pth"))
            torch.save(generator_optimizers[i].state_dict(),
                       os.path.join(filepath_models, "gen_optim_" + str(i) + "_" + str(epoch) + ".pth"))
            torch.save(discriminators[i].state_dict(),
                       os.path.join(filepath_models, "disc_" + str(i) + "_" + str(epoch) + ".pth"))
            torch.save(discriminator_optimizers[i].state_dict(),
                       os.path.join(filepath_models, "disc_optim_" + str(i) + "_" + str(epoch) + ".pth"))
            torch.save(aux_regs[i].state_dict(), os.path.join(filepath_models,
                                                              "aux_reg_" + str(i) + "_" + str(epoch) + ".pth"))
            torch.save(aux_reg_optimizers[i].state_dict(),
                       os.path.join(filepath_models, "aux_reg_optim_" + str(i) + "_" + str(epoch) + ".pth"))

        # save router
        torch.save(router_network.state_dict(), os.path.join(filepath_models, f"router_network_{str(epoch)}.pth"))
        torch.save(router_optimizer.state_dict(), os.path.join(filepath_models, f"router_network_optim_{str(epoch)}.pth"))
    except Exception as e:
        print(f"Error saving models: {e}")


def save_models_and_architectures(filepath_models, n_experts, aux_regs, aux_reg_optimizers,
                                  generators, generator_optimizers, discriminators, discriminator_optimizers,
                                  router_network, router_optimizer, epoch):
    try:
        for i in range(n_experts):
            torch.save(generators[i],
                       os.path.join(filepath_models, "gen_" + str(i) + "_" + str(epoch) + ".pth"))
            torch.save(generator_optimizers[i].state_dict(),
                       os.path.join(filepath_models, "gen_optim_" + str(i) + "_" + str(epoch) + ".pth"))
            torch.save(discriminators[i],
                       os.path.join(filepath_models, "disc_" + str(i) + "_" + str(epoch) + ".pth"))
            torch.save(discriminator_optimizers[i].state_dict(),
                       os.path.join(filepath_models, "disc_optim_" + str(i) + "_" + str(epoch) + ".pth"))
            torch.save(aux_regs[i], os.path.join(filepath_models,
                                                              "aux_reg_" + str(i) + "_" + str(epoch) + ".pth"))
            torch.save(aux_reg_optimizers[i].state_dict(),
                       os.path.join(filepath_models, "aux_reg_optim_" + str(i) + "_" + str(epoch) + ".pth"))

        # save router
        torch.save(router_network, os.path.join(filepath_models, f"router_network_{str(epoch)}.pth"))
        torch.save(router_optimizer.state_dict(), os.path.join(filepath_models, f"router_network_optim_{str(epoch)}.pth"))
    except Exception as e:
        print(f"Error saving models: {e}")