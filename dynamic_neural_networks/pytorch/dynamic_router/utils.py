import os
import numpy as np
from scipy.stats import wasserstein_distance
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List

import tqdm


def get_channel_masks(input_array: np.ndarray):
    """
    Returns masks of 5 for input array.

    input_array: Array of shape(N, M)
    """

    # Create a copy of the input array to use as the mask
    mask = np.ones_like(input_array)
    n, m = input_array.shape

    # Define the pattern of checks
    pattern = np.array([[0, 1], [1, 0]])

    # Fill the input array with the pattern
    for i in range(n):
        for j in range(m):
            mask[i, j] = pattern[i % 2, j % 2]

    mask5 = np.ones_like(input_array) - mask

    # Divide the mask into four equal rectangles
    rows, cols = mask.shape
    mid_row, mid_col = rows // 2, cols // 2

    mask1 = mask.copy()
    mask2 = mask.copy()
    mask3 = mask.copy()
    mask4 = mask.copy()

    mask4[mid_row:, :] = 0
    mask4[:, :mid_col] = 0

    mask2[:, :mid_col] = 0
    mask2[:mid_row, :] = 0

    mask3[mid_row:, :] = 0
    mask3[:, mid_col:] = 0

    mask1[:, mid_col:] = 0
    mask1[:mid_row, :] = 0

    return mask1, mask2, mask3, mask4, mask5


def sum_channels_parallel(data: np.ndarray):
    """
    Calculates the sum of 5 channels of input images. Each Input image is divided into 5 sections.

    data: Array of shape(x, N, M)
        Array of x images of the same size.
    """
    mask1, mask2, mask3, mask4, mask5 = get_channel_masks(data[0])

    ch1 = (data * mask1).sum(axis=1).sum(axis=1)
    ch2 = (data * mask2).sum(axis=1).sum(axis=1)
    ch3 = (data * mask3).sum(axis=1).sum(axis=1)
    ch4 = (data * mask4).sum(axis=1).sum(axis=1)
    ch5 = (data * mask5).sum(axis=1).sum(axis=1)

    return zip(ch1, ch2, ch3, ch4, ch5)


def get_max_value_image_coordinates(img):
    """

    :param img: Input image of any shape
    :return: Tuple with (X, Y) coordinates
    """
    return np.unravel_index(np.argmax(img), img.shape)


def create_dir(path, SAVE_EXPERIMENT_DATA):
    if SAVE_EXPERIMENT_DATA:
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)


def save_scales(model_name, scaler_means, scaler_scales, filepath):
    out_fnm = f"{model_name}_scales.txt"
    res = "#means"
    for mean_ in scaler_means:
        res += "\n" + str(mean_)
    res += "\n\n#scales"
    for scale_ in scaler_scales:
        res += "\n" + str(scale_)

    with open(filepath+out_fnm, mode="w") as f:
        f.write(res)


# def calculate_ws_ch_proton_model(n_calc, x_test, y_test, generator,
#                                  ch_org, noise_dim, device, batch_size=64):
#     ws = np.zeros(5)
#
#     # Ensure y_test is a PyTorch tensor
#     y_test = torch.tensor(y_test, device=device)
#
#     num_samples = x_test.shape[0]
#     num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate number of batches
#
#     for j in range(n_calc):  # Perform few calculations of the ws distance
#         batch_ws = np.zeros(5)
#
#         for batch_idx in range(num_batches):
#             start_idx = batch_idx * batch_size
#             end_idx = min(start_idx + batch_size, num_samples)
#
#             noise = torch.randn(end_idx - start_idx, noise_dim, device=device)
#             noise_cond = y_test[start_idx:end_idx]
#
#             # Generate results using the generator
#             with torch.no_grad():
#                 results = generator(noise, noise_cond).cpu().numpy()
#
#             results = np.exp(results) - 1
#
#             try:
#                 ch_gen = np.array(results).reshape(-1, 56, 30)
#                 ch_gen = pd.DataFrame(sum_channels_parallel(ch_gen)).values
#                 for i in range(5):
#                     if len(ch_org[start_idx:end_idx, i]) > 0 and len(ch_gen[:, i]) > 0:
#                         batch_ws[i] += wasserstein_distance(ch_org[start_idx:end_idx, i], ch_gen[:, i])
#             except ValueError as e:
#                 print('Error occurred while generating')
#                 print(e)
#
#         ws += batch_ws / num_batches  # Average the batch WS distances
#
#     ws = ws / n_calc  # Average over the number of calculations
#     ws_mean = ws.mean()
#     print("ws mean", f'{ws_mean:.2f}', end=" ")
#     for n, score in enumerate(ws):
#         print("ch" + str(n + 1), f'{score:.2f}', end=" ")
#     return ws_mean


# def calculate_ws_ch_proton_model(n_calc, x_test, y_test, generator,
#                                  ch_org, noise_dim, device):
#     num_samples = x_test.shape[0]
#     ws = np.zeros(5)
#     for j in range(n_calc):  # Perform few calculations of the ws distance
#         noise = torch.randn(num_samples, noise_dim, device=device)
#         noise_cond = torch.tensor(y_test, device=device)
#
#         # Generate results using the generator
#         with torch.no_grad():
#             results = generator(noise, noise_cond).detatch().numpy()
#         results = np.exp(results) - 1
#         try:
#             ch_gen = np.array(results).reshape(-1, 56, 30)
#             ch_gen = pd.DataFrame(sum_channels_parallel(ch_gen)).values
#             for i in range(5):
#                 ws[i] += wasserstein_distance(ch_org[:, i], ch_gen[:, i])
#         except ValueError as e:
#             print('Error occurred while generating')
#             print(e)
#
#     ws = ws / n_calc  # Average over the number of calculations
#     ws_mean = ws.mean()
#     print("ws mean", f'{ws_mean:.2f}', end=" ")
#     for n, score in enumerate(ws):
#         print("ch" + str(n + 1), f'{score:.2f}', end=" ")
#     return ws_mean

def calculate_joint_ws_across_experts(n_calc, x_tests: List, y_tests: List, generators: List,
                                      ch_org, ch_org_expert, noise_dim, device, batch_size=64, n_experts=3):
    """
    Calculates the WS distance across the whole distribution.
    """
    # if length of data is not the same, raise an error
    if len(x_tests) != len(y_tests) or len(x_tests) != len(generators):
        raise ValueError("Length of data is not the same")

    # gather all predictions
    ws = np.zeros(5)
    ws_exp = np.zeros((n_experts, 5))  # ws for each expert
    for j in range(n_calc):  # Perform few calculations of the ws distance
        ch_gen_all = []  # for gathering the whole generated distribution of pixels
        ch_gen_expert = []
        for generator_idx in range(len(generators)):  # for all generators
            y_test_temp = torch.tensor(y_tests[generator_idx], device=device)
            num_samples = x_tests[generator_idx].shape[0]

            if num_samples == 0:
                continue
            results_all = get_predictions_from_generator_results(batch_size, num_samples, noise_dim,
                                                                 device, y_test_temp, generators[generator_idx])
            print(f"for generator {generator_idx}. Samples generated: {results_all.shape}, real_samples: {num_samples}")
            ch_gen_smaller = pd.DataFrame(sum_channels_parallel(results_all)).values
            ch_gen_expert.append(ch_gen_smaller.copy())
            ch_gen_all.extend(ch_gen_smaller.copy())
        ch_gen_all = np.array(ch_gen_all)  # all generated predictions
        print("shape of all generated:", ch_gen_all.shape)
        for i in range(5):
            ws[i] = ws[i] + wasserstein_distance(ch_org[:, i], ch_gen_all[:, i])  # for all generations
            # Calculate separate WS distance for expert
            for exp_idx in range(len(generators)):
                ws_exp[exp_idx][i] += wasserstein_distance(ch_org_expert[exp_idx][:, i], ch_gen_expert[exp_idx][:, i])

        ws = np.array(ws)  # across 5 runs
        ws_exp = np.array(ws_exp)  # across 5 runs
    ws = ws / n_calc  # average per calculation
    ws_exp = ws_exp / n_calc  # average per calculation
    ws_mean = ws.mean()
    print("WS shape of expert 0",  ws_exp[0].shape)
    ws_mean_0 = ws_exp[0].mean()
    ws_mean_1 = ws_exp[1].mean()
    ws_mean_2 = ws_exp[2].mean()
    print("ws mean", f'{ws_mean:.2f}', end=" ")
    for n, score in enumerate(ws):
        print("ch" + str(n + 1), f'{score:.2f}', end=" ")
    return ws_mean, ws_mean_0, ws_mean_1, ws_mean_2


def get_predictions_from_generator_results(batch_size, num_samples, noise_dim,
                                           device, y_test, generator):
    num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate number of batches
    results_all = np.zeros((num_samples, 56, 30))
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        noise = torch.randn(end_idx - start_idx, noise_dim, device=device)
        noise_cond = y_test[start_idx:end_idx]

        # Generate results using the generator
        with torch.no_grad():
            generator.eval()
            results = generator(noise, noise_cond).cpu().numpy()

        results = np.exp(results) - 1
        results_all[start_idx:end_idx] = results.reshape(-1, 56, 30)
    return results_all


# Define the loss function
def regressor_loss(real_coords, fake_coords, scaler_poz, AUX_STRENGTH):
    # Ensure real_coords and fake_coords are on the same device
    # real_coords = real_coords.to(fake_coords.device)

    # Use in-place scaling if the scaler provides the scale and mean attributes
    # scale = torch.tensor(scaler_poz.scale_, device=fake_coords.device, dtype=torch.float32)
    # mean = torch.tensor(scaler_poz.mean_, device=fake_coords.device, dtype=torch.float32)
    #
    # # Scale fake_coords directly using PyTorch operations
    # fake_coords_scaled = (fake_coords - mean) / scale

    # Compute the MAE loss
    return F.mse_loss(fake_coords, real_coords) * AUX_STRENGTH


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


def evaluate_router(router_network, y_test, expert_number_test, accuracy_metric, precision_metric, recall_metric, f1_metric):
    router_network.eval()
    with torch.no_grad():
        predicted_expert = router_network(y_test)
        _, predicted_labels = torch.max(predicted_expert, 1)

        accuracy = accuracy_metric(predicted_labels, expert_number_test).cpu().item()
        precision = precision_metric(predicted_labels, expert_number_test).cpu().item()
        recall = recall_metric(predicted_labels, expert_number_test).cpu().item()
        f1 = f1_metric(predicted_labels, expert_number_test).cpu().item()

    return accuracy, precision, recall, f1


def sdi_gan_regularization(fake_latent, fake_latent_2, noise, noise_2, std, DI_STRENGTH):
    # Calculate the absolute differences and their means along the batch dimension
    abs_diff_latent = torch.mean(torch.abs(fake_latent - fake_latent_2), dim=1)
    abs_diff_noise = torch.mean(torch.abs(noise - noise_2), dim=1)

    # Compute the division term
    div = abs_diff_latent / (abs_diff_noise + 1e-5)

    # Calculate the div_loss
    div_loss = std * DI_STRENGTH / (div + 1e-5)

    # Calculate the final div_loss
    div_loss = torch.mean(std) * torch.mean(div_loss)

    return div_loss


def intensity_regularization(gen_im_proton, intensity_proton, scaler_intensity, IN_STRENGTH):
    """
    Computes the intensity regularization loss for generated images, returning the loss, the sum of intensities per image,
    and the mean and standard deviation of the intensity across the batch.

    Args:
        gen_im_proton (torch.Tensor): A tensor of generated images with shape [batch_size, channels, height, width].
        intensity_proton (torch.Tensor): A tensor representing the target intensity values for the batch, with shape [batch_size].
        scaler_intensity (object): A scaler object used to normalize the intensity values, typically an instance of a
                                   scaler from `sklearn.preprocessing` (e.g., `StandardScaler`).
        IN_STRENGTH (float): A scalar that controls the strength of the intensity regularization in the final loss.

    Returns:
        torch.Tensor: The intensity regularization loss, calculated as the Mean Absolute Error (MAE) between the scaled
                      sum of the intensities in the generated images and the target intensities, multiplied by `IN_STRENGTH`.
        torch.Tensor: The sum of intensities in each generated image, with shape [n_samples, 1].
        torch.Tensor: The standard deviation of the scaled intensity values across the batch.
        torch.Tensor: The mean of the scaled intensity values across the batch.
    """

    # Sum the intensities in the generated images
    sum_all_axes_p = torch.sum(gen_im_proton, dim=[2, 3], keepdim=True)  # Sum along all but batch dimension
    sum_all_axes_p = sum_all_axes_p.reshape(-1, 1)  # Scale and reshape back to (batch_size, 1)

    # Sum the intensities in the generated images
    gen_im_proton_rescaled = torch.exp(gen_im_proton.clone().detach()) - 1
    sum_all_axes_p_rescaled = torch.sum(gen_im_proton_rescaled, dim=[2, 3], keepdim=True)  # Sum along all but batch dimension
    sum_all_axes_p_rescaled = sum_all_axes_p_rescaled.reshape(-1, 1)  # Scale and reshape back to (batch_size, 1)

    # Compute mean and std as PyTorch tensors
    std_intensity_scaled = sum_all_axes_p_rescaled.std()
    mean_intensity_scaled = sum_all_axes_p_rescaled.mean()
    # remove the log from the predictions
    # print(sum_all_axes_p_rescaled)
    # print(mean_intensity_scaled)

    # DELETED THE SCALING

    # Manually scale using the parameters from the fitted MinMaxScaler
    # data_min = torch.tensor(scaler_intensity.data_min_, device=gen_im_proton.device, dtype=torch.float32)
    # data_max = torch.tensor(scaler_intensity.data_max_, device=gen_im_proton.device, dtype=torch.float32)
    #
    # sum_all_axes_p_scaled = (sum_all_axes_p - data_min) / (data_max - data_min)
    # Ensure intensity_proton is correctly shaped and on the same device
    intensity_proton = intensity_proton.view(-1, 1).to(gen_im_proton.device)  # Ensure it is of shape [batch_size, 1]

    # Calculate MAE loss
    mae_value_p = F.l1_loss(sum_all_axes_p, intensity_proton)
    return IN_STRENGTH * mae_value_p, sum_all_axes_p, std_intensity_scaled.detach(), mean_intensity_scaled.detach()


def generate_and_save_images(model, epoch, noise, noise_cond, x_test,
                             photon_sum_proton_min, photon_sum_proton_max,
                             device, random_generator):
    SUPTITLE_TXT = f"\nModel: SDI-GAN proton data from generator {random_generator}" \
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
        predictions = model(noise, noise_cond).cpu().numpy()  # Move to CPU for numpy operations

    fig, axs = plt.subplots(2, 6, figsize=(15, 5))
    fig.suptitle(SUPTITLE_TXT, x=0.1, horizontalalignment='left')

    for i in range(0, 12):
        if i < 7:
            x = x_test[20 + i].reshape(56, 30)
        else:
            x = predictions[i - 6].reshape(56, 30)
        im = axs[i // 6, i % 6].imshow(x, cmap='gnuplot')
        axs[i // 6, i % 6].axis('off')
        fig.colorbar(im, ax=axs[i // 6, i % 6])

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return fig


def calculate_expert_distribution_loss(gating_probs, features, lambda_reg=0.1):
    """
    Calculate the regularization loss for the router network to encourage balanced task distribution among experts.

    Args:
        gating_probs (torch.Tensor): The gating probabilities for each sample and expert with shape (batch_size, num_experts).
        features (torch.Tensor): The feature representations of the inputs with shape (batch_size, feature_dim).
        lambda_reg (float): The regularization strength.

    Returns:
        torch.Tensor: The calculated regularization loss.
    """
    # reshape the features from shape (batch_size) to (batch_size, 1)
    # Compute the pairwise Euclidean distance between the features
    pairwise_distances = torch.cdist(features, features, p=2)  # Shape: (batch_size, batch_size)
    # print(pairwise_distances.shape)

    # Assuming gating_probs is originally of type Long
    # print(gating_probs.shape)

    # Compute the gating similarities (dot product of gating probabilities for each pair of samples)
    gating_similarities = torch.matmul(gating_probs, gating_probs.T)  # Shape: (batch_size, batch_size)

    # print(gating_similarities.shape, pairwise_distances.shape)
    # Regularization loss: sum of (gating_similarities * pairwise_distances)
    reg_loss = torch.sum(gating_similarities * pairwise_distances) / gating_similarities.size(0)

    # Apply the regularization strength
    reg_loss = lambda_reg * reg_loss

    return reg_loss


def calculate_entropy(p):
    """
    Calculate entropy of a probability distribution p.
    """
    return -torch.sum(p * torch.log(p + 1e-9), dim=-1)


def calculate_expert_utilization_entropy(gating_probs, ENT_STRENGTH=0.1):
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
    return entropy * ENT_STRENGTH



