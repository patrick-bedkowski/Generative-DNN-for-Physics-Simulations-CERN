import os
import numpy as np
from scipy.stats import wasserstein_distance
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List
from sklearn.model_selection import StratifiedKFold

from utils import get_predictions_from_generator_results


def get_mean_std_from_expert_genrations(noise_cond, expert, device, batch_size=64, noise_dim=10):
    num_samples = noise_cond.shape[0]
    results_all = get_predictions_from_generator_results(batch_size, num_samples, noise_dim,
                                                         device, noise_cond, expert)
    print(f"Num of generated samples: {results_all.shape[0]}")
    photonsum_on_all_generated_images = np.sum(results_all, axis=(1, 2))
    photonsum_mean_generated_images = photonsum_on_all_generated_images.mean()
    photonsum_std_generated_images = photonsum_on_all_generated_images.std()
    return photonsum_mean_generated_images, photonsum_std_generated_images, photonsum_on_all_generated_images


def plot_proton_photonsum_histogreams(data_0, data_1, data_2, save_path=None):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize for better spacing

    axs[0].hist(data_0, bins=30, color='blue', alpha=0.7)
    axs[0].set_title('Data Condition 0')
    axs[0].set_xlabel('Proton Photon Sum')
    axs[0].set_ylabel('Frequency')
    axs[0].set_xscale('log')  # Set x-axis to logarithmic scale
    axs[0].set_yscale('log')  # Set y-axis to logarithmic scale

    axs[1].hist(data_1, bins=30, color='green', alpha=0.7)
    axs[1].set_title('Data Condition 1')
    axs[1].set_xlabel('Proton Photon Sum')
    axs[1].set_ylabel('Frequency')
    axs[1].set_xscale('log')  # Set x-axis to logarithmic scale
    axs[1].set_yscale('log')  # Set y-axis to logarithmic scale

    axs[2].hist(data_2, bins=30, color='red', alpha=0.7)
    axs[2].set_title('Data Condition 2')
    axs[2].set_xlabel('Proton Photon Sum')
    axs[2].set_ylabel('Frequency')
    axs[2].set_xscale('log')  # Set x-axis to logarithmic scale
    axs[2].set_yscale('log')  # Set y-axis to logarithmic scale

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_proton_photonsum_histogreams_shared(data_0, data_1, data_2, save_path=None):
    """
    Plots overlaid log-log histograms for three datasets and saves the figure.

    Parameters:
    - data1, data2, data3: Arrays or lists of data to plot.
    - save_path: String path where the plot will be saved.

    Returns:
    - fig: The figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figsize as needed

    ax.hist(data_0, bins=30, color='blue', alpha=0.5, label='Expert 0')
    ax.hist(data_1, bins=30, color='green', alpha=0.5, label='Expert 1')
    ax.hist(data_2, bins=30, color='red', alpha=0.5, label='Expert 2')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Proton Photon Sum')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of mean of generated predictions from expert.')

    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

    return fig

