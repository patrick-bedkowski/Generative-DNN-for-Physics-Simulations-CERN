import os
import numpy as np
from scipy.stats import wasserstein_distance
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.ticker as mtick


from utils import get_predictions_from_generator_results, calculate_ws_ch_proton_model, sum_channels_parallel


def get_mean_std_from_expert_genrations(noise_cond, expert, device, batch_size=64, noise_dim=10):
    num_samples = noise_cond.shape[0]
    results_all = get_predictions_from_generator_results(batch_size, num_samples, noise_dim,
                                                         device, noise_cond, expert)
    print(f"Num of generated samples: {results_all.shape[0]}")
    print(results_all.shape)
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


# Update Matplotlib rcParams
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': 15,                    # Set font size to 15pt
    'axes.labelsize': 15,               # Axis labels
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2,
    'text.usetex': False,
    'pgf.rcfonts': False,
})

def plot_proton_photonsum_histogreams_shared(data_0, data_1, data_2, save_path=None):
    """
    Plots outlined stacked histograms for three datasets using step histograms.

    Parameters:
    - data_0, data_1, data_2: Arrays or lists of data to plot.
    - save_path: String path where the plot will be saved.

    Returns:
    - fig: The figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 10))  # Adjust figsize as needed

    # Define the bins
    bins = np.linspace(min(np.min(data_0), np.min(data_1), np.min(data_2)),
                       max(np.max(data_0), np.max(data_1), np.max(data_2)), 51)

    # Compute the histogram values for each dataset
    hist_0, _ = np.histogram(data_0, bins=bins)
    hist_1, _ = np.histogram(data_1, bins=bins)
    hist_2, _ = np.histogram(data_2, bins=bins)
    # eps = 1e-8
    # hist_0 = np.maximum(hist_0, eps)
    # hist_1 = np.maximum(hist_1, eps)
    # hist_2 = np.maximum(hist_2, eps)
    #

    # whole_len = len(data_0) + len(data_1) + len(data_2)
    hist_0 = hist_0 / sum(data_0)
    hist_1 = hist_1 / sum(data_1)
    hist_2 = hist_2 / sum(data_2)

    # Optionally convert fraction to percentage

    # Plot step histograms
    ax.step(bins[:-1], hist_0, where='post', color='blue', label='Expert 1')
    ax.step(bins[:-1], hist_1, where='post', color='green', label='Expert 2')
    ax.step(bins[:-1], hist_2, where='post', color='red', label='Expert 3')
    # Set axis scales and labels
    # Format y-axis ticks as percents
    # ax.set_ylim([1e-4, 100])
    ax.set_yscale('log')
    # ax.set_yticks([1e-2, 1e-1, 1, 10, 80, 100])
    # Define what label you want on each
    # ax.set_yticklabels(['0.01%', '0.1%', '1%', '10%', '80%', '100%'])
    ax.set_ylabel('Frequency')

    # ax.set_xlim(201, 1900)
    # ax.set_ylim(0, 0.01)
    ax.set_xlabel('Proton Photon Sum')
    ax.set_title('Distribution of mean of generated predictions from experts.')
    ax.legend()

    # PErcentage of sampels
    # hist_0, _ = np.histogram(data_0, bins=bins)
    # hist_1, _ = np.histogram(data_1, bins=bins)
    # hist_2, _ = np.histogram(data_2, bins=bins)
    # # eps = 1e-8
    # # hist_0 = np.maximum(hist_0, eps)
    # # hist_1 = np.maximum(hist_1, eps)
    # # hist_2 = np.maximum(hist_2, eps)
    # #
    # hist_0 *= 100
    # hist_1 *= 100
    # hist_2 *= 100
    #
    # # whole_len = len(data_0) + len(data_1) + len(data_2)
    # hist_0 = hist_0 / len(data_0)
    # hist_1 = hist_1 / len(data_1)
    # hist_2 = hist_2 / len(data_2)
    #
    # # Optionally convert fraction to percentage
    #
    # # Plot step histograms
    # ax.step(bins[:-1], hist_0, where='post', color='blue', label='Expert 1')
    # ax.step(bins[:-1], hist_1, where='post', color='green', label='Expert 2')
    # ax.step(bins[:-1], hist_2, where='post', color='red', label='Expert 3')
    # # Set axis scales and labels
    # # Format y-axis ticks as percents
    # # ax.set_ylim([1e-4, 100])
    # ax.set_yscale('log')
    # ax.set_yticks([1e-2, 1e-1, 1, 10, 80, 100])
    # # Define what label you want on each
    # ax.set_yticklabels(['0.01%', '0.1%', '1%', '10%', '80%', '100%'])
    # ax.set_ylabel('Percentage')
    #
    # # ax.set_xlim(201, 1900)
    # # ax.set_ylim(0, 0.01)
    # ax.set_xlabel('Proton Photon Sum')
    # ax.set_title('Distribution of mean of generated predictions from experts.')
    # ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

    return fig


def make_histograms(noise_cond, expert, ch_org, device, noise_dim=9, batch_size=64):

    num_samples = noise_cond.shape[0]
    results_all = get_predictions_from_generator_results(batch_size, num_samples, noise_dim,
                                                         device, noise_cond, expert)


    ch_gen = np.array(results_all).reshape(-1, 56, 44)
    ch_gen = pd.DataFrame(sum_channels_parallel(ch_gen)).values
    original = ch_org
    expert_gen = ch_gen

    fig, axis = plt.subplots(5, 1, figsize=(10, 14), sharex=False, sharey=False)
    fig.suptitle("TEST", x=0.1, horizontalalignment='left')

    for i in range(5):
        bins = np.linspace(0, 1500, 250)
        axis[i].set_title("Kanał " + str(i + 1))
        axis[i].hist(original[:, i], bins, alpha=0.5, label='true', color="red")
        axis[i].hist(expert_gen[:, i], bins, alpha=0.5, label='generated', color="blue")
        axis[i].legend(loc='upper right')
        axis[i].set_ylabel('Liczba przykładów')
        axis[i].set_xlabel('Wartość kanału')
        axis[i].set_yscale('log')

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    plt.show()
