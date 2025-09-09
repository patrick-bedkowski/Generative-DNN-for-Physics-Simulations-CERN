import os
import matplotlib.pyplot as plt

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

import matplotlib.pyplot as plt
import numpy as np


def plot_proton_photonsum_histograms_shared(datasets, labels=None, save_path=None):
    """
    Plots outlined stacked histograms for multiple datasets using step histograms.

    Parameters:
    - datasets: A list of arrays or lists of data to plot.
    - labels: Optional list of labels for the datasets.
    - save_path: String path where the plot will be saved.

    Returns:
    - fig: The figure object containing the plot.
    """
    if len(datasets) == 0:
        raise ValueError("The datasets list must contain at least one dataset.")

    # Verify that labels, if provided, match the number of datasets
    if labels is not None and len(labels) != len(datasets):
        raise ValueError("Number of labels must match the number of datasets.")

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))  # Adjust figsize as needed

    # Define the bins based on the global min and max of all datasets
    all_data = np.concatenate(datasets)
    bins = np.linspace(np.min(all_data), np.max(all_data), 51)

    # Iterate over datasets and plot each dataset
    for i, data in enumerate(datasets):
        # Normalize the histogram values
        hist, _ = np.histogram(data, bins=bins)
        hist_normalized = hist  #/ np.sum(data)  # Frequency normalized by dataset sum

        # Use the label if provided, otherwise fallback to a default label
        label = labels[i] if labels is not None else f"Expert {i}"

        # Plot as a step histogram
        ax.step(bins[:-1], hist_normalized, where='post', label=label)

    # Set axis properties
    ax.set_yscale('log')  # Use logarithmic scale for the y-axis
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Proton Photon Sum')
    ax.set_title('Distribution of mean of generated predictions from experts.')
    ax.legend()

    # Save the plot if a save path is provided
    plt.tight_layout()
    if save_path:
        save_path = os.path.join(save_path, "proton_photonsum_histograms_shared.png")
        plt.savefig(save_path)

    return fig
