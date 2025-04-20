import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset, Subset

from data_transformations import transform_data_for_training, ZDCType, SCRATCH_PATH
from data_transformations import transform_data_for_training, ZDCType, SCRATCH_PATH
from training_setup import setup_experts, setup_router, load_checkpoint_weights
from training_utils import save_models_and_architectures
from utils import (sum_channels_parallel, calculate_ws_ch_proton_model,
                   calculate_joint_ws_across_experts,
                   create_dir, save_scales, evaluate_router,
                   intensity_regularization, sdi_gan_regularization,
                   generate_and_save_images,
                   calculate_expert_distribution_loss,
                   regressor_loss, calculate_expert_utilization_entropy,
                   StratifiedBatchSampler, plot_cond_pca_tsne, plot_expert_heatmap,
                   calculate_adaptive_load_balancing_loss)
from scipy.stats import wasserstein_distance, entropy
from sklearn.metrics import mean_absolute_error, r2_score


DATA_IMAGES_PATH = "/net/tscratch/people/plgpbedkowski/data/data_proton_photonsum_proton_1_2312.pkl"
DATA_COND_PATH = "/net/tscratch/people/plgpbedkowski/data/data_cond_photonsum_proton_1_2312.pkl"
DATA_POSITIONS_PATH = "/net/tscratch/people/plgpbedkowski/data/data_coord_proton_photonsum_proton_1_2312.pkl"
INPUT_IMAGE_SHAPE = (56, 30)
BATCH_SIZE = 256
LR_AUX = 1e-3
NUM_EPOCHS = 150


# Feature Extractor based on previous discriminator architecture
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.name = "feature_extractor_v1"
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 16, kernel_size=(3, 3)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

    def forward(self, x):
        x = self.conv_layers(x)
        features = x.mean([2, 3])  # Global average pooling
        return features


# Auxiliary Regressor that predicts coordinates
class AuxiliaryRegressor(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(AuxiliaryRegressor, self).__init__()
        self.name = "aux-reg-v1"
        self.feature_extractor = FeatureExtractor()

        self.flatten = nn.Flatten()
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # input shape torch.Size([256, 56, 30])
        x = x.unsqueeze(1)  # Adds dimension at index 1
        # The convolutional layer in your FeatureExtractor class expects 4D input in the format:
        # [BATCHSIZE, CHANNELS, HEIGHT, WIDTH]
        features = self.feature_extractor(x)  # out feature ext torch.Size([256, 16])
        coords = self.regressor(features)
        return coords

    @staticmethod
    def regressor_loss(real_coords, fake_coords):
        # Compute the MSE loss
        return F.mse_loss(fake_coords, real_coords)


# Training function for one epoch
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for real_images, _, _, _, _, coords in dataloader:
        images = real_images.to(device)
        coords = coords.to(device)

        optimizer.zero_grad()
        generated_positions = model(images)
        loss = criterion(coords, generated_positions)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # # Add latent space monitoring to training loop
        # if batch_idx % 50 == 0:
        #     with torch.no_grad():
        #         features = model.feature_extractor(images)
        #         wandb.log({"latent_features": wandb.Histogram(features.cpu())})
        #
        #         # Gradient flow analysis
        #         grads = [p.grad.abs().mean() for p in model.parameters()]
        #         wandb.log({"gradients": wandb.Histogram(grads)})

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


# Validation function
def validate(model, dataloader, criterion, scaler_poz, epoch, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    ede_list = []

    with torch.no_grad():
        for real_images, _, _, _, _, coords in dataloader:
            real_images = real_images.to(device)
            coords = coords.to(device)

            outputs = model(real_images)
            loss = criterion(coords, outputs)
            running_loss += loss.item() * real_images.size(0)

            # Inverse transform for metrics
            unscaled_preds = scaler_poz.inverse_transform(outputs.cpu())
            unscaled_true = scaler_poz.inverse_transform(coords.cpu())

            # Calculate Euclidean Distance Error
            ede = np.sqrt(np.sum((unscaled_preds - unscaled_true) ** 2, axis=1))
            ede_list.extend(ede.tolist())

            all_preds.extend(unscaled_preds.tolist())
            all_targets.extend(unscaled_true.tolist())

    # Convert to numpy arrays
    preds_array = np.array(all_preds)
    targets_array = np.array(all_targets)

    # Regression Metrics
    mae = mean_absolute_error(targets_array, preds_array)
    r2 = r2_score(targets_array, preds_array)
    rmse = np.sqrt(mean_squared_error(targets_array, preds_array))

    # Distribution Metrics
    wasserstein_x = wasserstein_distance(targets_array[:, 0], preds_array[:, 0])
    wasserstein_y = wasserstein_distance(targets_array[:, 1], preds_array[:, 1])

    # KL Divergence (using histogram approximation)
    hist_true, x_edges, y_edges = np.histogram2d(targets_array[:, 0], targets_array[:, 1], bins=50)
    hist_pred, _, _ = np.histogram2d(preds_array[:, 0], preds_array[:, 1], bins=(x_edges, y_edges))

    # Add small epsilon to avoid zero probabilities
    hist_true = hist_true.ravel() + 1e-10
    hist_pred = hist_pred.ravel() + 1e-10

    kl_div = entropy(hist_true, hist_pred)

    # Visualization plots
    fig = plot_predictions(targets_array, preds_array, epoch)
    error_heatmap = plot_error_heatmap(targets_array, preds_array)
    cumulative_error = plot_cumulative_error(ede_list)
    return_data = {
        'val_loss': running_loss / len(dataloader.dataset),
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'wasserstein_x': wasserstein_x,
        'wasserstein_y': wasserstein_y,
        'kl_div': kl_div,
        'ede_mean': np.mean(ede_list),
        'ede_std': np.std(ede_list),
        'prediction_plot': wandb.Image(fig),
        'error_heatmap': wandb.Image(error_heatmap),
        'cumulative_error': wandb.Image(cumulative_error)
    }

    # Add 3D visualization (only create every 5 epochs to save computation)
    if epoch % 5 == 0:
        features_3d_plot = plot_features_3d(model, dataloader, device)
        return_data['features_3d'] = wandb.Image(features_3d_plot)
        plt.close(features_3d_plot)

    # close the figures
    plt.close(fig)
    plt.close(error_heatmap)
    plt.close(cumulative_error)
    return return_data


def plot_predictions(targets, predictions, epoch):
    fig = plt.figure(figsize=(10, 10))  # Create explicit figure
    targets = np.array(targets)
    predictions = np.array(predictions)

    plt.scatter(targets[:, 0], targets[:, 1], c='blue', label='Ground Truth')
    plt.scatter(predictions[:, 0], predictions[:, 1], c='red', label='Predictions')
    plt.xlim([-10, 80])
    plt.ylim([-10, 40])
    plt.legend()
    plt.title(f'Coordinate Predictions - Epoch {epoch}')
    return fig  # Return figure object instead of pyplot state


def plot_error_heatmap(targets, preds):
    fig = plt.figure(figsize=(10, 8))  # Explicit figure creation
    errors = np.sqrt(np.sum((targets - preds) ** 2, axis=1))
    hb = plt.hexbin(targets[:, 0], targets[:, 1], C=errors, gridsize=50, cmap='inferno')
    plt.colorbar(hb, label='Euclidean Error')
    plt.title('Spatial Error Distribution Heatmap')
    return fig


def plot_cumulative_error(ede_list):
    fig = plt.figure(figsize=(10, 6))  # Create new figure
    sorted_errors = np.sort(ede_list)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    plt.plot(sorted_errors, cumulative, marker='.', linestyle='none')
    plt.xlabel('Euclidean Distance Error')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)
    return fig


# Add this new function to plot 3D features
def plot_features_3d(model, dataloader, device, max_samples=1000):
    """Extract and visualize features in 3D space"""
    features = []
    coords = []
    count = 0

    model.eval()
    with torch.no_grad():
        for real_images, _, _, _, _, target_coords in dataloader:
            if count >= max_samples:
                break

            real_images = real_images.to(device)

            # Get features from feature extractor
            if real_images.dim() == 3:
                real_images = real_images.unsqueeze(1)

            batch_features = model.feature_extractor(real_images).cpu().numpy()
            # batch_coords = scaler_poz.inverse_transform(target_coords)

            features.append(batch_features)
            coords.append(target_coords)

            count += real_images.shape[0]

    # Stack all batches
    features = np.vstack(features)[:max_samples]
    coords = np.vstack(coords)[:max_samples]

    print("feature shape: ", features.shape)

    # Reduce to 3D using PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    features_3d = pca.fit_transform(features)

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        features_3d[:, 0],
        features_3d[:, 1],
        features_3d[:, 2],
        c=coords[:, 0],  # Color by X coordinate
        cmap='viridis',
        alpha=0.7
    )

    plt.colorbar(scatter, label='X Coordinate Position')
    ax.set_title('Feature Extractor 3D Representation')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')

    return fig

def plot_distributions(train_coords, val_coords):
    """Plot coordinate distributions comparison"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].hist(train_coords[:, 0], bins=50, alpha=0.5, label='Train')
    ax[0].hist(val_coords[:, 0], bins=50, alpha=0.5, label='Val')
    ax[0].set_title('X-coordinate Distribution')
    ax[0].legend()

    ax[1].hist(train_coords[:, 1], bins=50, alpha=0.5, label='Train')
    ax[1].hist(val_coords[:, 1], bins=50, alpha=0.5, label='Val')
    ax[1].set_title('Y-coordinate Distribution')
    ax[1].legend()

    plt.tight_layout()
    return plt


def main():
    NAME = "working_auxiliary_regressor"

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data paths (update these to your actual paths)
    data = pd.read_pickle(DATA_IMAGES_PATH)
    data_cond = pd.read_pickle(DATA_COND_PATH)#[:20000]
    data_posi = pd.read_pickle(DATA_POSITIONS_PATH)

    photon_sum_proton_min, photon_sum_proton_max = data_cond.proton_photon_sum.min(), data_cond.proton_photon_sum.max()
    print('Loaded positions: ', data_posi.shape, "max:", data_posi.values.max(), "min:", data_posi.values.min())
    TAGS_RUN = ["param_sweep",
                f"proton_min_{photon_sum_proton_min}",
                f"proton_max_{photon_sum_proton_max}"]

    DATE_STR = datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f")[:-3]
    wandb_run_name = f"{NAME}_{DATE_STR}"
    EXPERIMENT_DIR_NAME = f"experiments/aux_reg/{wandb_run_name}"
    EXPERIMENT_DIR_NAME = os.path.join(SCRATCH_PATH, EXPERIMENT_DIR_NAME)
    CHECKPOINT_DATA_INDICES_FILE = None
    SAVE_EXPERIMENT_DATA = True


    x_train, x_test, x_train_2, x_test_2, y_train, y_test, std_train, std_test, \
    intensity_train, intensity_test, positions_train, positions_test, expert_number_train, \
    expert_number_test, scaler_poz, data_cond_names, filepath_models = transform_data_for_training(
        data_cond, data,
        data_posi,
        EXPERIMENT_DIR_NAME,
        ZDCType.PROTON,
        SAVE_EXPERIMENT_DATA,
        load_data_file_from_checkpoint=CHECKPOINT_DATA_INDICES_FILE)

    positions_train = scaler_poz.fit_transform(positions_train)
    positions_test = scaler_poz.transform(positions_test)

    # Separate datasets for each expert
    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(x_train_2),
                                  torch.tensor(y_train), torch.tensor(std_train),
                                  torch.tensor(intensity_train), torch.tensor(positions_train))

    # limited_train_dataset = Subset(train_dataset, range(10000))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(x_test_2),
                                 torch.tensor(y_test), torch.tensor(std_test),
                                 torch.tensor(intensity_test), torch.tensor(positions_test))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define model
    feature_dim = 16  # From the FeatureExtractor
    output_dim = 2  # Assuming 2D coordinates (x, y)
    model = AuxiliaryRegressor(feature_dim, output_dim).to(device)

    # Define loss function and optimizer
    criterion = model.regressor_loss
    optimizer = optim.Adam(model.parameters(), lr=LR_AUX)

    # Enhanced config tracking in main()
    config_wandb = {
        "aux_reg_architecture": model.name,
        "feature_extractor_architecture": model.feature_extractor.name,
        "optimizer": optimizer.__class__.__name__,
        "learning_rate": LR_AUX,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        # System info
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        # Model complexity
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        # Optimizer details
        "optimizer_state": {
            param_group_idx: {k: v for k, v in param_group.items() if k != 'params'}
            for param_group_idx, param_group in enumerate(optimizer.param_groups)
        }
    }

    # Initialize wandb
    run = wandb.init(
        project="auxiliary_regressor",
        entity="bedkowski-patrick",
        name=wandb_run_name,
        config=config_wandb,
        tags=TAGS_RUN
    )

    run.log_code("aux_regressor_test.py")

    # Log model architecture to wandb
    wandb.watch(model)

    # Training loop
    best_rmse = float('inf')
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, test_loader, criterion, scaler_poz, epoch, device)

        # Log all metrics
        log_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            **val_metrics
        }
        wandb.log(log_data)

        # Save best model
        if val_metrics['rmse'] < best_rmse:
            best_rmse = val_metrics['rmse']
            save_path = os.path.join(EXPERIMENT_DIR_NAME, "best_model.pth")
            torch.save({
                'model_state': model.state_dict(),
                'architecture': model.__class__.__name__,
                'feature_extractor': model.feature_extractor.name
            }, save_path)
            wandb.save(save_path)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val RMSE: {val_metrics['rmse']:.4f} | "
              f"MAE: {val_metrics['mae']:.4f} | "
              f"R²: {val_metrics['r2']:.4f}")

    # Final evaluation with best model
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state'])
    final_metrics = validate(model, test_loader, criterion, scaler_poz, device)
    wandb.log({'final_metrics': final_metrics})

    wandb.finish()


if __name__ == "__main__":
    main()
