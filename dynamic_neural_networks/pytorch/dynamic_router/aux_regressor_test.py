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


LR_AUX = 1e-3


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
# Modified validate function with correct scaling
def validate(model, dataloader, criterion, scaler_poz, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for real_images, _, _, _, _, coords in dataloader:
            real_images = real_images.to(device)
            coords = coords.to(device)

            outputs = model(real_images)

            # Calculate loss in SCALED space
            loss = criterion(coords, outputs)
            running_loss += loss.item() * real_images.size(0)

            # Inverse transform only for metrics
            unscaled_preds = scaler_poz.inverse_transform(outputs.cpu())
            unscaled_true = scaler_poz.inverse_transform(coords.cpu())

            all_preds.extend(unscaled_preds.tolist())
            all_targets.extend(unscaled_true.tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

    return epoch_loss, rmse, all_preds, all_targets


# Function to visualize predicted vs actual coordinates
def plot_predictions(targets, predictions, epoch):
    plt.figure(figsize=(10, 10))
    targets = np.array(targets)
    predictions = np.array(predictions)

    plt.scatter(targets[:, 0], targets[:, 1], c='blue', label='Ground Truth')
    plt.scatter(predictions[:, 0], predictions[:, 1], c='red', label='Predictions')
    plt.xlim([-10, 80])
    plt.ylim([-10, 40])
    plt.legend()
    plt.title(f'Coordinate Predictions - Epoch {epoch}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    return plt


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



DATA_IMAGES_PATH = "/net/tscratch/people/plgpbedkowski/data/data_proton_photonsum_proton_1_2312.pkl"
DATA_COND_PATH = "/net/tscratch/people/plgpbedkowski/data/data_cond_photonsum_proton_1_2312.pkl"
DATA_POSITIONS_PATH = "/net/tscratch/people/plgpbedkowski/data/data_coord_proton_photonsum_proton_1_2312.pkl"
INPUT_IMAGE_SHAPE = (56, 30)
BATCH_SIZE = 256


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

    config_wandb = {
           "aux_reg_architecture": model.name,
           "feature_extractor_architecture": model.feature_extractor.name,
    }

    # Initialize wandb
    run = wandb.init(
        project="auxiliary_regressor",
        entity="bedkowski-patrick",
        name=wandb_run_name,
        config=config_wandb,
        tags=TAGS_RUN
    )


    # Log model architecture to wandb
    wandb.watch(model)

    # Training loop
    num_epochs = 100
    best_rmse = float('inf')

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_rmse, predictions, targets = validate(model, test_loader, criterion, scaler_poz, device)

        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_rmse': val_rmse,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        # Plot coordinate distributions
        dist_plot = plot_distributions(positions_train, positions_test)
        wandb.log({"coordinate_distributions": wandb.Image(dist_plot)})
        dist_plot.close()

        # # Save best model
        # if val_rmse < best_rmse:
        #     best_rmse = val_rmse
        #     torch.save(model.state_dict(), 'best_auxiliary_regressor.pth')
        #     wandb.save('best_auxiliary_regressor.pth')

        # Visualize predictions every 5 epochs
        if epoch % 5 == 0:
            plt = plot_predictions(targets, predictions, epoch)
            wandb.log({"prediction_plot": wandb.Image(plt)})
            plt.close()

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}")

    # # Final evaluation with best model
    # model.load_state_dict(torch.load('best_auxiliary_regressor.pth'))
    # _, final_rmse, final_preds, final_targets = validate(model, test_loader, criterion, scaler_poz, device)

    wandb.finish()


if __name__ == "__main__":
    main()
