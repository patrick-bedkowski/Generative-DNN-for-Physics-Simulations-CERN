import logging
import os

from torch.utils.data import DataLoader, TensorDataset, Subset
import torch

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from enum import Enum
from expertsim.utils.utils import (DIR_INFO, DIR_MODELS, create_dir, save_scales, save_train_test_indices,
                                   load_train_test_indices)


class ZDCType(Enum):
    PROTON = "proton"
    NEUTRON = "neutron"


def get_dataset(cfg):
    """
    Either gets proton or neutron dataset
    Gets cfg.dataset config.
    """
    if cfg.limit_samples is not None:
        data = pd.read_pickle(cfg.dataset.DATA_IMAGES_PATH)[:cfg.limit_samples]
        data_cond = pd.read_pickle(cfg.dataset.DATA_COND_PATH)[:cfg.limit_samples]
        data_posi = pd.read_pickle(cfg.dataset.DATA_POSITIONS_PATH)[:cfg.limit_samples]
    else:
        data = pd.read_pickle(cfg.dataset.DATA_IMAGES_PATH)
        data_cond = pd.read_pickle(cfg.dataset.DATA_COND_PATH)
        data_posi = pd.read_pickle(cfg.dataset.DATA_POSITIONS_PATH)

    def filter_data_by_intensity(cfg, data, data_cond, data_posi):
        """
        Filters datasets in place based on intensity thresholds.
        """
        # Corrected key checks (assuming correct config keys; fix based on actual cfg)
        min_intensity_threshold = cfg.dataset.MIN_INTENSITY_THRESHOLD
        max_intensity_threshold = cfg.dataset.MAX_INTENSITY_THRESHOLD

        if min_intensity_threshold is not None:
            logging.info(f"Filtering data with min intensity threshold: {min_intensity_threshold}")  # Use logging.info
            if cfg.dataset.zdc_type == 'proton':
                mask_min = data_cond['proton_photon_sum'] >= min_intensity_threshold
            else:
                mask_min = data_cond['neutron_photon_sum'] >= min_intensity_threshold
            data_cond.drop(data_cond[~mask_min].index, inplace=True)
            data = data[mask_min]  # Note: data might not have index alignment; adjust if needed
            data_posi.drop(data_posi[~mask_min].index, inplace=True)
            data_cond.reset_index(drop=True, inplace=True)
            data_posi.reset_index(drop=True, inplace=True)

        if max_intensity_threshold is not None:
            logging.info(f"Filtering data with max intensity threshold: {max_intensity_threshold}")
            if cfg.dataset.zdc_type == 'proton':
                mask_max = data_cond['proton_photon_sum'] <= max_intensity_threshold  # Corrected logic (use <= for max)
            else:
                mask_max = data_cond['neutron_photon_sum'] <= max_intensity_threshold  # Corrected logic (use <= for max)
            data_cond.drop(data_cond[~mask_max].index, inplace=True)
            data = data[mask_max]
            data_posi.drop(data_posi[~mask_max].index, inplace=True)
            data_cond.reset_index(drop=True, inplace=True)
            data_posi.reset_index(drop=True, inplace=True)

        return data, data_cond, data_posi

    data, data_cond, data_posi = filter_data_by_intensity(cfg, data, data_cond, data_posi)

    n_samples = cfg.dataset.read_n_samples
    if n_samples is not None:
        if cfg.dataset.zdc_type == 'proton':
            values = data_cond['proton_photon_sum']
        else:
            values = data_cond['neutron_photon_sum']

        # Define number of bins
        n_bins = 1000  # You can tune this number

        # Assign each sample to a bin
        bins = pd.qcut(values, q=n_bins, duplicates='drop')

        # Collect indices for samples selected uniformly from bins
        selected_indices = []

        # Number of samples per bin (try to split equally)
        samples_per_bin = max(1, n_samples // n_bins)

        for bin_interval in bins.unique():
            # Get all indices in this bin
            bin_indices = data_cond[bins == bin_interval].index.to_list()

            # If bin has fewer samples than needed, take all
            n_take = min(samples_per_bin, len(bin_indices))

            # Select uniformly (random without replacement) from bin
            selected = np.random.choice(bin_indices, size=n_take, replace=False)
            selected_indices.extend(selected)

        # If total selected less than n_samples, sample remainder randomly from all data
        if len(selected_indices) < n_samples:
            remaining = n_samples - len(selected_indices)
            all_indices = set(data_cond.index)
            remaining_indices = list(all_indices - set(selected_indices))
            extra_samples = np.random.choice(remaining_indices, size=remaining, replace=False)
            selected_indices.extend(extra_samples)

        # Use selected indices to subset datasets
        samples_indices = np.array(selected_indices)
        data = data[samples_indices]
        data_cond = data_cond.loc[samples_indices].reset_index(drop=True)
        data_posi = data_posi.loc[samples_indices].reset_index(drop=True)
        logging.info(f"Sampling {n_samples} uniform samples based on photon_sum distribution.")

    if cfg.dataset.zdc_type == 'proton':
        photon_sum_proton_min, photon_sum_proton_max = data_cond.proton_photon_sum.min(), data_cond.proton_photon_sum.max()
        cfg.photon_sum_min = float(photon_sum_proton_min)
        cfg.photon_sum_max = float(photon_sum_proton_max)
    else:
        photon_sum_proton_min, photon_sum_proton_max = data_cond.neutron_photon_sum.min(), data_cond.neutron_photon_sum.max()
        cfg.photon_sum_min = float(photon_sum_proton_min)
        cfg.photon_sum_max = float(photon_sum_proton_max)

    logging.info(f"Photon sum min: {photon_sum_proton_min}, max: {photon_sum_proton_max}")

    return data, data_cond, data_posi

def transform_data_for_training(cfg, data, data_cond, data_posi):
    experiments_dir_name = cfg.train.save_experiments_dir  # top directory with the experiments
    experiment_dir_name = cfg.config.experiment_dir  # earlier created directory
    print("config experiment difr:", experiment_dir_name)
    # if checkpoint is not set
    if cfg.train.checkpoint_experiment_dir is not None:
        experiment_dir_name = os.path.join(experiments_dir_name, experiment_dir_name)

    dir_info = DIR_INFO.format(EXPERIMENT_DIR_NAME=experiment_dir_name)
    dir_models = DIR_MODELS.format(EXPERIMENT_DIR_NAME=experiment_dir_name)
    cfg.train.dir_info = dir_info
    cfg.train.dir_models = dir_models

    zdc_type = cfg.dataset.zdc_type

    # GROUP CONDITIONAL DATA
    data_cond["cond"] = data_cond["Energy"].astype(str) + "|" + data_cond["Vx"].astype(str) + "|" + data_cond[
        "Vy"].astype(str) + "|" + data_cond["Vz"].astype(str) + "|" + data_cond["Px"].astype(str) + "|" + data_cond[
                            "Py"].astype(str) + "|" + data_cond["Pz"].astype(str) + "|" + data_cond["mass"].astype(
        str) + "|" + data_cond["charge"].astype(str)
    data_cond_id = data_cond[["cond"]].reset_index()
    ids = data_cond_id.merge(data_cond_id.sample(frac=1), on=["cond"], how="inner").groupby("index_x").first()
    ids = ids["index_y"]

    # data = np.log1p(data).astype(np.float32)
    data = data.astype(np.float32)
    indices = np.arange(len(data))
    data_2 = data[ids]
    data_cond = data_cond.drop(columns="cond")

    # Diversity regularization
    if zdc_type == ZDCType.PROTON.value:
        expert_number = data_cond.expert_number  # experts assignments for every sample (e.g. 0, 1, 2 for 3 experts)

        scaler = MinMaxScaler()
        std = data_cond["std_proton"].values.reshape(-1, 1)
        std = np.float32(std)
        std = scaler.fit_transform(std)

        # Intensity regularization
        intensity = data_cond["proton_photon_sum"].values.reshape(-1, 1)
        intensity = np.float32(intensity)

        data_cond = data_cond.drop(columns=["std_proton", "proton_photon_sum",
                                            'group_number_proton', 'expert_number'])
    elif zdc_type == ZDCType.NEUTRON.value:
        scaler = MinMaxScaler()
        print(data_cond.columns)
        std = data_cond["std"].values.reshape(-1, 1)
        std = np.float32(std)
        std = scaler.fit_transform(std)

        # Intensity regularization
        intensity = data_cond["neutron_photon_sum"].values.reshape(-1, 1)
        intensity = np.float32(intensity)

        # data_cond = data_cond.drop(columns=["std", "neutron_photon_sum"])
        data_cond = data_cond.drop(columns=["std", "neutron_photon_sum", "group_number"])
    else:
        raise ValueError("Unsupported ZDC type! Choose either proton or neutron.")

    # Auxiliary regressor
    scaler_poz = StandardScaler()
    data_xy = np.float32(data_posi.copy()[["max_x", "max_y"]])
    # data_xy = scaler_poz.fit_transform(data_xy)  # don't scale the auxiliary regressor coordinates

    data_cond_names = data_cond.columns
    scaler_cond = StandardScaler()
    data_cond = scaler_cond.fit_transform(data_cond.astype(np.float32))

    # TODO: implement logging here
    # logging.info('Load positions:', data_xy.shape, "cond max", data_xy.max(), "min", data_xy.min())
    # logging.info("std max", std.max(), "min", std.min())

    # Return data for training based on the saved indices
    if cfg.train.checkpoint_experiment_dir and cfg.train.epoch_to_load:
        train_indices, test_indices = load_train_test_indices(dir_info)
        x_train, x_test, x_train_2, x_test_2, y_train, y_test, std_train, std_test, \
        intensity_train, intensity_test, positions_train, positions_test = data[train_indices], data[test_indices], \
                                                  data_2[train_indices], data_2[test_indices], \
                                                  data_cond[train_indices], data_cond[test_indices], \
                                                  std[train_indices], std[test_indices],\
                                                  intensity[train_indices], intensity[test_indices], \
                                                  data_xy[train_indices], data_xy[test_indices]

        return x_train, x_test, x_train_2, x_test_2, y_train, y_test, std_train, std_test, \
        intensity_train, intensity_test, positions_train, positions_test, scaler_poz, data_cond_names, dir_models
    elif (cfg.train.checkpoint_experiment_dir is None) != (cfg.train.epoch_to_load is None):
        raise ValueError("You should set both checkpoint_experiment_dir and epoch_to_load parameters!")
    else:
        if zdc_type == ZDCType.PROTON.value:
            x_train, x_test, x_train_2, x_test_2, y_train, y_test, std_train, std_test, \
            intensity_train, intensity_test, positions_train, positions_test, \
            expert_number_train, expert_number_test, train_indices, test_indices = train_test_split(
                data, data_2, data_cond, std, intensity, data_xy, expert_number.values, indices,
                test_size=cfg.dataset.test_size, shuffle=cfg.dataset.shuffle_train_test_split)

            print("Data shapes:", x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        elif zdc_type == ZDCType.NEUTRON.value:
            x_train, x_test, x_train_2, x_test_2, y_train, y_test, std_train, std_test, \
            intensity_train, intensity_test, positions_train, positions_test,\
            train_indices, test_indices = train_test_split(
                data, data_2, data_cond, std, intensity, data_xy,  indices,
                test_size=cfg.dataset.test_size, shuffle=cfg.dataset.shuffle_train_test_split)

            expert_number_train, expert_number_test = np.zeros(len(train_indices)), np.zeros(len(test_indices))
            print("Data shapes:", x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        else:
            raise ValueError("Unsupported ZDC type!")

        # Save scales
        if cfg.train.save_experiment_data:
            create_dir(dir_info)
            save_scales(f"{zdc_type}", scaler_cond.mean_, scaler_cond.scale_, dir_info)
            create_dir(dir_models)
            save_train_test_indices(dir_info, train_indices=train_indices, test_indices=test_indices)
        else:
            dir_models = None

        if zdc_type == ZDCType.NEUTRON.value:
            return x_train, x_test, x_train_2, x_test_2, y_train, y_test, std_train, std_test, \
                   intensity_train, intensity_test, positions_train, positions_test, expert_number_train, expert_number_test, \
                   scaler_poz, data_cond_names, dir_models
        elif zdc_type == ZDCType.PROTON.value:
            return x_train, x_test, x_train_2, x_test_2, y_train, y_test, std_train, std_test, \
                   intensity_train, intensity_test, positions_train, positions_test, expert_number_train, \
                   expert_number_test, scaler_poz, data_cond_names, dir_models


def get_train_test_data_loaders(cfg):
    data, data_cond, data_posi = get_dataset(cfg)

    if cfg.dataset.zdc_type == ZDCType.PROTON.value:
        x_train, x_test, x_train_2, x_test_2, y_train, y_test, std_train, std_test, \
        intensity_train, intensity_test, positions_train, positions_test, expert_number_train, \
        expert_number_test, scaler_poz, data_cond_names, dir_models = transform_data_for_training(cfg, data, data_cond,
                                                                                                  data_posi)

        train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(x_train_2),
                                      torch.tensor(y_train), torch.tensor(std_train),
                                      torch.tensor(intensity_train), torch.tensor(positions_train))
        test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(x_test_2),
                                     torch.tensor(y_test), torch.tensor(std_test),
                                     torch.tensor(intensity_test), torch.tensor(positions_test))
        train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=8,
                                  pin_memory=True,  # Faster GPU transfer
                                  persistent_workers=True,  # Keep workers alive
                                  prefetch_factor=4  # Prefetch batches
                                  )
        test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=8,
                                 pin_memory=True, persistent_workers=True, prefetch_factor=4)

        cfg.data_cond_names = list(data_cond_names)
        return train_loader, test_loader

    elif cfg.dataset.zdc_type == ZDCType.NEUTRON.value:
        x_train, x_test, x_train_2, x_test_2, y_train, y_test, std_train, std_test, \
        intensity_train, intensity_test, positions_train, positions_test, expert_number_train, \
        expert_number_test, scaler_poz, data_cond_names, dir_models =\
            transform_data_for_training(cfg, data, data_cond, data_posi)

        train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(x_train_2),
                                      torch.tensor(y_train), torch.tensor(std_train),
                                      torch.tensor(intensity_train), torch.tensor(positions_train))
        test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(x_test_2),
                                     torch.tensor(y_test), torch.tensor(std_test),
                                     torch.tensor(intensity_test), torch.tensor(positions_test))
        train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=2,
                                  pin_memory=True,  # Faster GPU transfer
                                  persistent_workers=True,  # Keep workers alive
                                  prefetch_factor=2  # Prefetch batches
                                  )
        test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=2,
                                 pin_memory=True, persistent_workers=True, prefetch_factor=2)

        cfg.data_cond_names = list(data_cond_names)
        return train_loader, test_loader
    else:
        raise ValueError("Unsupported ZDC type! Choose either proton or neutron.")
