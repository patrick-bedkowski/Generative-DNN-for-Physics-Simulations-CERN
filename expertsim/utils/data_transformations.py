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
                                   load_train_test_indices, TRAIN_TEST_INDICES_FILENAME)


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
            mask_min = data_cond['proton_photon_sum'] >= min_intensity_threshold
            data_cond.drop(data_cond[~mask_min].index, inplace=True)
            data = data[mask_min].reset_index(drop=True)  # Note: data might not have index alignment; adjust if needed
            data.drop(data_posi[~mask_min].index, inplace=True)
            data_cond.reset_index(drop=True, inplace=True)
            data_posi.reset_index(drop=True, inplace=True)

        if max_intensity_threshold is not None:
            logging.info(f"Filtering data with max intensity threshold: {max_intensity_threshold}")
            mask_max = data_cond['proton_photon_sum'] <= max_intensity_threshold  # Corrected logic (use <= for max)
            data_cond.drop(data_cond[~mask_max].index, inplace=True)
            data = data[mask_max].reset_index(drop=True)
            data_posi.drop(data_posi[~mask_max].index, inplace=True)
            data_cond.reset_index(drop=True, inplace=True)
            data_posi.reset_index(drop=True, inplace=True)

        return data, data_cond, data_posi

    data, data_cond, data_posi = filter_data_by_intensity(cfg, data, data_cond, data_posi)

    n_samples = cfg.dataset.read_n_samples
    if n_samples is not None:
        samples_indices = np.random.choice(data_cond.index, size=n_samples, replace=False)
        data = data[samples_indices]
        data_cond = data_cond.loc[samples_indices].reset_index(drop=True)
        data_posi = data_posi.loc[samples_indices].reset_index(drop=True)
        logging.info(f"Sampling {n_samples} random samples from the dataset.")

    return data, data_cond, data_posi


def transform_data_for_training(cfg, data, data_cond, data_posi):
    experiments_dir_name = cfg.train.save_experiments_dir
    experiment_dir_name = cfg.config.experiment_dir

    # if checkpoint is not set
    if cfg.train.checkpoint_experiment_dir is not None:
        experiment_dir_name = os.path.join(experiments_dir_name, experiment_dir_name)

    dir_info = DIR_INFO.format(EXPERIMENT_DIR_NAME=experiment_dir_name)
    dir_models = DIR_MODELS.format(EXPERIMENT_DIR_NAME=experiment_dir_name)

    zdc_type = cfg.dataset.zdc_type

    # GROUP CONDITIONAL DATA
    data_cond["cond"] = data_cond["Energy"].astype(str) + "|" + data_cond["Vx"].astype(str) + "|" + data_cond[
        "Vy"].astype(str) + "|" + data_cond["Vz"].astype(str) + "|" + data_cond["Px"].astype(str) + "|" + data_cond[
                            "Py"].astype(str) + "|" + data_cond["Pz"].astype(str) + "|" + data_cond["mass"].astype(
        str) + "|" + data_cond["charge"].astype(str)
    data_cond_id = data_cond[["cond"]].reset_index()
    ids = data_cond_id.merge(data_cond_id.sample(frac=1), on=["cond"], how="inner").groupby("index_x").first()
    ids = ids["index_y"]

    data = np.log(data + 1).astype(np.float32)
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
        std = data_cond["std"].values.reshape(-1, 1)
        std = np.float32(std)
        std = scaler.fit_transform(std)

        # Intensity regularization
        intensity = data_cond["neutron_photon_sum"].values.reshape(-1, 1)
        intensity = np.float32(intensity)

        data_cond = data_cond.drop(columns=["std", "neutron_photon_sum",
                                            'group_number'])
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
    # logging.info("intensity max", intensity.max(), "min", intensity.min())

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
                   intensity_train, intensity_test, positions_train, positions_test, scaler_poz, data_cond_names, dir_models
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
        train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

        return train_loader, test_loader

    elif cfg.dataset.zdc_type == ZDCType.NEUTRON.value:
        x_train, x_test, x_train_2, x_test_2, y_train, y_test, std_train, std_test, \
        intensity_train, intensity_test, positions_train, positions_test, scaler_poz, data_cond_names, dir_models =\
            transform_data_for_training(cfg, data, data_cond, data_posi)

        # TODO: Write the proprer return code for NEUTRON. FURTHERMORE standarize the datasets to be well
        # structured together
    else:
        raise ValueError("Unsupported ZDC type! Choose either proton or neutron.")
