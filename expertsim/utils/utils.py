import os
import numpy as np
from datetime import datetime

DIR_INFO = "{EXPERIMENT_DIR_NAME}/info/"  # dir for storing scales and indices of samples
DIR_MODELS = "{EXPERIMENT_DIR_NAME}/models/"


TRAIN_TEST_INDICES_FILENAME = "train_test_indices.npz"


def load_train_test_indices(checkpoint_data_filepath_dir: str):
    try:
        checkpoint_data_load_file = os.path.join(checkpoint_data_filepath_dir, TRAIN_TEST_INDICES_FILENAME)
        data_indices = np.load(checkpoint_data_load_file)
        print("Data train-test-split indices loaded!")
        return data_indices["train_indices"], data_indices["test_indices"]
    except FileNotFoundError:
        print("No data train-test-split indices found!")
        raise FileNotFoundError


def create_dir(path):
    is_exist = os.path.exists(path)
    if not is_exist:
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


def save_train_test_indices(filepath_dir: str, train_indices: np.array, test_indices: np.array):
    filepath = os.path.join(filepath_dir, TRAIN_TEST_INDICES_FILENAME)
    np.savez(filepath, train_indices=train_indices, test_indices=test_indices)
    print("Data train-test-split indices saved to", filepath)


def append_experiment_dir_to_cfg(cfg):
    date_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f")
    run_name_with_date = f"{cfg.config.run_name}_{date_str}"
    cfg.config.date = date_str  # save in the current folder
    cfg.wandb.run_name = run_name_with_date
    if cfg.train.save_experiments_dir is None:
        cfg.config.experiment_dir = run_name_with_date  # save in the current folder
    else:
        cfg.config.experiment_dir = os.path.join(cfg.train.save_experiments_dir, run_name_with_date)
