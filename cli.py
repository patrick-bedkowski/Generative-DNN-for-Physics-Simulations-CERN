"""
CLI for launching experiments. Supports configs from expertsim/config/ and experiments/configs/.
Usage: python cli.py [--config <full_path>] [--override key=value ...]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

from expertsim.train.loop import train  # Adjust based on your training entry point
from expertsim.utils.data_transformations import get_train_test_data_loaders
from expertsim.utils.utils import append_experiment_dir_to_cfg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

import torch
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set environment variable for better CUDA errors
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def parse_args(args: List[str] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run Mixture-of-Experts GAN experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default="expertsim/config/default.yaml",  # Default in expertsim/config/
        help="Full path to the YAML config file (defaults to expertsim/config/default.yaml)"
    )
    parser.add_argument(
        "--override",
        "-o",
        nargs="*",
        default=[],
        help="Override config keys, e.g., train.epochs=100 model.n_experts=5"
    )
    return parser.parse_args(args)


def load_config(
    config_path: str,
    overrides: List[str],
) -> DictConfig:
    """
    Load the specified configuration dynamically using Hydra.

    Args:
        config_path: Full or relative path to the YAML config file.
        overrides: List of 'key=value' overrides.

    Returns:
        Loaded DictConfig with overrides applied.
    """
    # Resolve to absolute path relative to repo root
    repo_root = Path(__file__).parent
    p = repo_root / config_path
    if not p.exists():
        raise FileNotFoundError(f"Config file not found at {p}")

    config_dir = str(Path(config_path).parent)
    config_name = p.stem

    # Initialize Hydra with the derived directory
    hydra.initialize(
        version_base=None,
        config_path=config_dir,
    )

    # Compose the specific config by name
    user_cfg = hydra.compose(
        config_name=config_name,
        overrides=overrides,
    )

    # Resolve any interpolations
    OmegaConf.resolve(user_cfg)
    OmegaConf.set_struct(user_cfg, False)  # Allow modifications
    logger.info(f"Loaded config from {p} with overrides: {overrides}")
    return user_cfg


def main():
    args = parse_args(sys.argv[1:])

    try:
        cfg = load_config(args.config, args.override)

        # Initialize W&B logger
        # get name of the run
        append_experiment_dir_to_cfg(cfg)

        train_loader, test_loader = get_train_test_data_loaders(cfg)

        train(cfg, train_loader, test_loader)

        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"Error during run: {str(e)}")
        raise e
        sys.exit(1)


if __name__ == "__main__":
    main()
