import pandas as pd
import numpy as np
import argparse
import os

import torch
import torch.optim as optim

from utils import (
    dataset,
    models,
    test,
    train,
    utils,
    visualisation,
)

DATA_DIR  = os.path.join(os.path.abspath('.'), "data")
IMAGE_DIR = os.path.join(os.path.abspath("."), "images")
MODEL_DIR = os.path.join(os.path.abspath("."), "checkpoints")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure that all operations are deterministic for reproducibility, even on GPU (if used)
utils.set_seed(42)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

def main(config):
    """Centralised"""

    model = models.load_model(model_name=config["auto_encoder"]["type"], params=config["auto_encoder"]["args"])
    model.to(DEVICE)

    print(model)
    print("loading dataset")
    train_loader, val_loader, test_loader = dataset.load_data_ae(
        data_path=DATA_DIR,
        batch_size=config["data_loader_ae"]["args"]["batch_size"],
    )
    print("Dataset loaded!")

    # criterion = getattr(torch.nn, config["loss"]["type"])(**config["loss"]["args"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        required=True,
        help="Config file path. (default: None)"
    )
    args = parser.parse_args()

    config = utils.read_json(args.config)
    main(config)
