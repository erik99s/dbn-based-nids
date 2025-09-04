from sklearn.metrics import classification_report, f1_score
import pandas as pd
import numpy as np
import argparse
import logging
import os

import torch
import torch.optim as optim

import sys

from logger import setup_logging
from utils import (
    dataset,
    models,
    test,
    train,
    utils,
    visualisation,
)

LOG_CONFIG_PATH = os.path.join(os.path.abspath("."),"logger", "logger_config.json")
LOG_DIR   = os.path.join(os.path.abspath("."), "logs")
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

    # Configure logging module
    utils.mkdir(LOG_DIR)
    setup_logging(save_dir=LOG_DIR, log_config=LOG_CONFIG_PATH)

    # loading the Auto Encoder

    logging.info("loading dataset for AE")
    _, _, test_loader = dataset.load_data_ae(
        data_path=DATA_DIR,
        batch_size=config["data_loader_ae"]["args"]["batch_size"],
    )
    logging.info("Datasets loaded")

    model = models.load_model(model_name=config["model"]["type"], params=config["model"]["args"])
    model.load_state_dict(torch.load("./stored_models/dbn_model_benign.pth"))
    model.to(DEVICE)

   
    criterion = getattr(torch.nn, config["loss"]["type"])(**config["loss"]["args"])
    optimizer = [getattr(torch.optim, config["optimizer"]["type"])(params=m.parameters(), **config["optimizer"]["args"])
        for m in model.models
    ]
       
    test_history = test(
        model=model,
        auto_encoder = model,
        criterion=criterion,
        criterionAE=criterion,
        test_loader=test_loader,
        device=DEVICE
    )



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
