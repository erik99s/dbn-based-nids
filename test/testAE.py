from sklearn.metrics import classification_report, f1_score
import pandas as pd
import numpy as np
import argparse
import logging
import os

import torch

import torch.nn as nn

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
    _, _, _, _, test_loader = dataset.load_data(
        data_path=DATA_DIR,
        batch_size=config["data_loader_ae"]["args"]["batch_size"],
        index = 1
    )
    logging.info("Datasets loaded")
   
    auto_encoder = models.load_model(model_name=config["auto_encoder_2nd"]["type"], params=config["auto_encoder_2nd"]["args"])
    auto_encoder.load_state_dict(torch.load("./stored_models/autoencoder_model_25_15.pth"))
    auto_encoder.to(DEVICE)

    criterion = nn.MSELoss(reduction='mean')

    print(auto_encoder)

    result = auto_encoder.test(
        criterion=criterion,        
        test_loader=test_loader,
        device=DEVICE
    )

    print(result)

    print("first test done")

    test_loader = dataset.load_filtered(
        data_path=DATA_DIR,
    )

    auto_encoder = models.load_model(model_name=config["auto_encoder_1st"]["type"], params=config["auto_encoder_1st"]["args"])
    auto_encoder.load_state_dict(torch.load("./stored_models/autoencoder_model_2nd_120.pth"))
    auto_encoder.to(DEVICE)

    print(auto_encoder)

    result = auto_encoder.testZero(
        criterion=criterion,        
        test_loader=test_loader,
        device=DEVICE
    )

    print(result)





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
