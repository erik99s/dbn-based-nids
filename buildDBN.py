import pandas as pd
import numpy as np
import argparse
import os

from logger import setup_logging

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

LOG_CONFIG_PATH = os.path.join(os.path.abspath("."), "logger", "logger_config.json")
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
    print("im here")

    utils.mkdir(LOG_DIR)
    setup_logging(save_dir=LOG_DIR, log_config=LOG_CONFIG_PATH)

    model = models.load_model(model_name=config["model"]["type"], params=config["model"]["args"])
    model.to(DEVICE)

    criterion = getattr(torch.nn, config["loss"]["type"])(**config["loss"]["args"])
    optimizer = [
            getattr(torch.optim, config["optimizer"]["type"])(params=m.parameters(), **config["optimizer"]["args"])
            for m in model.models
        ]
    
    print("model, criterion and optimizer loaded")
 
    train_loader, valid_loader, _, _, test_loader = dataset.load_data(
        data_path=DATA_DIR,
        batch_size=config["data_loader_ae"]["args"]["batch_size"],
        index = 1
    )

    print("start pre-training model")
    model.fit(train_loader)


    print("start training model")
    train_history = train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=config["trainer"]["num_epochs"],
        device=DEVICE
    )

    torch.save(model.state_dict(), "dbn_model.pth")

    print("done training")

    test_history = test(
        model=model,
        criterion=criterion,
        test_loader=test_loader,
        device=DEVICE
    )

    print("testing done")

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
