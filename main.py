from sklearn.metrics import classification_report, f1_score
import pandas as pd
import numpy as np
import argparse
import logging
import os

import torch
import torch.optim as optim

from logger import setup_logging
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

    # Configure logging module
    utils.mkdir(LOG_DIR)
    setup_logging(save_dir=LOG_DIR, log_config=LOG_CONFIG_PATH)

    # loading the deep belief network
    logging.info(f'######## Training the {config["name"]} model ########')
    model = models.load_model(model_name=config["model"]["type"], params=config["model"]["args"])
    model.to(DEVICE)

    # loading the Auto Encoder
    auto_encoder = models.load_model(model_name=config["auto_encoder"]["type"], params=config["auto_encoder"]["args"])
    auto_encoder.to(DEVICE)

    logging.info("loading dataset for AE")
    train_loader, valid_loader, test_loader = dataset.load_data_ae(
        data_path=DATA_DIR,
        batch_size=config["data_loader_ae"]["args"]["batch_size"],
    )
    logging.info("Datasets loaded")
    criterion = getattr(torch.nn, config["loss"]["type"])(**config["loss"]["args"])
    if config["model"]["type"] == "DBN":
        optimizer = [
            getattr(torch.optim, config["optimizer"]["type"])(params=m.parameters(), **config["optimizer"]["args"])
            for m in model.models
        ]
        
        # Pre-train the DBN model
        logging.info("Start pre-training the DBN...")
        model.fit(train_loader)
    else:
        optimizer = [getattr(torch.optim, config["optimizer"]["type"])(params=model.parameters(), **config["optimizer"]["args"])]
    
    criterionAE = getattr(torch.nn, config["lossAE"]["type"])(**config["loss"]["args"])
    optimizerAE = getattr(torch.optim, config["optimizer"]["type"])( params=auto_encoder.parameters(), **config["optimizer"]["args"])

    logging.info("Start training the DBN...")
    train_history = train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=config["trainer"]["num_epochs"],
        device=DEVICE
    )
    print(model)
    logging.info(f'{config["name"]} model trained!')

    torch.save(model.state_dict(), "dbn_model_.pth")
   
    logging.info("start traning the AE")
    auto_encoder.fit(
        criterion=criterionAE,
        optimizer=optimizerAE,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=DEVICE
    )
    logging.info("AE trained")

    torch.save(auto_encoder.state_dict(), "autoencoder_model.pth")



    train_output_true = train_history["train"]["output_true"]
    train_output_pred = train_history["train"]["output_pred"]
    valid_output_true = train_history["valid"]["output_true"]
    valid_output_pred = train_history["valid"]["output_pred"]

    """
    printed from processing
    0    407115
    3     64142
    5     11346
    2      1672
    6       464
    1       398
    4         8
    """


    labels = ['Benign', 'Bot', 'Brute Force', 'DoS', 'PortScan']
    # labels = ['Benign', 'Known']

    """
    logging.info('Training Set -- Classification Report')
    logging.info(classification_report(
        y_true=train_output_true,
        y_pred=train_output_pred,
        target_names=labels
    ))
    """
    
    visualisation.plot_confusion_matrix(
        y_true=train_output_true,
        y_pred=train_output_pred,
        labels=labels,
        save=True,
        save_dir=IMAGE_DIR,
        filename=f'{config["name"]}_train_confusion_matrix.pdf'
    )

    ## Validation Set results
    """
    logging.info('Validation Set -- Classification Report')
    logging.info(classification_report(
        y_true=valid_output_true,
        y_pred=valid_output_pred,
        target_names=labels
    ))
    """

    visualisation.plot_confusion_matrix(
        y_true=valid_output_true,
        y_pred=valid_output_pred,
        labels=labels,
        save=True,
        save_dir=IMAGE_DIR,
        filename=f'{config["name"]}_train_confusion_matrix.pdf'
    )


    logging.info(f'Evaluate {config["name"]} model')
    test_history = test(
        model=model,
        auto_encoder = auto_encoder,
        criterion=criterion,
        criterionAE=criterionAE,
        test_loader=test_loader,
        device=DEVICE
    )


    test_output_true = test_history["test"]["output_true"]
    test_output_pred = test_history["test"]["output_pred"]
    test_output_pred_prob = test_history["test"]["output_pred_prob"]

    labels = ['Benign', 'Bot', 'Brute Force', 'DoS', 'PortScan', 'ZeroDay']
    # labels = ['Benign', 'Known', 'ZeroDay']
    
    ## Testing Set results
    logging.info(f'Testing Set -- Classification Report {config["name"]}\n')
    logging.info(classification_report(
        y_true=test_output_true,
        y_pred=test_output_pred,
        target_names=labels
    ))

    utils.mkdir(IMAGE_DIR)
    visualisation.plot_confusion_matrix(
        y_true=test_output_true,
        y_pred=test_output_pred,
        labels=labels,  
        save=True,
        save_dir=IMAGE_DIR,
        filename=f'{config["name"]}_test_confusion_matrix.pdf'
    )

    y_test = pd.get_dummies(test_output_true).values
    y_score = np.array(test_output_pred_prob)

    # Plot ROC curve
    visualisation.plot_roc_curve(
        y_test=y_test,
        y_score=y_score,
        labels=labels,
        save=True,
        save_dir=IMAGE_DIR,
        filename=f'{config["name"]}_roc_curve.pdf'
    )

    # Plot Precision vs. Recall curve
    visualisation.plot_precision_recall_curve(
        y_test=y_test,
        y_score=y_score,
        labels=labels,
        save=True,
        save_dir=IMAGE_DIR,
        filename=f'{config["name"]}_prec_recall_curve.pdf'
    )

    path = os.path.join(MODEL_DIR, f'{config["name"]}.pt')
    utils.mkdir(MODEL_DIR)
    print("im here")
    torch.save({
        'epoch': config["trainer"]["num_epochs"],
        'model_state_dict': model.state_dict(),
    }, path)
    print("model saved")



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
