from sklearn.metrics import classification_report, f1_score
import pandas as pd
import numpy as np
import argparse
import logging
import os

import torch
import torch.nn as nn 
import torch.optim as optim

import sys

from tqdm import tqdm

from collections import Counter
from torch.utils.data.dataset import Dataset

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
   
    auto_encoder = models.load_model(model_name=config["auto_encoder"]["type"], params=config["auto_encoder"]["args"])
    auto_encoder.load_state_dict(torch.load("./stored_models/autoencoder_model.pth"))
    auto_encoder.to(DEVICE)

    dbn = models.load_model(model_name=config["model"]["type"], params=config["model"]["args"])
    dbn.load_state_dict(torch.load("./stored_models/dbn_model_attack.pth"))
    dbn.to(DEVICE)

    criterion = getattr(torch.nn, config["loss"]["type"])(**config["loss"]["args"])
    criterionAE = getattr(torch.nn, config["lossAE"]["type"])(**config["loss"]["args"])

    print(auto_encoder)
    print(dbn)
   

    AE_lossBenign = []
    AE_lossZero = []
    AE_labels = []

    predicted = 0

    featuresList = []
    labelsList = []

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    print(test_loader)
    with torch.no_grad():
        for (inputs, labels) in tqdm(test_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            labels = labels.squeeze(1)

             # reconstructs the data using the Auto encoder
            outputs = auto_encoder(inputs)
            lossAE = criterionAE(outputs, labels)

            # reconstruction DBN using reconstruct one
            # reconstructed_DBN = model.reconstructOne(inputs)
            if lossAE.item() > 0.6:
                predicted = labels.item()
                AE_labels.append(labels)
                featuresList.append(inputs)
                labelsList.append(labels)

            else:
                predicted = 0
            
            if labels == 0:
                AE_lossBenign.append(lossAE.item()) 
                # DBN_rec_lossBenign.append(reconstructed_DBN)
            else:
                AE_lossZero.append(lossAE.item())
                
                # DBN_rec_lossZero.append(reconstructed_DBN)

            if predicted != 0 and labels.item() != 0:   # Both say attack
                TP += 1
            elif predicted != 0 and labels.item() == 0: # Predicted attack, true benign
                FP += 1
            elif predicted == 0 and labels.item() == 0: # Both say benign
                TN += 1
            else:                                       # Predicted benign, true attack
                FN += 1

    
    print(labelsList)
    print(featuresList)

    featuresList.to_pickle(os.path.join(DATA_DIR, f'processed', 'filtered/features_list.pkl'))
    labelsList.to_pickle(os.path.join(DATA_DIR, f'processed', 'filtered/labels_list.pkl'))

    filtered_test_data = dataset.CICIDSDataset(
        features_file=featuresList,
        target_file=labelsList,
        transform=torch.tensor,
        target_transform=torch.tensor
    )

    filtered_test_loader = torch.utils.data.Dataloader(
        dataset = filtered_test_data
    )
    
    values = [t.item() for t in AE_labels]
    counts = Counter(values)
    print(counts)

    

            
    print(f"True positive: {TP}")
    print(f"False positive: {FP}")
    print(f"True negative: {TN}")
    print(f"False negative: {FN}")

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2*(precision*recall)/(precision+recall)

    print(precision)
    print(recall)
    print(f1_score)

    DBN_lossBenign = []
    DBN_lossZero = []

            
    with torch.no_grad():
        for (inputs, labels) in tqdm(filtered_test_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            labels = labels.squeeze(1)

            # calls the DBN on the input
            outputs = dbn(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.cpu().item()
            test_steps += 1

            loss_DBN = criterion(outputs,labels)


            # reconstruction DBN using reconstruct one
            # reconstructed_DBN = model.reconstructOne(inputs)
            if lossAE.item() > 0.6:
                above += 1
            
            if labels == 0:
                DBN_lossBenign.append(loss_DBN)
                # DBN_rec_lossBenign.append(reconstructed_DBN)
            else:
                DBN_lossZero.append(loss_DBN)
                # DBN_rec_lossZero.append(reconstructed_DBN)
            """   
            else:
                DBN_lossAttack.append(loss_DBN)
                AE_lossAttack.append(lossAE.item())
                # DBN_rec_lossAttack.append(reconstructed_DBN)
            """ 
            _, predicted = torch.max(outputs.data, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            test_output_pred += outputs.argmax(1).cpu().tolist()
            test_output_true += labels.tolist()
            test_output_pred_prob += nn.functional.softmax(outputs, dim=0).cpu().tolist()

    
    
    labels = ['Benign', 'ZeroDay']
    # labels = ['Benign', 'Known', 'ZeroDay']
    
    ## Testing Set results
    """
    logging.info(f'Testing Set -- Classification Report {config["name"]}\n')
    logging.info(classification_report(
        y_true=test_output_true,
        y_pred=test_output_pred,
        target_names=labels
    ))
    """
   

    utils.mkdir(IMAGE_DIR)
    visualisation.plot_confusion_matrix(
        y_true=test_output_true,
        y_pred=test_output_pred,
        labels=labels,  
        save=True,
        save_dir=IMAGE_DIR,
        filename=f'{config["name"]}_test_confusion_matrix.pdf'
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
