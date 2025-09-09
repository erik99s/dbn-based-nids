from tqdm import tqdm
import logging

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np


def test(
    model: torch.nn.Module,
    auto_encoder: torch.nn.Module,
    criterion: torch.nn.Module,
    criterionAE: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
):
    """Validate the network.

    Parameters
    ----------
    model: torch.nn.ModuleList
        Neural network model used in this example.

    test_loader: torch.utils.data.DataLoader
        DataLoader used in testing.

    device: torch.device
        (Default value = torch.device("cpu"))
        Device where the network will be trained within a client.

    Returns
    -------
        Tuple containing the history, and a detailed report.

    """

    
    model.eval()

    history = {
        'test': {
            'total': 0,
            'loss': 0.0,
            'accuracy': 0.0,
            'output_pred': [],
            'output_true': [],
            'output_pred_prob': []
        }
    }

    test_loss = 0.0
    test_steps = 0
    test_total = 0
    test_correct = 0

    test_output_pred = []
    test_output_true = []
    test_output_pred_prob = []

    DBN_lossBenign = []
    DBN_lossAttack = []
    DBN_lossZero = []

    AE_lossBenign = []
    AE_lossAttack = []
    AE_lossZero = []

    DBN_rec_lossBenign = []
    DBN_rec_lossAttack = []
    DBN_rec_lossZero = []

    listOfPredictedAttacks = []

    above = 0

    

    mseList = []
    mseListZero = []

    with torch.no_grad():
        for (inputs, labels) in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze(1)

            # calls the DBN on the input
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.cpu().item()
            test_steps += 1

            loss_DBN = criterion(outputs,labels)

             # reconstructs the data using the Auto encoder
            reconstructed = auto_encoder(inputs)
            lossAE = criterionAE(reconstructed, labels)



            # reconstruction DBN using reconstruct one
            # reconstructed_DBN = model.reconstructOne(inputs)
            if lossAE.item() > 0.6:
                above += 1
            
            if labels == 0:
                DBN_lossBenign.append(loss_DBN)
                AE_lossBenign.append(lossAE.item())
                # DBN_rec_lossBenign.append(reconstructed_DBN)
            else:
                DBN_lossZero.append(loss_DBN)
                AE_lossZero.append(lossAE.item())
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
    
    print(above)

    plt.figure(figsize=(12, 8))

    # Plot benign
    plt.subplot(3, 1, 1)
    plt.plot(range(len(AE_lossBenign)), AE_lossBenign, label="Benign", color="green")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss (Benign)")
    plt.legend()

    # Plot attack
    plt.subplot(3, 1, 2)
    plt.plot(range(len(AE_lossAttack)), AE_lossAttack, label="Attacks", color="red")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss (Attacks)")
    plt.legend()

    # Plot zero-day
    plt.subplot(3, 1, 3)
    plt.plot(range(len(AE_lossZero)), AE_lossZero, label="Zero-day", color="blue")
    plt.xlabel("Index (sample)")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss (Zero-day)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("reconstruction_losses_AE.png", dpi=300)

    plt.close()


    plt.figure(figsize=(12, 8))

    # Plot benign
    plt.subplot(3, 1, 1)
    plt.plot(range(len(DBN_lossBenign)), DBN_lossBenign, label="Benign", color="green")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss (Benign)")
    plt.legend()

    # Plot attack
    plt.subplot(3, 1, 2)
    plt.plot(range(len(DBN_lossAttack)), DBN_lossAttack, label="Attacks", color="red")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss (Attacks)")
    plt.legend()

    # Plot zero-day
    plt.subplot(3, 1, 3)
    plt.plot(range(len(DBN_lossZero)), DBN_lossZero, label="Zero-day", color="blue")
    plt.xlabel("Index (sample)")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss (Zero-day)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("reconstruction_losses_DBN.png", dpi=300)

    plt.close()

    """
    plt.figure(figsize=(12, 8))

    # Plot benign
    plt.subplot(3, 1, 1)
    plt.plot(range(len(DBN_rec_lossBenign)), DBN_rec_lossBenign, label="Benign", color="green")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss (Benign)")
    plt.legend()

    # Plot attack
    plt.subplot(3, 1, 2)
    plt.plot(range(len(DBN_rec_lossAttack)), DBN_rec_lossAttack, label="Attacks", color="red")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss (Attacks)")
    plt.legend()

    # Plot zero-day
    plt.subplot(3, 1, 3)
    plt.plot(range(len(DBN_rec_lossZero)), DBN_rec_lossZero, label="Zero-day", color="blue")
    plt.xlabel("Index (sample)")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss (Zero-day)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("reconstruction_losses_rec_DBN.png", dpi=300)

    plt.close()

    """

    

    history['test']['total'] = test_total
    history['test']['loss'] = test_loss/test_steps
    history['test']['accuracy'] = test_correct/test_total
    history['test']['output_pred'] = test_output_pred
    history['test']['output_true'] = test_output_true
    history['test']['output_pred_prob'] = test_output_pred_prob

    logging.info(f'Test loss: {test_loss/test_steps}, Test accuracy: {test_correct/test_total}')

    return history
