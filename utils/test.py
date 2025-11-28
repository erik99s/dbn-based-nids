from tqdm import tqdm
import logging

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

def test(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
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
    result = {
            'test': {
                'avg_test_loss': 0.0,
                'tp': 0.0,
                'tn': 0.0,
                'fp': 0.0,
                'fn': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        }

    model.eval()

    test

    loss_list_mse = []
    loss_list_dbn = []
    label_list = []
    total_test_loss = 0
    loss_Benign = []
    loss_Benign_mse = []
    loss_Attack = []
    loss_Attack_mse = []
    loss_Zero = []
    loss_Zero_mse = []

    tp = 0
    fp = 0 
    tn = 0 
    fn = 0

    percentile: int = 75

    with torch.no_grad():
        for (inputs, labels) in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze(1)

            # calls the DBN on the input
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_test_loss += loss.cpu().item()

            loss = criterion(outputs,labels)
            mse, _ = model.reconstructOne(inputs)

            loss_list_dbn.append(loss.item())
            loss_list_mse.append(mse.item())
            label_list.append(labels.item())

            
            if labels.item() == 0:
                loss_Benign.append(loss.item())
                loss_Benign_mse.append(mse.item())
            elif labels.item() == 5:
                loss_Zero.append(loss.item())
                loss_Zero_mse.append(mse.item())
            else:
                loss_Attack.append(loss.item())
                loss_Attack_mse.append(mse.item())
                                
    avg_test_loss = total_test_loss/len(test_loader)
    print(total_test_loss)
    print(avg_test_loss)

    threshold_dbn = np.percentile(loss_list_dbn, percentile)
    threshold_mse = np.percentile(loss_list_mse, percentile)

    print(f"threshold_dbn: {threshold_dbn}")
    print(f"threshold_mse: {threshold_mse}")

    for loss, label in zip(loss_list_mse, label_list):
        if loss < threshold_mse:
            # prediction = Benign
            if label == 0:
                tn += 1
            else:
                fn +=1
        else:
            # prediction = Attack
            if label == 0:
                fp += 1
            else:
                tp += 1

        
    print(f"values:  tp: {tp} tn: {tn} fp: {fp} fn: {fn}")
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)


    plt.figure(figsize=(12,8))

    plt.hist(loss_Benign_mse, bins='auto', alpha=0.5, label='Benign')   
    plt.hist(loss_Attack_mse, bins='auto', alpha=0.5, label='Attacks')
    plt.hist(loss_Zero_mse, bins='auto', alpha=0.5, label='ZeroDay')
    plt.xlim(0.05, 0.35)
    plt.title("reconstuction hist")
    plt.legend()
    plt.savefig("reconstruction hist_dbn_mse.png", dpi=300)


    result['test']['avg_test_loss'] = avg_test_loss
    result['test']['tp'] = tp
    result['test']['tn'] = tn
    result['test']['fp'] = fp
    result['test']['fn'] = fn
    result['test']['precision'] = precision
    result['test']['recall'] = recall
    result['test']['f1_score'] = f1_score

    return result