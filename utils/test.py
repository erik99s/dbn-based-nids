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

    reconstruct_mse, reconstruct_vprobs, batch_list = model.reconstruct(test_loader)

    print("reconstructed")
    print(reconstruct_mse)
    print(reconstruct_vprobs)
    print(batch_list)
    # print(reconstruct_vprobs.shape)
    # print(reconstruct_mse.type)

    plt.figure()
    plt.plot(batch_list)

    # Add optional labels and title
    plt.title("Reconstruction Error Over Time")
    plt.xlabel("Batch")
    plt.ylabel("Error (MSE)")

    # Display the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.savefig("reconstruction_error_plot.png")
    plt.close()


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

    with torch.no_grad():
        for (inputs, labels) in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze(1)

            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            test_loss += loss.cpu().item()
            test_steps += 1

            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            test_output_pred += outputs.argmax(1).cpu().tolist()
            test_output_true += labels.tolist()
            test_output_pred_prob += nn.functional.softmax(outputs, dim=0).cpu().tolist()
    """
    reconstruction_mse, visable_probs = model.reconstruct(test_loader)
    print("Reconstruction MSE:", reconstruction_mse.item())
    """
    

    history['test']['total'] = test_total
    history['test']['loss'] = test_loss/test_steps
    history['test']['accuracy'] = test_correct/test_total
    history['test']['output_pred'] = test_output_pred
    history['test']['output_true'] = test_output_true
    history['test']['output_pred_prob'] = test_output_pred_prob

    logging.info(f'Test loss: {test_loss/test_steps}, Test accuracy: {test_correct/test_total}')

    return history
