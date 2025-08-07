from tqdm import tqdm
import logging

import torch


def train(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    device: torch.device,
):
    """Train the network.

    Parameters
    ----------
    model: torch.nn.Module
        Neural network model used in this example.

    optimizer: torch.optim
        Optimizer.

    train_loader: torch.utils.data.DataLoader
        DataLoader used in training.

    valid_loader: torch.utils.data.DataLoader
        DataLoader used in validation.

    num_epochs: int
        Number of epochs to run in each round.

    device: torch.device
        (Default value = torch.device("cpu"))
        Device where the network will be trained within a client.

    Returns
    -------
        Tuple containing the history, the train_loss, the train_accuracy, the valid_loss, and the valid_accuracy.

    """

    model.to(device)

    history = {
        'train': {
            'total': 0,
            'loss': [],
            'accuracy': [],
            'output_pred': [],
            'output_true': []
        },
        'valid': {
            'total': 0,
            'loss': [],
            'accuracy': [],
            'output_pred': [],
            'output_true': []
        }
    }

    for epoch in range(1, num_epochs+1):

        ########################################
        ##             TRAIN LOOP             ##
        ########################################

        model.train()

        train_loss = 0.0
        train_steps = 0
        train_total = 0
        train_correct = 0

        train_output_pred = []
        train_output_true = []

        reconstruct_mse, reconstruct_vprobs = model.reconstruct(train_loader)
        print("reconstructed")
        print(reconstruct_mse)
        print(reconstruct_vprobs)
        print(reconstruct_vprobs.shape)

        logging.info(f"Epoch {epoch}/{num_epochs}:")
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze(1)

            print(inputs.shape)

            # zero the parameter gradients
            for opt in optimizer:
                opt.zero_grad()

            # Passing the batch down the model
            outputs = model(inputs)
            
            # forward + backward + optimize
            loss = criterion(outputs, labels)
            loss.backward()

            # For every possible optimizer performs the gradient update
            for opt in optimizer:
                opt.step()

            train_loss += loss.cpu().item()
            train_steps += 1

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_output_pred += outputs.argmax(1).cpu().tolist()
            train_output_true += labels.tolist()

        
        history['train']['total'] = train_total
        history['train']['loss'].append(train_loss/train_steps)
        history['train']['accuracy'].append(train_correct/train_total)
        history['train']['output_pred'] = train_output_pred
        history['train']['output_true'] = train_output_true


        logging.info(f'loss: {train_loss/train_steps} - acc: {train_correct/train_total}')

    logging.info(f"Finished Training")

    return history