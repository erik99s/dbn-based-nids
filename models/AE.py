from tqdm import tqdm
import logging

import torch
import torch.nn as nn
from keras.models import Sequential
from keras.layers import LSTM, Input, Dropout
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Model

import torch.optim as optim

import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

class AE(nn.Module): 
    def __init__(self,
                 n_visible=49,
                 n_hidden=(25,10,5),
                 n_classes=2,
                 learning_rate=1e-3,
                 batch_size=64,
                 num_epochs=10,
                 device="cpu"):
        """Initialization method.

        Parameters
        ----------
            n_visible (int): Amount of visible units.
            n_hidden (tuple): Amount of hidden units per layer.
            n_classes (int): Number of classes.
            learning_rate (int): 
            batch_size (tuple): Batch size per layer.
            num_epochs (tuple): Number of epochs per layer.
        """

        super(AE, self).__init__()

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.encoder = nn.Sequential(
            nn.Linear(n_visible, n_hidden[0]),
            nn.ReLU(),
            nn.Linear(n_hidden[0], n_hidden[1]),
            nn.ReLU(),
            nn.Linear(n_hidden[1], n_hidden[2]),
            nn.ReLU(),
            nn.Linear(n_hidden[2], n_hidden[3]),
            nn.ReLU(),
            nn.Linear(n_hidden[3], n_classes)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(n_classes, n_hidden[3]),
            nn.ReLU(),
            nn.Linear(n_hidden[3], n_hidden[2]),
            nn.ReLU(),
            nn.Linear(n_hidden[2], n_hidden[1]),
            nn.ReLU(),
            nn.Linear(n_hidden[1], n_hidden[0]),
            nn.ReLU(),
            nn.Linear(n_hidden[0], n_visible)
        )
        # For every possible layer

        # Creating the Fully Connected layer to append on top of DBNs
       
    
    def mse(self, batch):
        pass

    def forward(self, batch):
        return self.decoder(self.encoder(batch))



    def fit(
            self: torch.nn.Module,
            criterion: torch.nn.Module,
            optimizer: torch.optim,
            train_loader: torch.utils.data.DataLoader,
            valid_loader: torch.utils.data.DataLoader,
            device
        ):
        for epoch in range(self.num_epochs):
            self.train()

            logging.info(f"Epoch {epoch}/{self.num_epochs}:")
            total_loss = 0
            total_train_loss = 0
            # training loop
            for batch , _ in tqdm(train_loader):
                batch = batch.to(device)
                optimizer.zero_grad()
                reconstructed = self.forward(batch)   
                loss = criterion(reconstructed, batch)
                loss.backward()                 # computes grads for ALL params
                optimizer.step()           # updates ALL params
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            print(f"[Epoch {epoch+1}/{self.num_epochs}] Loss: {total_loss/len(train_loader):.6f}")

            # validation loop
            # Commented for now as it was used to figure out the amount of epochs needed
            """
            self.eval()
            total_val_loss = 0
            with torch.no_grad():   # no gradient computation
                for batch , _ in tqdm(valid_loader):
                    batch = batch.to(device)
                    reconstructed = self.forward(batch)
                    loss = criterion(reconstructed, batch)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(valid_loader)
            
            print(f"[Epoch {epoch+1}/{self.num_epochs}] "
            f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

            """


    def testing(
            self: torch.nn.Module,
            criterion: torch.nn.Module,
            test_loader: torch.utils.data.DataLoader,
            device
        ):

        self.eval()

        losslist = []
        total_test_loss = 0
        total_test_loss_benign = 0
        total_test_loss_zero = 0

        mse1 = 0
        mse2 = 0

        lossBenign = []
        lossZero = []
        with torch.no_grad():
            for input, labels in tqdm(test_loader):
                input = input.to(device)
                reconstructed = self.forward(input)
                loss = criterion(reconstructed, input)
                losslist.append(loss.item())
                total_test_loss += loss.item() 
                if labels.item() == 0:
                    total_test_loss_benign += loss.item()
                    lossBenign.append(loss.item())
                    mse1 += torch.sum(torch.pow(input - reconstructed, 2))
                else:
                    total_test_loss_zero += loss.item()
                    lossZero.append(loss.item())
                    mse2 = torch.sum(torch.pow(input - reconstructed, 2))
            avg_test_loss = total_test_loss/len(test_loader)
            avg_test_loss_benign = total_test_loss_benign/len(lossBenign)
            avg_test_loss_zero = total_test_loss_zero/len(lossZero)
            avg_mse_benign = mse1 / len(lossBenign)
            avg_mse_zero = mse2 / len(lossZero)
        
        print( f"Test Loss: {avg_test_loss:.6f}")
        print( f"Test Loss: {avg_test_loss_benign:.6f}")
        print( f"Test Loss: {avg_test_loss_zero:.6f}")

        print( f"Test MSE: {avg_mse_benign:.6f}")
        print( f"Test MSE: {avg_mse_zero:.6f}")


    def testingWithAttacks(
            self: torch.nn.Module,
            criterion: torch.nn.Module,
            test_loader: torch.utils.data.DataLoader,
            device
        ):

        self.eval()

        losslist = []
        total_test_loss = 0
        total_test_loss_benign = 0
        total_test_loss_attacks = 0
        total_test_loss_zero = 0

        mse1 = 0
        mse2 = 0
        mse3 = 0

        lossBenign = []
        lossAttack = []
        lossZero = []

        with torch.no_grad():
            for input, labels in tqdm(test_loader):
                input = input.to(device)
                reconstructed = self(input)
                loss = criterion(reconstructed, input)
                losslist.append(loss.item())
                total_test_loss += loss.item() 
                if labels.item() == 0:
                    total_test_loss_benign += loss.item()
                    lossBenign.append(loss.item())
                    mse1 += torch.sum(torch.pow(input - reconstructed, 2))
                elif labels.item() == 5:
                    total_test_loss_zero += loss.item()
                    lossZero.append(loss.item())
                    mse2 += torch.sum(torch.pow(input - reconstructed, 2))
                else:
                    total_test_loss_attacks += loss.item()
                    lossAttack.append(loss.item())
                    mse3 += torch.sum(torch.pow(input - reconstructed,2))                  
            avg_test_loss = total_test_loss/len(test_loader)
            avg_test_loss_benign = total_test_loss_benign/len(lossBenign)
            avg_test_loss_attack = total_test_loss_attacks/len(lossAttack)
            avg_test_loss_zero = total_test_loss_zero/len(lossZero)
            avg_mse_benign = mse1 / len(lossBenign)
            avg_mse_attack = mse3 / len(lossAttack)
            avg_mse_zero = mse2 / len(lossZero)

            maxBenign = min(lossBenign)
            minBenign = max(lossBenign)
            minZero = min(lossZero)
            maxZero = max(lossZero)
            minAttack = min(lossAttack)
            maxAttack = max(lossAttack)

        
        print( f"Test Loss: {avg_test_loss:.6f}")
        print( f"Test Loss: {avg_test_loss_benign:.6f}")
        print( f"Test Loss: {avg_test_loss_attack:.6f}")
        print( f"Test Loss: {avg_test_loss_zero:.6f}")
        
        print( f"Test MSE: {avg_mse_benign:.6f}")
        print( f"Test MSE: {avg_mse_attack:.6f}")
        print( f"Test MSE: {avg_mse_zero:.6f}")

        print( f"{maxBenign} : {minBenign} : {minZero} : {maxZero} : {minAttack} : {maxAttack}")

        plt.figure(figsize=(12, 8))

        # Plot benign
        plt.subplot(3, 1, 1)
        plt.plot(range(len(lossBenign)), lossBenign, label="Benign", color="green")
        plt.ylabel("Loss")
        plt.title("Reconstruction Loss (Benign)")
        plt.legend()

        # Plot attack
        plt.subplot(3, 1, 2)
        plt.plot(range(len(lossAttack)), lossAttack, label="Attacks", color="red")
        plt.ylabel("Loss")
        plt.title("Reconstruction Loss (Attacks)")
        plt.legend()

        # Plot zero-day
        plt.subplot(3, 1, 3)
        plt.plot(range(len(lossZero)), lossZero, label="Zero-day", color="blue")
        plt.xlabel("Index (sample)")
        plt.ylabel("Loss")
        plt.title("Reconstruction Loss (Zero-day)")
        plt.legend()

        plt.tight_layout()
        plt.savefig("reconstruction_losses.png", dpi=300)
    

        


    
    
        """
        plt.hist(lossBenign.detach().cpu().numpy(), bins=50, alpha=0.5, label="benign")
        plt.hist(lossZero.detach().cpu().numpy(), bins=50, alpha=0.5, label="zero-day")
        plt.legend()
        plt.show()
        """
    
            
    """    
    def train(self, train_loader):
        for epoch in range(1, self.num_epochs+1):

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            model.to(device)

            model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch in dataloader:
                    batch = batch[0].to(device)  # if dataloader returns (data, labels)
                    optimizer.zero_grad()
                    reconstructed = model(batch)
                    loss = criterion(reconstructed, batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.6f}")
    
        # Define Autoencoder model
    
        model = Sequential()
        model.add(LSTM(128, activation='relu', input_shape=(), return_sequence=True))
        model.add(LSTM(64,activation='relu', input_shape=(), return_sequence=False))
        model.add(RepeatVector())
        model.add(LSTM(64,activation='relu', input_shape=(), return_sequence=True))
        model.add(LSTM(128, activation='relu', input_shape=(), return_sequence=True))

        model.compile(optimizer='adam', loss ='mse')
        model.summary()
    
    """