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
            nn.Linear(n_visible, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, n_classes)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(n_classes, 5),
            nn.ReLU(),
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 25),
            nn.ReLU(),
            nn.Linear(25, n_visible),
            nn.Sigmoid()
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


    def testing(
            self: torch.nn.Module,
            criterion: torch.nn.Module,
            test_loader: torch.utils.data.DataLoader,
            device
        ):

        self.eval()

        losslist = []
        avg_test_loss = 0
        with torch.no_grad():
            for input, labels in tqdm(test_loader):
                input = input.to(device)
                reconstructed = self.forward(input)
                loss = criterion(reconstructed, input)
                losslist.append(loss)
                total_val_loss += loss.item()
            avg_test_loss = total_val_loss/len(test_loader)
        
        print( f"Train Loss: {avg_test_loss:.6f} | Val Loss: {avg_test_loss:.6f}")
    
            
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