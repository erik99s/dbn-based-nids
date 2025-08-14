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

import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

class AE(nn.Module): 
    def __init__(self,
                 n_visible=49,
                 n_hidden=(25,10,5),
                 n_classes=2,
                 learning_rate=(1e-3,1e-3,1e-3),
                 batch_size=(64, 64, 64),
                 num_epochs=(10, 10, 10),
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

        # For every possible layer

        # Creating the Fully Connected layer to append on top of DBNs
        self.fc = nn.Linear(self.n_hidden[-1], self.n_classes)
    

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


    def fit(self, train_loader):

        self.encoder = nn.Sequential(
            nn.Linear(49, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(),
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10,25),
            nn.ReLU(),
            nn.Linear(25, 49),
            nn.Sigmoid())
            

        model = AE()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(lr=1e-3, weight_decay=1e-5))

        # Define Autoencoder model
        """
        model = Sequential()
        model.add(LSTM(128, activation='relu', input_shape=(), return_sequence=True))
        model.add(LSTM(64,activation='relu', input_shape=(), return_sequence=False))
        model.add(RepeatVector())
        model.add(LSTM(64,activation='relu', input_shape=(), return_sequence=True))
        model.add(LSTM(128, activation='relu', input_shape=(), return_sequence=True))

        model.compile(optimizer='adam', loss ='mse')
        model.summary()
        """
    
    def train(self, trainloader):
        num_epochs = 1
        output = []
        for epoch in range(num_epochs):
            for input in trainloader:
                pass
            print(f'Epoch: {epoch+1}')
