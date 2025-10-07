from tqdm import tqdm
import logging

import torch
import torch.nn as nn

import torch.optim as optim

import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

class AE(nn.Module): 
    def __init__(self,
                 n_visible=49,
                 n_hidden=[128,64],
                 n_classes=32,
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

        """
        self.encoder = nn.Sequential(
            nn.Linear(n_visible, n_hidden[0]),
            nn.ReLU(),
            nn.Linear(n_hidden[0],n_classes)
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_classes,n_hidden[0]),
            nn.ReLU(),
            nn.Linear(n_hidden[0],n_visible)
        )

    
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

        """

        self.encoder = nn.Sequential(
            nn.Linear(49, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 8)
)
        # Decoder layers
        
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 49),
            nn.Sigmoid()  # use Sigmoid if your inputs are scaled to [0, 1]; otherwise use ReLU
        )

        # For every possible layer

        # Creating the Fully Connected layer to append on t op of DBNs
       
    
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
        print(self)
        for epoch in range(self.num_epochs):
            logging.info(f"Epoch {epoch+1}/{self.num_epochs}:")
            self.train()
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

    def test(
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
            for inputs, labels in tqdm(test_loader):
                inputs = inputs.to(device)
                reconstructed = self(inputs)
                loss = criterion(reconstructed, inputs)
                losslist.append(loss.item())
                total_test_loss += loss.item() 
                if labels.item() == 0:
                    total_test_loss_benign += loss.item()
                    lossBenign.append(loss.item())
                    mse1 += torch.sum(torch.pow(inputs - reconstructed, 2))
                elif labels.item() == 5:
                    total_test_loss_zero += loss.item()
                    lossZero.append(loss.item())
                    mse2 += torch.sum(torch.pow(inputs - reconstructed, 2))
                else:
                    total_test_loss_attacks += loss.item()
                    lossAttack.append(loss.item())
                    mse3 += torch.sum(torch.pow(inputs - reconstructed,2))  
                               
            avg_test_loss = total_test_loss/len(test_loader)
            avg_test_loss_benign = total_test_loss_benign/len(lossBenign)
            avg_test_loss_attack = total_test_loss_attacks/len(lossAttack)
            avg_test_loss_zero = total_test_loss_zero/len(lossZero)
            avg_mse_benign = mse1 / len(lossBenign)
            avg_mse_attack = mse3 / len(lossAttack)
            avg_mse_zero = mse2 / len(lossZero)

            minBenign = min(lossBenign)
            maxBenign = max(lossBenign)
            minZero = min(lossZero)
            maxZero = max(lossZero)
            minAttack = min(lossAttack)
            maxAttack = max(lossAttack)
                
            avg_test_loss = total_test_loss/len(test_loader)
            print(total_test_loss)
            print(avg_test_loss)
            

        print( f"Test Loss: {avg_test_loss:.6f}")
        print( f"Test Loss: {avg_test_loss_benign:.6f}")
        print( f"Test Loss: {avg_test_loss_attack:.6f}")
        print( f"Test Loss: {avg_test_loss_zero:.6f}")
        
        print( f"Test MSE: {avg_mse_benign:.6f}")
        print( f"Test MSE: {avg_mse_attack:.6f}")
        print( f"Test MSE: {avg_mse_zero:.6f}")

        print( f"{minBenign} : {maxBenign} : {minZero} : {maxZero} : {minAttack} : {maxAttack}")




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
    
        plt.close()

        plt.figure(figsize=(12,8))

        plt.hist(lossBenign, bins=100, alpha=0.5, label='Benign')   
        plt.hist(lossAttack, bins=100, alpha=0.5, label='Attacks')
        plt.hist(lossZero, bins=100, alpha=0.5, label='ZeroDay')
        plt.title("reconstuction hist")
        plt.legend()
        plt.savefig("reconstruction hist.png", dpi=300)

        """
        plt.figure(figsize=(12,8))

        plt.plot(range(len(losslist)),losslist,label='loss', color='green')
        plt.ylabel('loss')
        plt.title('Reconstruction Loss Aute encoder')
        plt.legend()
        plt.tight_layout
        plt.savefig("reconstructio_loss_batch.png", dpi=300)
        """

    
    
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