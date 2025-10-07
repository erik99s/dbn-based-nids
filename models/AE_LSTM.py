    
from tqdm import tqdm
import logging

import torch
import torch.nn as nn

import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

class AE_LSTM(nn.Module):
    def __init__(self,
                n_features=49,
                layers=[128,64],
                latent_dim=64,
                learning_rate=1e-3,
                timesteps = 1,
                batch_size=64,
                num_epochs=20,
                device="cpu"):
        
        super(AE_LSTM, self).__init__()

        self.n_features = n_features
        self.n_layers = layers
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device

        # ---------------- Encoder ----------------
        # Sequential container with two stacked LSTMs
        self.encoder = nn.Sequential(
            nn.LSTM(input_size=n_features, hidden_size=layers[0], num_layers=1, batch_first=True),
            # The second LSTM compresses the sequence further
            nn.LSTM(input_size=layers[0], hidden_size=layers[1], num_layers=1, batch_first=True)
        )

        # Linear layer to project final hidden state -> latent space (bottleneck)
        self.latent = nn.Linear(layers[1], latent_dim)

        # ---------------- Decoder ----------------
        # Mirror of encoder: latent -> hidden -> output
        self.decoder_lstm1 = nn.LSTM(input_size=latent_dim, hidden_size=layers[1], num_layers=1, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(input_size=layers[1], hidden_size=layers[0], num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(layers[0], n_features)

    def forward(self, x):
        """
        x shape: (batch, timesteps, n_features)
        """

        # --- Encoder ---
        enc_out1, _ = self.encoder[0](x)
        enc_out2, (h, c) = self.encoder[1](enc_out1)

        # Use the last hidden state as the latent representation
        z = self.latent(h[-1])  # (batch, latent_dim)

        # --- Decoder ---
        # Repeat latent vector across all timesteps (similar to Keras RepeatVector)
        z_repeated = z.unsqueeze(1).repeat(1, self.timesteps, 1)

        # Decode
        dec_out1, _ = self.decoder_lstm1(z_repeated)
        dec_out2, _ = self.decoder_lstm2(dec_out1)

        # Reconstruct features
        x_recon = self.output_layer(dec_out2)

        return x_recon
    

    def train_model(
            self: torch.nn.Module,
            train_loader:torch.utils.data.DataLoader, 
            valid_loader:torch.utils.data.DataLoader, 
            optimizer: torch.optim, 
            criterion: torch.nn.Module, 
            device
        ):
    
        
        for epoch in range(self.num_epochs):
            logging.info(f"Epoch {epoch+1}/{self.num_epochs}:")
            self.train()
            train_loss = 0.0
            for X_batch, _ in tqdm(train_loader):
                X_batch = X_batch.unsqueeze(1)
                X_batch = X_batch[0].to(device) if isinstance(X_batch, (list, tuple)) else X_batch.to(device)
                optimizer.zero_grad()
                X_pred = self(X_batch)
                loss = criterion(X_pred, X_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, _ in tqdm(valid_loader):
                    X_batch = X_batch.unsqueeze(1)
                    X_batch = X_batch[0].to(device) if isinstance(X_batch, (list, tuple)) else X_batch.to(device)
                    X_pred = self(X_batch)
                    loss = criterion(X_pred, X_batch)
                    val_loss += loss.item()

        print(f"Epoch [{epoch+1}/{self.num_epochs}]  Train Loss: {train_loss/len(train_loader):.6f}  "
            f"Val Loss: {val_loss/len(valid_loader):.6f}")

    def test_model(
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
                inputs = inputs.unsqueeze(1)
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
        plt.savefig("reconstruction_losses_LSTM.png", dpi=300)
    
        plt.close()

        plt.figure(figsize=(12,8))

        plt.hist(lossBenign, bins=100, alpha=0.5, label='Benign')   
        plt.hist(lossAttack, bins=100, alpha=0.5, label='Attacks')
        plt.hist(lossZero, bins=100, alpha=0.5, label='ZeroDay')
        plt.title("reconstuction hist")
        plt.legend()
        plt.savefig("reconstruction hist.png", dpi=300)