from tqdm import tqdm
import logging

import torch
import torch.nn as nn

import torch.optim as optim

import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import os



DATA_DIR  = os.path.join(os.path.abspath("."), "data")

class AE_2nd(nn.Module): 
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

        super(AE_2nd, self).__init__()

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        """
        self.encoder = nn.Sequential(
            nn.Linear(49, 42),
            nn.ReLU(),
            nn.Linear(42, 35),
            nn.ReLU(),
            nn.Linear(35, 25),
        )   
        # Decoder layers
        
        self.decoder = nn.Sequential(
            nn.Linear(25, 35),
            nn.ReLU(),
            nn.Linear(35, 42),
            nn.ReLU(),
            nn.Linear(42, 49),
            nn.Sigmoid()  # use Sigmoid if your inputs are scaled to [0, 1]; otherwise use ReLU
        )

        self.encoder = nn.Sequential(
            nn.Linear(49, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )   
        # Decoder layers
        
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64,49),
            nn.Sigmoid()  # use Sigmoid if your inputs are scaled to [0, 1]; otherwise use ReLU
        )
        """
        self.encoder = nn.Sequential(
            nn.Linear(49, 25),
            nn.ReLU(),
            nn.Linear(25, 12),
        )   
        # Decoder layers
        
        self.decoder = nn.Sequential(
            nn.Linear(12, 25),
            nn.ReLU(),
            nn.Linear(25, 49),
            nn.Sigmoid()  # use Sigmoid if your inputs are scaled to [0, 1]; otherwise use ReLU
        )
        """
        self.encoder = nn.Sequential(
            nn.Linear(49, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )   
        # Decoder layers
        
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 49),
            nn.Sigmoid()  # use Sigmoid if your inputs are scaled to [0, 1]; otherwise use ReLU
        )
        """
       
    
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
            total_train_loss = 0
            # training loop
            for batch , _ in tqdm(train_loader):
                batch = batch.to(device)
                optimizer.zero_grad()
                reconstructed = self.forward(batch)   
                loss = criterion(reconstructed, batch)
                loss.backward()                 # computes grads for ALL params
                optimizer.step()           # updates ALL params
                total_train_loss += loss
            
            avg_train_loss = total_train_loss / len(train_loader)
            print(f"[Epoch {epoch+1}/{self.num_epochs}] Loss: {avg_train_loss:.6f}")

            # validation loop
            # Commented for now as it was used to figure out the amount of epochs needed
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

    def test(
            self: torch.nn.Module,
            criterion: torch.nn.Module,
            test_loader: torch.utils.data.DataLoader,
            device
        ):

        result = {
        'test': {
            'avg_test_loss': 0.0,
            'tp': 0.0,
            'tn': 0.0,
            'fp': 0.0,
            'fn': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    }

        self.eval()

        loss_list = []
        label_list = []
        features_list = []
        total_test_loss = 0
        loss_Benign = []
        loss_Attack = []
        loss_Zero = []

        tp = 0
        fp = 0 
        tn = 0 
        fn = 0

        percentile: int = 70

        with torch.no_grad():
            for inputs, label in tqdm(test_loader):
                inputs = inputs.to(device)
                output = self(inputs)
                loss = criterion(output, inputs)
                loss_list.append((loss.item()))
                label_list.append(label.item())
                features_list.append(inputs.detach().cpu().numpy().squeeze())
                total_test_loss += loss.item() 
                if label.item() == 0:
                    loss_Benign.append(loss.item())
                elif label.item() == 5:
                    loss_Zero.append(loss.item())
                else:
                    loss_Attack.append(loss.item())
                               
            avg_test_loss = total_test_loss/len(test_loader)
            print(total_test_loss)
            print(avg_test_loss)

            threshold = np.percentile(loss_list, percentile)

            print(f"threshold: {threshold}" )


            anomaly_features = []
            anomaly_labels = []
            for loss,feature, label in zip(loss_list,features_list, label_list):
                if loss < threshold:
                    # prediction = Benign
                    if label == 0:
                        tn += 1
                    else:
                        fn +=1
                else:
                    anomaly_features.append(feature)
                    anomaly_labels.append(label)
                    # prediction = Attack
                    if label == 0:
                        fp += 1
                    else:
                        tp += 1

            
            print(f"values:  tp: {tp} tn: {tn} fp: {fp} fn: {fn}")

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1_score = 2 * ((precision * recall) / (precision + recall + 1e-8))

            df_features = pd.DataFrame(anomaly_features)
            df_labels = pd.DataFrame(anomaly_labels, columns=["label"])

            df_features.to_pickle(os.path.join(DATA_DIR, 'filtered', 'features/features.pkl'))
            df_labels.to_pickle(os.path.join(DATA_DIR, 'filtered', 'labels/labels.pkl'))
            print("Saved anomalous samples to anomalous_features.pkl and anomalous_labels.pkl")

        plt.figure(figsize=(12,8))

        plt.hist(loss_Benign, bins='auto', alpha=0.5, label='Benign')   
        plt.hist(loss_Attack, bins='auto', alpha=0.5, label='Attacks')
        plt.hist(loss_Zero, bins='auto', alpha=0.5, label='ZeroDay')
        plt.xlim(0, 0.2)
        # plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.3f}')
        plt.title("reconstuction hist")
        plt.legend()
        plt.savefig("reconstruction hist_1st.png", dpi=300)

        result['test']['avg_test_loss'] = avg_test_loss
        result['test']['tp'] = tp
        result['test']['tn'] = tn
        result['test']['fp'] = fp
        result['test']['fn'] = fn
        result['test']['accuracy'] = accuracy
        result['test']['precision'] = precision
        result['test']['recall'] = recall
        result['test']['f1_score'] = f1_score

        return result

    def testZero(
            self: torch.nn.Module,
            criterion: torch.nn.Module,
            test_loader: torch.utils.data.DataLoader,
            device
        ):

        result = {
        'test': {
            'avg_test_loss': 0.0,
            'tp': 0.0,
            'tn': 0.0,
            'fp': 0.0,
            'fn': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    }

        self.eval()

        loss_list = []
        label_list = []

        total_test_loss = 0
        loss_Benign = []
        loss_Zero = []

        tp = 0
        fp = 0 
        tn = 0 
        fn = 0

        percentile: int = 82

        with torch.no_grad():
            for inputs, label in tqdm(test_loader):
                inputs = inputs.to(device)
                output = self(inputs)
                loss = criterion(output, inputs)
                loss_list.append((loss.item()))
                label_list.append(label)
                total_test_loss += loss.item() 
                if label.item() == 4:
                    loss_Zero.append(loss.item())
                else:
                    loss_Benign.append(loss.item())
                               
            avg_test_loss = total_test_loss/len(test_loader)
            print(total_test_loss)
            print(avg_test_loss)

            threshold = np.percentile(loss_list, percentile)
            print(f"threshold: {threshold}" )


            for loss, label in zip(loss_list, label_list):
                if loss < threshold:
                    # prediction = Benign
                    if label == 4:
                        fn += 1
                    else:
                        tn +=1
                else:
                    # prediction = Attack
                    if label == 4:
                        tp += 1
                    else:
                        fp += 1

            
            print(f"values:  tp: {tp} tn: {tn} fp: {fp} fn: {fn}")
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1_score = 2 * ((precision * recall) / (precision + recall + 1e-8))

        print( f"Test Loss: {avg_test_loss:.6f}")
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1 score: {f1_score}")
        
        plt.figure(figsize=(12,8))

        plt.hist(loss_Benign, bins='auto', alpha=0.5, label='Benign')   
        plt.hist(loss_Zero, bins='auto', alpha=0.5, label='ZeroDay')

        plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.3f}')
        plt.title("reconstuction hist")
        plt.legend()
        plt.savefig("reconstruction hist_2nd.png", dpi=300)
    
        result['test']['avg_test_loss'] = avg_test_loss
        result['test']['tp'] = tp
        result['test']['tn'] = tn
        result['test']['fp'] = fp
        result['test']['fn'] = fn
        result['test']['accuracy'] = accuracy
        result['test']['precision'] = precision
        result['test']['recall'] = recall
        result['test']['f1_score'] = f1_score

        return result