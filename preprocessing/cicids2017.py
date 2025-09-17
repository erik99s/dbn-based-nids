import pandas as pd
import numpy as np
import glob
import os
import logging

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


DATA_DIR  = os.path.join(os.path.abspath("."), "data")

class CICIDS2017Preprocessor(object):
    def __init__(self, data_path, training_size, validation_size, testing_size, tp):
        self.data_path = data_path
        self.training_size = training_size
        self.validation_size = validation_size
        self.testing_size = testing_size
        self.tp = tp
        
        self.data = None
        self.features = None
        self.label = None

    def read_data(self):
        """"""
        filenames = glob.glob(os.path.join(self.data_path, 'raw', '*.csv'))
        datasets = [pd.read_csv(filename) for filename in filenames]

        # Remove white spaces and rename the columns
        for dataset in datasets:
            dataset.columns = [self._clean_column_name(column) for column in dataset.columns]

        # Concatenate the datasets
        self.data = pd.concat(datasets, axis=0, ignore_index=True)
        self.data.drop(labels=['fwd_header_length.1'], axis= 1, inplace=True)

    def _clean_column_name(self, column):
        """"""
        column = column.strip(' ')
        column = column.replace('/', '_')
        column = column.replace(' ', '_')
        column = column.lower()
        return column

    def remove_duplicate_values(self):
        """"""
        # Remove duplicate rows
        self.data.drop_duplicates(inplace=True, keep=False, ignore_index=True)

    def remove_missing_values(self):
        """"""
        # Remove missing values
        self.data.dropna(axis=0, inplace=True, how="any")

    def remove_infinite_values(self):
        """"""
        # Replace infinite values to NaN
        self.data.replace([-np.inf, np.inf], np.nan, inplace=True)

        # Remove infinte values
        self.data.dropna(axis=0, how='any', inplace=True)

    def remove_constant_features(self, threshold=0.01):
        """"""
        # Standard deviation denoted by sigma (σ) is the average of the squared root differences from the mean.
        data_std = self.data.std(numeric_only=True)

        # Find Features that meet the threshold
        constant_features = [column for column, std in data_std.iteritems() if std < threshold]

        # Drop the constant features
        self.data.drop(labels=constant_features, axis=1, inplace=True)

    def remove_correlated_features(self, threshold=0.98):
        """"""
        # Correlation matrix
        data_corr = self.data.corr()

        # Create & Apply mask
        mask = np.triu(np.ones_like(data_corr, dtype=bool))
        tri_df = data_corr.mask(mask)

        # Find Features that meet the threshold
        correlated_features = [c for c in tri_df.columns if any(tri_df[c] > threshold)]

        # Drop the highly correlated features
        self.data.drop(labels=correlated_features, axis=1, inplace=True)

    def group_labels(self):
        """"""

        print(self.tp)

        if self.tp == 0 or self.tp == 2 or self.tp == 1:
            print("why did i end up here?")
            attack_group = {
                'BENIGN': 'Benign',
                'PortScan': 'PortScan', 
                'DDoS': 'DoS',
                'DoS Hulk': 'DoS',
                'DoS GoldenEye': 'DoS',
                'DoS slowloris': 'DoS', 
                'DoS Slowhttptest': 'DoS',
                'Heartbleed': 'ZeroDay',
                'FTP-Patator': 'Brute Force',
                'SSH-Patator': 'Brute Force',
                'Bot': 'Bot',
                'Web Attack � Brute Force': 'ZeroDay',
                'Web Attack � Sql Injection': 'ZeroDay',
                'Web Attack � XSS': 'ZeroDay',
                'Infiltration': 'ZeroDay'
            }
        else:
        # Proposed Groupings
            print("correct grouping")
            attack_group = {
                'BENIGN': 'Benign',
                'PortScan': 'Attack', 
                'DDoS': 'Attack',
                'DoS Hulk': 'Attack',
                'DoS GoldenEye': 'Attack',
                'DoS slowloris': 'Attack', 
                'DoS Slowhttptest': 'Attack',
                'Heartbleed': 'Attack',
                'FTP-Patator': 'Attack',
                'SSH-Patator': 'Attack',
                'Bot': 'Attack',
                'Web Attack � Brute Force': 'Attack',
                'Web Attack � Sql Injection': 'Attack',
                'Web Attack � XSS': 'Attack',
                'Infiltration': 'Attack'
            }
    
        

        # Create grouped label column
        self.data['label_category'] = self.data['label'].map(lambda x: attack_group[x])
        
    def train_valid_test_split(self):
        """
        tp == 0 : original code
        tp == 1 : training and validation set with only benign samples
        tp == 2 : traning and validation set with only attack samples
        
        testing set containing unseed samples of benign data, attack data and the zero day attacks. 

        self.data: all attack classes
        benign: only benign traffic
        attack: only attack data with zero days excluded
        zeroday: only zeroday attack data
        """

        self.benign = self.data[self.data['label_category'] == 'Benign']
        self.benignAttack = self.data[self.data['label_category'] != 'ZeroDay']
        self.attacks = self.data[~self.data['label_category'].isin(['ZeroDay', 'Benign'])]
        self.zeroday = self.data[self.data['label_category'] == 'ZeroDay']
        self.grouped = self.data[self.data['label_category'] != 'Benign']
        

        # original code generating training, validation and testing set containing all type of network traffic data
        if self.tp == 0: 
            self.features = self.data.drop(labels=['label', 'label_category'], axis=1)
            self.labels = self.data['label_category']
            X_train, X_test, y_train, y_test = train_test_split(
                self.features,
                self.labels,
                test_size=(self.validation_size+self.testing_size),
                random_state=42,
                stratify=self.labels
            )
            X_test, X_val, y_test, y_val = train_test_split(
                X_test,
                y_test,
                test_size=self.testing_size / (self.validation_size + self.testing_size),
                random_state=42
            )
        # training and validation dataset with only benign samples
        elif self.tp == 1:
            self.features = self.benign.drop(labels=['label', 'label_category'], axis=1)
            self.labels = self.benign['label_category']
            X_train, X_test, y_train, y_test = train_test_split(
                self.features,
                self.labels,
                test_size=(self.validation_size+self.testing_size),
                random_state=42,
                stratify=self.labels
            )
            X_test, X_val, y_test, y_val = train_test_split(
                X_test,
                y_test,
                test_size=self.testing_size / (self.validation_size + self.testing_size),
                random_state=42
            )
            
            y_testAttack = self.grouped['label_category']
            X_testAttack = self.grouped.drop(labels=['label', 'label_category'], axis=1)

            print(type(X_testAttack))

            X_test = pd.concat([X_test,X_testAttack])
            y_test = pd.concat([y_test,y_testAttack]) 

            print(y_test.value_counts())


        # training and validation dataset with only attack samples
        elif self.tp == 2:
            self.features = self.attacks.drop(labels=['label', 'label_category'], axis=1)
            self.labels = self.attacks['label_category']
            X_train, X_test, y_train, y_test = train_test_split(
                self.features,
                self.labels,
                test_size=(self.validation_size+self.testing_size),
                random_state=42,
                stratify=self.labels
            )
            X_test, X_val, y_test, y_val = train_test_split(
                X_test,
                y_test,
                test_size=self.testing_size / (self.validation_size + self.testing_size),
                random_state=42
            )

            y_testZero = self.zeroday['label_category']
            X_testZero = self.zeroday.drop(labels=['label', 'label_category'], axis=1)

            print(type(X_testZero))

            X_test = pd.concat([X_test,X_testZero])
            y_test = pd.concat([y_test,y_testZero]) 

            print(y_test.value_counts())


        
        print(y_train.value_counts())
        print(y_val.value_counts())
        print(y_test.value_counts())

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def scale(self, training_set, validation_set, testing_set):
        """"""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = training_set, validation_set, testing_set
        
        categorical_features = self.features.select_dtypes(exclude=["number"]).columns
        numeric_features = self.features.select_dtypes(exclude=[object]).columns


        preprocessor = ColumnTransformer(transformers=[
            ('categoricals', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'), categorical_features),
            ('numericals', QuantileTransformer(), numeric_features)
        ])

        # Preprocess the features
        # columns = numeric_features.tolist()

        X_train = pd.DataFrame(preprocessor.fit_transform(X_train))
        X_val = pd.DataFrame(preprocessor.transform(X_val))
        X_test = pd.DataFrame(preprocessor.transform(X_test))


        # Preprocess the labels
        all_labels = pd.concat([y_train, y_val, y_test])
        all_classes = sorted(all_labels.unique())

        le = LabelEncoder()
        le.classes_ = np.array(all_classes)

        y_train = pd.DataFrame(le.transform(y_train), columns=["label"])
        y_val = pd.DataFrame(le.transform(y_val), columns=["label"])
        y_test = pd.DataFrame(le.transform(y_test), columns=["label"])

        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

        print(y_train['label'].value_counts())
        print(y_val['label'].value_counts())
        print(y_test['label'].value_counts())


        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":

    cicids2017 = CICIDS2017Preprocessor(
        data_path=DATA_DIR,
        training_size=0.6,
        validation_size=0.2,
        testing_size=0.2,
        tp = 1, # change this for different processings
    )

    # Read datasets
    cicids2017.read_data()

    # Remove NaN, -Inf, +Inf, Duplicates
    cicids2017.remove_duplicate_values()
    cicids2017.remove_missing_values
    cicids2017.remove_infinite_values()

    # Drop constant & correlated features
    cicids2017.remove_constant_features()
    cicids2017.remove_correlated_features()

    # Create new label category
    cicids2017.group_labels()

    # Split & Normalise data sets
    training_set, validation_set, testing_set            = cicids2017.train_valid_test_split()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = cicids2017.scale(training_set, validation_set, testing_set)
    
    # Save the results
    X_train.to_pickle(os.path.join(DATA_DIR, f'processed{cicids2017.tp}', 'train/train_features.pkl'))
    X_val.to_pickle(os.path.join(DATA_DIR, f'processed{cicids2017.tp}', 'val/val_features.pkl'))
    X_test.to_pickle(os.path.join(DATA_DIR, f'processed{cicids2017.tp}', 'test/test_features.pkl'))

    y_train.to_pickle(os.path.join(DATA_DIR, f'processed{cicids2017.tp}', 'train/train_labels.pkl'))
    y_val.to_pickle(os.path.join(DATA_DIR, f'processed{cicids2017.tp}', 'val/val_labels.pkl'))
    y_test.to_pickle(os.path.join(DATA_DIR, f'processed{cicids2017.tp}', 'test/test_labels.pkl'))


    """
    Name: label_category, dtype: int64
    Train: 1455004, Val: 485002, Test: 485721
    Train: 1455004, Val: 485002, Test: 485721
    0    1221303
    3     192155
    4      34383
    2       5997
    1       1166
    Name: label, dtype: int64
    0    406947
    3     64266
    4     11440
    2      2005
    1       344
    Name: label, dtype: int64
    0    407255
    3     63837
    4     11482
    2      1994
    5       720
    1       433
    Name: label, dtype: int64

    """