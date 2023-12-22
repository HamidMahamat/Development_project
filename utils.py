import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import re
import scipy
import math
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from itertools import product
from abc import ABC, abstractmethod

#__________________________________Main Functions used in main.ipynb____________________________________

def preprocessing(df):
    """Function for preprocessing: converting data to appropriate type, data cleaning,
    and categorical features one hot encoding.

    Args:
        df (DataFrame): dataset

    Returns:
        preprocessed_df: preprocessed dataset
    """
    # droping id column
    if "id" in df.columns:
        df = df.drop(["id"], axis=1)
    # preprocessing dataset
    new_df = convert_to_appropriate_type(df)
    cleaned_df = deal_with_NA_values(new_df, 0.1, 0.1)
    preprocessed_df = one_hot_encoding(cleaned_df)
    return preprocessed_df

def prepare_for_training(preprocessed_df):
    """Preparing dataset for training: splitting dataset into train and test,
    normalizing and handling the skewness of the dataset, and splitting train for cross
    validation.

    Args:
        preprocessed_df (DataFrame): preprocessed dataset
    Returns:
        train_df (DataFrame): part of the dataset to use for training
        test_df (DataFrame): part of the dataset to use for testing
        cross_val_split (iterator): iterator generating indexes of train
        and val elements for cross validation. 
    """
    # splitting and normalizing data
    normalized_train_df, normalized_test_df = normalization(preprocessed_df)
    # handling skewness
    train_df, test_df = handle_skewness(normalized_train_df, normalized_test_df)
    # splitting train data for cross validation
    kf = KFold(n_splits=4)
    cross_val_split = list(kf.split(train_df))
    return train_df, test_df, cross_val_split


def model_training(trainer, train_data, cross_val_split, target):
    """Training a model on the train_dataset. 

    Args:
        trainer (Trainer): specific trainer for a binary classification method.
        (See abstract class Trainer below)
        train_data (Dataframe)

    """
    # splitting target from features
    X_train, y_train = splitting_target(train_data, target)
    # training a model on the train dataset using the provided trainer
    trainer.train(X_train, y_train, cross_val_split)

def compare_results(models, test_data, target):
    """Testing a list of models on the test dataset and displaying results comparision in the form of a table and a bar plot.  

    Args:
        models (list): list containing the trained models to test 
        test_data (DataFrame): test dataset
        target (string): name of the target column
    """

    # testing the different models on the test dataset 
    models_scores = {
    'KNN': model_testing(models[0], test_data, target),
    'Decision_Tree': model_testing(models[1], test_data, target),
    'XG_boost': model_testing(models[2], test_data, target),
    'Logistic Regression': model_testing(models[3], test_data, target),
    'MLP': model_testing(models[4], test_data, target)
    }

    # Tabular Comparison
    scores_df = pd.DataFrame.from_dict(models_scores, orient='index', columns=['Test F1 Score'])
    print(scores_df)

    # Bar plot Comparision
    model_names = list(models_scores.keys())
    scores = list(models_scores.values())
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, scores, color='skyblue')
    plt.title('Comparison of Test Scores for Different Models')
    plt.ylabel('Test F1 Score')
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


#_____________________________________________Subfunctions_______________________________________________

def model_testing(model, test_data, target):
    """Test a model on the test dataset and returns the F1 Score.

    Args:
        model (Trainer): Trainer containing the trained model
        test_data (DataFrame): test dataset
        target (string): Target column name

    Returns:
        _float_ : resulting F1 score
    """
    # splitting target from features
    X_test, y_test = splitting_target(test_data, target)
    # returning f1 score
    return model.test(X_test, y_test)

def remove_weird_characters(text):
    """remove all problematic characters (eg. /t /n or ?) from a text

    Args:
        text (string)

    Returns:
        cleaned_text (string): text with no problematic characters
    """
    # Define the pattern for weird characters using regular expressions
    cleaned_text = text
    cleaned_text = cleaned_text.strip()
    cleaned_text = cleaned_text.replace('\t', "")
    cleaned_text = cleaned_text.replace(" ", "")
    pattern = r'[/[^\w.]|_/g]'  # This pattern allows letters, numbers, and spaces

    # Replace weird characters with an empty string
    if not re.search(pattern, text) or text == "nan":
        cleaned_text = np.nan
    return cleaned_text

def convert_to_appropriate_type(df):
    """convert strings to appropriate types

    Args:
        df (DataFrame): data

    Returns:
        new_df: new dataframe
    """

    new_df = copy.deepcopy(df)
    non_int_columns=[]
    
    # coverting concerned columns types into int
    for column in df.columns:
        new_df[column] = df[column].apply(lambda x: remove_weird_characters(str(x)))
        try:
            new_df[column] = new_df[column].astype(int)
        except ValueError:
            non_int_columns.append(column)
    # coverting concerned columns types into floats 
    for column in non_int_columns:
        try:
            new_df[column] = new_df[column].astype(float)
        except ValueError:
            pass
    return new_df

def deal_with_NA_values(df, r, r_float):
    """This function deals with NA values:
    - if the number of NA values is high (according to ratio r) create a new category "unknown"
    - else: replace by the majoritary category

    Args:
        df (_DataFrame_): data
        column (_string_): name of the column
        r (_float_): ratio between the number of NA values and the number of values in minority category
        r_float (_float_) : ratio between number of categories and length of the column. In order to consider if the column is categorical or not
    Returns:
        df (_pd_dataframe_): dataframe containing the column with dealed Nan problem
    """
    cleaned_df = copy.copy(df)
    for column in df.columns:
        categories_counts = df[column].value_counts(dropna=True)
        Nan_counts = df[column].size - sum(categories_counts)
        
        if df[column].dtype == 'float' : 
            if all([round(val)==val for val in df[column] if not np.isnan(val)]) : # Int type condition 
                # Verify if its categorical or take any float values
                
                # Categorical float value
                if categories_counts.size/df[column].size <= r_float : 
                    if Nan_counts/df[column].size <= r : # Don't create a new class
                        cleaned_df[column]= df[column].fillna(categories_counts.idxmax())
                    else :
                        cleaned_df[column] = df[column].fillna(round(np.mean(df[column])))
                    
                else :
                    # When it's not categorical compute the mean 
                    cleaned_df[column] = df[column].fillna(round(np.mean(df[column])))
            else :
                
                if categories_counts.size/df[column].size <= r_float : 
                    if Nan_counts/df[column].size <= r : # Don't create a new class
                        cleaned_df[column]= df[column].fillna(categories_counts.idxmax())
                    else :
                        cleaned_df[column] = df[column].fillna(np.mean(df[column]))
                    
                else :
                    # When it's not categorical compute the mean 
                    cleaned_df[column] = df[column].fillna(np.mean(df[column]))
                

        else :
            # We consider as categorical all column with other non float type
            if Nan_counts/df[column].size <= r : # Don't create a new class
                cleaned_df[column] = df[column].fillna(categories_counts.idxmax())
            else :
                cleaned_df[column] = df[column].fillna("unknown")
    return cleaned_df

def one_hot_encoding(df):
    """Apply one hot encoding on categorical columns of df.

    Args:
        df (DataFrame): data
        columns (list): name of columns to be one hot encoded
        target (string): name of target column. If target is not present in the df, specify None.

    Returns:
        DataFrame: new one hot encoded dataframe
    """
    df_copy = copy.deepcopy(df)
    # define columns to encode
    columns_to_encode = []
    for column in df.columns:
        if df[column].dtype == "O":
            if df[column].value_counts(dropna=True).size>2:
                columns_to_encode.append(column)
            else:
                label_encoder = LabelEncoder()
                df_copy[column] = label_encoder.fit_transform(df[column])
    
    if len(columns_to_encode)!=0:
        # Initialize OneHotEncoder
        onehot_encoder = OneHotEncoder(sparse=False)

        # Fit and transform the data
        encoded_data = onehot_encoder.fit_transform(df_copy[columns_to_encode])

        # Create a DataFrame with the encoded data
        encoded_df = pd.DataFrame(encoded_data, columns=onehot_encoder.get_feature_names_out(columns_to_encode))
        # Concatenate the encoded DataFrame with the original DataFrame
        df_encoded = pd.concat([df_copy.drop(columns_to_encode, axis=1), encoded_df], axis=1)
    else:
        df_encoded = df_copy

    return df_encoded

def data_splitting(df):
    """Split the data into train and test datasets 

    Args:
        df (DataFrame): dataset

    Returns:
        train_df (Dataframe): train data
        test_df (Dataframe): test data
    """
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df

def normalization(df):
    """scale the dataset and resolve skewness.

    Args:
        df (DataFrame): dataset

    Returns:
        DataFrame: normalized dataset
    """
    # splitting data
    train_df, test_df = data_splitting(df)

    # creating copy of the datasets
    train_normalized_df = copy.deepcopy(train_df)
    test_normalized_df = copy.deepcopy(test_df)


    #retriving the numerical columns to scale
    numerical_columns = []
    for column in df.columns:
        if df[column].dtype != "O" and df[column].value_counts().size>2:
            numerical_columns.append(column)

    # scaling the datasets
    stdScaler = StandardScaler()
    train_normalized_df[numerical_columns] = stdScaler.fit_transform(train_normalized_df[numerical_columns])
    test_normalized_df[numerical_columns] = stdScaler.transform(test_normalized_df[numerical_columns])
    return train_normalized_df, test_normalized_df

def handle_skewness(train_df, test_df):
    """
    handle very skewed features

    Args:
        train_df (Dataframe): train_dataset
        train_df (Dataframe): train_dataset

    """
    non_skewed_train_df = copy.deepcopy(train_df)
    non_skewed_test_df = copy.deepcopy(test_df)
    for column in non_skewed_train_df.columns:
        if non_skewed_train_df[column].dtype != "O" and non_skewed_train_df[column].value_counts().size>2:
            skew= scipy.stats.skew(non_skewed_train_df[column], axis=0)
            if skew>=1 or skew<=-1:
                try:
                    non_skewed_train_df[column] = non_skewed_train_df[column].apply(math.log1p)
                    non_skewed_test_df[column] = non_skewed_test_df[column].apply(math.log1p)
                except ValueError:
                    pass
    return non_skewed_train_df, non_skewed_test_df

def splitting_target(df, target):
    """split the target from the features

    Args:
        df (Dataframe): data
        target (string): target column name

    Returns:
        X (Dataframe): features
        y (Dataframe): target
    """
    X = df.drop([target], axis=1)
    y = df[target]
    return X, y

#__________________________________Trainer Class for each binary classification method_____________________________

class Trainer(ABC):
    """
    A Trainer Class is a class which trains and tests models using a specific binary classification method.
 
    This is the abstract Class to be implemented by every Trainer Class. Should contain the following attributes
    and methods.

    Attributes:
        model: best trained model
        val_score: best obtained validation score after training

    Methods:
        train: train different models using the same binary classification method and
          save the one performing the best cross validation score
        test: test the saved model on a test dataset and returns the score
    """

    @abstractmethod
    def train(self, X_train, y_train, cross_val_split):
        """
        Args:
            X_train (DataFrame): train dataset features
            y_train (DataFrame): train dataset target
            cross_val_split (List): List of indexes of cross validation splits.
        """
        pass

    @abstractmethod
    def test(self, X_test, y_test):
        """
        Args:
            X_test (DataFrame): test dataset features
            y_test (DataFrame): test dataset target
        """
        pass

class KNN_trainer(Trainer):
    """Trainer using KNN method. This trainer uses cross validation
      to find the best "n_neighbors" hyperparameter. """
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.val_score = - np.inf
        self.n_neighbors_list = np.linspace(1, 10, 10)
    
    def train(self, X_train, y_train, cross_val_split):
        for i in range(len(self.n_neighbors_list)-1):
            n_neighbors = int(self.n_neighbors_list[i])
            model = KNeighborsClassifier(n_neighbors= n_neighbors)
            val_scores = []
            for train_index, val_index in cross_val_split:
                # Split data into training and validation sets using the indices obtained from kf.split
                X, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
                y, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
                model.fit(X, y)
                predictions = model.predict(X_val)
                val_score = f1_score(predictions, y_val)
                val_scores.append(val_score)
            med_val_score =  np.median(val_scores)
            if self.val_score < med_val_score:
                self.model = model
                previous_score = self.val_score
                self.val_score = med_val_score
                print(f"Median cross validation score increased from {previous_score} to {self.val_score}"
                    f" with n_neighbors={n_neighbors}. Saving model... ")
    
    def test(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        score = f1_score(predictions, y_test)
        return score

class DecisionTree_trainer(Trainer):
    """Trainer using Decision Tree method. This trainer uses cross validation
      to find the best "maximum depth" hyperparameter. """
    def __init__(self):
        self.model = DecisionTreeClassifier()
        self.val_score = -np.inf
        self.n_depths = 10
        self.depths = np.linspace(1, 10, self.n_depths)
    
    def train(self, X_train, y_train, cross_val_split):
        for i in range(len(self.depths)-1):
            max_depth = int(self.depths[i])
            model = DecisionTreeClassifier(max_depth= max_depth)
            val_scores = []
            for train_index, val_index in cross_val_split:
                # Split data into training and validation sets using the indices obtained from kf.split
                X, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
                y, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
                model.fit(X, y)
                predictions = model.predict(X_val)
                val_score = f1_score(predictions, y_val)
                val_scores.append(val_score)
            med_val_score =  np.median(val_scores)
            if self.val_score < med_val_score:
                self.model = model
                previous_score = self.val_score
                self.val_score = med_val_score
                print(f"Median cross validation score increased from {previous_score} to {self.val_score}"
                    f" with max_depth={max_depth}. Saving model... ")
            if self.val_score == 1:
                break 
    
    def test(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        score = f1_score(predictions, y_test)
        return score


class XG_boost_trainer(Trainer):
    """Trainer using XG_boost method. This trainer uses cross validation and grid search
    to find the best combination of following hyperparameters: learning rate, max_depth, n_estimators
    and alpha coefficient for L1 regularization (feature selection). """
    def __init__(self):
        self.model = xgb.XGBClassifier(subsample=0.8, colsample_bytree=0.8, random_state=42)
        self.val_score = -np.inf
        hyperparameters = {
            'learning_rate': [0.1, 0.01, 0.001],
            'max_depth': [3, 5, 7],
            'n_estimators': [50, 100, 200],
            "reg_alpha": [0, 0.001, 0.01, 0.05, 0.1]
        }

        # all combinations of hyperparameters
        self.param_combinations = list(product(*hyperparameters.values()))

    def train(self, X_train, y_train, cross_val_split):
        for params in self.param_combinations:
            # Unpack parameters for XGBoost model
            xgb_params = {
                'learning_rate': params[0],
                'max_depth': params[1],
                'n_estimators': params[2],
                # Add other hyperparameters here
            }
            model = xgb.XGBClassifier(**xgb_params)
            val_scores = []
            for train_index, val_index in cross_val_split:
                # Split data into training and validation sets using the indices obtained from kf.split
                X, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
                y, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
                model.fit(X, y)
                predictions = model.predict(X_val)
                val_score = f1_score(predictions, y_val)
                val_scores.append(val_score)
            med_val_score =  np.median(val_scores)
            if self.val_score < med_val_score:
                self.model = model
                previous_score = self.val_score
                self.val_score = med_val_score
                print(f"Median cross validation score increased from {previous_score} to {self.val_score}"
                    f" with learning_rate={xgb_params['learning_rate']}, max_depth={xgb_params['max_depth']}, n_estimators={xgb_params['n_estimators']}. Saving model... ")
            if self.val_score == 1:
                break 
    
    def test(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        score = f1_score(predictions, y_test)
        return score

class logistic_regression(nn.Module):
    """ Model architecture for logistic regression"""
    def __init__(self, input_size):
        super(logistic_regression,self).__init__()
        self.fc1 = nn.Linear(input_size, 2)
        
    def forward(self,x):
        x = self.fc1(x)
        return x
    
class logistic_regression_trainer(Trainer):
    """Trainer using Logistic regression method. This trainer uses cross validation and grid search
    to find the best combination of following hyperparameters: learning rate,  n_epochs
    and alpha coefficient for L1 regularization (feature selection). """

    def __init__(self, input_size):
        self.input_size = input_size
        self.model = logistic_regression(input_size)
        self.batch_size = 20 # how many samples per batch to load
        self.criterion = nn.CrossEntropyLoss() # specify loss function (categorical cross-entropy)
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr = 0.01) # specify optimizer (stochastic gradient descent) and learning rate
        self.n_epochs = 100
        self.batch_size = 30
        self.val_score = - np.inf

        hyperparameters = {
            'learning_rate': [0.1, 0.01],
            'n_epochs': [100, 150],
            "reg_alpha": [0, 0.1, 0.01, 0.001]
        }

        # all combinations of hyperparameters
        self.param_combinations = list(product(*hyperparameters.values()))



    def train(self, X_train, y_train, cross_val_split):
        # train the model
        for params in self.param_combinations:
            model = logistic_regression(self.input_size)
            alpha_L1 = params[2]
            self.n_epochs = params[1]
            lr = params[0]
            self.optimizer = torch.optim.SGD(model.parameters(),lr = lr, weight_decay=alpha_L1)
            val_scores = []
            for train_index, val_index in cross_val_split:
                # Split data into training and validation sets using the indices obtained from kf.split
                X, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
                y, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
                # create train dataset and train loader
                features_tensor = torch.tensor(X.values, dtype=torch.float32)
                labels_tensor = torch.tensor(y.values, dtype=torch.long)
                train_data = TensorDataset(features_tensor, labels_tensor)
                train_loader = DataLoader(train_data, batch_size = self.batch_size)
                for epoch in range(self.n_epochs):
                    train_loss = 0 # monitor losses
                    
                    # train the model
                    model.train() # prep model for training
                    for data, label in train_loader:
                        self.optimizer.zero_grad() # clear the gradients of all optimized variables
                        output = model(data) # forward pass: compute predicted outputs by passing inputs to the model
                        loss = self.criterion(output, label) # calculate the loss
                        loss.backward() # backward pass: compute gradient of the loss with respect to model parameters
                        self.optimizer.step() # perform a single optimization step (parameter update)
                        train_loss += loss.item() * data.size(0) # update running training loss
                    train_loss /= len(train_loader.sampler)   
                
                features_tensor_val = torch.tensor(X_val.values, dtype=torch.float32)
                labels_tensor_val = torch.tensor(y_val.values, dtype=torch.long)
                val_data = TensorDataset(features_tensor_val, labels_tensor_val)
                val_loader = DataLoader(val_data, batch_size = self.batch_size)
                # validating the model
                model.eval()
                true_labels = []
                predicted_labels = []
                for data, label in val_loader:
                    with torch.no_grad():
                        outputs = model(data) # forward pass: compute predicted outputs by passing inputs to the model
                    # Convert outputs to predicted labels
                    _, preds = torch.max(outputs, 1)
                    true_labels.extend(label.tolist())
                    predicted_labels.extend(preds.tolist())
                val_score = f1_score(true_labels, predicted_labels, average='weighted')
                val_scores.append(val_score)
            med_val_score = np.median(val_scores)
            if self.val_score < med_val_score:
                self.model = model
                previous_score = self.val_score
                self.val_score = med_val_score
                print(f"Median cross validation score increased from {previous_score} to {self.val_score} with learning_rate={lr}, n_epochs={self.n_epochs}, L1_regularization_coeff={alpha_L1}. Saving model... ")
            if self.val_score == 1:
                    break 

    def test(self, X_test, y_test):
        features_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        labels_tensor = torch.tensor(y_test.values, dtype=torch.long)
        test_data = TensorDataset(features_tensor, labels_tensor)
        test_loader = DataLoader(test_data, batch_size = self.batch_size)
        true_labels = []
        predicted_labels = []
        # initialize lists to monitor test loss and accuracy
        self.model.eval() # prep model for evaluation
        for data, label in test_loader:
            with torch.no_grad():
                outputs = self.model(data) # forward pass: compute predicted outputs by passing inputs to the model
            # Convert outputs to predicted labels
            _, preds = torch.max(outputs, 1)
            true_labels.extend(label.tolist())
            predicted_labels.extend(preds.tolist())
        score = f1_score(true_labels, predicted_labels, average='weighted')
        return score

class MLP_model(nn.Module):
    """ MLP (multi layer perceptron) model architecture """
    def __init__(self, input_size):
        super(MLP_model,self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x) 
        x = self.relu(x)
        x = self.fc3(x)
        return x
            
class MLP_trainer(Trainer):
    """Trainer using MLP (multi layer perceptron) method. This trainer uses cross validation and grid search
    to find the best combination of following hyperparameters: learning rate,  n_epochs
    and alpha coefficient for L1 regularization (feature selection). """
    def __init__(self, input_size):
        self.input_size = input_size
        self.model = MLP_model(input_size)
        self.batch_size = 20 # how many samples per batch to load
        self.criterion = nn.CrossEntropyLoss() # specify loss function (categorical cross-entropy)
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr = 0.01) # specify optimizer (stochastic gradient descent) and learning rate
        self.n_epochs = 150
        self.batch_size = 30
        self.val_score = - np.inf

        hyperparameters = {
            'learning_rate': [0.1, 0.01],
            'n_epochs': [100, 150],
            "reg_alpha": [0, 0.1, 0.01, 0.001]
        }

        # all combinations of hyperparameters
        self.param_combinations = list(product(*hyperparameters.values()))


    def train(self, X_train, y_train, cross_val_split):
        # train the model
        for params in self.param_combinations:
            model = MLP_model(self.input_size)
            alpha_L1 = params[2]
            self.n_epochs = params[1]
            lr = params[0]
            self.optimizer = torch.optim.SGD(model.parameters(),lr = lr, weight_decay=alpha_L1)
            val_scores = []
            for train_index, val_index in cross_val_split:
                # Split data into training and validation sets using the indices obtained from kf.split
                X, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
                y, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
                # create train dataset and train loader
                features_tensor = torch.tensor(X.values, dtype=torch.float32)
                labels_tensor = torch.tensor(y.values, dtype=torch.long)
                train_data = TensorDataset(features_tensor, labels_tensor)
                train_loader = DataLoader(train_data, batch_size = self.batch_size)
                for epoch in range(self.n_epochs):
                    train_loss = 0 # monitor losses
                    
                    # train the model
                    model.train() # prep model for training
                    for data, label in train_loader:
                        self.optimizer.zero_grad() # clear the gradients of all optimized variables
                        output = model(data) # forward pass: compute predicted outputs by passing inputs to the model
                        loss = self.criterion(output, label) # calculate the loss
                        loss.backward() # backward pass: compute gradient of the loss with respect to model parameters
                        self.optimizer.step() # perform a single optimization step (parameter update)
                        train_loss += loss.item() * data.size(0) # update running training loss
                    train_loss /= len(train_loader.sampler)
                    
                
                features_tensor_val = torch.tensor(X_val.values, dtype=torch.float32)
                labels_tensor_val = torch.tensor(y_val.values, dtype=torch.long)
                val_data = TensorDataset(features_tensor_val, labels_tensor_val)
                val_loader = DataLoader(val_data, batch_size = self.batch_size)
                # validating the model
                model.eval()
                true_labels = []
                predicted_labels = []
                for data, label in val_loader:
                    with torch.no_grad():
                        outputs = model(data) # forward pass: compute predicted outputs by passing inputs to the model
                    # Convert outputs to predicted labels
                    _, preds = torch.max(outputs, 1)
                    true_labels.extend(label.tolist())
                    predicted_labels.extend(preds.tolist())
                val_score = f1_score(true_labels, predicted_labels, average='weighted')
                val_scores.append(val_score)
            med_val_score = np.median(val_scores)
            if self.val_score < med_val_score:
                self.model = model
                previous_score = self.val_score
                self.val_score = med_val_score
                print(f"Median cross validation score increased from {previous_score} to {self.val_score} with learning_rate={lr}, n_epochs={self.n_epochs}, L1_regularization_coeff={alpha_L1}. Saving model... ")
            if self.val_score == 1:
                break 
    def test(self, X_test, y_test):
        features_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        labels_tensor = torch.tensor(y_test.values, dtype=torch.long)
        test_data = TensorDataset(features_tensor, labels_tensor)
        test_loader = DataLoader(test_data, batch_size = self.batch_size)
        true_labels = []
        predicted_labels = []
        # initialize lists to monitor test loss and accuracy
        self.model.eval() # prep model for evaluation
        for data, label in test_loader:
            with torch.no_grad():
                outputs = self.model(data) # forward pass: compute predicted outputs by passing inputs to the model
            # Convert outputs to predicted labels
            _, preds = torch.max(outputs, 1)
            true_labels.extend(label.tolist())
            predicted_labels.extend(preds.tolist())
        score = f1_score(true_labels, predicted_labels, average='weighted')
        return score