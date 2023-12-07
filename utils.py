import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import re
import scipy
import math
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


"""
1 - Hamid: enlever NA values pour chaque colonne (arguments, le nom de la colonne, p, )
-> compter le nombre de catégories
-> si le nombre de Nan <p*min(N autres catégories) avec p petit, alors remplacer par catégorie makoritaire
-> sinon mettre une nouvelle catégorie 
2- Yassine: one-hot encoding for categories
-> take only columns with string elements
-> apply get dummy
-> remove original columns and concat with new ones
-> return the new df
3-Hamid: PCA
4- Yassine: splitting data
5- Yassine: visualizing balance between classes
6_ Yassine: normalization
7- models (classes with same methods?)
8 - training (different functions for different models?)
9- testing (score?)
"""

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
            

def depep_cleaning_visualization(df):
    columns = df.columns
    
    for column in columns :
        if df[column].dtype == 'O' :
            print(df[column].value_counts(dropna=True))
        else :
            pass
            

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

def handle_skewness(df):
    """handle very skewed features

    Args:
        df (Sataframe): dataset

    Returns:
        Dataframe: non skewed dataset
    """
    non_skewed_df = copy.deepcopy(df)
    for column in non_skewed_df.columns:
        if non_skewed_df[column].dtype != "O" and non_skewed_df[column].value_counts().size>2:
            skew= scipy.stats.skew(non_skewed_df[column], axis=0)
            if skew>=1 or skew<=-1:
                non_skewed_df[column] = non_skewed_df[column].apply(math.log1p)
    return non_skewed_df
    
