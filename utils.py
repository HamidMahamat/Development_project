import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

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
    # Define the pattern for weird characters using regular expressions
    cleaned_text = text
    cleaned_text = cleaned_text.strip()
    cleaned_text = cleaned_text.replace('\t', "")
    cleaned_text = cleaned_text.replace(" ", "")
    if '\t' in cleaned_text:
        print(cleaned_text)
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
    columns = df.columns

    new_df = copy.deepcopy(df)
    non_int_columns=[]
    

    for column in df.columns:
        new_df[column] = df[column].apply(lambda x: remove_weird_characters(str(x)))
        try:
            new_df[column] = new_df[column].astype(int)
        except ValueError:
            non_int_columns.append(column)
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
    # converting 
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
        if df[column].dtype != "float":
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
    
