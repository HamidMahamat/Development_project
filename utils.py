import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

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
            # Verify if its categorical or take any float values
            
            # Categorical float value
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
            

def one_hot_encoding(df, target, columns=[]):
    """Apply one hot encoding on categorical columns of df.

    Args:
        df (DataFrame): data
        columns (list): name of columns to be one hot encoded
        target (string): name of target column. If target is not present in the df, specify None.

    Returns:
        DataFrame: new one hot encoded dataframe
    """
    if target is not None:
        df = df.drop(target, axis=1)
    if columns == []:
        encoded_columns = pd.get_dummies(df)
    else:
        encoded_columns = pd.get_dummies(df, columns=columns)
    df = df.drop(encoded_columns.columns.tolist(), axis=1)
    df_encoded = pd.concat([df, encoded_columns], axis=1)
    return df_encoded
    
