import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
4- Yassine:splitting data
5- visualizing balance between classes
6_ normalization
7- models (classes with same methods?)
8 - training (different functions for different models?)
9- testing (score?)
"""

def deal_with_NA_values(df, column, r):
    """This function deals with NA values:
    - if the number of NA values is high (according to ratio r) create a new category "unknown"
    - else: replace by the majoritary category

    Args:
        df (_DataFrame_): data
        column (_string_): name of the column
        r (_float_): ratio between the number of NA values and the number of values in minority category

    Returns:
        _type_: _description_
    """

    categories_counts = df[column].value_counts(dropna=True)
    
    return categories_counts


def one_hot_encoding(df):
    """Apply one hot encoding on categorical columns of df.

    Args:
        df (DataFrame): data

    Returns:
        DataFrame: new one hot encoded dataframe
    """
    categorical_columns = df.select_dtypes(include=['object'])
    return categorical_columns